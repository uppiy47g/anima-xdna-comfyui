"""Model-free npu2 QK/scale experiment; never changes the production runtime."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import triton
import triton.language as tl
from triton.backends.amd_triton_npu.config import config_context
from triton.backends.amd_triton_npu.driver import NPUDriver, detect_npu_version
from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform
from triton.backends.amd_triton_npu.multilaunch import NPUChain

from examples.qk_scale_model_probe import (
    AVAILABLE_MODES as MODES, BF16_OUTPUT_MODES, finish_qk_output,
)


@triton.jit
def qk_kernel(
    Q, K, C,
    N: tl.constexpr,
    HEAD: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
):
    rows = tl.program_id(0) * 256 + tl.arange(0, 256)
    columns = tl.program_id(1) * 256 + tl.arange(0, 256)
    inner = tl.arange(0, 256)
    q = tl.load(Q + HEAD * 1024 * 256 + rows[:, None] * 256 + inner[None, :])
    k = tl.load(K + HEAD * 256 * N + inner[:, None] * N + columns[None, :])
    scores = tl.dot(q, k)
    if MODE == "scale_fp32" or MODE == "direct_bf16":
        scores = scores * SCALE
    elif MODE == "legacy_bf16":
        # Match the existing BF16 QK output, then BF16 host scaling boundary.
        scores = (scores.to(tl.bfloat16).to(tl.float32) * SCALE).to(tl.bfloat16)
    tl.store(C + HEAD * 1024 * N + rows[:, None] * N + columns[None, :], scores)


def errors(actual, expected):
    delta = actual.float() - expected.float()
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "normalized_rms": (delta.norm() / expected.float().norm()).item(),
        "bitwise_equal": torch.equal(actual, expected),
        "finite": bool(torch.isfinite(actual).all()),
    }


def make_inputs(keys, heads, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.zeros(heads, 1024, 256, dtype=torch.bfloat16)
    k = torch.zeros(heads, 256, keys, dtype=torch.bfloat16)
    q[:, :, :128] = torch.randn(heads, 1024, 128, generator=generator)
    k[:, :128, :] = torch.randn(heads, 128, keys, generator=generator)
    return q, k


def bf16_numpy(tensor):
    import ml_dtypes

    return tensor.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)


def run_variant(q, k, mode, script, work_dir, warmups, runs):
    heads, _, keys = k.shape
    bf16_output = mode in BF16_OUTPUT_MODES
    dtype = np.dtype(np.float32)
    if bf16_output:
        import ml_dtypes

        dtype = np.dtype(ml_dtypes.bfloat16)
    output = np.empty((heads, 1024, keys), dtype=dtype)
    template = torch.empty(
        heads, 1024, keys,
        dtype=torch.bfloat16 if bf16_output else torch.float32,
    )
    chain = NPUChain(
        f"anima_qk_probe_{heads}_{keys}_{mode}",
        air_project_path=str(work_dir / "air_project"),
    )
    try:
        for head in range(heads):
            chain.add(
                qk_kernel,
                grid=(4, keys // 256),
                arg_map={0: 0, 1: 1, 2: 2},
                args=(q, k, template),
                constexprs={
                    "N": keys, "HEAD": head, "MODE": mode, "SCALE": 128**-0.5,
                },
                transform_script=str(script),
            )
        arrays = [bf16_numpy(q), bf16_numpy(k), output]

        def execute():
            started = time.perf_counter()
            got = chain.run(
                arrays,
                bo_key="shape",
                intermediate_indices={2},
                output_indices={2},
            )[2]
            chain_ms = (time.perf_counter() - started) * 1000
            result, phases, unscaled = finish_qk_output(got, mode)
            phases["chain_ms"] = chain_ms
            return result, phases, unscaled

        started = time.perf_counter()
        result, _, _ = execute()
        first_ms = (time.perf_counter() - started) * 1000
        print(f"{keys=} {mode}: compile/first {first_ms:.1f} ms", flush=True)
        for _ in range(warmups):
            execute()
        samples = []
        phase_samples = []
        for _ in range(runs):
            started = time.perf_counter()
            result, phases, unscaled = execute()
            samples.append((time.perf_counter() - started) * 1000)
            phase_samples.append(phases)
        return result.clone(), {
            "compile_and_first_ms": first_ms,
            "warm_ms": samples,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "phase_samples_ms": phase_samples,
            "phase_medians_ms": {
                name: statistics.median(sample[name] for sample in phase_samples)
                for name in phase_samples[0]
            },
            "unscaled_bf16_sha256": (
                hashlib.sha256(unscaled.view(torch.uint16).numpy().tobytes()).hexdigest()
                if unscaled is not None else None
            ),
            "dispatches_per_call": 1,
            "h2d_bytes_per_call": q.numel() * 2 + k.numel() * 2,
            "d2h_visible_bytes_per_call": output.nbytes,
        }
    finally:
        chain.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, choices=(1, 16), default=16)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--modes", nargs="+", choices=MODES,
        default=("baseline", "scale_fp32", "legacy_bf16"),
    )
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0 or args.threads < 0:
        parser.error("runs must be positive; warmups and threads must be non-negative")
    if args.modes[0] != "baseline" or len(set(args.modes)) != len(args.modes):
        parser.error("modes must begin with baseline and contain no duplicates")
    if "epilogue" not in inspect.signature(generate_matmul_transform).parameters:
        raise RuntimeError("Triton-XDNA #171 epilogue generator is required")
    if args.threads:
        torch.set_num_threads(args.threads)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    scripts = {}
    for epilogue in (False, True):
        path = work_dir / ("epilogue.mlir" if epilogue else "plain.mlir")
        script = generate_matmul_transform(
            l1_m=64, l1_n=64, l2_k=64, epilogue=epilogue,
        )
        path.write_text(
            script.encode("ascii", errors="replace").decode("ascii"),
            encoding="ascii",
        )
        scripts[epilogue] = path
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "input": "seeded synthetic BF16 Q/K, unit-scale normal, no model weights",
        "heads": args.heads,
        "head_dim": 128,
        "padded_head_dim": 256,
        "seed": args.seed,
        "warmups": args.warmups,
        "runs": args.runs,
        "modes": list(args.modes),
        "torch_threads": torch.get_num_threads(),
        "timing_scope": (
            "NPUChain input staging, dispatch/wait, output sync, and host "
            "BF16 conversion/scaling; excludes input generation/padding, "
            "softmax, AV, and complete image generation"
        ),
        "phase_semantics": {
            "chain_ms": "NPUChain input copy, sync, dispatch/wait and output readback; not pure compute.",
            "host_view_ms": "NumPy-to-Torch wrapper and BF16 bit reinterpretation, without numerical conversion.",
            "host_cast_ms": "Host numerical FP32-to-BF16 conversion; exactly zero when output is already BF16.",
            "host_scale_ms": "Remaining BF16 host scale for baseline and QK-only BF16 output.",
            "rounding": "qk_bf16 keeps CPU scaling; direct_bf16 requests accumulator*scale then BF16 store without an explicit pre-scale BF16 cast.",
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch", "triton-xdna", "mlir-air", "mlir-aie-no-rtti", "llvm-aie",
            )
        },
        "generator_sha256": hashlib.sha256(
            Path(inspect.getfile(generate_matmul_transform)).read_bytes()
        ).hexdigest(),
        "transform_library_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                (Path(inspect.getfile(generate_matmul_transform)).parent
                 / "transform_library").glob("*.mlir")
            )
        },
        "cpu_elementwise_gate": {"rtol": 0.02, "atol": 0.03},
        "cases": [],
    }
    report_path = work_dir / "results.json"

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    previous_cwd = Path.cwd()
    previous_driver = triton.runtime.driver._active
    try:
        target = detect_npu_version(runtime="xrt")
        if target != "npu2":
            raise RuntimeError(f"npu2 required, received {target}")
        report["target"] = target
        triton.runtime.driver.set_active(NPUDriver("xrt"))
        os.chdir(work_dir)
        with config_context(
            air_project_path=str(work_dir / "air_project"),
            transform_tiling_script=str(scripts[False]),
        ):
            for keys in (1024, 512):
                q, k = make_inputs(keys, args.heads, args.seed)
                oracle = torch.bmm(q.float(), k.float())
                legacy_oracle = oracle.to(torch.bfloat16) * (128**-0.5)
                direct_oracle = (oracle * (128**-0.5)).to(torch.bfloat16)
                baseline = None
                baseline_unscaled_hash = None
                for mode in args.modes:
                    result, timing = run_variant(
                        q, k, mode, scripts[mode != "baseline"], work_dir,
                        args.warmups, args.runs,
                    )
                    if baseline is None:
                        baseline = result
                        baseline_unscaled_hash = timing["unscaled_bf16_sha256"]
                    reference = (
                        direct_oracle if mode in ("scale_fp32", "direct_bf16")
                        else legacy_oracle
                    )
                    case = {
                        "attention": "self" if keys == 1024 else "cross",
                        "q_shape": [args.heads, 1024, 128],
                        "k_shape": [args.heads, keys, 128],
                        "mode": mode,
                        **timing,
                        "vs_cpu_matching_rounding": errors(result, reference),
                        "vs_unfused_control": errors(result, baseline),
                        "vs_cpu_legacy_rounding": errors(result, legacy_oracle),
                        "vs_cpu_direct_rounding": errors(result, direct_oracle),
                        "cpu_rounding_order_difference": errors(direct_oracle, legacy_oracle),
                        "unscaled_bf16_equal_to_control": (
                            timing["unscaled_bf16_sha256"] == baseline_unscaled_hash
                            if timing["unscaled_bf16_sha256"] is not None else None
                        ),
                    }
                    try:
                        torch.testing.assert_close(
                            result, reference, rtol=0.02, atol=0.03,
                        )
                    except AssertionError as error:
                        case["cpu_gate_passed"] = False
                        case["cpu_gate_error"] = str(error)
                    else:
                        case["cpu_gate_passed"] = True
                    report["cases"].append(case)
                    save()
                    print(json.dumps(case), flush=True)
        report["status"] = (
            "passed" if all(case["cpu_gate_passed"] for case in report["cases"])
            else "completed_with_cpu_gate_failures"
        )
        report["execution_completed"] = True
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        os.chdir(previous_cwd)
        triton.runtime.driver.set_active(previous_driver)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
