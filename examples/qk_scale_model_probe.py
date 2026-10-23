"""Isolated, fixed-seed ComfyUI experiment for the QK/scale epilogue."""

import argparse
import ast
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap
import time


MODES = ("baseline", "scale_fp32", "legacy_bf16")
AVAILABLE_MODES = ("baseline", "qk_bf16", "scale_fp32", "direct_bf16", "legacy_bf16")
BF16_OUTPUT_MODES = ("qk_bf16", "direct_bf16", "legacy_bf16")
POSITIVE = "a cheerful anime girl under cherry blossoms, detailed illustration, soft daylight"
NEGATIVE = "worst quality, low quality, blurry, jpeg artifacts"


def finish_qk_output(got, mode):
    import numpy as np
    import torch

    if mode not in AVAILABLE_MODES:
        raise ValueError(f"unknown output mode: {mode}")
    started = time.perf_counter()
    if mode in BF16_OUTPUT_MODES:
        result = torch.from_numpy(got.view(np.uint16)).view(torch.bfloat16)
    else:
        result = torch.from_numpy(got)
    phases = {"host_view_ms": (time.perf_counter() - started) * 1000}
    phases["host_cast_ms"] = 0.0
    if mode not in BF16_OUTPUT_MODES:
        started = time.perf_counter()
        result = result.to(torch.bfloat16)
        phases["host_cast_ms"] = (time.perf_counter() - started) * 1000
    unscaled = None
    phases["host_scale_ms"] = 0.0
    if mode in ("baseline", "qk_bf16"):
        unscaled = result
        started = time.perf_counter()
        result = result * (128**-0.5)
        phases["host_scale_ms"] = (time.perf_counter() - started) * 1000
    return result, phases, unscaled


def tensor_error(actual, expected):
    import torch

    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape:
        raise ValueError(f"comparison shape mismatch: {actual.shape} != {expected.shape}")
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise ValueError("non-finite comparison tensor")
    delta = actual - expected
    reference_norm = expected.norm().item()
    if reference_norm == 0:
        raise ValueError("zero-norm comparison reference")
    return {
        "max_abs": delta.abs().max().item(),
        "mean_abs": delta.abs().mean().item(),
        "normalized_rms": delta.norm().item() / reference_norm,
        "bitwise_equal": torch.equal(actual, expected),
    }


def decoded_image_rgb(image):
    import torch

    if image.ndim == 5 and image.shape[1] == 1:
        image = image[:, 0]
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
        raise ValueError(f"expected one decoded RGB frame, got {tuple(image.shape)}")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("non-finite decoded image")
    return (image[0].float().cpu().clamp(0, 1).numpy() * 255).astype("uint8")


def profile_summary(steps):
    fields = (
        "wall_ms", "dispatches", "h2d_bytes", "d2h_bytes", "host_copy_ms",
        "h2d_sync_ms", "kernel_sync_ms", "d2h_sync_ms", "weight_population_bytes",
    )
    groups = defaultdict(lambda: {field: 0 for field in fields})
    stages = defaultdict(lambda: {field: 0 for field in fields})
    low = defaultdict(lambda: defaultdict(float))
    for step in steps:
        for block in step["blocks"]:
            for stage in block["stages"]:
                name = stage["name"]
                group = (
                    "self_attention" if name.startswith("attn1.") else
                    "cross_attention" if name.startswith("attn2.") else
                    "mlp" if name.startswith("ff.") else
                    "adaln" if name.startswith(("norm1.", "norm2.", "norm3.")) else
                    "other"
                )
                for field in fields:
                    groups[group][field] += stage[field]
                    stages[name][field] += stage[field]
            for name, values in block["low_level"].items():
                for field, value in values.items():
                    low[name][field] += value
    return {
        "groups": dict(groups),
        "stages": dict(stages),
        "xrt_by_stage": dict(low),
        "dispatches": sum(v["dispatches"] for v in stages.values()),
        "instrumented_sync_bytes": {
            key: sum(v.get(key, 0) for v in low.values())
            for key in ("h2d_sync_bytes", "d2h_sync_bytes")
        },
    }


class Probe:
    def __init__(self, work_dir, air_manifests=None):
        self.work_dir = work_dir
        self.mode = "baseline"
        self.stage = "outside_stage"
        self.enabled = False
        self.capture = False
        self.replay = False
        self.step = 0
        self.steps = []
        self.low = defaultdict(lambda: defaultdict(float))
        self.chains = {}
        self.air_attention = {}
        self.mutations = []
        if air_manifests:
            from examples.air_attention import AIRAttention
            try:
                for keys, path in air_manifests.items():
                    self.air_attention[keys] = AIRAttention(path)
            except Exception:
                self.restore()
                raise

    def patch(self, obj, name, replacement):
        self.mutations.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def restore(self):
        for obj, name, original in reversed(self.mutations):
            setattr(obj, name, original)
        for chain in self.chains.values():
            chain.close()
        for attention in self.air_attention.values():
            attention.close()

    def timed(self, category, function, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            values = self.low[self.stage]
            values[category] += (time.perf_counter() - started) * 1000
            values[category + "_calls"] += 1
            if category in ("h2d_sync_ms", "d2h_sync_ms"):
                byte_key = category.replace("_ms", "_bytes")
                values[byte_key] += function.__self__.size()

    def instrument_xrt(self, cls):
        original = cls.run
        source = textwrap.dedent(inspect.getsource(original))
        tree = ast.parse(source)

        class Timer(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                function = node.func
                if not isinstance(function, ast.Attribute):
                    return node
                category = None
                if function.attr == "sync":
                    direction = ast.unparse(node.args[0])
                    category = (
                        "h2d_sync_ms" if direction.endswith("TO_DEVICE")
                        else "d2h_sync_ms"
                    )
                elif function.attr == "copyto":
                    category = "host_copy_ms"
                elif function.attr == "wait2":
                    category = "xrt_wait_ms"
                elif function.attr == "start":
                    category = "xrt_start_ms"
                if category is None:
                    return node
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_probe_time_call", ctx=ast.Load()),
                        args=[ast.Constant(category), function, *node.args],
                        keywords=node.keywords,
                    ),
                    node,
                )

        tree = ast.fix_missing_locations(Timer().visit(tree))
        namespace = {**original.__globals__, "_probe_time_call": self.timed}
        exec(compile(tree, "<qk-probe-xrt-instrumentation>", "exec"), namespace)
        self.patch(cls, "run", namespace["run"])

    def fused_qk(self, session, query, key, operation_key):
        import numpy as np
        import torch
        import ml_dtypes
        from triton.backends.amd_triton_npu.multilaunch import NPUChain
        from examples.qk_scale_epilogue_probe import qk_kernel, bf16_numpy
        from anima_xdna_poc.xdna import DispatchResult

        heads, rows, inner = query.shape
        columns = key.shape[-1]
        if (heads, rows, inner, columns) not in (
            (16, 1024, 128, 1024), (16, 1024, 128, 512),
        ):
            raise ValueError(f"unexpected real QK shape: {query.shape}, {key.shape}")
        q = torch.zeros(heads, rows, 256, dtype=torch.bfloat16)
        k = torch.zeros(heads, 256, columns, dtype=torch.bfloat16)
        q[:, :, :inner] = query
        k[:, :inner, :] = key
        signature = (self.mode, columns)
        chain = self.chains.get(signature)
        bf16_output = self.mode in BF16_OUTPUT_MODES
        if chain is None:
            chain = NPUChain(
                f"anima_qk_probe_{heads}_{columns}_{self.mode}",
                air_project_path=str(self.work_dir / "fused_air_project"),
            )
            template = torch.empty(
                heads, rows, columns,
                dtype=torch.bfloat16 if bf16_output else torch.float32,
            )
            for head in range(heads):
                chain.add(
                    qk_kernel, grid=(4, columns // 256),
                    arg_map={0: 0, 1: 1, 2: 2}, args=(q, k, template),
                    constexprs={
                        "N": columns, "HEAD": head, "MODE": self.mode,
                        "SCALE": 128**-0.5,
                    },
                    transform_script=str(self.work_dir / "qk_epilogue.mlir"),
                )
            self.chains[signature] = chain
        output = np.empty(
            (heads, rows, columns),
            dtype=ml_dtypes.bfloat16 if bf16_output else np.float32,
        )
        started = time.perf_counter()
        got = chain.run(
            [bf16_numpy(q), bf16_numpy(k), output], bo_key="shape",
            intermediate_indices={2}, output_indices={2},
        )[2]
        elapsed = (time.perf_counter() - started) * 1000
        if bf16_output:
            logical = torch.from_numpy(got.view(np.uint16)).view(torch.bfloat16)
            output_bf16 = logical
        else:
            logical = torch.from_numpy(got)
            output_bf16 = logical.to(torch.bfloat16)
        return DispatchResult(
            output_bf16, logical, elapsed, None,
            q.numel() * 2 + k.numel() * 2, output.nbytes,
            resident_hit=True, kernel_sync_ms=elapsed,
        )

    def install(self):
        import anima_xdna_poc.block as block_module
        import anima_xdna_poc.chain as chain_module
        from anima_xdna_poc.xdna import XDNASession, _ResidentKernelRunner
        from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner

        self.instrument_xrt(MultiLaunchRunner)
        self.instrument_xrt(_ResidentKernelRunner)
        execution = block_module._BlockExecution
        old_host = execution.host

        def fused_host(execution_self, name, operation):
            if execution_self.runner_device != "xdna2":
                return old_host(execution_self, name, operation)
            if self.mode == "air_attention":
                if name.endswith(".mask"):
                    raise ValueError("AIR attention prototype does not support masks")
                if name.endswith((".qk_scale", ".softmax")):
                    scores = inspect.getclosurevars(operation).nonlocals["scores"]
                    return old_host(execution_self, name, lambda: scores)
            if self.mode in ("scale_fp32", "direct_bf16", "legacy_bf16") and name.endswith(".qk_scale"):
                captured = inspect.getclosurevars(operation).nonlocals
                return old_host(execution_self, name, lambda: captured["scores"])
            return old_host(execution_self, name, operation)

        self.patch(execution, "host", fused_host)
        for method_name in ("linear", "batched", "qkv"):
            original = getattr(execution, method_name)

            def scoped(execution_self, name, *args, _original=original,
                       _kind=method_name, **kwargs):
                previous = self.stage
                self.stage = name + ".qkv_chain" if _kind == "qkv" else name
                try:
                    return _original(execution_self, name, *args, **kwargs)
                finally:
                    self.stage = previous

            self.patch(execution, method_name, scoped)
        if self.air_attention:
            original_batched = execution.batched
            pending = {}

            def attention_batched(execution_self, name, inputs, weights):
                from anima_xdna_poc.block import StageMetric
                import torch

                if execution_self.runner_device != "xdna2":
                    return original_batched(execution_self, name, inputs, weights)
                key = (id(execution_self), name.split(".")[0])
                if name.endswith(".qk_batched"):
                    pending[key] = (inputs, weights.transpose(1, 2))
                    if self.mode == "air_attention":
                        # Only a handle to Q crosses the skipped host stages.
                        # No score/probability-shaped host tensor is created.
                        return inputs
                if name.endswith(".av_batched") and self.mode == "air_attention":
                    query, key_tensor = pending.pop(key)
                    attention = self.air_attention[key_tensor.shape[1]]
                    previous = self.stage
                    self.stage = name.replace(".av_batched", ".air_attention")
                    started = time.perf_counter()
                    try:
                        output = attention.run(query, key_tensor, weights)
                    finally:
                        elapsed = (time.perf_counter() - started) * 1000
                        self.stage = previous
                    shape = attention.shape
                    execution_self.metrics.append(StageMetric(
                        name.replace(".av_batched", ".air_attention"), "xdna2",
                        elapsed, 1,
                        4 * shape.heads * shape.queries * shape.keys * shape.depth,
                        shape.h2d_bytes + shape.d2h_bytes,
                        shape.h2d_bytes, shape.d2h_bytes,
                    ))
                    return output
                output = original_batched(execution_self, name, inputs, weights)
                if name.endswith(".av_batched"):
                    query, key_tensor = pending.pop(key)
                    block = int(execution_self.weights.prefix.split(".")[-2])
                    if (
                        self.capture and self.mode == "baseline"
                        and self.step - 1 in (0, 3) and block in (0, 13, 27)
                    ):
                        directory = self.work_dir / "attention_inputs"
                        directory.mkdir(exist_ok=True)
                        torch.save({
                            "query": query.clone(), "key": key_tensor.clone(),
                            "value": weights.clone(), "control_output": output.clone(),
                            "step": self.step - 1, "block": block, "attention": key[1],
                        }, directory / f"step{self.step - 1}_block{block}_{key[1]}.pt")
                return output

            self.patch(execution, "batched", attention_batched)
        original_head = XDNASession.dispatch_head_chain

        def head(session, query, key, operation_key):
            if self.mode != "baseline" and operation_key.endswith(".qk_batched"):
                return self.fused_qk(session, query, key, operation_key)
            return original_head(session, query, key, operation_key)

        self.patch(XDNASession, "dispatch_head_chain", head)
        original_block = chain_module.run_xdna_block

        def block(*args, **kwargs):
            self.low = defaultdict(lambda: defaultdict(float))
            started = time.perf_counter()
            result = original_block(*args, **kwargs)
            wall_ms = (time.perf_counter() - started) * 1000
            if self.enabled:
                self.steps[-1]["blocks"].append({
                    "index": kwargs["block_index"],
                    "wall_ms": wall_ms,
                    "stages": [asdict(metric) for metric in result.metrics],
                    "low_level": {k: dict(v) for k, v in self.low.items()},
                })
            return result

        self.patch(chain_module, "run_xdna_block", block)
        original_range = chain_module.AnimaXDNAChainRuntime.run_range

        def run_range(runtime, inputs, start=0, end=28):
            import torch

            index = self.step
            self.step += 1
            if self.enabled:
                self.steps.append({"step": index, "blocks": []})
            if self.capture and self.mode == "baseline" and not self.replay:
                torch.save(inputs, self.work_dir / f"input_{index}.pt")
            result = original_range(runtime, inputs, start, end)
            if self.enabled:
                self.steps[-1]["chain_wall_ms"] = result.wall_ms
            if self.capture:
                outputs = torch.stack([block.result.output for block in result.blocks])
                if self.mode == "baseline" and not self.replay:
                    torch.save(outputs, self.work_dir / f"baseline_blocks_{index}.pt")
                else:
                    baseline = torch.load(
                        self.work_dir / f"baseline_blocks_{index}.pt",
                        weights_only=True,
                    )
                    self.steps[-1]["vs_baseline"] = [
                        {"block": i, **tensor_error(output, baseline[i])}
                        for i, output in enumerate(outputs)
                    ]
            return result

        self.patch(chain_module.AnimaXDNAChainRuntime, "run_range", run_range)

    def reset(self, mode, capture=False, replay=False):
        self.mode, self.capture, self.replay = mode, capture, replay
        self.step, self.steps = 0, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=700755787614899)
    parser.add_argument(
        "--modes", nargs="+", choices=(*AVAILABLE_MODES, "air_attention"), default=MODES,
    )
    parser.add_argument("--air-self-manifest", type=Path)
    parser.add_argument("--air-cross-manifest", type=Path)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    if args.modes[0] != "baseline" or len(set(args.modes)) != len(args.modes):
        parser.error("modes must begin with baseline and contain no duplicates")
    modes = tuple(args.modes)
    if "air_attention" in modes and (
        args.air_self_manifest is None or args.air_cross_manifest is None
    ):
        parser.error("air_attention requires both AIR manifests")
    work_dir = args.work_dir.resolve()
    if (work_dir / "input_0.pt").exists():
        raise FileExistsError("Refusing to overwrite an existing real-input capture")
    work_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.comfy_root.resolve()))
    saved_argv = sys.argv
    sys.argv = [str(args.comfy_root / "main.py"), "--cpu", "--disable-all-custom-nodes"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args
    import torch
    # Match the validated allocator order without changing ComfyUI itself.
    os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    sys.argv = saved_argv
    import nodes
    import comfy.sd
    import comfy.sample
    import comfy.model_management as management
    import comfy.utils
    import folder_paths
    from PIL import Image
    from triton.backends.amd_triton_npu.matmul_transform import generate_matmul_transform
    from anima_xdna_poc.block import run_cpu_block
    from anima_xdna_poc.block_checkpoint import AnimaBlockWeights
    from anima_xdna_poc.checkpoint_schema import detect_checkpoint_schema
    from anima_xdna_poc.xdna import probe as hardware_probe
    from comfyui_xdna_nodes.model_wrapper import (
        LoadAnimaBF16, LoadAttachAnimaXDNAModel, ATTACHMENT_KEY,
    )

    ready = hardware_probe()
    print(ready, flush=True)
    script = generate_matmul_transform(l1_m=64, l1_n=64, l2_k=64, epilogue=True)
    (work_dir / "qk_epilogue.mlir").write_text(
        script.encode("ascii", errors="replace").decode("ascii"), encoding="ascii",
    )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "conditions": {
            "checkpoint": args.checkpoint.name, "text_encoder": args.text_encoder.name,
            "vae": args.vae.name, "seed": args.seed, "size": [512, 512],
            "steps": 4, "sampler": "euler", "scheduler": "simple", "cfg": 1,
            "denoise": 1, "batch": 1, "positive": POSITIVE, "negative": NEGATIVE,
            "qkv_chaining": True, "block_dispatch_baseline": 19,
            "modes": list(modes),
            "lora": False, "torch_threads": torch.get_num_threads(),
            "allocator_order": "MIMALLOC_PURGE_DELAY=0 after torch import",
        },
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "triton-xdna", "mlir-air", "mlir-aie-no-rtti", "llvm-aie")
        },
        "source": {
            "repository_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
            ).strip(),
            "comfy_commit": subprocess.check_output(
                ["git", "-C", str(args.comfy_root), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "epilogue_transform_sha256": hashlib.sha256(
                (work_dir / "qk_epilogue.mlir").read_bytes(),
            ).hexdigest(),
        },
        "counter_semantics": {
            "stage_wall": "Existing StageMetric spans; compilation and setup appear on first use.",
            "stage_kernel_sync": "Legacy field: NPUChain includes copies/sync; do not add to wall time.",
            "low_level": "Timed original pyxrt sync/start/wait2 and numpy.copyto calls via AST wrappers; includes their Python crossing overhead.",
            "sync_bytes": "Exact BO size per sync, cache-maintenance exposure, not discrete-device bus traffic.",
            "capture": "Contains tensor capture/comparison overhead; excluded from performance samples.",
            "timing": "Direct ComfyUI sampler, warm conditioning, VAE decode separate; not browser/API prompt wall.",
        },
        "cold": {}, "captures": [], "fixed_input_replays": [],
        "cpu_oracles": [], "timing_runs": [],
    }
    probe = Probe(
        work_dir,
        air_manifests=(
            {1024: args.air_self_manifest, 512: args.air_cross_manifest}
            if "air_attention" in modes else None
        ),
    )
    runtime = None

    def save():
        (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def sample(model, positive, negative, capture=False):
        latent = torch.zeros(1, 16, 1, 64, 64)
        noise = comfy.sample.prepare_noise(latent, args.seed)
        callbacks = []

        def callback(step, denoised, current, total):
            if capture:
                callbacks.append({
                    "step": step, "denoised": denoised.detach().cpu().clone(),
                    "current": current.detach().cpu().clone(),
                })

        started = time.perf_counter()
        result = comfy.sample.sample(
            model, noise, 4, 1.0, "euler", "simple", positive, negative, latent,
            denoise=1.0, callback=callback, disable_pbar=True, seed=args.seed,
        )
        wall_ms = (time.perf_counter() - started) * 1000
        if probe.step != 4 or (capture and len(callbacks) != 4):
            raise RuntimeError(f"Expected four real model evaluations, got {probe.step}")
        return result, callbacks, wall_ms

    try:
        with torch.inference_mode():
            started = time.perf_counter()
            clip = comfy.sd.load_clip(
                ckpt_paths=[str(args.text_encoder.resolve())],
                embedding_directory=[],
                clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
                model_options={"load_device": torch.device("cpu"), "offload_device": torch.device("cpu")},
            )
            positive = nodes.CLIPTextEncode().encode(clip, POSITIVE)[0]
            negative = nodes.CLIPTextEncode().encode(clip, NEGATIVE)[0]
            report["cold"]["text_load_encode_ms"] = (time.perf_counter() - started) * 1000
            management.unload_all_models()
            del clip
            gc.collect()
            folder_paths.add_model_folder_path("diffusion_models", str(args.checkpoint.parent))
            started = time.perf_counter()
            model = LoadAnimaBF16().load("diffusion_models:" + args.checkpoint.name)[0]
            wrapped, _ = LoadAttachAnimaXDNAModel().attach(
                model, str(args.checkpoint.resolve()), False,
                cache_dir=str(work_dir / "packed_cache"), qkv_chaining=True,
            )
            runtime = wrapped.get_attachment(ATTACHMENT_KEY).runtime
            report["cold"]["load_attach_ms"] = (time.perf_counter() - started) * 1000
            report["model_fingerprint"] = runtime.snapshot()["model_fingerprint"]
            probe.install()
            probe.enabled = True
            probe.reset("baseline", capture=True)
            baseline_latent, baseline_callbacks, wall = sample(wrapped, positive, negative, True)
            torch.save(baseline_latent, work_dir / "baseline_latent.pt")
            torch.save(baseline_callbacks, work_dir / "baseline_callbacks.pt")
            report["cold"]["baseline_sampler_ms"] = wall
            report["captures"].append({
                "mode": "baseline", "sampler_wall_ms": wall,
                "steps": probe.steps, "profile": profile_summary(probe.steps),
            })
            save()
            print("Baseline real four-step inputs captured", flush=True)
            # Warm fused artifacts/BOs independently of every measurement.
            first_inputs = torch.load(work_dir / "input_0.pt", weights_only=False)
            for mode in modes[1:]:
                probe.reset(mode)
                runtime.open().run_range(first_inputs)
                print(f"{mode} warm-up complete", flush=True)

            for mode in modes[1:]:
                probe.reset(mode, capture=True)
                latent, callbacks, wall = sample(wrapped, positive, negative, True)
                torch.save(latent, work_dir / f"{mode}_latent.pt")
                torch.save(callbacks, work_dir / f"{mode}_callbacks.pt")
                report["captures"].append({
                    "mode": mode, "sampler_wall_ms": wall,
                    "steps": probe.steps, "profile": profile_summary(probe.steps),
                    "final_latent_vs_baseline": tensor_error(latent, baseline_latent),
                    "callbacks_vs_baseline": [
                        {"step": i, "denoised": tensor_error(c["denoised"], baseline_callbacks[i]["denoised"]),
                         "current": tensor_error(c["current"], baseline_callbacks[i]["current"])}
                        for i, c in enumerate(callbacks)
                    ],
                })
                save()
                print(f"{mode} independent sampler captured", flush=True)

            for mode in modes[1:]:
                probe.reset(mode, capture=True, replay=True)
                for step in range(4):
                    inputs = torch.load(work_dir / f"input_{step}.pt", weights_only=False)
                    runtime.open().run_range(inputs)
                report["fixed_input_replays"].append({"mode": mode, "steps": probe.steps})
                save()
                print(f"{mode} same-input replay complete", flush=True)

            parameters = dict(model.model.diffusion_model.named_parameters())
            schema = detect_checkpoint_schema(parameters)
            for step in range(4):
                inputs = torch.load(work_dir / f"input_{step}.pt", weights_only=False)
                hidden = inputs.hidden_states
                baseline = torch.load(work_dir / f"baseline_blocks_{step}.pt", weights_only=True)
                cpu_outputs = []
                comparisons = []
                cpu_ms = []
                for index in range(28):
                    prefix = f"transformer_blocks.{index}."
                    tensors = {
                        key[len(prefix):]: parameters[source].detach()
                        for key, source in schema.canonical_to_source.items()
                        if key.startswith(prefix)
                    }
                    weights = AnimaBlockWeights(prefix, tensors)
                    block_inputs = runtime.open()._next_inputs(inputs, hidden)
                    started = time.perf_counter()
                    result = run_cpu_block(runtime.open().config, weights, block_inputs)
                    cpu_ms.append((time.perf_counter() - started) * 1000)
                    hidden = result.output
                    cpu_outputs.append(hidden)
                    comparisons.append({"block": index, **tensor_error(baseline[index], hidden)})
                torch.save(torch.stack(cpu_outputs), work_dir / f"cpu_blocks_{step}.pt")
                report["cpu_oracles"].append({
                    "step": step, "block_ms": cpu_ms, "baseline_vs_cpu": comparisons,
                })
                save()
                print(f"CPU oracle step {step} complete", flush=True)

            # Same-input gates against CPU isolate QK changes from sampler feedback.
            for replay in report["fixed_input_replays"]:
                mode = replay["mode"]
                probe.reset(mode)
                for step in range(4):
                    inputs = torch.load(work_dir / f"input_{step}.pt", weights_only=False)
                    chain = runtime.open().run_range(inputs)
                    cpu = torch.load(work_dir / f"cpu_blocks_{step}.pt", weights_only=True)
                    comparisons = [
                        {"block": i, **tensor_error(block.result.output, cpu[i])}
                        for i, block in enumerate(chain.blocks)
                    ]
                    replay["steps"][step]["vs_cpu"] = comparisons
                    replay["steps"][step]["turbo_cpu_gate"] = {
                        "max_intermediate_nrms": 0.10, "max_final_nrms": 0.02,
                        "passed": max(c["normalized_rms"] for c in comparisons) <= 0.10
                        and comparisons[-1]["normalized_rms"] <= 0.02,
                    }
                save()

            for repeat in range(args.runs):
                # Rotate order to vary each variant's measurement position.
                offset = repeat % len(modes)
                order = modes[offset:] + modes[:offset]
                for mode in order:
                    probe.reset(mode)
                    latent, _, wall = sample(wrapped, positive, negative)
                    item = {
                        "repeat": repeat, "mode": mode, "sampler_wall_ms": wall,
                        "steps": probe.steps, "profile": profile_summary(probe.steps),
                        "final_latent_vs_baseline": tensor_error(latent, baseline_latent),
                    }
                    report["timing_runs"].append(item)
                    save()
                    print(f"timed {repeat} {mode}: {wall / 1000:.3f} s", flush=True)

            # Decode the captured latents only after sampler profiling.
            state, metadata = comfy.utils.load_torch_file(str(args.vae.resolve()), return_metadata=True)
            vae = comfy.sd.VAE(sd=state, metadata=metadata)
            del state
            for mode in modes:
                latent = torch.load(work_dir / f"{mode}_latent.pt", weights_only=True)
                started = time.perf_counter()
                image = vae.decode(latent)
                decode_ms = (time.perf_counter() - started) * 1000
                rgb = decoded_image_rgb(image)
                Image.fromarray(rgb).save(work_dir / f"{mode}.png")
                if mode == "baseline":
                    baseline_rgb = rgb.copy()
                item = next(c for c in report["captures"] if c["mode"] == mode)
                item["vae_decode_ms"] = decode_ms
                item["image_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
                item["image_rgb_mae_vs_baseline"] = float(
                    abs(rgb.astype("float32") - baseline_rgb.astype("float32")).mean()
                )
                save()
            report["timing_medians_ms"] = {
                mode: statistics.median(
                    r["sampler_wall_ms"] for r in report["timing_runs"] if r["mode"] == mode
                )
                for mode in modes
            }
            report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        probe.restore()
        if runtime is not None:
            runtime.close()
        management.unload_all_models()


if __name__ == "__main__":
    main()
