"""Compare complete AIR and host-round-trip attention on private Q/K/V captures."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import torch

from examples.air_attention import AIRAttention
from examples.qk_scale_model_probe import Probe, tensor_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-manifest", type=Path, required=True)
    parser.add_argument("--cross-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        parser.error("runs must be positive and warmups non-negative")
    paths = sorted(args.input_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError("no private Q/K/V captures")
    os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    torch.set_num_threads(10)
    from anima_xdna_poc.xdna import XDNASession
    from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner

    probe = Probe(None)
    probe.enabled = True
    probe.instrument_xrt(MultiLaunchRunner)
    runners = {}
    report = {
        "status": "running", "runs": args.runs, "warmups": args.warmups,
        "timing_scope": "Whole QK-scale-softmax-AV call including host preparation, copies, sync, wait, output conversion; CPU references and comparisons outside timing.",
        "ordering": "Alternate control/AIR order on each repeat; both warmed independently.",
        "reference": "Actual captured NPU Q/K/V; control performs two existing NPU head-chain dispatches with host scale and softmax.",
        "cpu_gate": {"max_attention_output_nrms": 0.02, "new_prototype_acceptance": True},
        "cases": [],
    }

    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        runners[1024] = AIRAttention(args.self_manifest)
        runners[512] = AIRAttention(args.cross_manifest)
        with torch.inference_mode(), XDNASession() as session:
            for path in paths:
                case = torch.load(path, weights_only=True)
                q, k, v = (case[name] for name in ("query", "key", "value"))
                air_runner = runners[k.shape[1]]

                def execute(mode):
                    probe.low = defaultdict(lambda: defaultdict(float))
                    probe.stage = "attention." + mode
                    started = time.perf_counter()
                    if mode == "baseline":
                        scores = session.dispatch_head_chain(
                            q, k.transpose(1, 2), "attention.qk",
                        ).output_bf16
                        scores = scores * (128**-0.5)
                        probabilities = torch.softmax(scores.float(), dim=-1).bfloat16()
                        output = session.dispatch_head_chain(
                            probabilities, v, "attention.av",
                        ).output_bf16
                    else:
                        output = air_runner.run(q, k, v)
                    elapsed = (time.perf_counter() - started) * 1000
                    return output, elapsed, dict(probe.low[probe.stage])

                outputs = {}
                first = {}
                stable = {"baseline": True, "air_attention": True}
                for mode in ("baseline", "air_attention"):
                    output, elapsed, _ = execute(mode)
                    first[mode] = elapsed
                    for _ in range(args.warmups):
                        execute(mode)
                    outputs[mode] = output.clone()
                samples = {"baseline": [], "air_attention": []}
                for repeat in range(args.runs):
                    order = ("baseline", "air_attention")
                    if repeat % 2:
                        order = tuple(reversed(order))
                    for mode in order:
                        output, elapsed, low = execute(mode)
                        samples[mode].append({"wall_ms": elapsed, "xrt": low})
                        stable[mode] = stable[mode] and torch.equal(output, outputs[mode])
                        outputs[mode] = output.clone()
                scores = torch.bmm(q.float(), k.float().transpose(1, 2)).bfloat16()
                probs = torch.softmax((scores * (128**-0.5)).float(), dim=-1).bfloat16()
                cpu = torch.bmm(probs.float(), v.float()).bfloat16()
                cpu_fp32 = torch.nn.functional.scaled_dot_product_attention(
                    q.float(), k.float(), v.float(),
                ).bfloat16()
                item = {
                    "step": case["step"], "block": case["block"],
                    "attention": case["attention"],
                    "q_shape": list(q.shape), "kv_shape": list(k.shape),
                    "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "first_call_ms": first, "timing_samples": samples,
                    "repeat_outputs_bitwise_stable": stable,
                    "timing_medians_ms": {
                        mode: statistics.median(s["wall_ms"] for s in values)
                        for mode, values in samples.items()
                    },
                    "control_vs_captured": tensor_error(outputs["baseline"], case["control_output"]),
                    "air_vs_control": tensor_error(outputs["air_attention"], outputs["baseline"]),
                    "baseline_vs_cpu": tensor_error(outputs["baseline"], cpu),
                    "air_vs_cpu": tensor_error(outputs["air_attention"], cpu),
                    "air_vs_fp32_sdpa": tensor_error(outputs["air_attention"], cpu_fp32),
                }
                item["cpu_gate_passed"] = (
                    item["air_vs_cpu"]["normalized_rms"]
                    <= report["cpu_gate"]["max_attention_output_nrms"]
                )
                if not item["control_vs_captured"]["bitwise_equal"]:
                    raise RuntimeError(f"control attention does not reproduce captured output: {path.name}")
                if not all(stable.values()):
                    raise RuntimeError(f"repeated attention outputs are unstable: {path.name}")
                report["cases"].append(item)
                save()
                print(
                    case["step"], case["block"], case["attention"],
                    item["timing_medians_ms"], item["air_vs_cpu"], flush=True,
                )
        report["status"] = "completed"
        report["all_attention_cpu_gates_passed"] = all(
            c["cpu_gate_passed"] for c in report["cases"]
        )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()
        probe.restore()
        for runner in runners.values():
            runner.close()


if __name__ == "__main__":
    main()
