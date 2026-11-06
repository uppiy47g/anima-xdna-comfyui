"""Opt-in runtime integration for the measured INT8/AIR sampler profile.

The implementation reuses the same block-boundary transformations validated by
the end-to-end harness. It is installed only when the explicit profile
environment variable is set; unsupported artifacts or boundaries fail loudly.
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import block, chain
from .xdna import _ResidentKernelRunner
from .fused_middle import FusedMiddle
from .fused_io import FusedIO
from .fused_qk import FusedQK, reference_prepare
from .fused_attention import (
    FusedAttention,
    dequantize_reference,
    prepare_reference,
    quantize_reference,
)
from .int8_attention_runner import Int8QKVRunner


PROFILE = "int8-attn-l1k64-nq8pp"
SYNC_FIELDS = ("host_copy_ms", "h2d_sync_ms", "kernel_sync_ms", "d2h_sync_ms")
COUNT_FIELDS = ("dispatches", "h2d_bytes", "d2h_bytes")
TUNED_DIRS = {
    "in": "i8-proj_in-l2k512-l1k64",
    "out": "i8-proj_out-kloop-l2k512-l1k64",
    "attn-out": "i8-sq2048-l1k64",
    "qkv-self": "i8-qkv-shared-l1k64",
    "qkv-cross": "i8-qkv-cross-l1k64",
}
ATTN_INT8_MODES = {PROFILE}
FULL_ATTN_MODES = {PROFILE}
TUNED_MODES = {PROFILE}
QK_MODES = {PROFILE}
IO_MODES = {PROFILE}


def quant_rows(tensor):
    value = tensor.float()
    scale = value.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    return torch.round(value / scale).clamp_(-127, 127).to(torch.int8), scale


def bitwise_equal(got, expected):
    if got.dtype != expected.dtype or got.shape != expected.shape:
        return False
    return torch.equal(got.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def artifact_key(mode, stage):
    if mode not in TUNED_MODES or stage not in TUNED_DIRS:
        raise ValueError(f"Unsupported INT8 artifact stage: {mode}/{stage}")
    return "l1k64/" + stage


def schedule_name(mode):
    if mode not in TUNED_MODES:
        raise ValueError(f"Unsupported INT8 schedule: {mode}")
    return "l1k64"


def _artifact_paths():
    root = Path(os.environ["ANIMA_XDNA_INT8_ARTIFACT_DIR"]).expanduser().resolve()
    paths = {f"l1k64/{stage}": root / dirname for stage, dirname in TUNED_DIRS.items()}
    gemm_shapes = {
        "in": {"rows": 1024, "columns": 8192, "inner": 2048},
        "out": {"rows": 1024, "columns": 2048, "inner": 8192},
        "attn-out": {"rows": 1024, "columns": 2048, "inner": 2048},
    }
    for stage, shape in gemm_shapes.items():
        path = paths[f"l1k64/{stage}"]
        manifest_path = path / "manifest.json"
        elf = path / "aie.elf"
        if not manifest_path.is_file() or not elf.is_file() or not (path / "elf_kernel_name.txt").is_file():
            raise FileNotFoundError(f"Missing INT8 artifact l1k64/{stage}: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "compiled" or manifest.get("dtype") != "i8"
                or manifest.get("shape") != shape or manifest.get("schedule", {}).get("l1_k") != 64):
            raise ValueError(f"Incompatible INT8 GEMM artifact {stage}: {path}")
        if manifest.get("elf_sha256") and hashlib.sha256(elf.read_bytes()).hexdigest() != manifest["elf_sha256"]:
            raise ValueError(f"INT8 GEMM artifact hash mismatch: {path}")
    for stage, shared in (("qkv-self", True), ("qkv-cross", False)):
        path = paths[f"l1k64/{stage}"]
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or not (path / "aie.elf").is_file():
            raise FileNotFoundError(f"Missing INT8 artifact l1k64/{stage}: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "compiled" or manifest.get("dtype") != "i8"
                or manifest.get("kind") != "native_int8_qkv" or manifest.get("shared_input") is not shared):
            raise ValueError(f"Incompatible INT8 QKV artifact {stage}: {path}")
    return paths


class QuantizedActivation:
    def __init__(self, values, scales):
        self.values, self.scales = values, scales


class AttentionAccumulator:
    def __init__(self, value, row_scale, weight_scale):
        self.value, self.row_scale, self.weight_scale = value, row_scale, weight_scale


def install_profile(profile=PROFILE):
    """Install the explicit optimized profile once, refusing unknown or incomplete setups."""
    global _runtime
    if profile != PROFILE:
        raise ValueError(f"Unsupported optimized profile: {profile!r}")
    if os.environ.get("ANIMA_XDNA_ATTENTION_BACKEND") != "air":
        raise RuntimeError("The optimized INT8 profile requires ANIMA_XDNA_ATTENTION_BACKEND=air")
    if os.environ.get("ANIMA_XDNA_ADALN_DEVICE") != "host":
        raise RuntimeError("The optimized INT8 profile requires ANIMA_XDNA_ADALN_DEVICE=host")
    if not os.environ.get("ANIMA_XDNA_AIR_ATTENTION_DIR"):
        raise RuntimeError("The optimized INT8 profile requires ANIMA_XDNA_AIR_ATTENTION_DIR")
    if not os.environ.get("ANIMA_XDNA_GEMM_LAYOUT_DIR"):
        raise RuntimeError("The optimized INT8 profile requires ANIMA_XDNA_GEMM_LAYOUT_DIR")
    air_directory = Path(os.environ["ANIMA_XDNA_AIR_ATTENTION_DIR"]).expanduser().resolve()
    for name, keys in (("self", 1024), ("cross", 512)):
        manifest_path = air_directory / name / "manifest.json"
        elf = air_directory / name / "aie.elf"
        if not manifest_path.is_file() or not elf.is_file():
            raise FileNotFoundError(f"Missing optimized AIR {name} artifact: {manifest_path.parent}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shape = manifest.get("shape", {})
        variant = manifest.get("variant", {})
        if (shape.get("keys") != keys or shape.get("query_cores") != 8
                or variant.get("kv_buffers") != 2 or variant.get("q_mode") != "direct"):
            raise ValueError(f"Incompatible optimized AIR {name} manifest: {manifest_path}")
        if hashlib.sha256(elf.read_bytes()).hexdigest() != manifest.get("elf_sha256"):
            raise ValueError(f"Optimized AIR {name} ELF hash mismatch: {elf}")
    gemm_layout = Path(os.environ["ANIMA_XDNA_GEMM_LAYOUT_DIR"]).expanduser().resolve()
    if not gemm_layout.is_dir():
        raise FileNotFoundError(f"Missing optimized GEMM layout directory: {gemm_layout}")
    if _runtime is not None:
        return _runtime
    torch.set_num_threads(int(os.environ.get("ANIMA_XDNA_NUM_THREADS", "10")))
    runtime = OptimizedInt8Runtime(_artifact_paths())
    _runtime = runtime
    try:
        runtime.install()
    except Exception:
        _runtime = None
        raise
    print(f"[Anima XDNA] Optimized experimental profile active: {PROFILE}; BF16 AIR attention is unchanged.")
    return _runtime


_runtime = None


class OptimizedInt8Runtime:
    def __init__(self, artifact_paths):
        self.mode = PROFILE
        self.artifacts = artifact_paths
        self.current = None
        self.int8_runners = {}
        self.int8_templates = {}
        self.int8_scales = {}
        self.middle_input = None
        self.middle_scale = None
        self.pending_exit = None
        self.fused = FusedMiddle()
        self.io = FusedIO()
        self.qk = FusedQK()
        self.attention_host = FusedAttention()
        self.qkv_runners = {}
        self.attention_scales = {}
        self.attention_weights = {}
        self.verify_qk = False
        self.verify_attention = True
        self.qk_verified_calls = defaultdict(int)
        self.attention_verified_calls = defaultdict(int)
        self._originals = None
        self.reset()

    def close(self):
        self.uninstall()
        for runner in self.int8_runners.values():
            runner.close()
        for runner in self.qkv_runners.values():
            runner.close()
        self.int8_runners.clear()
        self.qkv_runners.clear()
        self.attention_weights.clear()
        self.attention_scales.clear()

    def reset(self):
        self.stages = defaultdict(lambda: defaultdict(float))
        self.blocks = defaultdict(float)
        self.sub = defaultdict(float)
        self.forwards = []
        self.dispatches = 0
        self.block_wall_ms = 0.0
        self.stage_sum_ms = 0.0

    def timed(self, key, function):
        started = time.perf_counter()
        try:
            return function()
        finally:
            self.sub[key] += (time.perf_counter() - started) * 1000

    def int8_project(self, value, stage):
        index, cache = self.current
        akey = artifact_key(self.mode, stage)
        key = (akey, index, stage)
        name, inner, columns = (
            ("ff.net.0.proj.weight", 2048, 8192) if stage == "in" else ("ff.net.2.weight", 8192, 2048)
        )
        if akey not in self.int8_runners:
            self.int8_runners[akey] = _ResidentKernelRunner(str(self.artifacts[akey]))
            self.int8_templates[akey] = (
                np.empty((inner, columns), dtype=np.int8),
                np.empty((1024, columns), dtype=np.int32),
            )
        runner = self.int8_runners[akey]
        weight, output = self.int8_templates[akey]
        if key not in self.int8_scales:
            packed = torch.cat([cache.tensor(f"{index}:{name}:k{k}")
                                for k in range(0, inner, 2048)], dim=0)
            q_weight, weight_scale = quant_rows(packed.T)
            weight = q_weight.T.contiguous().numpy()
            self.int8_scales[key] = weight_scale.T.contiguous()
        if stage == "in":
            if isinstance(value, QuantizedActivation):
                q, scale = value.values, value.scales
            else:
                q, scale = self.timed("int8_in.quantize_ms", lambda: quant_rows(value.reshape(1024, inner)))
        else:
            if value.dtype != torch.int8 or self.middle_scale is None:
                raise ValueError("INT8 proj_out requires quantized GELU output and scale")
            q, scale = value.reshape(1024, inner), self.middle_scale
        started = time.perf_counter()
        result, profile = runner.run(
            [q.numpy(), weight, output], str(key), static_indices={1},
            scratch_indices={2}, output_index=2,
        )
        self.sub[f"int8_{stage}.npu_run_ms"] += (time.perf_counter() - started) * 1000
        accumulator = torch.from_numpy(result)
        weight_scale = self.int8_scales[key]
        if stage == "in":
            self.middle_input = (accumulator, scale, weight_scale)
            return accumulator.reshape(1, 1024, columns), profile
        if self.mode in IO_MODES:
            self.pending_exit = (accumulator, scale, weight_scale)
            return accumulator.reshape(1, 1024, columns), profile
        dequantized = self.timed(
            "int8_out.dequant_ms", lambda: accumulator.float().mul_(scale).mul_(weight_scale).bfloat16())
        return dequantized.reshape(1, 1024, columns), profile

    def entry(self, execution, hidden, embedded, temb, prefix="norm3"):
        """Fuse modulation and quantization; LayerNorm and the GEMV pair stay unchanged."""
        if execution.adaln_runner is None:
            raise ValueError("int8-io requires host AdaLN (ANIMA_XDNA_ADALN_DEVICE=host)")
        activated = execution.host(prefix + ".silu", lambda: F.silu(embedded))
        pair_names = (prefix + ".linear_1.weight", prefix + ".linear_2.weight")
        modulation = execution.host(
            prefix + ".adaln_linear_pair", lambda: execution.adaln_runner(activated, pair_names))

        def normalize():
            combined = modulation + temb
            shift, scale, gate = combined.chunk(3, dim=-1)
            normalized = F.layer_norm(hidden, (execution.config.hidden_size,), eps=1e-6)
            values, scales = self.timed(
                "io.modulate_quantize_ms" if prefix == "norm3" else prefix + ".modulate_quantize_ms",
                lambda: self.io.modulate_quantize(normalized, shift, scale))
            if self.verify_attention and prefix != "norm3":
                reference = normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                self.check_boundary(prefix + ".entry_quantize", (values, scales), quant_rows(reference.squeeze(0)))
            return QuantizedActivation(values, scales), gate.unsqueeze(1)

        return execution.host(prefix + ".layer_norm_modulation", normalize)

    def exit(self, gate, hidden):
        accumulator, row_scale, weight_scale = self.pending_exit
        self.pending_exit = None
        return self.timed(
            "io.dequant_gated_residual_ms",
            lambda: self.io.dequant_gated_residual(accumulator, row_scale, weight_scale, gate, hidden),
        )

    def check_boundary(self, name, actual, expected):
        if len(actual) != len(expected) or not all(bitwise_equal(a, b) for a, b in zip(actual, expected)):
            raise AssertionError(f"{self.mode}/{name} changed reference bits at block {self.current[0]}")
        self.attention_verified_calls[f"{self.mode}/{name}"] += 1

    def attention_weight(self, prefix, names):
        index, cache = self.current
        key = f"{schedule_name(self.mode)}:{cache.manifest['cache_key']}:{index}:{prefix}"
        if key in self.attention_scales:
            return key, self.attention_weights.get(key), self.attention_scales[key]
        weights, scales = [], []
        for name in names:
            packed = cache.tensor(f"{index}:{name}:k0")
            quantized, scale = quant_rows(packed.T)
            weights.append(quantized.T.contiguous().numpy())
            scales.append(scale.T.contiguous())
        self.attention_scales[key] = scales
        if self.verify_attention:
            self.attention_weights[key] = weights
        return key, weights, scales

    def check_projection(self, name, accumulator, activation, weight):
        # Full random/saturation matrices are checked separately; every real
        # warmup dispatch additionally checks spread-out rows/channels exactly.
        rows = [0, activation.shape[0] // 2, activation.shape[0] - 1]
        columns = list(range(0, 2048, 67)) + [2047]
        expected = activation[rows].double() @ torch.from_numpy(weight[:, columns]).double()
        actual = accumulator[rows][:, columns]
        self.check_boundary(name, (actual,), (expected.to(torch.int32),))

    def attention_qkv(self, execution, prefix, hidden, context):
        if not isinstance(hidden, QuantizedActivation):
            raise TypeError("INT8 QKV requires the fused quantized norm input")
        shared = prefix == "attn1"
        started = time.perf_counter()
        if shared:
            if hidden is not context:
                raise ValueError("Self QKV requires a shared quantized input")
            kv = hidden
        else:
            kv = QuantizedActivation(*self.timed(
                prefix + ".context_quantize_ms", lambda: self.attention_host.quantize(context)))
        names = tuple(prefix + ".to_" + part + ".weight" for part in ("q", "k", "v"))
        key, weights, scales = self.attention_weight(prefix + ".qkv", names)
        rkey = (schedule_name(self.mode), prefix)
        if rkey not in self.qkv_runners:
            self.qkv_runners[rkey] = Int8QKVRunner(
                self.artifacts[artifact_key(self.mode, "qkv-self" if shared else "qkv-cross")], shared)
        runner = self.qkv_runners[rkey]
        query_np = hidden.values.numpy()
        context_np = query_np if shared else kv.values.numpy()
        accumulators, profile = runner.run(query_np, context_np, weights, key)
        self.sub[prefix + ".int8_qkv_runner_ms"] += profile["runner_run_ms"]
        row_scales = (hidden.scales, kv.scales, kv.scales)
        if self.verify_attention:
            for name, acc, activation, weight in zip(
                ("q", "k", "v"), accumulators, (hidden.values, kv.values, kv.values), weights,
            ):
                self.check_projection(prefix + "." + name, acc, activation, weight)
            if not shared:
                self.check_boundary(prefix + ".context_quantize", (kv.values, kv.scales),
                                    quantize_reference(context.reshape(512, 1024)))
        execution.metrics.append(block.StageMetric(
            name=prefix + ".qkv_chain", device="xdna",
            wall_ms=(time.perf_counter() - started) * 1000, dispatches=1,
            estimated_flops=sum(2 * m * n * k for m, n, k in runner.shapes),
            transfer_bytes=profile["h2d_bytes"] + profile["d2h_bytes"],
            **{name: profile[name] for name in (
                "h2d_bytes", "d2h_bytes", "allocation_count", "resident_hits", "weight_population_bytes",
            )},
        ))
        return accumulators, row_scales, scales

    def attention_out(self, prefix, activation):
        if not isinstance(activation, QuantizedActivation):
            raise TypeError("INT8 output projection requires the fused AIR layout/quantization input")
        stage = artifact_key(self.mode, "attn-out")
        if stage not in self.int8_runners:
            self.int8_runners[stage] = _ResidentKernelRunner(str(self.artifacts[stage]))
            self.int8_templates[stage] = (
                np.empty((2048, 2048), dtype=np.int8), np.empty((1024, 2048), dtype=np.int32),
            )
        key, weights, scales = self.attention_weight(prefix + ".out", (prefix + ".to_out.0.weight",))
        placeholder, output = self.int8_templates[stage]
        started = time.perf_counter()
        result, profile = self.int8_runners[stage].run(
            [activation.values.numpy(), weights[0] if weights is not None else placeholder, output],
            key, static_indices={1}, scratch_indices={2}, output_index=2,
        )
        self.sub[prefix + ".int8_out_runner_ms"] += (time.perf_counter() - started) * 1000
        accumulator = torch.from_numpy(result)
        if self.verify_attention:
            self.check_projection(prefix + ".out", accumulator, activation.values, weights[0])
        return AttentionAccumulator(accumulator, activation.scales, scales[0]), profile

    def attention_quantize_layout(self, prefix, output):
        values, scales = self.attention_host.quantize(output, head_major=True)
        if self.verify_attention:
            expected = quantize_reference(output.transpose(0, 1).reshape(1024, 2048))
            self.check_boundary(prefix + ".layout_quantize", (values, scales), expected)
        return QuantizedActivation(values, scales)

    def attention_exit(self, prefix, attended, gate, hidden):
        if not isinstance(attended, AttentionAccumulator):
            return hidden + gate * attended
        result = self.timed(
            prefix + ".dequant_gated_residual_ms",
            lambda: self.io.dequant_gated_residual(
                attended.value, attended.row_scale, attended.weight_scale, gate, hidden),
        )
        if self.verify_attention:
            projected = dequantize_reference(attended.value, attended.row_scale, attended.weight_scale)
            self.check_boundary(prefix + ".dequant_residual", (result,), (hidden + gate * projected.unsqueeze(0),))
        return result

    def run_io(self, execution, inputs):
        """_BlockExecution.run with the MLP entry/exit fused; everything else is unchanged."""
        hidden = inputs.hidden_states
        checkpoints = []
        if self.mode in FULL_ATTN_MODES:
            normalized, gate = self.entry(execution, hidden, inputs.embedded_timestep, inputs.temb, "norm1")
        else:
            normalized, gate = execution.adaln("norm1", hidden, inputs.embedded_timestep, inputs.temb)
        attended = execution.attention("attn1", normalized, normalized, inputs.image_rotary_emb, None)
        hidden = execution.host("attn1.gated_residual", lambda: self.attention_exit("attn1", attended, gate, hidden))
        checkpoints.append(("self_attention", hidden))

        if self.mode in FULL_ATTN_MODES:
            normalized, gate = self.entry(execution, hidden, inputs.embedded_timestep, inputs.temb, "norm2")
        else:
            normalized, gate = execution.adaln("norm2", hidden, inputs.embedded_timestep, inputs.temb)
        attended = execution.attention(
            "attn2", normalized, inputs.encoder_hidden_states, None, inputs.attention_mask)
        hidden = execution.host("attn2.gated_residual", lambda: self.attention_exit("attn2", attended, gate, hidden))
        checkpoints.append(("cross_attention", hidden))

        activation, gate = self.entry(execution, hidden, inputs.embedded_timestep, inputs.temb)
        accumulator = execution.linear("ff.proj_in", activation, "ff.net.0.proj.weight")
        quantized = execution.host("ff.gelu", lambda: accumulator)
        accumulator = execution.linear("ff.proj_out", quantized, "ff.net.2.weight")
        hidden = execution.host("ff.gated_residual", lambda: self.exit(gate, hidden))
        checkpoints.append(("feed_forward", hidden))
        return block.BlockResult(hidden, tuple(execution.metrics), checkpoints=tuple(checkpoints))

    def install(self):
        if self._originals is not None:
            return
        from comfyui_xdna_nodes import model_wrapper

        self._model_wrapper = model_wrapper
        original_block = chain.run_xdna_block
        original_linear = block._BlockExecution.linear
        original_host = block._BlockExecution.host
        original_run = block._BlockExecution.run
        original_attention = block._BlockExecution.attention
        original_forward = model_wrapper.SharedRuntime.forward
        self._originals = (
            original_block, original_linear, original_host, original_run,
            original_attention, original_forward,
        )
        runtime = self

        def run_block(*a, **kw):
            runtime.current = (kw["block_index"], kw["packed_cache"])
            started = time.perf_counter()
            try:
                result = original_block(*a, **kw)
            finally:
                runtime.current = None
            wall = (time.perf_counter() - started) * 1000
            runtime.block_wall_ms += wall
            runtime.blocks[kw["block_index"]] += wall
            runtime.dispatches += result.dispatch_count
            for metric in result.metrics:
                row = runtime.stages[metric.name]
                row["count"] += 1
                row["wall_ms"] += metric.wall_ms
                runtime.stage_sum_ms += metric.wall_ms
                for field in SYNC_FIELDS + COUNT_FIELDS:
                    row[field] += getattr(metric, field)
            return result

        def linear(execution, name, value, weight_name):
            attention_out = name in ("attn1.out", "attn2.out") and runtime.mode in ATTN_INT8_MODES
            if not attention_out and name not in ("ff.proj_in", "ff.proj_out"):
                return original_linear(execution, name, value, weight_name)
            if not attention_out and weight_name not in ("ff.net.2.weight", "ff.net.0.proj.weight"):
                raise ValueError(weight_name)
            started = time.perf_counter()
            if attention_out:
                prefix = name.split(".")[0]
                if weight_name != prefix + ".to_out.0.weight":
                    raise ValueError(weight_name)
                result, profile = runtime.attention_out(prefix, value)
            else:
                result, profile = runtime.int8_project(value, "in" if name == "ff.proj_in" else "out")
            execution.metrics.append(block.StageMetric(
                name=name, device="xdna", wall_ms=(time.perf_counter() - started) * 1000,
                dispatches=1, estimated_flops=2 * 1024 * 2048 * (2048 if attention_out else 8192),
                transfer_bytes=profile["h2d_bytes"] + profile["d2h_bytes"],
                resident_hits=int(profile["resident_hit"]),
                **{k: profile[k] for k in (
                    "h2d_bytes", "d2h_bytes", "allocation_count", "host_copy_ms",
                    "h2d_sync_ms", "kernel_sync_ms", "d2h_sync_ms",
                    "weight_population_ms", "weight_population_bytes",
                )},
            ))
            return result

        def host(execution, name, operation):
            if name != "ff.gelu":
                return original_host(execution, name, operation)

            def middle():
                accumulator, row_scale, weight_scale = runtime.middle_input
                quantized, scale = runtime.fused(accumulator, row_scale, weight_scale)
                runtime.middle_scale = scale
                return quantized.reshape(1, 1024, 8192)

            return original_host(execution, name, middle)

        def run(execution, inputs):
            if runtime.mode in IO_MODES:
                return runtime.run_io(execution, inputs)
            return original_run(execution, inputs)

        def attention(execution, prefix, hidden, context, rotary, mask):
            if runtime.mode not in QK_MODES:
                return original_attention(execution, prefix, hidden, context, rotary, mask)
            if mask is not None or execution.attention_runner is None:
                raise ValueError("Fused QK experiment requires unmasked AIR attention; no fallback")
            weights = (execution.weights[prefix + ".norm_q.weight"], execution.weights[prefix + ".norm_k.weight"])
            if runtime.mode in FULL_ATTN_MODES:
                accumulators, row_scales, weight_scales = runtime.attention_qkv(execution, prefix, hidden, context)
                prepared = execution.host(
                    prefix + ".qkv_prepare",
                    lambda: runtime.attention_host.prepare(accumulators, row_scales, weight_scales, *weights, rotary),
                )
                if runtime.verify_attention:
                    expected = prepare_reference(accumulators, row_scales, weight_scales, *weights, rotary)
                    runtime.check_boundary(prefix + ".prepare", prepared, expected)
            else:
                query, key, value = execution.qkv(prefix, hidden, context)
                prepared = execution.host(
                    prefix + ".qkv_prepare",
                    lambda: runtime.qk(query, key, value, *weights, rotary),
                )
                if runtime.verify_qk:
                    expected = reference_prepare(query, key, value, *weights, rotary)
                    for name, actual, reference in zip(("query", "key", "value"), prepared, expected):
                        if not bitwise_equal(actual, reference):
                            raise AssertionError(f"{prefix} {name} fused preparation changed bits at block {runtime.current[0]}")
                    runtime.qk_verified_calls[prefix] += 1
            started = time.perf_counter()
            output = execution.attention_runner(*prepared)
            wall = (time.perf_counter() - started) * 1000
            h2d = sum(x.numel() for x in prepared) * 2
            d2h = output.numel() * 2
            execution.metrics.append(block.StageMetric(
                prefix + ".air_attention", execution.runner_device, wall, 1,
                4 * execution.config.num_heads * prepared[0].shape[1] * prepared[1].shape[1]
                * execution.config.head_dim, h2d + d2h, h2d, d2h,
            ))
            if runtime.mode in ATTN_INT8_MODES:
                attended = execution.host(
                    prefix + ".av_layout_quantize", lambda: runtime.attention_quantize_layout(prefix, output))
            else:
                attended = execution.host(
                    prefix + ".av_layout",
                    lambda: output.transpose(0, 1).reshape(1, hidden.shape[1], execution.config.hidden_size),
                )
            return execution.linear(prefix + ".out", attended, prefix + ".to_out.0.weight")

        def forward(runtime_self, *a, **kw):
            started = time.perf_counter()
            output = original_forward(runtime_self, *a, **kw)
            wall = (time.perf_counter() - started) * 1000
            d = runtime_self.diagnostics
            runtime.forwards.append({
                "wall_ms": wall, "preprocess_ms": d.last_preprocess_ms, "chain_ms": d.last_chain_ms,
                "postprocess_ms": d.last_postprocess_ms, "total_ms": d.last_total_ms,
            })
            if runtime.verify_attention and len(runtime.forwards) == 4:
                runtime.verify_attention = False
                runtime.attention_weights.clear()
                print("[Anima XDNA] Optimized INT8 boundary checks passed for four Anima forwards.")
            return output

        chain.run_xdna_block = run_block
        block._BlockExecution.linear = linear
        block._BlockExecution.host = host
        block._BlockExecution.run = run
        block._BlockExecution.attention = attention
        model_wrapper.SharedRuntime.forward = forward
        print("[Anima XDNA] Optimized INT8/AIR execution hooks installed.")

    def uninstall(self):
        if self._originals is None:
            return
        (
            chain.run_xdna_block,
            block._BlockExecution.linear,
            block._BlockExecution.host,
            block._BlockExecution.run,
            block._BlockExecution.attention,
            self._model_wrapper.SharedRuntime.forward,
        ) = self._originals
        self._originals = None

    def instrument_air(self, backend):
        if backend is None or not backend.runners:
            raise AssertionError("AIR runners must exist before instrumentation")
        for keys, attention in backend.runners.items():
            inner = attention.runner
            original = inner.run

            def timed_run(*a, _original=original, _keys=keys, **kw):
                started = time.perf_counter()
                try:
                    return _original(*a, **kw)
                finally:
                    self.sub[f"air.runner_run_ms.{_keys}"] += (time.perf_counter() - started) * 1000

            inner.run = timed_run
