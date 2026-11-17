from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from anima_xdna_poc import block
from anima_xdna_poc.int8_runtime import (
    PROFILE,
    AttentionAccumulator,
    OptimizedInt8Runtime,
    QuantizedActivation,
    _artifact_paths,
    bitwise_equal,
    install_profile,
    quant_rows,
    require_int8_tokens,
)
from anima_xdna_poc.errors import UnsupportedTensor
from anima_xdna_poc.vae_optimization import _restrided_upsample_forward, optimize_vae

_PROFILE = {
    "h2d_bytes": 0, "d2h_bytes": 0, "allocation_count": 0, "host_copy_ms": 0.0, "h2d_sync_ms": 0.0,
    "kernel_sync_ms": 0.0, "d2h_sync_ms": 0.0, "weight_population_ms": 0.0,
    "weight_population_bytes": 0, "resident_hit": True,
}


class _FakeNPU:
    """One launch in flight; each (stage, slot) output is overwritten in place by its next launch."""

    def __init__(self, events):
        self.events = events
        self.buffers = {}
        self.busy = None

    def launch(self, stage, slot, compute, label):
        assert self.busy is None, f"{label} launched while {self.busy} is still running"
        self.events.append(f"start:{label}")
        buffer = self.buffers.setdefault((stage, slot or 0), torch.zeros_like(compute))
        buffer.copy_(compute)
        self.busy = label

        def finish():
            assert self.busy == label
            self.busy = None
            self.events.append(f"finish:{label}")
            return buffer

        return finish

class OptimizedProfileTests(unittest.TestCase):
    def test_quant_rows_uses_symmetric_per_row_int8(self):
        value = torch.tensor([[0.0, -2.0, 1.0], [1e-14, -1e-14, 0.0]])

        quantized, scales = quant_rows(value)

        self.assertEqual(quantized.dtype, torch.int8)
        self.assertTrue(torch.equal(quantized, torch.tensor([[0, -127, 64], [1, -1, 0]], dtype=torch.int8)))
        self.assertTrue(torch.equal(scales, torch.tensor([[2.0 / 127], [1e-12 / 127]])))

    def test_bitwise_equal_requires_matching_shape_dtype_and_bits(self):
        value = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
        same = value.clone()
        changed = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)

        self.assertTrue(bitwise_equal(value, same))
        self.assertFalse(bitwise_equal(value, changed))
        self.assertFalse(bitwise_equal(value, value.float()))
        self.assertFalse(bitwise_equal(value, value[:1]))

    def test_artifact_paths_rejects_missing_root(self):
        with patch.dict(os.environ, {"ANIMA_XDNA_INT8_ARTIFACT_DIR": "not-an-artifact-directory"}):
            with self.assertRaises(FileNotFoundError):
                _artifact_paths()

    def test_profile_requires_explicit_runtime_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "ATTENTION_BACKEND=air"):
                install_profile(PROFILE)

    def test_profile_accepts_only_the_validated_token_counts(self):
        require_int8_tokens(SimpleNamespace(hidden_states=torch.empty(1, 1024, 1)))
        require_int8_tokens(SimpleNamespace(hidden_states=torch.empty(1, 4096, 1)))
        for tokens in (512, 2048, 9216):
            with self.assertRaisesRegex(UnsupportedTensor, "1024 or 4096 image tokens"):
                require_int8_tokens(SimpleNamespace(hidden_states=torch.empty(1, tokens, 1)))

    def _air_directory(self, root, self4096=None):
        for name, keys in (("self", 1024), ("cross", 512), ("self4096", 4096)):
            if name == "self4096" and self4096 is None:
                continue
            directory = root / name
            directory.mkdir()
            elf = directory / "aie.elf"
            elf.write_bytes(name.encode())
            manifest = {
                "shape": {"keys": keys, "query_cores": 8},
                "variant": {"kv_buffers": 2, "q_mode": "direct"},
                "elf_sha256": hashlib.sha256(name.encode()).hexdigest(),
            }
            if name == "self4096":
                manifest["shape"]["keys"] = self4096
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _install_with_air_directory(self, root):
        environment = {
            "ANIMA_XDNA_ATTENTION_BACKEND": "air",
            "ANIMA_XDNA_ADALN_DEVICE": "host",
            "ANIMA_XDNA_AIR_ATTENTION_DIR": str(root),
            "ANIMA_XDNA_GEMM_LAYOUT_DIR": str(root),
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "anima_xdna_poc.int8_runtime._artifact_paths", side_effect=FileNotFoundError("artifacts")
        ):
            install_profile(PROFILE)

    def test_profile_validates_the_optional_self4096_attention_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._air_directory(root)
            with self.assertRaisesRegex(FileNotFoundError, "artifacts"):
                self._install_with_air_directory(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._air_directory(root, self4096=4096)
            with self.assertRaisesRegex(FileNotFoundError, "artifacts"):
                self._install_with_air_directory(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._air_directory(root, self4096=1024)
            with self.assertRaisesRegex(ValueError, "Incompatible optimized AIR self4096 manifest"):
                self._install_with_air_directory(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._air_directory(root, self4096=4096)
            (root / "self4096" / "aie.elf").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "self4096 ELF hash mismatch"):
                self._install_with_air_directory(root)

    def _tiled_runtime(self, overlap=False):
        runtime = OptimizedInt8Runtime.__new__(OptimizedInt8Runtime)
        runtime.sub = defaultdict(float)
        runtime.overlap = overlap
        runtime.verify_attention = False
        runtime.mode = PROFILE
        return runtime

    def test_tiled_modulate_quantize_assembles_each_1024_row_tile(self):
        class FakeIO:
            rows, columns = 1024, 4

            def __init__(self):
                self.calls = []

            def modulate_quantize(self, normalized, shift, scale):
                self.calls.append(normalized.data_ptr())
                values = normalized[0].float().to(torch.int8)
                return values, normalized[0, :, :1].float()

        runtime = self._tiled_runtime()
        runtime.io = FakeIO()
        normalized = (torch.arange(4096 * 4).reshape(1, 4096, 4) % 100).bfloat16()

        values, scales = runtime.modulate_quantize(normalized, None, None)

        self.assertEqual(len(runtime.io.calls), 4)
        self.assertTrue(torch.equal(values, normalized[0].float().to(torch.int8)))
        self.assertTrue(torch.equal(scales, normalized[0, :, :1].float()))
        self.assertEqual(runtime.modulate_quantize(normalized[:, :1024].contiguous(), None, None)[0].shape, (1024, 4))

    def test_tiled_feed_forward_runs_every_tile_into_its_own_output_rows(self):
        runtime = self._tiled_runtime()
        seen = []

        def fake_exit(gate, hidden, out=None):
            seen.append(hidden.shape)
            out.copy_(hidden + gate)

        runtime.exit = fake_exit
        execution = SimpleNamespace(
            linear=lambda name, value, weight_name: seen.append((name, value)) or value,
            host=lambda name, operation: operation(),
        )
        activation = QuantizedActivation(
            torch.arange(4096 * 2, dtype=torch.int32).reshape(4096, 2).to(torch.int8), torch.ones(4096, 1))
        hidden = torch.arange(4096 * 2).reshape(1, 4096, 2).bfloat16()
        gate = torch.ones(1, 1, 2).bfloat16()

        result = runtime.feed_forward_tiled(execution, block._row_tiles(4096), activation, gate, hidden)

        self.assertTrue(torch.equal(result, hidden + gate))
        projections = [item for item in seen if isinstance(item[0], str)]
        self.assertEqual([name for name, _ in projections], ["ff.proj_in", "ff.proj_out"] * 4)
        tiles = [value for name, value in projections if name == "ff.proj_in"]
        for index, tile in enumerate(tiles):
            self.assertTrue(torch.equal(tile.values, activation.values[index * 1024:(index + 1) * 1024]))

    def test_overlapped_feed_forward_orders_jobs_and_reads_current_slots(self):
        for count in (1, 2, 3, 4):
            events = []
            npu = _FakeNPU(events)
            runtime = self._tiled_runtime(overlap=True)
            starts = {"in": 0, "out": 0}
            gelus, exits = [0], [0]
            gelu_values = torch.zeros(1024, 2, dtype=torch.int8)
            gelu_scales = torch.zeros(1024, 1)

            def project_start(value, stage, slot=None, scale=None):
                index = starts[stage]
                starts[stage] += 1
                if stage == "in":
                    compute, row_scale = value.values.float() + 1, value.scales
                else:
                    compute, row_scale = value.reshape(1024, 2).float() * 3, scale
                finish = npu.launch(stage, slot, compute, f"{stage}{index}")
                return lambda: ((finish(), row_scale, torch.ones(1, 2)), dict(_PROFILE, exposed_ms=0.0))

            def host(name, operation):
                if name != "ff.gelu":
                    return operation()
                events.append(f"gelu{gelus[0]}")
                gelus[0] += 1
                accumulator, scale, _ = runtime.middle_input
                gelu_values.copy_((accumulator * 2).to(torch.int8))
                gelu_scales.copy_(scale + 1)
                runtime.middle_scale = gelu_scales
                return gelu_values.reshape(1, 1024, 2)

            def exit_(gate, hidden, out=None):
                events.append(f"exit{exits[0]}")
                exits[0] += 1
                accumulator, scale, _ = runtime.pending_exit
                runtime.pending_exit = None
                out.copy_((hidden.float() + accumulator.reshape(1, 1024, 2) * scale.reshape(1, 1024, 1)).bfloat16())

            runtime.int8_project_start = project_start
            runtime.exit = exit_
            execution = SimpleNamespace(host=host, metrics=[])
            rows = 1024 * count
            activation = QuantizedActivation(
                (torch.arange(rows * 2).reshape(rows, 2) % 31).to(torch.int8),
                torch.arange(rows).reshape(rows, 1).float() % 3 + 1,
            )
            hidden = (torch.arange(rows * 2).reshape(1, rows, 2) % 17).bfloat16()

            result = runtime.feed_forward_tiled(execution, block._row_tiles(rows), activation, None, hidden)

            for start, end in block._row_tiles(rows):
                accumulator = (activation.values[start:end].float() + 1) * 2
                middle = accumulator.to(torch.int8).reshape(1024, 2).float() * 3
                scale = activation.scales[start:end] + 1
                expected = (hidden[:, start:end].float() + middle.reshape(1, 1024, 2) * scale.reshape(1, 1024, 1)).bfloat16()
                self.assertTrue(torch.equal(result[:, start:end], expected), (count, start))
            self.assertEqual([m.name for m in execution.metrics].count("ff.proj_out"), count)
            if count == 1:
                self.assertEqual(events, ["start:in0", "finish:in0", "gelu0", "start:out0", "finish:out0", "exit0"])
            if count == 4:
                self.assertEqual(events, [
                    "start:in0", "finish:in0", "start:in1", "gelu0", "finish:in1", "start:out0", "gelu1",
                    "finish:out0", "start:in2", "exit0", "finish:in2", "start:out1", "gelu2",
                    "finish:out1", "start:in3", "exit1", "finish:in3", "start:out2", "gelu3",
                    "finish:out2", "start:out3", "exit2", "finish:out3", "exit3",
                ])

    def test_overlapped_attention_matches_sequential_and_hides_host_work(self):
        def run(overlap):
            events = []
            npu = _FakeNPU(events)
            runtime = self._tiled_runtime(overlap=overlap)
            counts = {"qkv": 0, "prepare": 0, "quantize": 0, "out": 0, "exit": 0}

            def count(name):
                counts[name] += 1
                return counts[name] - 1

            def qkv_start(execution, prefix, tile, context, slot=None):
                finish = npu.launch("qkv", slot, tile.values.float() + 1, f"qkv{count('qkv')}")
                return lambda: ((finish(),), tile.scales, None)

            def prepare(accumulators, row_scales, weight_scales, *rest):
                events.append(f"prepare{count('prepare')}")
                query = accumulators[0].reshape(1, 1024, 1).expand(16, 1024, 128).bfloat16()
                return query, query * 2, query * 3

            def quantize(prefix, tile):
                events.append(f"quantize{count('quantize')}")
                values = tile[0, :, :1].float().clamp(-100, 100).to(torch.int8)
                return QuantizedActivation(values, torch.ones(1024, 1))

            def out_start(prefix, attended, slot=None):
                finish = npu.launch("out", slot, attended.values.float() * 3 + 1, f"out{count('out')}")
                return lambda: (
                    AttentionAccumulator(finish(), attended.scales, torch.ones(1, 1)),
                    dict(_PROFILE, exposed_ms=0.0),
                )

            def attention_exit(prefix, accumulator, gate, hidden, out=None):
                events.append(f"exit{count('exit')}")
                out.copy_((hidden.float() + gate.float() * accumulator.value.reshape(1, 1024, 1)).bfloat16())
                return out

            def air(query, key, value):
                events.append("air")
                return (query.float() + key.float() + value.float()).bfloat16()

            class FakeAIR:
                """Resident AIR buffers; each output slot is overwritten by its next launch."""

                def __init__(self):
                    self.queries = [torch.zeros(16, 1024, 128, dtype=torch.bfloat16) for _ in range(4)]
                    self.key = torch.zeros(16, 4096, 128, dtype=torch.bfloat16)
                    self.value = torch.zeros_like(self.key)

                def resident(self, tiles):
                    return tuple(self.queries[:tiles]), self.key, self.value

                def start(self, tile, slot, upload_kv):
                    assert upload_kv == (tile == 0)
                    rows = slice(tile * 1024, (tile + 1) * 1024)
                    compute = (self.queries[tile].float() + self.key[:, rows].float()
                               + self.value[:, rows].float()).bfloat16()
                    return npu.launch("air", slot, compute, f"air{tile}")

                def finish(self, pending):
                    return pending()

            class Backend:
                def __call__(self, query, key, value):
                    return air(query, key, value)

                @staticmethod
                def runner_for(tokens, keys):
                    assert (tokens, keys) == (4096, 4096)
                    return fake_air

            fake_air = FakeAIR()
            runtime.attention_qkv_start = qkv_start
            runtime.attention_host = SimpleNamespace(prepare=prepare)
            runtime.attention_quantize_layout = quantize
            runtime.attention_out_start = out_start
            runtime.attention_exit = attention_exit
            execution = SimpleNamespace(
                weights={"attn1.norm_q.weight": 1, "attn1.norm_k.weight": 1},
                config=SimpleNamespace(num_heads=16, head_dim=128), runner_device="xdna",
                attention_runner=Backend(), metrics=[], host=lambda name, operation: operation(),
                linear=lambda name, attended, weight_name: out_start("attn1", attended)()[0],
            )
            normalized = QuantizedActivation(
                (torch.arange(4096).reshape(4096, 1) % 50).to(torch.int8), torch.ones(4096, 1))
            hidden = (torch.arange(4096 * 4).reshape(1, 4096, 4) % 13).bfloat16()
            gate = torch.full((1, 1, 4), 0.5).bfloat16()
            result = runtime.attention_tiled(
                execution, "attn1", block._row_tiles(4096), normalized, None, None, gate, hidden)
            return result, events, execution.metrics

        sequential, _, sequential_metrics = run(False)
        overlapped, events, metrics = run(True)

        self.assertTrue(torch.equal(overlapped, sequential))
        for collected in (sequential_metrics, metrics):
            air_metrics = [m for m in collected if m.name == "attn1.air_attention"]
            self.assertEqual([(m.dispatches, m.h2d_bytes, m.d2h_bytes) for m in air_metrics],
                             [(4, 3 * 16 * 4096 * 128 * 2, 16 * 4096 * 128 * 2)])
        self.assertEqual(events, [
            "start:qkv0",
            "finish:qkv0", "start:qkv1", "prepare0",
            "finish:qkv1", "start:qkv2", "prepare1",
            "finish:qkv2", "start:qkv3", "prepare2",
            "finish:qkv3", "prepare3",
            "start:air0",
            "finish:air0", "start:air1", "quantize0",
            "finish:air1", "start:out0", "quantize1",
            "finish:out0", "start:air2", "exit0",
            "finish:air2", "start:out1", "quantize2",
            "finish:out1", "start:air3", "exit1",
            "finish:air3", "start:out2", "quantize3",
            "finish:out2", "start:out3", "exit2",
            "finish:out3", "exit3",
        ])

    def test_two_stage_waits_for_the_job_in_flight_when_settling_fails(self):
        events = []
        npu = _FakeNPU(events)

        def settle(kind, index, outcome):
            raise RuntimeError("nonfinite")

        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            OptimizedInt8Runtime.two_stage(
                3, lambda kind, index: npu.launch(kind, index % 2, torch.zeros(1), f"{kind}{index}"), settle)
        self.assertEqual(events, ["start:in0", "finish:in0", "start:in1", "finish:in1"])
        self.assertIsNone(npu.busy)

    def test_vae_conversion_sets_bf16_and_channels_last(self):
        class FakeVAEModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv3d = torch.nn.Conv3d(2, 2, 1)
                self.conv2d = torch.nn.Conv2d(2, 2, 1)
                self.upsample = torch.nn.Upsample(scale_factor=2, mode="nearest-exact")

            def decode(self, value):
                return value

        model = FakeVAEModel()
        vae = SimpleNamespace(first_stage_model=model, vae_dtype=torch.float32)

        self.assertIs(optimize_vae(vae), vae)
        self.assertIs(vae.vae_dtype, torch.bfloat16)
        self.assertTrue(model.conv3d.weight.is_contiguous(memory_format=torch.channels_last_3d))
        self.assertTrue(model.conv2d.weight.is_contiguous(memory_format=torch.channels_last))
        value = torch.rand(1, 2, 1, 4, 5)
        self.assertTrue(model.decode(value).is_contiguous(memory_format=torch.channels_last_3d))
        self.assertIs(optimize_vae(vae), vae)

    def test_restride_preserves_upsample_values_and_layout(self):
        upsample = torch.nn.Upsample(scale_factor=2, mode="nearest-exact")
        value = torch.rand(1, 4, 3, 5).contiguous(memory_format=torch.channels_last)
        expected = upsample(value)

        actual = _restrided_upsample_forward(upsample.forward)(value)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))


if __name__ == "__main__":
    unittest.main()
