import importlib.util
import json
import hashlib
import os
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import torch

from examples.air_attention import AIRAttention, AttentionShape, build_module
from anima_xdna_poc.air_attention import AIRAttentionBackend
from anima_xdna_poc.block import _BlockExecution
from anima_xdna_poc.chain import AnimaXDNAChainRuntime
from anima_xdna_poc.errors import UnsupportedTensor


class AIRAttentionTests(unittest.TestCase):
    def test_live_deployment_keeps_quality_and_timing_limits_explicit(self):
        root = Path(__file__).resolve().parents[1]
        live = json.loads(
            (root / "docs" / "evidence" / "air-attention-live-deployment.json").read_text()
        )
        measured = json.loads(
            (root / "docs" / "evidence" / "air-attention-model-probe.json").read_text()
        )
        air_capture = next(c for c in measured["captures"] if c["mode"] == "air_attention")
        self.assertEqual(live["attention_backend"], "air_experimental")
        self.assertFalse(live["deployment"]["silent_fallback"])
        self.assertFalse(live["limitations"]["quality_pass"])
        self.assertFalse(live["limitations"]["twelve_second_goal_achieved"])
        self.assertEqual(live["native_replay"]["dispatches_per_step"], [476] * 4)
        self.assertTrue(live["native_replay"]["all_112_block_control_nrms_match_measured_air_prototype"])
        generation = live["normal_comfyui_generation"]
        self.assertEqual(generation["cold"]["image_rgb_sha256"], air_capture["image_rgb_sha256"])
        self.assertEqual(generation["cold"]["rgb_mae_vs_control"], air_capture["image_rgb_mae_vs_baseline"])
        self.assertTrue(generation["cold"]["pixel_equal_to_measured_air_prototype"])
        self.assertEqual(generation["resident"]["runtime_diagnostics"]["last_dispatches"], 476)
        self.assertEqual(generation["resident"]["status"], "success")

    def test_native_attention_skips_full_score_and_probability_pipeline(self):
        cfg = SimpleNamespace(num_heads=16, head_dim=128, hidden_size=2048)
        hidden = torch.zeros(1, 1024, 2048, dtype=torch.bfloat16)
        context = torch.zeros(1, 512, 2048, dtype=torch.bfloat16)
        output = torch.ones(16, 1024, 128, dtype=torch.bfloat16)
        attention = Mock(return_value=output)
        execution = _BlockExecution(
            cfg, None, Mock(), "xdna2", Mock(), attention_runner=attention,
        )
        execution.qkv = Mock(return_value=(hidden, context, context))
        execution.rms_norm = Mock(side_effect=lambda name, value, weight: value)
        execution.linear = Mock(side_effect=lambda name, value, weight: value)
        got = execution.attention("attn2", hidden, context, None, None)
        self.assertTrue(bool((got == 1).all()))
        args = attention.call_args.args
        self.assertEqual([tuple(a.shape) for a in args], [
            (16, 1024, 128), (16, 512, 128), (16, 512, 128),
        ])
        execution.batched_runner.assert_not_called()
        names = [m.name for m in execution.metrics]
        self.assertIn("attn2.air_attention", names)
        self.assertNotIn("attn2.softmax", names)
        self.assertNotIn("attn2.qk_scale", names)
        metric = next(m for m in execution.metrics if m.name == "attn2.air_attention")
        self.assertEqual((metric.dispatches, metric.h2d_bytes, metric.d2h_bytes),
                         (1, 8 * 1024**2, 4 * 1024**2))
        with self.assertRaisesRegex(UnsupportedTensor, "masks"):
            execution.attention("attn2", hidden, context, None, torch.ones(1))
        attention.assert_called_once()

    def test_configuration_selects_air_or_explicit_baseline(self):
        env = {
            "ANIMA_XDNA_AIR_ATTENTION_DIR": "private-artifacts",
            "ANIMA_XDNA_ATTENTION_BACKEND": "air",
        }
        with patch("anima_xdna_poc.chain.load_checkpoint_config"), patch.dict(os.environ, env):
            runtime = AnimaXDNAChainRuntime(Path("unused"))
            self.assertEqual(runtime.attention_backend, "air_experimental")
            self.assertIsInstance(runtime._air_attention, AIRAttentionBackend)
            runtime.close()
            with patch.dict(os.environ, {"ANIMA_XDNA_ATTENTION_BACKEND": "baseline"}):
                runtime = AnimaXDNAChainRuntime(Path("unused"))
                self.assertEqual(runtime.attention_backend, "baseline")
                self.assertIsNone(runtime._air_attention)
                runtime.close()
            with patch.dict(os.environ, {"ANIMA_XDNA_AIR_ATTENTION_DIR": ""}):
                with self.assertRaisesRegex(UnsupportedTensor, "requires"):
                    AnimaXDNAChainRuntime(Path("unused"))

    def test_public_default_is_baseline_even_with_air_artifacts(self):
        with patch("anima_xdna_poc.chain.load_checkpoint_config"), patch.dict(
            os.environ, {"ANIMA_XDNA_AIR_ATTENTION_DIR": "private-artifacts"}, clear=True,
        ):
            runtime = AnimaXDNAChainRuntime(Path("unused"))
            self.assertEqual(runtime.attention_backend, "baseline")
            self.assertIsNone(runtime._air_attention)
            runtime.close()
            del os.environ["ANIMA_XDNA_AIR_ATTENTION_DIR"]
            runtime = AnimaXDNAChainRuntime(Path("unused"))
            self.assertEqual(runtime.attention_backend, "baseline")
            runtime.close()
            os.environ["ANIMA_XDNA_ATTENTION_BACKEND"] = "unknown"
            with self.assertRaisesRegex(UnsupportedTensor, "unknown attention backend"):
                AnimaXDNAChainRuntime(Path("unused"))

    def test_public_launcher_and_prerelease_keep_explicit_opt_in(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "examples" / "run_comfyui_xdna.ps1").read_text()
        self.assertIn('[string]$AttentionBackend = "baseline"', launcher)
        self.assertIn("$env:ANIMA_XDNA_ATTENTION_BACKEND = $AttentionBackend", launcher)
        self.assertIn('if ($AttentionBackend -eq "air")', launcher)
        self.assertIn("-AttentionBackend air requires -AIRAttentionDirectory.", launcher)
        self.assertIn('version = "0.2.0a1"', (root / "pyproject.toml").read_text())
        notes = (root / "docs" / "releases" / "0.2.0-alpha.1.md").read_text()
        for warning in ("29.035%", "15.975/255", "27.095", "18.765", "28.98"):
            self.assertIn(warning, notes)

    def test_backend_rejects_masks_and_cleans_partial_loads(self):
        backend = AIRAttentionBackend(Path("private-artifacts"))
        config = SimpleNamespace(num_heads=16, head_dim=128)
        inputs = SimpleNamespace(
            hidden_states=torch.empty(1, 1024, 1),
            encoder_hidden_states=torch.empty(1, 512, 1),
            attention_mask=None,
        )
        backend.validate(config, inputs)
        inputs.attention_mask = torch.ones(1)
        with self.assertRaisesRegex(UnsupportedTensor, "masks"):
            backend.validate(config, inputs)
        first = Mock(shape=AttentionShape(1024))
        with patch("anima_xdna_poc.air_attention.AIRAttention", side_effect=[first, ValueError("bad ELF")]):
            with self.assertRaisesRegex(ValueError, "bad ELF"):
                backend._open()
        first.close.assert_called_once()
        self.assertEqual(backend.runners, {})

    def test_elf_is_digest_verified_before_opening_xrt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "aie.elf").write_bytes(b"test-elf")
            manifest = {
                "shape": AttentionShape(512).__dict__,
                "artifacts": {"elf_path": "aie.elf", "elf_kernel_name": "main:anima_attention"},
                "elf_sha256": hashlib.sha256(b"test-elf").hexdigest(),
            }
            file = path / "manifest.json"
            file.write_text(json.dumps(manifest), encoding="utf-8")
            module = SimpleNamespace(MultiLaunchRunner=Mock())
            with patch.dict("sys.modules", {"triton.backends.amd_triton_npu.multilaunch": module}):
                runner = AIRAttention(file)
                module.MultiLaunchRunner.assert_called_once_with(
                    str(path / "aie.elf"), "main:anima_attention",
                )
                runner.close()
                module.MultiLaunchRunner.reset_mock()
                (path / "aie.elf").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    AIRAttention(file)
                module.MultiLaunchRunner.assert_not_called()

    def test_recorded_sampler_dispatch_and_sync_reduction(self):
        root = Path(__file__).resolve().parents[1]
        evidence = json.loads(
            (root / "docs" / "evidence" / "air-attention-model-probe.json").read_text()
        )
        self.assertEqual(evidence["status"], "completed")
        self.assertFalse(evidence["methodology"]["production_enabled"])
        self.assertFalse(evidence["implementation"]["full_score_probability_host_buffers"])
        self.assertEqual(evidence["conditions"]["modes"], ["baseline", "air_attention"])
        self.assertEqual(evidence["conditions"]["seed"], 700755787614899)
        self.assertTrue(evidence["cpu_oracle_recovery"]["all_112_air_replay_comparisons_unchanged"])
        self.assertEqual(len(evidence["timing_runs"]), 6)
        for mode, dispatches, h2d, d2h in (
            ("baseline", 2128, 16485711872, 35173433344),
            ("air_attention", 1904, 7090470912, 17322475520),
        ):
            runs = [r for r in evidence["timing_runs"] if r["mode"] == mode]
            self.assertEqual(len(runs), 3)
            self.assertEqual(
                statistics.median(r["sampler_wall_ms"] for r in runs),
                evidence["timing_medians_ms"][mode],
            )
            for run in runs:
                self.assertEqual(run["dispatches_per_step"], [dispatches // 4] * 4)
                self.assertEqual(run["profile"]["dispatches"], dispatches)
                sizes = run["profile"]["instrumented_sync_bytes"]
                self.assertEqual(sizes["h2d_sync_bytes"], h2d)
                self.assertEqual(sizes["d2h_sync_bytes"], d2h)
                if mode == "air_attention":
                    stages = run["profile"]["stages"]
                    self.assertNotIn("attn1.qk_batched", stages)
                    self.assertNotIn("attn2.av_batched", stages)
                    self.assertEqual(stages["attn1.air_attention"]["dispatches"], 112)
                    self.assertEqual(stages["attn2.air_attention"]["dispatches"], 112)

    def test_recorded_accuracy_retains_failed_gates_and_all_blocks(self):
        root = Path(__file__).resolve().parents[1]
        evidence = json.loads(
            (root / "docs" / "evidence" / "air-attention-model-probe.json").read_text()
        )
        air = next(c for c in evidence["captures"] if c["mode"] == "air_attention")
        self.assertAlmostEqual(
            air["final_latent_vs_baseline"]["normalized_rms"], 0.29035156183571814,
        )
        self.assertFalse(air["final_latent_vs_baseline"]["bitwise_equal"])
        self.assertGreater(air["image_rgb_mae_vs_baseline"], 15)
        for comparison in (air, evidence["fixed_input_replays"][0]):
            self.assertEqual(len(comparison["steps"]), 4)
            for step in comparison["steps"]:
                self.assertEqual(
                    [c["block"] for c in step["vs_baseline"]], list(range(28)),
                )
        self.assertEqual(len(evidence["cpu_oracles"]), 4)
        for errors, gate in (
            [(o["baseline_vs_cpu"], o["turbo_cpu_gate"]) for o in evidence["cpu_oracles"]]
            + [(s["vs_cpu"], s["turbo_cpu_gate"]) for s in evidence["fixed_input_replays"][0]["steps"]]
        ):
            self.assertEqual([c["block"] for c in errors], list(range(28)))
            self.assertEqual(gate["max_intermediate_nrms"], 0.10)
            self.assertEqual(gate["max_final_nrms"], 0.02)
            self.assertEqual(
                gate["passed"],
                max(e["normalized_rms"] for e in errors) <= 0.10
                and errors[-1]["normalized_rms"] <= 0.02,
            )
        for run in evidence["timing_runs"]:
            if run["mode"] == "air_attention":
                self.assertEqual(
                    run["final_latent_vs_baseline"], air["final_latent_vs_baseline"],
                )

    def test_recorded_hotpath_measures_whole_calls_and_preserves_roundtrip_counts(self):
        root = Path(__file__).resolve().parents[1]
        evidence = json.loads(
            (root / "docs" / "evidence" / "air-attention-hotpath-probe.json").read_text()
        )
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual((evidence["runs"], evidence["warmups"]), (21, 3))
        self.assertEqual(len(evidence["cases"]), 12)
        self.assertFalse(evidence["all_attention_cpu_gates_passed"])
        self.assertEqual(evidence["summary"]["cpu_gate_passed_cases"], 3)
        for case in evidence["cases"]:
            self.assertTrue(case["control_vs_captured"]["bitwise_equal"])
            self.assertTrue(all(case["repeat_outputs_bitwise_stable"].values()))
            self.assertEqual(
                case["cpu_gate_passed"],
                case["air_vs_cpu"]["normalized_rms"] <= 0.02,
            )
            self_attention = case["attention"] == "attn1"
            shape = AttentionShape(1024 if self_attention else 512)
            for mode in ("baseline", "air_attention"):
                samples = case["timing_samples"][mode]
                self.assertEqual(len(samples), 21)
                self.assertEqual(
                    statistics.median(s["wall_ms"] for s in samples),
                    case["timing_medians_ms"][mode],
                )
                for sample in samples:
                    low = sample["xrt"]
                    self.assertEqual(low["xrt_start_ms_calls"], 2 if mode == "baseline" else 1)
                    self.assertEqual(low["xrt_wait_ms_calls"], 2 if mode == "baseline" else 1)
                    if mode == "air_attention":
                        self.assertEqual(low["h2d_sync_bytes"], shape.h2d_bytes)
                        self.assertEqual(low["d2h_sync_bytes"], shape.d2h_bytes)
                    else:
                        self.assertEqual(low["h2d_sync_bytes"], (64 if self_attention else 36) * 1024**2)
                        self.assertEqual(low["d2h_sync_bytes"], (96 if self_attention else 64) * 1024**2)

    def test_fixed_shape_and_host_sync_budget(self):
        for keys, h2d in ((512, 8 * 1024**2), (1024, 12 * 1024**2)):
            shape = AttentionShape(keys)
            self.assertEqual(shape.h2d_bytes, h2d)
            self.assertEqual(shape.d2h_bytes, 4 * 1024**2)
            self.assertEqual(shape.core_working_bytes, 53504)
            self.assertLess(shape.core_working_bytes + 4096, 64 * 1024)
        for kwargs in (
            {"keys": 256}, {"keys": 512, "depth": 64},
            {"keys": 512, "parallel_heads": 4},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    AttentionShape(**kwargs)

    def test_runner_has_only_qkv_and_output_buffers(self):
        attention = AIRAttention.__new__(AIRAttention)
        attention.shape = AttentionShape(512)

        def run(arrays, **kwargs):
            self.assertEqual(len(arrays), 4)
            self.assertTrue(all(a.flags.c_contiguous for a in arrays))
            self.assertEqual([a.nbytes for a in arrays], [4194304, 2097152, 2097152, 4194304])
            self.assertEqual(kwargs["intermediate_indices"], {3})
            self.assertEqual(kwargs["output_indices"], {3})
            arrays[3].fill(1)
            return {3: arrays[3]}

        attention.runner = Mock()
        attention.runner.run.side_effect = run
        q = torch.ones(16, 1024, 128, dtype=torch.bfloat16)
        k = torch.ones(16, 512, 128, dtype=torch.bfloat16)
        output = attention.run(q, k, k)
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(bool((output == 1).all()))
        attention.runner.run.assert_called_once()
        strided_q = q.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(strided_q.is_contiguous())
        self.assertTrue(torch.equal(attention.run(strided_q, k, k), output))
        for invalid in (q.float(), q[:, :512], q * float("nan")):
            with self.subTest(dtype=invalid.dtype, shape=invalid.shape):
                with self.assertRaises(ValueError):
                    attention.run(invalid, k, k)

    @unittest.skipUnless(importlib.util.find_spec("air"), "MLIR-AIR bindings not installed")
    def test_direct_air_module_keeps_intermediates_on_device(self):
        for keys in (512, 1024):
            with self.subTest(keys=keys), tempfile.TemporaryDirectory() as directory:
                module = build_module(AttentionShape(keys), Path(directory) / "attention.o")
                self.assertTrue(module.operation.verify())
                text = str(module)
                entry = next(
                    line for line in text.splitlines()
                    if "func.func @anima_attention(" in line
                )
                self.assertEqual(entry.count("memref<"), 4)
                self.assertNotIn(f"memref<16x1024x{keys}xbf16", text)
                for line in text.splitlines():
                    if "memref.alloc()" in line:
                        self.assertTrue(", 1 : i32>" in line or ", 2 : i32>" in line)
                self.assertIn("memref<32x64xbf16, 2 : i32>", text)
                self.assertIn("memref<32x128xbf16, 2 : i32>", text)
                self.assertIn("memref<128x128xbf16, 1 : i32>", text)
                for kernel in ("matmul_a_b_bf16", "fused_softmax", "matmul_g_b_bf16"):
                    self.assertIn("func.call @" + kernel, text)


if __name__ == "__main__":
    unittest.main()
