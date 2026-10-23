from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from anima_xdna_poc.block import StageMetric
from examples.qk_scale_model_probe import (
    Probe, decoded_image_rgb, profile_summary, tensor_error,
)


class _Direction:
    TO_DEVICE = 1
    FROM_DEVICE = 2


class _Buffer:
    def __init__(self):
        self.directions = []

    def size(self):
        return 64

    def sync(self, direction):
        self.directions.append(direction)


class _Run:
    def start(self):
        pass

    def wait2(self):
        pass


class _Runner:
    def __init__(self):
        self.buffer = _Buffer()

    def run(self):
        self.buffer.sync(_Direction.TO_DEVICE)
        run = _Run()
        run.start()
        run.wait2()
        self.buffer.sync(_Direction.FROM_DEVICE)
        return 7


class QKModelProbeTests(unittest.TestCase):
    def test_air_host_bypass_never_changes_cpu_oracle(self):
        from anima_xdna_poc.block import _BlockExecution

        probe = Probe(None)
        probe.mode = "air_attention"
        fake_module = SimpleNamespace(MultiLaunchRunner=_Runner)
        with patch.dict(
            "sys.modules",
            {"triton.backends.amd_triton_npu.multilaunch": fake_module},
        ):
            try:
                probe.install()
                execution = _BlockExecution.__new__(_BlockExecution)
                execution.metrics = []
                execution.runner_device = "host"
                scores = torch.tensor([[0.0, 1.0]])
                operation = lambda: torch.softmax(scores, dim=-1)
                self.assertTrue(torch.equal(
                    execution.host("attn1.softmax", operation), operation(),
                ))
                mask = lambda: scores.masked_fill(scores == 0, float("-inf"))
                self.assertTrue(torch.equal(
                    execution.host("attn1.mask", mask), mask(),
                ))
                execution.runner_device = "xdna2"
                self.assertIs(execution.host("attn1.softmax", operation), scores)
                with self.assertRaisesRegex(ValueError, "does not support masks"):
                    execution.host("attn1.mask", mask)
            finally:
                probe.restore()

    def test_recorded_qk_direct_output_preserves_every_real_block(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs" / "evidence" / "qk-direct-bf16-model-probe.json"
        )
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["conditions"]["modes"], ["baseline", "qk_bf16", "direct_bf16"])
        self.assertEqual(evidence["conditions"]["seed"], 700755787614899)
        self.assertTrue(all(evidence["cross_experiment_comparison"].values()))
        captures = {c["mode"]: c for c in evidence["captures"]}
        direct = captures["qk_bf16"]
        self.assertTrue(direct["final_latent_vs_baseline"]["bitwise_equal"])
        self.assertEqual(direct["image_rgb_sha256"], captures["baseline"]["image_rgb_sha256"])
        for callback in direct["callbacks_vs_baseline"]:
            self.assertTrue(callback["current"]["bitwise_equal"])
            self.assertTrue(callback["denoised"]["bitwise_equal"])
        replay = next(c for c in evidence["fixed_input_replays"] if c["mode"] == "qk_bf16")
        for comparison in (direct, replay):
            self.assertEqual(len(comparison["steps"]), 4)
            for step in comparison["steps"]:
                self.assertEqual(
                    [c["block"] for c in step["vs_baseline"]], list(range(28)),
                )
                self.assertTrue(all(c["bitwise_equal"] for c in step["vs_baseline"]))
                self.assertTrue(all(c["normalized_rms"] == 0 for c in step["vs_baseline"]))
        self.assertEqual(len(evidence["timing_runs"]), 9)
        for run in evidence["timing_runs"]:
            self.assertEqual(run["dispatches_per_step"], [532] * 4)
            if run["mode"] == "qk_bf16":
                self.assertTrue(run["final_latent_vs_baseline"]["bitwise_equal"])
        sizes = evidence["performance_summary"]
        self.assertEqual(
            sizes["baseline"]["instrumented_sync_bytes"]["d2h_sync_bytes"]
            - sizes["qk_bf16"]["instrumented_sync_bytes"]["d2h_sync_bytes"],
            5637144576,
        )
        for comparison in evidence["fixed_input_replays"]:
            self.assertEqual(
                [s["turbo_cpu_gate"]["passed"] for s in comparison["steps"]],
                [True, False, False, False],
            )

    def test_recorded_evidence_retains_all_steps_and_failed_gates(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs" / "evidence" / "qk-scale-model-probe.json"
        )
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["conditions"]["seed"], 700755787614899)
        self.assertEqual(evidence["conditions"]["size"], [512, 512])
        self.assertEqual(evidence["conditions"]["steps"], 4)
        self.assertFalse(evidence["methodology"]["production_enabled"])
        self.assertEqual(len(evidence["timing_runs"]), 9)
        for run in evidence["timing_runs"]:
            self.assertEqual(run["dispatches_per_step"], [532] * 4)
            self.assertEqual(run["profile"]["dispatches"], 2128)
            self.assertEqual(
                sum(g["weight_population_bytes"] for g in run["profile"]["groups"].values()),
                0,
            )
        self.assertEqual(len(evidence["cpu_oracles"]), 4)
        comparisons = [
            (oracle["baseline_vs_cpu"], oracle["turbo_cpu_gate"])
            for oracle in evidence["cpu_oracles"]
        ]
        self.assertEqual(
            [oracle["turbo_cpu_gate"]["passed"] for oracle in evidence["cpu_oracles"]],
            [True, False, False, False],
        )
        self.assertEqual(len(evidence["fixed_input_replays"]), 2)
        for replay in evidence["fixed_input_replays"]:
            self.assertEqual(len(replay["steps"]), 4)
            for step in replay["steps"]:
                self.assertEqual(
                    [c["block"] for c in step["vs_baseline"]], list(range(28)),
                )
                comparisons.append((step["vs_cpu"], step["turbo_cpu_gate"]))
        for errors, gate in comparisons:
            self.assertEqual([c["block"] for c in errors], list(range(28)))
            self.assertEqual(gate["max_intermediate_nrms"], 0.10)
            self.assertEqual(gate["max_final_nrms"], 0.02)
            self.assertEqual(
                gate["passed"],
                max(c["normalized_rms"] for c in errors) <= 0.10
                and errors[-1]["normalized_rms"] <= 0.02,
            )
        self.assertTrue(evidence["cross_variant_comparison"]["final_latents_bitwise_equal"])

    def test_decoded_image_accepts_only_one_finite_rgb_frame(self):
        image = torch.full((1, 2, 2, 3), 0.5)
        rgb = decoded_image_rgb(image)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertTrue((rgb == 127).all())
        self.assertTrue((decoded_image_rgb(image[:, None]) == rgb).all())
        for invalid in (
            torch.zeros(1, 2, 2, 2, 3),
            torch.zeros(2, 2, 2, 3),
            torch.zeros(1, 2, 2, 4),
            torch.full((1, 2, 2, 3), float("nan")),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(ValueError):
                    decoded_image_rgb(invalid)

    def test_tensor_comparison_checks_finite_shapes_and_reference(self):
        reference = torch.tensor([1.0, 2.0])
        error = tensor_error(reference, reference)
        self.assertTrue(error["bitwise_equal"])
        self.assertEqual(error["normalized_rms"], 0)
        for actual, expected in (
            (torch.ones(3), reference),
            (torch.tensor([float("nan"), 1.0]), reference),
            (reference, torch.zeros(2)),
        ):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(ValueError):
                    tensor_error(actual, expected)

    def test_stage_summary_keeps_wall_and_nested_xrt_times_separate(self):
        stages = [
            StageMetric("attn1.qk_batched", "xdna2", 10, 1, 0, 0, 128, 256),
            StageMetric("attn2.softmax", "host", 5, 0, 0, 0),
            StageMetric("norm1.silu", "host", 2, 0, 0, 0),
            StageMetric("ff.proj_out", "xdna2", 20, 4, 0, 0),
        ]
        report = profile_summary([{
            "blocks": [{
                "stages": [asdict(stage) for stage in stages],
                "low_level": {
                    "attn1.qk_batched": {
                        "xrt_wait_ms": 3,
                        "h2d_sync_bytes": 128,
                        "d2h_sync_bytes": 256,
                    },
                },
            }],
        }])
        self.assertEqual(report["dispatches"], 5)
        self.assertEqual(report["groups"]["self_attention"]["wall_ms"], 10)
        self.assertEqual(report["groups"]["cross_attention"]["wall_ms"], 5)
        self.assertEqual(report["groups"]["adaln"]["wall_ms"], 2)
        self.assertEqual(report["instrumented_sync_bytes"]["d2h_sync_bytes"], 256)
        self.assertEqual(report["xrt_by_stage"]["attn1.qk_batched"]["xrt_wait_ms"], 3)

    def test_ast_instrumentation_preserves_calls_and_collects_bytes(self):
        probe = Probe(None)
        probe.enabled = True
        probe.stage = "attn1.qk_batched"
        original = _Runner.run
        try:
            probe.instrument_xrt(_Runner)
            runner = _Runner()
            self.assertEqual(runner.run(), 7)
            self.assertEqual(runner.buffer.directions, [1, 2])
            measured = probe.low["attn1.qk_batched"]
            self.assertEqual(measured["h2d_sync_bytes"], 64)
            self.assertEqual(measured["d2h_sync_bytes"], 64)
            self.assertEqual(measured["xrt_wait_ms_calls"], 1)
            self.assertEqual(measured["xrt_start_ms_calls"], 1)
        finally:
            probe.restore()
        self.assertIs(_Runner.run, original)


if __name__ == "__main__":
    unittest.main()
