import json
from pathlib import Path
import unittest

import ml_dtypes
import numpy as np
import torch

from examples.qk_scale_model_probe import finish_qk_output


def bf16_array(tensor):
    return tensor.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)


class DirectBF16OutputTests(unittest.TestCase):
    def test_recorded_direct_output_cases_preserve_rounding_and_sizes(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "docs" / "evidence" / "qk-direct-bf16-probe.json"
        )
        evidence = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(evidence["execution_completed"])
        self.assertEqual(evidence["status"], "completed_with_cpu_gate_failures")
        self.assertEqual(evidence["cpu_elementwise_gate"], {"rtol": 0.02, "atol": 0.03})
        self.assertEqual(len(evidence["cases"]), 10)
        for attention in ("self", "cross"):
            cases = {
                case["mode"]: case for case in evidence["cases"]
                if case["attention"] == attention
            }
            baseline = cases["baseline"]
            direct = cases["qk_bf16"]
            self.assertTrue(direct["unscaled_bf16_equal_to_control"])
            self.assertTrue(direct["vs_unfused_control"]["bitwise_equal"])
            self.assertEqual(direct["vs_unfused_control"]["normalized_rms"], 0)
            for mode, case in cases.items():
                self.assertEqual(len(case["warm_ms"]), 21)
                self.assertEqual(len(case["phase_samples_ms"]), 21)
                self.assertFalse(case["cpu_gate_passed"])
                self.assertEqual(case["h2d_bytes_per_call"], baseline["h2d_bytes_per_call"])
                self.assertEqual(case["dispatches_per_call"], 1)
                if mode in ("qk_bf16", "direct_bf16", "legacy_bf16"):
                    self.assertEqual(
                        case["d2h_visible_bytes_per_call"] * 2,
                        baseline["d2h_visible_bytes_per_call"],
                    )
                    self.assertEqual(case["phase_medians_ms"]["host_cast_ms"], 0)
                else:
                    self.assertGreater(case["phase_medians_ms"]["host_cast_ms"], 0)
            self.assertEqual(direct["phase_medians_ms"]["host_cast_ms"], 0)
            self.assertGreater(direct["phase_medians_ms"]["host_scale_ms"], 0)
            self.assertFalse(cases["direct_bf16"]["vs_unfused_control"]["bitwise_equal"])

    def test_direct_qk_output_keeps_host_rounding_and_scaling(self):
        scores = np.array([0.999, 1.003, -1.007, 17.125], dtype=np.float32)
        expected, _, expected_unscaled = finish_qk_output(scores, "baseline")
        direct, phases, unscaled = finish_qk_output(bf16_array(expected_unscaled), "qk_bf16")
        self.assertTrue(torch.equal(direct, expected))
        self.assertTrue(torch.equal(unscaled, expected_unscaled))
        self.assertEqual(phases["host_cast_ms"], 0)

    def test_scaled_bf16_outputs_need_neither_host_cast_nor_scale(self):
        scores = torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16)
        for mode in ("direct_bf16", "legacy_bf16"):
            with self.subTest(mode=mode):
                got, phases, unscaled = finish_qk_output(bf16_array(scores), mode)
                self.assertTrue(torch.equal(got, scores))
                self.assertIsNone(unscaled)
                self.assertEqual(phases["host_cast_ms"], 0)
                self.assertEqual(phases["host_scale_ms"], 0)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            finish_qk_output(np.ones(1, dtype=np.float32), "unknown")


if __name__ == "__main__":
    unittest.main()
