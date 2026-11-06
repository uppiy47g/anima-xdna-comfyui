import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from anima_xdna_poc.int8_runtime import (
    PROFILE,
    _artifact_paths,
    bitwise_equal,
    install_profile,
    quant_rows,
)
from anima_xdna_poc.vae_optimization import _restrided_upsample_forward, optimize_vae


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
