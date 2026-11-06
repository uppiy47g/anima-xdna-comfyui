"""Strict fixed-shape host boundaries for native INT8 attention projections."""

import ctypes
from pathlib import Path

import torch

from .fused_qk import FusedQK, reference_prepare


def quantize_reference(value):
    value = value.float()
    scale = value.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
    return torch.round(value / scale).clamp_(-127, 127).to(torch.int8), scale


def dequantize_reference(accumulator, row_scale, weight_scale):
    return accumulator.float().mul_(row_scale).mul_(weight_scale).bfloat16()


def prepare_reference(accumulators, row_scales, weight_scales, query_weight, key_weight, rotary=None):
    values = [dequantize_reference(*parts) for parts in zip(accumulators, row_scales, weight_scales)]
    return reference_prepare(*values, query_weight, key_weight, rotary)


class FusedAttention:
    def __init__(self, threads=10):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer in 1..64")
        self.threads = threads
        self.library = ctypes.CDLL(str(Path(__file__).with_name("fused_attention.dll")))
        self.library.avx512_supported.restype = ctypes.c_int
        if not self.library.avx512_supported():
            raise RuntimeError("OS-enabled AVX-512 F/DQ/BW/VL is required")
        self.quantize_function = self.library.quantize_attention_input
        self.quantize_function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4
        self.quantize_function.restype = ctypes.c_int
        self.prepare_function = self.library.prepare_attention_i32
        self.prepare_function.argtypes = [ctypes.c_void_p] * 16 + [ctypes.c_int] * 3
        self.prepare_function.restype = ctypes.c_int
        FusedQK._check_reduction_contract()

    def quantize(self, tensor, head_major=False):
        if type(head_major) is not bool:
            raise TypeError("head_major must be bool")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("activation must be a torch.Tensor")
        if head_major:
            FusedQK._tensor(tensor, (16, 1024, 128), torch.bfloat16, "AIR output")
            rows, columns = 1024, 2048
        else:
            if tensor.ndim not in (2, 3):
                raise ValueError("activation must have two dimensions or batch-one three dimensions")
            rows, columns = tensor.shape[-2:]
            if rows not in (512, 1024) or columns not in (1024, 2048):
                raise ValueError("Unsupported activation shape")
            shape = (rows, columns) if tensor.ndim == 2 else (1, rows, columns)
            FusedQK._tensor(tensor, shape, torch.bfloat16, "activation")
        values = torch.empty(rows, columns, dtype=torch.int8)
        scales = torch.empty(rows, 1, dtype=torch.float32)
        code = self.quantize_function(
            tensor.data_ptr(), values.data_ptr(), scales.data_ptr(),
            rows, columns, int(head_major), self.threads,
        )
        if code:
            raise RuntimeError(f"Attention quantization rejected invalid/nonfinite input (code {code})")
        return values, scales

    def prepare(self, accumulators, row_scales, weight_scales, query_weight, key_weight, rotary=None):
        """Newly owned head-major BF16 outputs; INT32 inputs may borrow NPU BO storage."""
        for name, sequence in (("accumulators", accumulators), ("row_scales", row_scales),
                               ("weight_scales", weight_scales)):
            if not isinstance(sequence, (tuple, list)) or len(sequence) != 3:
                raise ValueError(f"{name} requires exactly three tensors")
        if not isinstance(accumulators[1], torch.Tensor) or accumulators[1].ndim != 2:
            raise ValueError("key accumulator must be two-dimensional")
        keys = accumulators[1].shape[0]
        if keys not in (512, 1024):
            raise ValueError("key accumulator requires 512 or 1024 tokens")
        for i, rows in enumerate((1024, keys, keys)):
            FusedQK._tensor(accumulators[i], (rows, 2048), torch.int32, f"accumulator[{i}]")
            FusedQK._tensor(row_scales[i], (rows, 1), torch.float32, f"row_scale[{i}]")
            FusedQK._tensor(weight_scales[i], (1, 2048), torch.float32, f"weight_scale[{i}]")
        for name, weight in (("query_weight", query_weight), ("key_weight", key_weight)):
            FusedQK._tensor(weight, (128,), torch.bfloat16, name)
        cosine = sine = None
        if rotary is not None:
            if keys != 1024 or not isinstance(rotary, tuple) or len(rotary) != 2:
                raise ValueError("RoPE requires self-attention and a (cosine, sine) tuple")
            cosine, sine = rotary
            for name, part in (("cosine", cosine), ("sine", sine)):
                FusedQK._tensor(part, (1024, 128), torch.float32, name)
        outputs = tuple(torch.empty(16, rows, 128, dtype=torch.bfloat16) for rows in (1024, keys, keys))
        code = self.prepare_function(
            *(x.data_ptr() for x in (*accumulators, *row_scales, *weight_scales, query_weight, key_weight)),
            cosine.data_ptr() if cosine is not None else None,
            sine.data_ptr() if sine is not None else None,
            *(x.data_ptr() for x in outputs), 1024, keys, self.threads,
        )
        if code:
            raise RuntimeError(f"INT8 QKV preparation rejected invalid scale/nonfinite result (code {code})")
        return outputs
