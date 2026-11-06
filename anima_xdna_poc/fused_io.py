"""Session-only AVX-512 fused MLP entry (modulate+quantize) and exit (dequant+gate+residual)."""

import ctypes
from pathlib import Path

import torch


def _row_vector(tensor, name, columns):
    if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a CPU BF16 tensor, got {tensor.device} {tensor.dtype}")
    if tensor.numel() != columns or tensor.stride(-1) != 1:
        raise ValueError(f"{name} must hold {columns} unit-stride values, got {tuple(tensor.shape)}")
    return tensor


class FusedIO:
    def __init__(self, rows=1024, columns=2048, threads=10):
        if rows <= 0 or columns <= 0 or columns % 16 or not 1 <= threads <= 64:
            raise ValueError("Positive dimensions, columns divisible by 16, and 1..64 threads required")
        self.library = ctypes.CDLL(str(Path(__file__).with_name("fused_io.dll")))
        self.library.avx512_supported.restype = ctypes.c_int
        if not self.library.avx512_supported():
            raise RuntimeError("The probe requires OS-enabled AVX-512 F/DQ/BW/VL")
        self.entry = self.library.modulate_quantize
        self.entry.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
        self.entry.restype = ctypes.c_int
        self.exit = self.library.dequant_gated_residual
        self.exit.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 3
        self.exit.restype = ctypes.c_int
        self.rows, self.columns, self.threads = rows, columns, threads
        self.scratch = torch.empty(2 + threads, columns, dtype=torch.float32)
        self.quantized = torch.empty(rows, columns, dtype=torch.int8)
        self.scales = torch.empty(rows, 1, dtype=torch.float32)

    def _tensor(self, tensor, dtype, shape, name):
        if tensor.device.type != "cpu" or tensor.dtype != dtype:
            raise ValueError(f"{name} must be a CPU {dtype} tensor, got {tensor.device} {tensor.dtype}")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {shape}, got {tuple(tensor.shape)}")

    def modulate_quantize(self, normalized, shift, scale):
        """Return borrowed (int8 [rows, columns], float32 [rows, 1]); one instance per caller."""
        self._tensor(normalized, torch.bfloat16, (1, self.rows, self.columns), "normalized")
        shift = _row_vector(shift, "shift", self.columns)
        scale = _row_vector(scale, "scale", self.columns)
        code = self.entry(
            normalized.data_ptr(), shift.data_ptr(), scale.data_ptr(),
            self.quantized.data_ptr(), self.scales.data_ptr(), self.scratch.data_ptr(),
            self.rows, self.columns, self.threads,
        )
        if code:
            raise RuntimeError(f"Fused modulate+quantize rejected invalid input or nonfinite output (code {code})")
        return self.quantized, self.scales

    def dequant_gated_residual(self, accumulator, row_scale, weight_scale, gate, hidden):
        """Return a new BF16 [1, rows, columns] tensor: hidden + gate * dequantized."""
        self._tensor(accumulator, torch.int32, (self.rows, self.columns), "accumulator")
        self._tensor(row_scale, torch.float32, (self.rows, 1), "row_scale")
        self._tensor(weight_scale, torch.float32, (1, self.columns), "weight_scale")
        self._tensor(hidden, torch.bfloat16, (1, self.rows, self.columns), "hidden")
        gate = _row_vector(gate, "gate", self.columns)
        output = torch.empty_like(hidden)
        code = self.exit(
            accumulator.data_ptr(), row_scale.data_ptr(), weight_scale.data_ptr(),
            gate.data_ptr(), hidden.data_ptr(), output.data_ptr(),
            self.rows, self.columns, self.threads,
        )
        if code:
            raise RuntimeError(f"Fused dequant+gated residual rejected invalid input or nonfinite output (code {code})")
        return output
