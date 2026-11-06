"""Session-only AVX-512 fused dequant/BF16 erf-GELU/per-token quant probe."""

import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F


class FusedMiddle:
    def __init__(self, rows=1024, columns=8192, threads=10):
        if rows <= 0 or columns <= 0 or columns % 16 or not 1 <= threads <= 64:
            raise ValueError("Positive dimensions, columns divisible by 16, and 1..64 threads required")
        self.library = ctypes.CDLL(str(Path(__file__).with_suffix(".dll")))
        self.library.avx512_supported.restype = ctypes.c_int
        if not self.library.avx512_supported():
            raise RuntimeError("The probe requires OS-enabled AVX-512 F/DQ/BW/VL")
        self.function = self.library.fused_middle
        self.function.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3
        self.function.restype = ctypes.c_int
        self.rows, self.columns, self.threads = rows, columns, threads
        patterns = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        self.lut = F.gelu(patterns, approximate="none").float().contiguous()
        self.scratch = torch.empty(threads, columns, dtype=torch.float32)
        self.output = torch.empty(rows, columns, dtype=torch.int8)
        self.scales = torch.empty(rows, 1, dtype=torch.float32)

    def __call__(self, accumulator, row_scale, weight_scale):
        for tensor, dtype, shape in (
            (accumulator, torch.int32, (self.rows, self.columns)),
            (row_scale, torch.float32, (self.rows, 1)),
            (weight_scale, torch.float32, (1, self.columns)),
        ):
            if tensor.device.type != "cpu" or tensor.dtype != dtype:
                raise ValueError(f"Expected contiguous CPU {dtype}, got {tensor.device} {tensor.dtype}")
            if tuple(tensor.shape) != shape or not tensor.is_contiguous():
                raise ValueError(f"Expected contiguous {shape}, got {tensor.shape}")
        code = self.function(*[
            tensor.data_ptr() for tensor in (
                accumulator, row_scale, weight_scale, self.lut, self.output,
                self.scales, self.scratch,
            )
        ], self.rows, self.columns, self.threads)
        if code:
            raise RuntimeError(f"Fused middle rejected invalid arguments/scales/nonfinite output (code {code})")
        # Borrowed buffers, consumed before the next call; one instance per caller.
        return self.output, self.scales
