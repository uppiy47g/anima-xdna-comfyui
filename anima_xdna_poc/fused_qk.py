"""Session-only CPU RMSNorm/RoPE fusion producing contiguous AIR Q/K/V inputs."""

import ctypes
from pathlib import Path

import torch


def reference_prepare(query, key, value, query_weight, key_weight, rotary=None):
    tensors = [x.view(1, -1, 16, 128).transpose(1, 2) for x in (query, key, value)]
    for i, weight in enumerate((query_weight, key_weight)):
        x = tensors[i]
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = (x * torch.rsqrt(variance + 1e-6)).bfloat16() * weight
        if rotary is not None:
            cosine, sine = (part[None, None] for part in rotary)
            first, second = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            rotated = torch.cat((-second, first), dim=-1)
            x = (x.float() * cosine + rotated.float() * sine).bfloat16()
        tensors[i] = x
    return tuple(x.squeeze(0).contiguous() for x in tensors)


class FusedQK:
    def __init__(self, threads=10):
        if not isinstance(threads, int) or not 1 <= threads <= 64:
            raise ValueError("threads must be an integer in 1..64")
        self.threads = threads
        self.library = ctypes.CDLL(str(Path(__file__).with_name("fused_qk.dll")))
        self.library.avx512_supported.restype = ctypes.c_int
        if not self.library.avx512_supported():
            raise RuntimeError("OS-enabled AVX-512 F/DQ/BW/VL is required")
        self.function = self.library.prepare_attention
        self.function.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_int] * 3
        self.function.restype = ctypes.c_int
        self._check_reduction_contract()

    @staticmethod
    def _check_reduction_contract():
        generator = torch.Generator().manual_seed(20261106)
        x = torch.randn(256, 128, generator=generator).bfloat16().float().square()
        chunks = x.reshape(256, 16, 8)
        sums = [torch.zeros(256, 8) for _ in range(4)]
        for i in range(16):
            sums[i % 4] = sums[i % 4] + chunks[:, i]
        combined = ((sums[0] + sums[1]) + sums[2]) + sums[3]
        total = torch.zeros(256)
        for i in range(8):
            total = total + combined[:, i]
        if not torch.equal(total / 128, x.mean(-1)):
            raise RuntimeError("PyTorch depth-128 reduction differs from the fused kernel; no fallback")

    @staticmethod
    def _tensor(tensor, shape, dtype, name):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cpu" or tensor.dtype != dtype:
            raise ValueError(f"{name} requires CPU {dtype}, got {tensor.device} {tensor.dtype}")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} requires contiguous {shape}, got {tuple(tensor.shape)}")

    def __call__(self, query, key, value, query_weight, key_weight, rotary=None):
        """Return newly owned Q/K/V [16, tokens, 128]; no borrowed output buffers."""
        if not isinstance(key, torch.Tensor) or key.ndim not in (2, 3) or key.shape[-2] not in (512, 1024):
            raise ValueError("key requires 512 or 1024 tokens")
        keys = key.shape[-2]
        for name, tensor, rows in (("query", query, 1024), ("key", key, keys), ("value", value, keys)):
            shape = (rows, 2048) if isinstance(tensor, torch.Tensor) and tensor.ndim == 2 else (1, rows, 2048)
            self._tensor(tensor, shape, torch.bfloat16, name)
        for name, weight in (("query_weight", query_weight), ("key_weight", key_weight)):
            self._tensor(weight, (128,), torch.bfloat16, name)
        cosine = sine = None
        if rotary is not None:
            if keys != 1024 or not isinstance(rotary, tuple) or len(rotary) != 2:
                raise ValueError("RoPE requires self-attention and a (cosine, sine) tuple")
            cosine, sine = rotary
            for name, tensor in (("cosine", cosine), ("sine", sine)):
                self._tensor(tensor, (1024, 128), torch.float32, name)
        outputs = tuple(torch.empty(16, rows, 128, dtype=torch.bfloat16) for rows in (1024, keys, keys))
        code = self.function(
            query.data_ptr(), key.data_ptr(), value.data_ptr(),
            query_weight.data_ptr(), key_weight.data_ptr(),
            cosine.data_ptr() if cosine is not None else None,
            sine.data_ptr() if sine is not None else None,
            *(x.data_ptr() for x in outputs), 1024, keys, self.threads,
        )
        if code:
            raise RuntimeError(f"Fused QK preparation rejected invalid/nonfinite input or output (code {code})")
        return outputs
