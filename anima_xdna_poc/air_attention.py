"""Explicit fixed-shape AIR attention runtime; no compiler or host fallback."""

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

from .errors import UnsupportedTensor


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttentionShape:
    keys: int
    heads: int = 16
    queries: int = 1024
    depth: int = 128
    query_tile: int = 32
    key_tile: int = 64
    query_cores: int = 4
    parallel_heads: int = 2

    def __post_init__(self):
        if (
            self.keys not in (512, 1024)
            or self.heads != 16
            or self.queries != 1024
            or self.depth != 128
            or self.query_tile != 32
            or self.key_tile != 64
            or self.query_cores != 4
            or self.parallel_heads not in (1, 2)
        ):
            raise ValueError("unsupported fixed Anima attention schedule")

    @property
    def query_group(self):
        return self.query_tile * self.query_cores

    @property
    def h2d_bytes(self):
        return 2 * self.heads * self.depth * (self.queries + 2 * self.keys)

    @property
    def d2h_bytes(self):
        return 2 * self.heads * self.queries * self.depth

    @property
    def core_working_bytes(self):
        return 2 * (
            self.query_tile * self.depth
            + 2 * self.query_tile * self.depth
            + self.key_tile * self.depth
            + self.query_tile * self.key_tile
            + self.query_tile * self.depth
            + 4 * self.query_tile
        )


class AIRAttention:
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve()
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        self.shape = AttentionShape(**self.manifest["shape"])
        artifacts = self.manifest["artifacts"]
        elf = Path(artifacts["elf_path"])
        if not elf.is_absolute():
            elf = path.parent / elf
        digest = self.manifest.get("elf_sha256")
        if digest is None or hashlib.sha256(elf.read_bytes()).hexdigest() != digest:
            raise ValueError(f"AIR ELF integrity check failed: {path}")
        if artifacts["elf_kernel_name"] != "main:anima_attention":
            raise ValueError(f"unexpected AIR kernel name: {path}")
        from triton.backends.amd_triton_npu.multilaunch import MultiLaunchRunner

        self.runner = MultiLaunchRunner(str(elf), artifacts["elf_kernel_name"])

    def run(self, query, key, value):
        import ml_dtypes
        import numpy as np
        import torch

        s = self.shape
        for name, tensor, rows in (
            ("query", query, s.queries), ("key", key, s.keys), ("value", value, s.keys),
        ):
            if tensor.shape != (s.heads, rows, s.depth) or tensor.dtype != torch.bfloat16:
                raise ValueError(f"unexpected {name} shape/dtype: {tensor.shape}, {tensor.dtype}")
            if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite CPU BF16")
        arrays = [
            tensor.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
            for tensor in (query, key, value)
        ] + [np.empty((s.heads, s.queries, s.depth), dtype=ml_dtypes.bfloat16)]
        got = self.runner.run(
            arrays, bo_key="attention", intermediate_indices={3}, output_indices={3},
        )[3]
        output = torch.from_numpy(got.view(np.uint16)).view(torch.bfloat16).clone()
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("AIR attention returned nonfinite output")
        return output

    def close(self):
        self.runner.unload()


class AIRAttentionBackend:
    """One persistent self/cross runner pair owned by a chain runtime."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.runners = {}

    def validate(self, config, inputs):
        if (
            config.num_heads != 16 or config.head_dim != 128
            or inputs.hidden_states.shape[1] != 1024
            or inputs.encoder_hidden_states.shape[1] != 512
        ):
            raise UnsupportedTensor(
                "AIR attention requires batch 1, 1024 image tokens, "
                "512 context tokens and 16 heads of depth 128"
            )
        if inputs.attention_mask is not None:
            raise UnsupportedTensor("AIR attention does not support masks; no fallback")

    def _open(self):
        if self.runners:
            return
        try:
            for keys, name in ((1024, "self"), (512, "cross")):
                runner = AIRAttention(self.directory / name / "manifest.json")
                self.runners[keys] = runner
                if runner.shape.keys != keys:
                    raise ValueError(f"AIR {name} manifest has incorrect key length")
        except Exception:
            self.close()
            raise
        LOGGER.warning(
            "[Anima XDNA] EXPERIMENTAL AIR attention ACTIVE: tiled QK/scale/softmax/AV, "
            "17 dispatches/block with QKV chaining. Known fixed-seed latent NRMS "
            "vs control 29.035%; not quality-equivalent. No silent fallback."
        )

    def __call__(self, query, key, value):
        keys = key.shape[1]
        if keys not in (512, 1024):
            raise UnsupportedTensor(f"unsupported AIR attention key length: {keys}")
        self._open()
        return self.runners[keys].run(query, key, value)

    def close(self):
        try:
            for runner in self.runners.values():
                runner.close()
        finally:
            self.runners.clear()
