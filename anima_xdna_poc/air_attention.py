"""Explicit fixed-shape AIR attention runtime; no compiler or host fallback."""

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

from .errors import UnsupportedTensor


LOGGER = logging.getLogger(__name__)


def _all_finite(tensor):
    """Reject NaN/Inf with one FP32 reduction instead of four elementwise passes.

    Any NaN or Inf propagates into the sum. A finite tensor whose FP32 sum
    overflows is also rejected, which fails loudly rather than silently.
    """
    import torch

    return bool(torch.isfinite(tensor.sum(dtype=torch.float32)))


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
            self.keys not in (512, 1024, 4096)
            or self.heads != 16
            or self.queries != 1024
            or self.depth != 128
            or self.query_tile != 32
            or self.key_tile != 64
            or self.query_cores not in (4, 8)
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
        self._init_resident()

    def _init_resident(self):
        self._query_bos, self._queries = [], []
        self._kv_bos, self._kv = [], []
        self._output_bos, self._outputs = [], []

    def _resident_buffer(self, shape):
        import numpy as np
        import torch

        count = 1
        for size in shape:
            count *= size
        bo = self.runner._xrt.ext.bo(self.runner.device, 2 * count)
        view = np.frombuffer(bo.map(), dtype=np.int16, count=count)
        return bo, torch.from_numpy(view).view(torch.bfloat16).reshape(shape)

    def resident(self, tiles):
        """BO-backed ``(query tiles, key, value)`` that ``start`` reads without a staging copy.

        The caller writes each launch's inputs into these tensors first. They are
        reused by every later call and stay valid until ``close``. The sequential
        ``run`` buffer set is released on every call; ``run`` re-creates it if needed.
        """
        s = self.shape
        if type(tiles) is not int or not 1 <= tiles <= 4:
            raise ValueError("AIR resident buffers support 1..4 query tiles")
        # The cross runner is shared with 512x512, whose sequential run() may have re-created it.
        self.runner._bos.pop("attention", None)
        try:
            if not self._outputs:
                for _ in range(2):
                    bo, view = self._resident_buffer((s.heads, s.keys, s.depth))
                    self._kv_bos.append(bo)
                    self._kv.append(view)
                to_device = self.runner._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
                for _ in range(2):
                    bo, view = self._resident_buffer((s.heads, s.queries, s.depth))
                    view.zero_()
                    bo.sync(to_device)
                    self._output_bos.append(bo)
                    self._outputs.append(view)
            while len(self._queries) < tiles:
                bo, view = self._resident_buffer((s.heads, s.queries, s.depth))
                self._query_bos.append(bo)
                self._queries.append(view)
        except Exception:
            self._release_resident()
            raise
        return tuple(self._queries[:tiles]), self._kv[0], self._kv[1]

    def start(self, tile, slot, upload_kv):
        """Launch AIR on resident query ``tile`` into output ``slot`` (0 or 1) without waiting.

        ``upload_kv`` syncs the resident K/V first; later tiles of the same call reuse
        them. Only one NPU job may be in flight; ``finish`` must follow.
        """
        if slot not in (0, 1) or not self._outputs or not 0 <= tile < len(self._queries):
            raise ValueError("AIR start requires resident buffers for this tile and slot 0 or 1")
        xrt = self.runner._xrt
        to_device = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        if upload_kv:
            for bo in self._kv_bos:
                bo.sync(to_device)
        self._query_bos[tile].sync(to_device)
        run = xrt.run(self.runner.kernel)
        for index, bo in enumerate((self._query_bos[tile], *self._kv_bos, self._output_bos[slot])):
            run.set_arg(index, bo)
        run.start()
        return run, slot

    def finish(self, pending):
        """Wait for ``start``; the returned view stays valid until that slot is launched again."""
        run, slot = pending
        run.wait2()
        self._output_bos[slot].sync(self.runner._xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        return self._outputs[slot]

    def _release_resident(self):
        # Drop tensor views before releasing the BO storage they alias.
        self._queries, self._kv, self._outputs = [], [], []
        self._query_bos, self._kv_bos, self._output_bos = [], [], []

    def run(self, query, key, value):
        """Attend ``query`` over the compiled key length.

        The kernel is fixed at ``shape.queries`` rows. A longer query (a multiple
        of that size) is dispatched in row tiles against the same K/V, which are
        uploaded once per call and reused by the remaining tiles.
        """
        import ml_dtypes
        import numpy as np
        import torch

        s = self.shape
        tokens = query.shape[1] if query.ndim == 3 else 0
        if tokens == 0 or tokens % s.queries:
            raise ValueError(f"unexpected query shape/dtype: {query.shape}, {query.dtype}")
        for name, tensor, rows in (
            ("query", query, tokens), ("key", key, s.keys), ("value", value, s.keys),
        ):
            if tensor.shape != (s.heads, rows, s.depth) or tensor.dtype != torch.bfloat16:
                raise ValueError(f"unexpected {name} shape/dtype: {tensor.shape}, {tensor.dtype}")
            if tensor.device.type != "cpu" or not _all_finite(tensor):
                raise ValueError(f"{name} must be finite CPU BF16")

        def as_bf16_array(tensor):
            return tensor.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

        k_array, v_array = as_bf16_array(key), as_bf16_array(value)
        scratch = np.empty((s.heads, s.queries, s.depth), dtype=ml_dtypes.bfloat16)
        output = torch.empty(query.shape, dtype=torch.bfloat16)
        for tile, first in enumerate(range(0, tokens, s.queries)):
            rows = slice(first, first + s.queries)
            # The first tile must upload K/V; static reuse is only valid within this call.
            reuse = {"static_indices": {1, 2}} if tile else {}
            got = self.runner.run(
                [as_bf16_array(query[:, rows]), k_array, v_array, scratch],
                bo_key="attention", intermediate_indices={3}, output_indices={3}, **reuse,
            )[3]
            output[:, rows] = torch.from_numpy(got.view(np.uint16)).view(torch.bfloat16)
        if not _all_finite(output):
            raise RuntimeError("AIR attention returned nonfinite output")
        return output

    def close(self):
        self._release_resident()
        self.runner.unload()


class AIRAttentionBackend:
    """Lazily opened AIR runners owned by a chain runtime.

    Cross-attention always uses the 512-key ``cross`` artifact. Self-attention
    uses the artifact compiled for the image token count: ``self`` at 1024
    tokens (512x512) and ``self4096`` at 4096 tokens (1024x1024). The runners
    are keyed by key length and opened the first time their resolution is used.
    """

    CONTEXT_TOKENS = 512
    SELF_ARTIFACTS = {1024: "self", 4096: "self4096"}

    def __init__(self, directory):
        self.directory = Path(directory)
        self.runners = {}
        self._announced = set()

    def validate(self, config, inputs):
        tokens = inputs.hidden_states.shape[1]
        if (
            config.num_heads != 16 or config.head_dim != 128
            or tokens not in self.SELF_ARTIFACTS
            or inputs.encoder_hidden_states.shape[1] != self.CONTEXT_TOKENS
        ):
            raise UnsupportedTensor(
                "AIR attention requires batch 1, 1024 or 4096 image tokens "
                "(512x512 or 1024x1024), 512 context tokens and 16 heads of "
                f"depth 128; got {tokens} image tokens. Use "
                "ANIMA_XDNA_ATTENTION_BACKEND=baseline for other resolutions"
            )
        if inputs.attention_mask is not None:
            raise UnsupportedTensor("AIR attention does not support masks; no fallback")

    def _open(self, image_tokens=1024):
        wanted = (
            (image_tokens, self.SELF_ARTIFACTS[image_tokens]),
            (self.CONTEXT_TOKENS, "cross"),
        )
        opened = []
        try:
            for keys, name in wanted:
                if keys in self.runners:
                    continue
                runner = AIRAttention(self.directory / name / "manifest.json")
                self.runners[keys] = runner
                opened.append(keys)
                if runner.shape.keys != keys:
                    raise ValueError(f"AIR {name} manifest has incorrect key length")
        except Exception:
            for keys in opened:
                self.runners.pop(keys).close()
            raise
        if image_tokens in self._announced:
            return
        self._announced.add(image_tokens)
        if image_tokens == 1024:
            LOGGER.warning(
                "[Anima XDNA] EXPERIMENTAL AIR attention ACTIVE: tiled QK/scale/softmax/AV, "
                "17 dispatches/block in the BF16 projection path with QKV chaining when AdaLN "
                "runs on XDNA; the optimized INT8 profile has a different dispatch count. "
                "Known fixed-seed latent NRMS vs control 29.035%; not quality-equivalent. "
                "No silent fallback."
            )
        else:
            LOGGER.warning(
                "[Anima XDNA] EXPERIMENTAL AIR attention ACTIVE at %d image tokens "
                "(1024x1024): self-attention and cross-attention each run as %d 1024-query "
                "dispatches per block against fully resident K/V. Known single-seed decoded "
                "RGB MAE vs the baseline-attention control 22.1/255; BF16 online softmax is "
                "not quality-equivalent. No silent fallback.",
                image_tokens, image_tokens // 1024,
            )

    def runner_for(self, tokens, keys):
        """The opened runner for ``tokens`` image queries over ``keys`` keys."""
        if tokens not in self.SELF_ARTIFACTS or keys not in (self.CONTEXT_TOKENS, tokens):
            raise UnsupportedTensor(
                f"unsupported AIR attention shape: {tokens} queries, {keys} keys"
            )
        self._open(tokens)
        return self.runners[keys]

    def __call__(self, query, key, value):
        return self.runner_for(query.shape[1], key.shape[1]).run(query, key, value)

    def close(self):
        self._announced.clear()
        try:
            for runner in self.runners.values():
                runner.close()
        finally:
            self.runners.clear()
