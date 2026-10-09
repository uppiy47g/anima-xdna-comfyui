"""Guarded ComfyUI MODEL wrapper replacing only Anima's 28-block loop."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Any, Optional
import weakref

import torch

from anima_xdna_poc.block import BlockInputs, rotary_embedding
from anima_xdna_poc.chain import AnimaXDNAChainRuntime
from anima_xdna_poc.checkpoint_schema import (
    canonical_keys,
    detect_checkpoint_schema,
    fingerprint_model_blocks,
    validated_variant,
)
from anima_xdna_poc.errors import PrototypeError, UnsupportedTensor
from anima_xdna_poc.weight_cache import fingerprint_effective_tensors


SUPPORTED_COMFY_COMMIT = "170594057a22673349ddf0a3d88624b7fa5865bb"
WRAPPER_KEY = "anima_xdna2_resident"
ATTACHMENT_KEY = "anima_xdna2_runtime"
SOURCE_PROVENANCE_KEY = "anima_xdna2_source_provenance"
AUTO_CHECKPOINT = "Auto (from MODEL)"
SUPPORTED_SHAPE = (1, 16, 1, 64, 64)


@dataclass
class RuntimeDiagnostics:
    state: str = "created"
    calls: int = 0
    chain_runs: int = 0
    resident_reuses: int = 0
    cold_setup_ms: float = 0.0
    source_loader_total_ms: float = 0.0
    source_loader_model_ms: float = 0.0
    attach_total_ms: float = 0.0
    attach_model_load_ms: float = 0.0
    lora_snapshot_total_ms: float = 0.0
    lora_snapshot_clone_ms: float = 0.0
    lora_snapshot_hash_ms: float = 0.0
    lora_base_identity_ms: float = 0.0
    lora_snapshot_bytes: int = 0
    lora_snapshot_tensor_count: int = 0
    identity_check_ms: float = 0.0
    first_call_total_ms: float = 0.0
    first_chain_ms: float = 0.0
    last_total_ms: float = 0.0
    last_preprocess_ms: float = 0.0
    last_chain_ms: float = 0.0
    last_postprocess_ms: float = 0.0
    last_input_bytes: int = 0
    last_output_bytes: int = 0
    last_dispatches: int = 0
    last_h2d_bytes: int = 0
    last_d2h_bytes: int = 0
    last_allocations: int = 0
    last_resident_hits: int = 0
    last_weight_population_ms: float = 0.0
    last_weight_population_bytes: int = 0
    last_error: Optional[str] = None
    cache: Optional[dict[str, Any]] = None
    source_schema: Optional[str] = None
    source_fingerprint: Optional[str] = None
    source_execution_fingerprint: Optional[str] = None
    source_dtypes: Optional[list[str]] = None
    source_normalization_message: Optional[str] = None
    lora_patch_count: int = 0
    lora_cache_message: Optional[str] = None
    model_schema: Optional[str] = None
    model_fingerprint: Optional[str] = None
    model_variant: Optional[str] = None
    qkv_chaining: bool = True
    model_block_parameter_count: int = 0
    model_block_parameter_dtypes: Optional[dict[str, int]] = None
    model_block_parameter_bytes: int = 0
    model_block_unique_storage_bytes: int = 0
    model_nonblock_parameter_bytes: int = 0
    model_fp32_block_memory_warning: Optional[str] = None
    model_block_parameters_released: bool = False
    model_block_memory_policy: str = "retain_shared_modelpatcher_parameters"


class SharedRuntime:
    """One single-tenant XRT runtime shared by ModelPatcher clones."""

    _npu_lock = threading.RLock()

    def __init__(
        self,
        checkpoint: Path,
        cache_dir: Optional[Path] = None,
        rebuild_cache: bool = False,
        qkv_chaining: bool = True,
        effective_tensor_provider=None,
        lora_patch_count: int = 0,
        base_model_fingerprint: Optional[str] = None,
        base_model_schema: Optional[str] = None,
        attach_model_load_ms: float = 0.0,
        source_loader_total_ms: float = 0.0,
        source_loader_model_ms: float = 0.0,
    ):
        self.checkpoint = Path(checkpoint)
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        self.qkv_chaining = qkv_chaining
        self.effective_tensor_provider = effective_tensor_provider
        self.lora_patch_count = lora_patch_count
        self.base_model_fingerprint = base_model_fingerprint
        self.base_model_schema = base_model_schema
        self._runtime: Optional[AnimaXDNAChainRuntime] = None
        self._refs = 1
        self._lock = threading.RLock()
        self.diagnostics = RuntimeDiagnostics()
        self.diagnostics.attach_model_load_ms = attach_model_load_ms
        self.diagnostics.source_loader_total_ms = source_loader_total_ms
        self.diagnostics.source_loader_model_ms = source_loader_model_ms
        if effective_tensor_provider is not None:
            self.diagnostics.lora_snapshot_total_ms = getattr(
                effective_tensor_provider,
                "snapshot_total_ms",
                0.0,
            )
            self.diagnostics.lora_snapshot_clone_ms = getattr(
                effective_tensor_provider,
                "snapshot_clone_ms",
                0.0,
            )
            self.diagnostics.lora_snapshot_hash_ms = getattr(
                effective_tensor_provider,
                "snapshot_hash_ms",
                0.0,
            )
            self.diagnostics.lora_base_identity_ms = getattr(
                effective_tensor_provider,
                "base_identity_ms",
                0.0,
            )
            self.diagnostics.lora_snapshot_bytes = getattr(
                effective_tensor_provider,
                "snapshot_bytes",
                0,
            )
            self.diagnostics.lora_snapshot_tensor_count = getattr(
                effective_tensor_provider,
                "snapshot_tensor_count",
                0,
            )

    def prepare(self, diffusion_model) -> None:
        with self._lock:
            if self._runtime is not None:
                return
            started = time.perf_counter()
            runtime = AnimaXDNAChainRuntime(
                self.checkpoint,
                cache_host_weights=False,
                weight_cache=True,
                cache_dir=self.cache_dir,
                rebuild_cache=self.rebuild_cache,
                qkv_chaining=self.qkv_chaining,
                effective_tensor_provider=self.effective_tensor_provider,
            )
            try:
                runtime.prepare_weight_cache()
                self.effective_tensor_provider = None
                identity = runtime.source_identity
                if identity is None:
                    raise RuntimeError("packed cache did not expose source identity")
                execution_identity = runtime.execution_identity
                if execution_identity is None:
                    raise RuntimeError(
                        "packed cache did not expose BF16 execution identity"
                    )
                identity_started = time.perf_counter()
                if self.base_model_fingerprint is None:
                    base_model_fingerprint, model_schema = (
                        fingerprint_model_blocks(diffusion_model)
                    )
                else:
                    base_model_fingerprint = self.base_model_fingerprint
                    model_schema = self.base_model_schema
                identity_check_ms = (
                    time.perf_counter() - identity_started
                ) * 1000
                source_fingerprint = identity["block_fingerprint"]
                base_execution_identity = runtime.base_execution_identity
                if not isinstance(base_execution_identity, dict):
                    base_execution_identity = execution_identity
                if base_execution_identity is None:
                    raise RuntimeError(
                        "packed cache did not expose base execution identity"
                    )
                source_execution_fingerprint = base_execution_identity[
                    "block_fingerprint"
                ]
                if base_model_fingerprint != source_execution_fingerprint:
                    raise RuntimeError(
                        "Anima MODEL/checkpoint mismatch: the connected MODEL "
                        "does not contain the same base 28-block weights as the "
                        f"XDNA source (MODEL {base_model_fingerprint[:16]}..., source "
                        f"{source_execution_fingerprint[:16]}...). Select the matching "
                        "Base or Turbo checkpoint; no dispatch was attempted."
                    )
                model_fingerprint = execution_identity["block_fingerprint"]
                storage_profile = _model_storage_profile(diffusion_model)
            except BaseException:
                self.effective_tensor_provider = None
                runtime.effective_tensor_provider = None
                runtime.close()
                raise
            self._runtime = runtime
            self.diagnostics.cold_setup_ms = (
                time.perf_counter() - started
            ) * 1000
            self.diagnostics.identity_check_ms = identity_check_ms
            self.diagnostics.state = "prepared"
            self.diagnostics.source_schema = identity["schema"]
            self.diagnostics.source_fingerprint = source_fingerprint
            self.diagnostics.source_execution_fingerprint = (
                source_execution_fingerprint
            )
            self.diagnostics.source_dtypes = execution_identity["source_dtypes"]
            self.diagnostics.model_schema = model_schema
            self.diagnostics.model_fingerprint = model_fingerprint
            self.diagnostics.model_variant = validated_variant(
                source_execution_fingerprint
            )
            self.diagnostics.lora_patch_count = self.lora_patch_count
            self.diagnostics.qkv_chaining = self.qkv_chaining
            self.diagnostics.model_block_parameter_count = storage_profile[
                "block_parameter_count"
            ]
            self.diagnostics.model_block_parameter_dtypes = storage_profile[
                "block_dtype_numel"
            ]
            self.diagnostics.model_block_parameter_bytes = storage_profile[
                "block_parameter_bytes"
            ]
            self.diagnostics.model_block_unique_storage_bytes = storage_profile[
                "block_unique_storage_bytes"
            ]
            self.diagnostics.model_nonblock_parameter_bytes = storage_profile[
                "nonblock_parameter_bytes"
            ]
            fp32_bytes = storage_profile["block_dtype_bytes"].get("float32", 0)
            if fp32_bytes:
                self.diagnostics.model_fp32_block_memory_warning = (
                    f"The attached Anima blocks retain {fp32_bytes} bytes of "
                    "FP32 parameters. Native Base/Turbo checkpoints are BF16; "
                    "use the repository's Load Anima (BF16) node to avoid "
                    "avoidable CPU weight expansion. We do not delete or "
                    "replace shared ModelPatcher parameters."
                )
            self.diagnostics.cache = (
                {
                    **asdict(runtime.cache_status),
                    "path": str(runtime.cache_status.path),
                }
                if runtime.cache_status is not None
                else None
            )
            if execution_identity["source_dtypes"] != ["BF16"]:
                action = (
                    "Reusing the verified"
                    if runtime.cache_status is not None
                    and runtime.cache_status.hit
                    else "Created a verified"
                )
                message = (
                    f"{action} BF16 packed cache for "
                    f"{'/'.join(execution_identity['source_dtypes'])} source "
                    "weights. The original checkpoint is unchanged; future "
                    "runs reuse this cache."
                )
                self.diagnostics.source_normalization_message = message
                print(f"[Anima XDNA] {message}")
            if self.lora_patch_count:
                action = (
                    "Reusing the verified"
                    if runtime.cache_status is not None
                    and runtime.cache_status.hit
                    else "Created a verified"
                )
                message = (
                    f"{action} BF16 packed cache for "
                    f"{self.lora_patch_count} Anima LoRA block patches. "
                    "The checkpoint and LoRA files are unchanged; identical "
                    "effective weights reuse this cache."
                )
                self.diagnostics.lora_cache_message = message
                print(f"[Anima XDNA] {message}")

    def acquire(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError("Anima XDNA runtime is already closed")
            self._refs += 1
        return self

    def release(self):
        with self._lock:
            self._refs -= 1
            if self._refs <= 0:
                self.close()

    def open(self) -> AnimaXDNAChainRuntime:
        with self._lock:
            if self._runtime is None:
                raise RuntimeError("Anima XDNA runtime was not identity-bound")
            if not self._runtime.is_open:
                self._runtime.__enter__()
                self.diagnostics.state = "open"
            return self._runtime

    def close(self):
        with self._lock:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self.diagnostics.state = "closed"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self.diagnostics)
            payload.update(
                {
                    "checkpoint": str(self.checkpoint),
                    "reference_count": self._refs,
                    "runtime_open": (
                        self._runtime is not None and self._runtime.is_open
                    ),
                    "last_d2h_host_visible_bytes": (
                        self.diagnostics.last_d2h_bytes
                    ),
                    "supported_shape": list(SUPPORTED_SHAPE),
                    "replacement_boundary": (
                        "after ComfyUI patch/time/Qwen/RoPE preprocessing; "
                        "before final_layer/unpatchify"
                    ),
                    "npu_weight_storage_accounting": (
                        "XRT BO population is reported separately and is not "
                        "included in CPU Parameter storage byte counts."
                    ),
                }
            )
            return payload

    @staticmethod
    def _interrupt():
        try:
            import comfy.model_management

            comfy.model_management.throw_exception_if_processing_interrupted()
        except ImportError:
            pass

    @staticmethod
    def _validate_model(model, x, context, kwargs):
        class_name = type(model).__name__
        module = type(model).__module__
        if class_name != "Anima" or not module.endswith("ldm.anima.model"):
            raise UnsupportedTensor(
                "Anima XDNA wrapper requires ComfyUI's comfy.ldm.anima.model.Anima; "
                f"received {module}.{class_name}"
            )
        if x.ndim != 5 or tuple(x.shape[1:]) != SUPPORTED_SHAPE[1:] or x.shape[0] not in (1, 2):
            raise UnsupportedTensor(
                "Anima XDNA currently supports a batch-1 latent (or the "
                "sampler's two-way CFG batch) with shape [B,16,1,64,64], "
                f"got {tuple(x.shape)}"
            )
        if tuple(context.shape) != (x.shape[0], 512, 1024):
            raise UnsupportedTensor(
                "Anima XDNA requires one adapted Qwen context per sampler "
                f"batch item with shape [B,512,1024], got {tuple(context.shape)}"
            )
        transformer_options = kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})
        replacements = transformer_options.get("patches_replace", {})
        if patches or replacements:
            names = sorted(set(patches) | set(replacements))
            raise UnsupportedTensor(
                "Anima XDNA cannot execute transformer patches/LoRAs yet; "
                "remove them before attaching the wrapper. Active patches: "
                + ", ".join(names)
            )
        required = (
            "prepare_embedded_sequence",
            "t_embedder",
            "t_embedding_norm",
            "final_layer",
            "unpatchify",
            "blocks",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing or len(model.blocks) != 28:
            raise UnsupportedTensor(
                "unsupported ComfyUI Anima API/version; expected 28 blocks and "
                f"pre/post methods, missing={missing}"
            )

    def forward(
        self,
        model,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps=None,
        padding_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_model(model, x, context, kwargs)
        self._interrupt()
        total_started = time.perf_counter()
        original_device = x.device
        original_dtype = x.dtype
        orig_shape = list(x.shape)

        pre_started = time.perf_counter()
        # The fixed 512x512 image shape is already divisible by [1,2,2].
        embedded, _comfy_rope, extra_position = model.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
        )
        if extra_position is not None:
            raise UnsupportedTensor(
                "extra per-block absolute position embeddings are unsupported"
            )
        timestep_tokens = timesteps.unsqueeze(1) if timesteps.ndim == 1 else timesteps
        embedded_timestep, temb = model.t_embedder[1](
            model.t_embedder[0](timestep_tokens).to(embedded.dtype)
        )
        embedded_timestep = model.t_embedding_norm(embedded_timestep)
        if temb is None:
            raise UnsupportedTensor("Anima XDNA requires AdaLN-LoRA timestep modulation")
        grid_shape = embedded.shape[1:4]
        if tuple(grid_shape) != (1, 32, 32):
            raise UnsupportedTensor(
                f"expected patch grid [1,32,32], got {tuple(grid_shape)}"
            )
        batch = x.shape[0]
        hidden = embedded.reshape(batch, 1024, 2048).to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        cpu_context = context.to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        cpu_timestep = embedded_timestep.reshape(batch, 2048).to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        cpu_temb = temb.reshape(batch, 6144).to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        rope = rotary_embedding(self.open().config, 1024)
        preprocess_ms = (time.perf_counter() - pre_started) * 1000

        self._interrupt()
        with self._npu_lock:
            runtime = self.open()
            chains = []
            for index in range(batch):
                block_inputs = BlockInputs(
                    hidden[index : index + 1],
                    cpu_context[index : index + 1],
                    cpu_timestep[index : index + 1],
                    cpu_temb[index : index + 1],
                    rope,
                    None,
                )
                chains.append(runtime.run_range(block_inputs))
                self._interrupt()
        self._interrupt()

        post_started = time.perf_counter()
        chain_output = torch.cat([chain.output for chain in chains], dim=0)
        residual = chain_output.reshape(batch, 1, 32, 32, 2048).to(
            device=original_device,
            dtype=context.dtype,
        )
        output_patches = model.final_layer(
            residual,
            embedded_timestep,
            adaln_lora_B_T_3D=temb,
        )
        output = model.unpatchify(output_patches)[
            :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
        ]
        output = output.to(device=original_device, dtype=original_dtype)
        postprocess_ms = (time.perf_counter() - post_started) * 1000

        with self._lock:
            metrics = tuple(
                metric
                for chain in chains
                for block in chain.blocks
                for metric in block.result.metrics
            )
            chain_wall_ms = sum(chain.wall_ms for chain in chains)
            dispatch_count = sum(chain.dispatch_count for chain in chains)
            call_total_ms = (time.perf_counter() - total_started) * 1000
            if self.diagnostics.calls == 0:
                self.diagnostics.first_call_total_ms = call_total_ms
                self.diagnostics.first_chain_ms = chain_wall_ms
            self.diagnostics.calls += 1
            self.diagnostics.chain_runs += batch
            self.diagnostics.resident_reuses = max(
                0, self.diagnostics.chain_runs - 1
            )
            self.diagnostics.last_preprocess_ms = preprocess_ms
            self.diagnostics.last_chain_ms = chain_wall_ms
            self.diagnostics.last_postprocess_ms = postprocess_ms
            self.diagnostics.last_total_ms = call_total_ms
            self.diagnostics.last_input_bytes = (
                hidden.numel() * hidden.element_size()
                + cpu_context.numel() * cpu_context.element_size()
                + batch * 2048 * 2
                + batch * 6144 * 2
            )
            self.diagnostics.last_output_bytes = (
                chain_output.numel() * chain_output.element_size()
            )
            self.diagnostics.last_dispatches = dispatch_count
            self.diagnostics.last_h2d_bytes = sum(
                metric.h2d_bytes for metric in metrics
            )
            self.diagnostics.last_d2h_bytes = sum(
                metric.d2h_bytes for metric in metrics
            )
            self.diagnostics.last_allocations = sum(
                metric.allocation_count for metric in metrics
            )
            self.diagnostics.last_resident_hits = sum(
                metric.resident_hits for metric in metrics
            )
            self.diagnostics.last_weight_population_ms = sum(
                metric.weight_population_ms for metric in metrics
            )
            self.diagnostics.last_weight_population_bytes = sum(
                metric.weight_population_bytes for metric in metrics
            )
            self.diagnostics.cache = (
                {
                    **asdict(runtime.cache_status),
                    "path": str(runtime.cache_status.path),
                }
                if runtime.cache_status is not None
                else None
            )
            self.diagnostics.last_error = None
        return output


class RuntimeAttachment:
    def __init__(self, runtime: SharedRuntime):
        self.runtime = runtime
        self._finalizer = weakref.finalize(self, runtime.release)

    def on_model_patcher_clone(self):
        return RuntimeAttachment(self.runtime.acquire())

    def cleanup(self):
        if self._finalizer.alive:
            self._finalizer()


class AnimaXDNADiffusionWrapper:
    def __init__(self, runtime: SharedRuntime):
        self.runtime = runtime

    def __call__(self, executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        try:
            return self.runtime.forward(
                executor.class_obj,
                x,
                timesteps,
                context,
                fps,
                padding_mask,
                **kwargs,
            )
        except PrototypeError as error:
            self.runtime.diagnostics.last_error = str(error)
            raise RuntimeError(f"Anima XDNA execution failed: {error}") from error

    def to(self, _device):
        return self

    def cleanup(self, **_kwargs):
        # ModelPatcher attachment owns the reference and cleanup lifecycle.
        pass


def _validate_patcher(model):
    required = (
        "clone",
        "add_wrapper_with_key",
        "set_attachments",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "unsupported ComfyUI ModelPatcher API; missing " + ", ".join(missing)
        )
    transformer_options = getattr(model, "model_options", {}).get(
        "transformer_options", {}
    )
    if transformer_options.get("patches") or transformer_options.get(
        "patches_replace"
    ):
        raise RuntimeError(
            "Anima XDNA cannot silently ignore active transformer patches"
        )
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("MODEL does not expose model.diffusion_model")
    if type(diffusion_model).__name__ != "Anima":
        raise RuntimeError(
            "Load/Attach Anima XDNA Model requires a ComfyUI Anima MODEL"
        )


def _effective_lora_provider(model):
    snapshot_started = time.perf_counter()
    snapshot_clone_ms = 0.0
    snapshot_hash_ms = 0.0
    snapshot_bytes = 0
    patches = getattr(model, "patches", {})
    block_patch_keys = {
        key
        for key in patches
        if isinstance(key, str)
        and key.startswith("diffusion_model.blocks.")
    }
    if not block_patch_keys:
        return None, 0, None, None
    if not hasattr(model, "get_key_patches"):
        raise RuntimeError(
            "unsupported ComfyUI ModelPatcher API; missing get_key_patches"
        )

    try:
        import comfy.float
        import comfy.lora
        import comfy.model_management
        import comfy.utils
        from comfy.weight_adapter.lora import LoRAAdapter
    except ImportError as error:
        raise RuntimeError(
            "ComfyUI LoRA APIs are unavailable; use the validated ComfyUI "
            f"commit {SUPPORTED_COMFY_COMMIT}."
        ) from error

    key_patches = model.get_key_patches("diffusion_model.")
    schema = detect_checkpoint_schema(key_patches.keys())
    canonical_to_model = dict(schema.canonical_to_source)
    canonical_model_keys = set(canonical_to_model.values())
    unsupported_keys = sorted(block_patch_keys - canonical_model_keys)
    if unsupported_keys:
        raise RuntimeError(
            "Anima XDNA cannot safely apply block patches outside the 560 "
            "supported weights: " + ", ".join(unsupported_keys[:4])
        )

    entries_by_canonical = {}
    snapshot_records = []
    patch_count = 0
    for canonical_key in canonical_keys():
        model_key = canonical_to_model[canonical_key]
        entries = key_patches[model_key]
        if not entries or len(entries[0]) != 2:
            raise RuntimeError(
                f"invalid ComfyUI patch entries for {model_key!r}"
            )
        snapshot_patches = []
        for patch_index, patch in enumerate(entries[1:]):
            if not isinstance(patch, tuple) or len(patch) != 5:
                raise RuntimeError(
                    f"unsupported ComfyUI patch structure for {model_key!r}"
                )
            strength, adapter, strength_model, offset, function = patch
            if (
                strength_model != 1.0
                or offset is not None
                or function is not None
            ):
                raise RuntimeError(
                    "Anima XDNA supports only additive block patches without "
                    f"model scaling, offsets, or functions; unsupported patch on "
                    f"{model_key!r}"
                )
            if not isinstance(strength, (int, float)) or not math.isfinite(
                float(strength)
            ):
                raise RuntimeError(
                    f"LoRA strength for {model_key!r} must be finite"
                )
            base = entries[0][0]
            if type(adapter) is tuple:
                is_norm = model_key.endswith(
                    (
                        ".self_attn.q_norm.weight",
                        ".self_attn.k_norm.weight",
                        ".cross_attn.q_norm.weight",
                        ".cross_attn.k_norm.weight",
                    )
                )
                if (
                    not is_norm
                    or len(adapter) != 2
                    or adapter[0] != "diff"
                    or type(adapter[1]) is not tuple
                    or len(adapter[1]) != 1
                    or not isinstance(adapter[1][0], torch.Tensor)
                ):
                    raise RuntimeError(
                        "Anima XDNA supports legacy diff patches only for "
                        f"attention q/k norm weights; unsupported patch on {model_key!r}"
                    )
                diff = adapter[1][0]
                if (
                    tuple(diff.shape) != tuple(base.shape)
                    or not bool(torch.isfinite(diff.float()).all())
                ):
                    raise RuntimeError(
                        "Anima XDNA requires finite, exact-shape attention "
                        f"norm diff weights on {model_key!r}"
                    )
                patch_count += 1
                clone_started = time.perf_counter()
                snapshot_diff = diff.detach().to(device="cpu").clone()
                snapshot_clone_ms += (
                    time.perf_counter() - clone_started
                ) * 1000
                snapshot_bytes += (
                    snapshot_diff.numel() * snapshot_diff.element_size()
                )
                hash_started = time.perf_counter()
                snapshot_sha256 = hashlib.sha256(
                    snapshot_diff.contiguous().view(torch.uint8).numpy()
                ).hexdigest()
                snapshot_hash_ms += (
                    time.perf_counter() - hash_started
                ) * 1000
                snapshot_patches.append(
                    (
                        float(strength),
                        ("diff", (snapshot_diff,)),
                        1.0,
                        None,
                        None,
                    )
                )
                snapshot_records.append(
                    {
                        "key": canonical_key,
                        "index": patch_index,
                        "type": "diff",
                        "strength": float(strength),
                        "shape": list(snapshot_diff.shape),
                        "dtype": str(snapshot_diff.dtype),
                        "sha256": snapshot_sha256,
                    }
                )
                continue
            if type(adapter) is not LoRAAdapter:
                raise RuntimeError(
                    "Anima XDNA currently supports only ordinary additive "
                    "LoRAAdapter patches and exact attention norm diff patches; "
                    f"unsupported patch on {model_key!r}"
                )
            weights = adapter.weights
            if not isinstance(weights, tuple) or len(weights) != 6:
                raise RuntimeError(
                    f"invalid LoRAAdapter weights for {model_key!r}"
                )
            up, down, alpha, mid, dora_scale, reshape = weights
            if (
                not isinstance(up, torch.Tensor)
                or not isinstance(down, torch.Tensor)
                or up.ndim != 2
                or down.ndim != 2
                or up.shape[1] != down.shape[0]
                or mid is not None
                or dora_scale is not None
                or reshape is not None
                or (
                    alpha is not None
                    and (
                        not isinstance(alpha, (int, float))
                        or not math.isfinite(float(alpha))
                    )
                )
                or not bool(torch.isfinite(up.float()).all())
                or not bool(torch.isfinite(down.float()).all())
            ):
                raise RuntimeError(
                    "Anima XDNA supports finite 2-D ordinary LoRA weights "
                    f"without mid/DoRA/reshape on {model_key!r}"
                )
            if up.numel() == 0 or down.numel() == 0 or (
                up.shape[0] * down.shape[1] != base.numel()
            ):
                raise RuntimeError(
                    f"LoRA rank/shape does not match {model_key!r}"
                )
            patch_count += 1
            snapshot_values = []
            for value in weights:
                if not isinstance(value, torch.Tensor):
                    snapshot_values.append(value)
                    continue
                clone_started = time.perf_counter()
                snapshot_value = value.detach().to(device="cpu").clone()
                snapshot_clone_ms += (
                    time.perf_counter() - clone_started
                ) * 1000
                snapshot_bytes += (
                    snapshot_value.numel() * snapshot_value.element_size()
                )
                snapshot_values.append(snapshot_value)
            snapshot_weights = tuple(snapshot_values)
            snapshot_adapter = LoRAAdapter(
                set(adapter.loaded_keys),
                snapshot_weights,
            )
            snapshot_patches.append(
                (
                    float(strength),
                    snapshot_adapter,
                    1.0,
                    None,
                    None,
                )
            )
            up_snapshot, down_snapshot = snapshot_weights[:2]
            hash_started = time.perf_counter()
            up_sha256 = hashlib.sha256(
                up_snapshot.contiguous().view(torch.uint8).numpy()
            ).hexdigest()
            down_sha256 = hashlib.sha256(
                down_snapshot.contiguous().view(torch.uint8).numpy()
            ).hexdigest()
            snapshot_hash_ms += (
                time.perf_counter() - hash_started
            ) * 1000
            snapshot_records.append(
                {
                    "key": canonical_key,
                    "index": patch_index,
                    "type": "lora",
                    "strength": float(strength),
                    "alpha": alpha,
                    "up_shape": list(up_snapshot.shape),
                    "up_dtype": str(up_snapshot.dtype),
                    "up_sha256": up_sha256,
                    "down_shape": list(down_snapshot.shape),
                    "down_dtype": str(down_snapshot.dtype),
                    "down_sha256": down_sha256,
                }
            )
        entries_by_canonical[canonical_key] = (
            model_key,
            [entries[0], *snapshot_patches],
        )

    cpu = torch.device("cpu")

    def base_provider(canonical_key: str) -> torch.Tensor:
        _model_key, entries = entries_by_canonical[canonical_key]
        base, convert = entries[0]
        tensor = comfy.model_management.cast_to_device(
            base, cpu, torch.bfloat16, copy=True
        )
        return convert(tensor, inplace=True).detach().contiguous()

    base_identity_started = time.perf_counter()
    base_identity = fingerprint_effective_tensors(
        base_provider,
        range(28),
    )
    base_identity_ms = (
        time.perf_counter() - base_identity_started
    ) * 1000

    def provider(canonical_key: str) -> torch.Tensor:
        model_key, entries = entries_by_canonical[canonical_key]
        base, convert = entries[0]
        compute_dtype = comfy.model_management.lora_compute_dtype(cpu)
        tensor = comfy.model_management.cast_to_device(
            base, cpu, compute_dtype, copy=True
        )
        tensor = convert(tensor, inplace=True)
        if entries[1:]:
            tensor = comfy.lora.calculate_weight(
                entries[1:],
                tensor,
                model_key,
            )
            tensor = comfy.float.stochastic_rounding(
                tensor,
                base.dtype,
                seed=comfy.utils.string_to_seed(model_key),
            )
        tensor = tensor.detach().to(torch.bfloat16).contiguous()
        if tuple(tensor.shape) != tuple(base.shape):
            raise RuntimeError(
                f"effective LoRA tensor shape changed for {model_key!r}"
            )
        if not bool(torch.isfinite(tensor.float()).all()):
            raise RuntimeError(
                f"effective LoRA tensor contains non-finite values: {model_key!r}"
            )
        return tensor

    snapshot_payload = json.dumps(
        snapshot_records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    provider.input_fingerprint = hashlib.sha256(snapshot_payload).hexdigest()
    provider.snapshot_tensor_count = sum(
        1
        for record in snapshot_records
        for name in record
        if name in ("sha256", "up_sha256", "down_sha256")
    )
    provider.snapshot_total_ms = (
        time.perf_counter() - snapshot_started
    ) * 1000
    provider.snapshot_clone_ms = snapshot_clone_ms
    provider.snapshot_hash_ms = snapshot_hash_ms
    provider.base_identity_ms = base_identity_ms
    provider.snapshot_bytes = snapshot_bytes

    return (
        provider,
        patch_count,
        base_identity["block_fingerprint"],
        schema.name,
    )


def _model_storage_profile(diffusion_model) -> dict[str, Any]:
    parameters = dict(diffusion_model.named_parameters())
    schema = detect_checkpoint_schema(parameters.keys())
    block_parameter_names = {
        schema.canonical_to_source[key] for key in canonical_keys()
    }
    block_dtype_numel: Counter[str] = Counter()
    block_dtype_bytes: Counter[str] = Counter()
    block_parameter_count = 0
    block_parameter_bytes = 0
    nonblock_parameter_bytes = 0
    storage_categories: dict[tuple[str, int, int], set[str]] = {}
    storage_sizes: dict[tuple[str, int, int], int] = {}
    for name, parameter in parameters.items():
        category = "block" if name in block_parameter_names else "nonblock"
        parameter_bytes = parameter.numel() * parameter.element_size()
        if category == "block":
            block_parameter_count += 1
            block_parameter_bytes += parameter_bytes
            block_dtype_numel[str(parameter.dtype).removeprefix("torch.")] += (
                parameter.numel()
            )
            block_dtype_bytes[str(parameter.dtype).removeprefix("torch.")] += (
                parameter_bytes
            )
        else:
            nonblock_parameter_bytes += parameter_bytes
        storage = parameter.untyped_storage()
        storage_key = (
            str(parameter.device),
            storage.data_ptr(),
            storage.nbytes(),
        )
        storage_categories.setdefault(storage_key, set()).add(category)
        storage_sizes[storage_key] = storage.nbytes()
    block_unique_storage_bytes = sum(
        storage_sizes[key]
        for key, categories in storage_categories.items()
        if categories == {"block"}
    )
    return {
        "block_parameter_count": block_parameter_count,
        "block_dtype_numel": dict(block_dtype_numel),
        "block_dtype_bytes": dict(block_dtype_bytes),
        "block_parameter_bytes": block_parameter_bytes,
        "block_unique_storage_bytes": block_unique_storage_bytes,
        "nonblock_parameter_bytes": nonblock_parameter_bytes,
    }


def _file_identity_token(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=False)
    try:
        stat = resolved.stat()
    except FileNotFoundError:
        identity = {"path": str(resolved), "missing": True}
    else:
        identity = {
            "path": str(resolved),
            "device": stat.st_dev,
            "file_id": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


_ANIMA_MODEL_CATEGORIES = ("diffusion_models", "checkpoints")


@dataclass(frozen=True)
class ModelSourceProvenance:
    selector: str
    path: str
    identity_token: str
    loader_total_ms: float = 0.0
    loader_model_ms: float = 0.0

    def on_model_patcher_clone(self):
        return self


def _anima_model_selector(value: str) -> tuple[str, str]:
    category, separator, name = value.partition(":")
    if not separator:
        return "diffusion_models", value
    if category not in _ANIMA_MODEL_CATEGORIES or not name:
        raise ValueError(
            "Anima model selector must be an unqualified diffusion model or "
            "'diffusion_models:<name>' / 'checkpoints:<name>'"
        )
    return category, name


def _auto_checkpoint_path(model) -> Path:
    get_attachment = getattr(model, "get_attachment", None)
    provenance = (
        get_attachment(SOURCE_PROVENANCE_KEY)
        if callable(get_attachment)
        else None
    )
    if not isinstance(provenance, ModelSourceProvenance):
        raise RuntimeError(
            "Auto checkpoint selection requires a MODEL loaded by "
            "Load Anima (BF16). Use that loader or enter the matching "
            "checkpoint path manually."
        )
    path = Path(provenance.path)
    if not path.is_file():
        raise RuntimeError(
            f"Auto-selected Anima checkpoint does not exist: {path}"
        )
    if _file_identity_token(path) != provenance.identity_token:
        raise RuntimeError(
            "The auto-selected Anima checkpoint changed after the MODEL was "
            "loaded. Reload the MODEL before attaching XDNA."
        )
    return path


def _resolve_attach_checkpoint(model, checkpoint: str) -> Path:
    if checkpoint.strip() == AUTO_CHECKPOINT:
        return _auto_checkpoint_path(model)
    return Path(checkpoint)


class LoadAnimaBF16:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        models = [
            f"{category}:{name}"
            for category in _ANIMA_MODEL_CATEGORIES
            for name in folder_paths.get_filename_list(category)
        ]
        return {
            "required": {
                "unet_name": (models,)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Anima XDNA 2"

    @classmethod
    def IS_CHANGED(cls, unet_name):
        import folder_paths

        category, name = _anima_model_selector(unet_name)
        path = folder_paths.get_full_path_or_raise(category, name)
        return _file_identity_token(Path(path))

    def load(self, unet_name):
        import comfy.sd
        import folder_paths

        loader_started = time.perf_counter()
        category, name = _anima_model_selector(unet_name)
        path = folder_paths.get_full_path_or_raise(category, name)
        model_load_started = time.perf_counter()
        model = comfy.sd.load_diffusion_model(
            path,
            model_options={"dtype": torch.bfloat16},
        )
        loader_model_ms = (
            time.perf_counter() - model_load_started
        ) * 1000
        diffusion_model = model.model.diffusion_model
        if type(diffusion_model).__name__ != "Anima":
            raise RuntimeError(
                "Load Anima (BF16) requires a native ComfyUI Anima checkpoint"
            )
        profile = _model_storage_profile(diffusion_model)
        if profile["block_dtype_bytes"].get("bfloat16", 0) != profile[
            "block_parameter_bytes"
        ]:
            raise RuntimeError(
                "ComfyUI did not retain all Anima transformer-block Parameters "
                "as BF16; refusing to load a memory-expanded MODEL."
            )
        set_attachment = getattr(model, "set_attachments", None)
        if not callable(set_attachment):
            raise RuntimeError(
                "ComfyUI ModelPatcher attachments are unavailable; use the "
                f"validated ComfyUI commit {SUPPORTED_COMFY_COMMIT}."
            )
        resolved = Path(path).resolve()
        set_attachment(
            SOURCE_PROVENANCE_KEY,
            ModelSourceProvenance(
                selector=unet_name,
                path=str(resolved),
                identity_token=_file_identity_token(resolved),
                loader_total_ms=(
                    time.perf_counter() - loader_started
                ) * 1000,
                loader_model_ms=loader_model_ms,
            ),
        )
        return (model,)


class LoadAttachAnimaXDNAModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "checkpoint": (
                    "STRING",
                    {"default": AUTO_CHECKPOINT},
                ),
                "rebuild_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": ""}),
                "qkv_chaining": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "runtime_status")
    FUNCTION = "attach"
    CATEGORY = "Anima XDNA 2"

    @classmethod
    def IS_CHANGED(
        cls,
        checkpoint,
        rebuild_cache,
        cache_dir="",
        qkv_chaining=True,
        model=None,
        **_kwargs,
    ):
        checkpoint_identity = (
            AUTO_CHECKPOINT
            if checkpoint.strip() == AUTO_CHECKPOINT
            else _file_identity_token(Path(checkpoint))
        )
        identity = {
            "checkpoint": checkpoint_identity,
            "rebuild_cache": bool(rebuild_cache),
            "cache_dir": str(Path(cache_dir).expanduser().resolve(strict=False))
            if cache_dir.strip()
            else None,
            "qkv_chaining": bool(qkv_chaining),
        }
        return json.dumps(identity, sort_keys=True, separators=(",", ":"))

    def attach(
        self, model, checkpoint, rebuild_cache, cache_dir="", qkv_chaining=True
    ):
        attach_started = time.perf_counter()
        _validate_patcher(model)
        path = _resolve_attach_checkpoint(model, checkpoint)
        if not path.is_file():
            raise RuntimeError(f"Anima checkpoint does not exist: {path}")
        try:
            import comfy.model_management
        except ImportError as error:
            raise RuntimeError(
                "ComfyUI model management is unavailable"
            ) from error
        model_load_started = time.perf_counter()
        comfy.model_management.load_models_gpu([model])
        attach_model_load_ms = (
            time.perf_counter() - model_load_started
        ) * 1000
        get_attachment = getattr(model, "get_attachment", None)
        provenance = (
            get_attachment(SOURCE_PROVENANCE_KEY)
            if callable(get_attachment)
            else None
        )
        source_loader_total_ms = (
            provenance.loader_total_ms
            if isinstance(provenance, ModelSourceProvenance)
            else 0.0
        )
        source_loader_model_ms = (
            provenance.loader_model_ms
            if isinstance(provenance, ModelSourceProvenance)
            else 0.0
        )
        (
            effective_tensor_provider,
            lora_patch_count,
            base_model_fingerprint,
            base_model_schema,
        ) = (
            _effective_lora_provider(model)
        )
        patched = model.clone()
        runtime = SharedRuntime(
            path,
            Path(cache_dir) if cache_dir.strip() else None,
            rebuild_cache,
            qkv_chaining,
            effective_tensor_provider,
            lora_patch_count,
            base_model_fingerprint,
            base_model_schema,
            attach_model_load_ms,
            source_loader_total_ms,
            source_loader_model_ms,
        )
        try:
            runtime.prepare(model.model.diffusion_model)
        except Exception:
            runtime.close()
            raise
        runtime.diagnostics.attach_total_ms = (
            time.perf_counter() - attach_started
        ) * 1000
        cache = runtime.diagnostics.cache or {}
        cache_timings = cache.get("timings") or {}
        print(
            "[Anima XDNA] initial-load timings ms: "
            f"source_model={runtime.diagnostics.source_loader_model_ms:.1f}, "
            f"source_total={runtime.diagnostics.source_loader_total_ms:.1f}, "
            f"attach_load={runtime.diagnostics.attach_model_load_ms:.1f}, "
            f"snapshot={runtime.diagnostics.lora_snapshot_total_ms:.1f} "
            f"(clone={runtime.diagnostics.lora_snapshot_clone_ms:.1f}, "
            f"hash={runtime.diagnostics.lora_snapshot_hash_ms:.1f}, "
            f"base_identity={runtime.diagnostics.lora_base_identity_ms:.1f}), "
            f"cache={float(cache_timings.get('total_ms', 0.0)):.1f} "
            f"(fingerprint={float(cache_timings.get('fingerprint_ms', 0.0)):.1f}, "
            f"pack={float(cache_timings.get('pack_ms', 0.0)):.1f}, "
            f"verify={float(cache_timings.get('verify_ms', 0.0)):.1f}), "
            f"verify_lease_hit={cache_timings.get('verification_lease_hit')}, "
            "verify_saved_bytes="
            f"{int(cache_timings.get('verification_lease_saved_bytes', 0))}, "
            f"cold_setup={runtime.diagnostics.cold_setup_ms:.1f}, "
            f"attach_total={runtime.diagnostics.attach_total_ms:.1f}, "
            f"cache_hit={cache.get('hit')}, reason={cache.get('reason')}"
        )
        attachment = RuntimeAttachment(runtime)
        patched.set_attachments(ATTACHMENT_KEY, attachment)
        try:
            import comfy.patcher_extension
        except ImportError as error:
            attachment.cleanup()
            raise RuntimeError(
                "ComfyUI patcher_extension is unavailable; use a supported "
                f"ComfyUI checkout (validated commit {SUPPORTED_COMFY_COMMIT})"
            ) from error
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            AnimaXDNADiffusionWrapper(runtime),
        )
        return patched, json.dumps(runtime.snapshot(), indent=2)


class AnimaXDNARuntimeStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "runtime_status")
    OUTPUT_NODE = True
    FUNCTION = "status"
    CATEGORY = "Anima XDNA 2"

    def status(self, model):
        attachment = (
            model.get_attachment(ATTACHMENT_KEY)
            if hasattr(model, "get_attachment")
            else None
        )
        if not isinstance(attachment, RuntimeAttachment):
            return model, json.dumps(
                {"state": "not attached", "error": "MODEL has no Anima XDNA runtime"},
                indent=2,
            )
        return model, json.dumps(attachment.runtime.snapshot(), indent=2)


class AnimaXDNAUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "Anima XDNA 2"

    def unload(self, model):
        attachment = (
            model.get_attachment(ATTACHMENT_KEY)
            if hasattr(model, "get_attachment")
            else None
        )
        if isinstance(attachment, RuntimeAttachment):
            attachment.cleanup()
            return model, "Anima XDNA runtime reference released"
        return model, "MODEL had no Anima XDNA runtime"


NODE_CLASS_MAPPINGS = {
    "LoadAnimaBF16": LoadAnimaBF16,
    "LoadAttachAnimaXDNAModel": LoadAttachAnimaXDNAModel,
    "AnimaXDNARuntimeStatus": AnimaXDNARuntimeStatus,
    "AnimaXDNAUnload": AnimaXDNAUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAnimaBF16": "Load Anima (BF16)",
    "LoadAttachAnimaXDNAModel": "Load/Attach Anima XDNA Model",
    "AnimaXDNARuntimeStatus": "Anima XDNA Runtime Status",
    "AnimaXDNAUnload": "Unload Anima XDNA Runtime",
}
