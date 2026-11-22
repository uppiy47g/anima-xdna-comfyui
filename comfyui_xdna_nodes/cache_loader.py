"""CPU host layers plus verified XDNA block weights, without a second block MODEL."""

from pathlib import Path
import time
import weakref

import torch
from safetensors import safe_open

import comfy.model_detection
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import WrappersMP
import folder_paths

from anima_xdna_poc.checkpoint_schema import detect_checkpoint_schema
from anima_xdna_poc.errors import UnsupportedTensor
from .model_wrapper import (
    ATTACHMENT_KEY,
    SOURCE_PROVENANCE_KEY,
    WRAPPER_KEY,
    AnimaXDNADiffusionWrapper,
    ModelSourceProvenance,
    RuntimeAttachment,
    SharedRuntime,
    _anima_model_selector,
    _effective_lora_provider,
    _file_identity_token,
)


_SOURCE_PREFIXES = {
    "native-net": "net.",
    "native-model-diffusion-model": "model.diffusion_model.",
    "comfyui-anima-module": "",
    "comfyui-model-wrapper": "diffusion_model.",
}
_MODEL_PREFIX = "diffusion_model."
_BLOCK_PREFIX = _MODEL_PREFIX + "blocks."


class CacheOnlyBlock(torch.nn.Module):
    """Parameter-free block whose shape-only weights stay visible to LoRA key mapping."""

    def __init__(self, placeholders=None):
        super().__init__()
        # A plain dict, not buffers: Module.to() and ComfyUI weight loading skip it.
        self.placeholders = dict(placeholders or {})
        self._placeholder_roots = frozenset(name.split(".", 1)[0] for name in self.placeholders)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Not AttributeError: module lookups that skip missing layers (bypass LoRA)
            # would otherwise silently drop block patches.
            if name in self.__dict__.get("_placeholder_roots", ()):
                raise RuntimeError(
                    f"Cache-backed Anima block submodule {name!r} does not exist; block "
                    "weights live only in the XDNA cache. Use the standard Load LoRA "
                    "nodes; bypass LoRA and module injection are unsupported."
                ) from None
            raise

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name, tensor in self.placeholders.items():
            destination[prefix + name] = tensor

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Cache-backed Anima blocks require the XDNA runtime; CPU fallback is unavailable."
        )


class CacheRuntimeAttachment(RuntimeAttachment):
    """Runtime reference tagged with the block-patch set its packed cache contains."""

    def __init__(self, runtime: SharedRuntime, patch_identity=None):
        super().__init__(runtime)
        self.patch_identity = patch_identity

    def on_model_patcher_clone(self):
        return CacheRuntimeAttachment(self.runtime.acquire(), self.patch_identity)


class CacheBackedDiffusionWrapper(AnimaXDNADiffusionWrapper):
    """Dispatch to the runtime selected by the sampling ModelPatcher's pre_run."""

    def __init__(self):
        super().__init__(None)

    def _runtime_for(self, executor) -> SharedRuntime:
        runtime = getattr(executor.class_obj, "xdna_active_runtime", None)
        if runtime is None:
            raise RuntimeError(
                "Cache-backed Anima has no active XDNA runtime; run it through a ComfyUI sampler"
            )
        return runtime


class _CheckpointBlockSource:
    """Base block tensors read on demand from the identity-checked checkpoint."""

    def __init__(self, handle, diffusion_model, block_fingerprint: str):
        schema = detect_checkpoint_schema(handle.keys())
        if schema.name not in _SOURCE_PREFIXES:
            raise UnsupportedTensor("Anima checkpoint schema changed after loading")
        self._handle = handle
        self._prefix = _SOURCE_PREFIXES[schema.name]
        self.block_fingerprint = block_fingerprint
        self.placeholders = {
            f"{_BLOCK_PREFIX}{index}.{name}": tensor
            for index, block in enumerate(diffusion_model.blocks)
            for name, tensor in block.placeholders.items()
        }

    def key_patches(self, patches):
        def unchanged(tensor, **_kwargs):
            return tensor

        return {
            key: [(placeholder, unchanged), *patches.get(key, ())]
            for key, placeholder in self.placeholders.items()
        }

    def tensor(self, model_key: str) -> torch.Tensor:
        # Same BF16 normalization as weight_cache.fingerprint_execution_source.
        source_key = self._prefix + model_key[len(_MODEL_PREFIX):]
        return self._handle.get_tensor(source_key).to(torch.bfloat16).contiguous()


def _suspend_resident_runtime(diffusion_model, keep=None):
    reference = getattr(diffusion_model, "xdna_resident_runtime", None)
    resident = reference() if reference is not None else None
    if resident is not None and resident is not keep:
        resident.suspend()


class CacheBackedModelPatcher(ModelPatcher):
    def model_size(self):
        if self.size > 0:
            return self.size
        # Shape-only block placeholders own no host memory.
        self.size = sum(
            tensor.nbytes
            for tensor in self.model.state_dict().values()
            if tensor.device.type != "meta"
        )
        return self.size

    def get_key_patches(self, filter_prefix=None):
        raise RuntimeError(
            "Cache-backed Anima has no ComfyUI block weights to merge or extract. "
            "Use Load Anima (BF16) with the original checkpoint for model merging."
        )

    def add_hook_patches(self, hook, patches, strength_patch=1.0, strength_model=1.0):
        raise RuntimeError("Cache-backed Anima does not support weight hooks")

    def model_state_dict_for_saving(self, model=None, prefix=""):
        raise RuntimeError(
            "Cache-backed Anima is an execution-only MODEL, not a complete checkpoint. "
            "Use the original checkpoint for saving or merging."
        )

    def _validate_cache_contract(self):
        if self.load_device.type != "cpu" or self.offload_device.type != "cpu":
            raise RuntimeError("Cache-backed Anima requires CPU load/offload devices")
        if self.hook_patches or self.weight_wrapper_patches or self.injections:
            raise RuntimeError(
                "Cache-backed Anima cannot execute weight hooks, weight wrappers, or injections"
            )
        attachment = self.get_attachment(ATTACHMENT_KEY)
        if (
            not isinstance(attachment, CacheRuntimeAttachment)
            or not attachment._finalizer.alive
        ):
            raise RuntimeError("Cache-backed Anima runtime was unloaded; reload the MODEL")
        return attachment

    def _ensure_runtime(self) -> CacheRuntimeAttachment:
        """Bind the runtime whose packed cache matches this clone's block patches."""
        attachment = self._validate_cache_contract()
        has_block_patches = any(
            isinstance(key, str) and key.startswith(_BLOCK_PREFIX) for key in self.patches
        )
        wanted = self.patches_uuid if has_block_patches else None
        if attachment.patch_identity == wanted:
            return attachment
        base = attachment.runtime
        provenance = self.get_attachment(SOURCE_PROVENANCE_KEY)
        if not isinstance(provenance, ModelSourceProvenance):
            raise RuntimeError("Cache-backed Anima MODEL lost its source provenance; reload the MODEL")
        base_fingerprint = base.diagnostics.source_execution_fingerprint
        if base_fingerprint is None:
            raise RuntimeError("Cache-backed Anima runtime has no verified base identity")
        path = Path(provenance.path)

        def require_unchanged_source():
            if _file_identity_token(path) != provenance.identity_token:
                raise RuntimeError("Anima checkpoint changed after loading; reload the MODEL")

        require_unchanged_source()
        diffusion = self.model.diffusion_model
        # Free resident XRT weights before materializing another packed cache.
        _suspend_resident_runtime(diffusion)
        started = time.perf_counter()
        runtime = None
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                provider, patch_count = None, 0
                if has_block_patches:
                    source = _CheckpointBlockSource(handle, diffusion, base_fingerprint)
                    provider, patch_count, _, _ = _effective_lora_provider(
                        self, base_source=source
                    )
                runtime = SharedRuntime(
                    path,
                    base.cache_dir,
                    base.rebuild_cache,
                    base.qkv_chaining,
                    effective_tensor_provider=provider,
                    lora_patch_count=patch_count,
                    source_loader_total_ms=base.diagnostics.source_loader_total_ms,
                    source_loader_model_ms=base.diagnostics.source_loader_model_ms,
                )
                runtime.prepare(diffusion, cache_backed=True)
            if runtime.diagnostics.source_execution_fingerprint != base_fingerprint:
                raise RuntimeError(
                    "Anima checkpoint no longer matches the loaded MODEL; reload the MODEL"
                )
            require_unchanged_source()
            runtime.diagnostics.attach_total_ms = (time.perf_counter() - started) * 1000
        except BaseException:
            if runtime is not None:
                runtime.close()
            raise
        replacement = CacheRuntimeAttachment(runtime, wanted)
        self.set_attachments(ATTACHMENT_KEY, replacement)
        attachment.cleanup()
        return replacement

    def load(
        self, device_to=None, lowvram_model_memory=0,
        force_patch_weights=False, full_load=False,
    ):
        self._validate_cache_contract()
        if device_to is not None and device_to.type != "cpu":
            raise RuntimeError("Cache-backed Anima host layers must stay on CPU")
        return super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)

    def pre_run(self):
        attachment = self._ensure_runtime()
        result = super().pre_run()
        diffusion = self.model.diffusion_model
        _suspend_resident_runtime(diffusion, keep=attachment.runtime)
        diffusion.xdna_resident_runtime = weakref.ref(attachment.runtime)
        diffusion.xdna_active_runtime = attachment.runtime
        return result

    def cleanup(self):
        diffusion = getattr(self.model, "diffusion_model", None)
        attachment = self.get_attachment(ATTACHMENT_KEY)
        if diffusion is not None and getattr(
            diffusion, "xdna_active_runtime", None
        ) is getattr(attachment, "runtime", None):
            diffusion.xdna_active_runtime = None
        return super().cleanup()


def _load_host_model(path: Path):
    with safe_open(path, framework="pt", device="cpu") as source:
        schema = detect_checkpoint_schema(source.keys())
        if schema.name not in _SOURCE_PREFIXES:
            raise UnsupportedTensor(
                "Cache-backed loader requires a native ComfyUI Anima safetensors checkpoint"
            )
        prefix = _SOURCE_PREFIXES[schema.name]
        dtypes = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
        shapes = {}
        for key in source.keys():
            if not key.startswith(prefix):
                continue
            view = source.get_slice(key)
            dtype = view.get_dtype()
            if dtype not in dtypes:
                raise UnsupportedTensor(f"Unsupported Anima tensor dtype {dtype}: {key}")
            shapes[key[len(prefix):]] = torch.empty(
                view.get_shape(), dtype=dtypes[dtype], device="meta"
            )
        block_keys = {key for key in shapes if key.startswith("blocks.")}
        expected_blocks = {key[len(prefix):] for key in schema.canonical_to_source.values()}
        if block_keys != expected_blocks:
            raise UnsupportedTensor("Cache-backed loader requires exactly the 560 Anima block tensors")
        config = comfy.model_detection.model_config_from_unet(
            shapes, "", metadata=source.metadata()
        )
        expected_config = {
            "image_model": "anima", "num_blocks": 28, "model_channels": 2048,
            "in_channels": 16, "out_channels": 16, "num_heads": 16,
        }
        if config is None or any(
            config.unet_config.get(key) != value for key, value in expected_config.items()
        ) or config.quant_config is not None:
            raise UnsupportedTensor("Cache-backed loader requires the native Anima 2B architecture")
        cpu = torch.device("cpu")
        compute_dtype = comfy.model_management.unet_manual_cast(
            torch.bfloat16, cpu, config.supported_inference_dtypes
        )
        config.set_inference_dtype(torch.bfloat16, compute_dtype, device=cpu)
        # Construct only host layers; do not allocate blocks and then discard them.
        config.unet_config["num_blocks"] = 0
        try:
            model = config.get_model({}, "", device=cpu)
        finally:
            config.unet_config["num_blocks"] = 28
        diffusion = model.diffusion_model
        if type(diffusion).__name__ != "Anima" or len(diffusion.blocks):
            raise RuntimeError("Unsupported ComfyUI Anima constructor: expected zero host blocks")
        host_state = diffusion.state_dict()
        expected_host = set(host_state)
        host_keys = shapes.keys() - block_keys
        if host_keys != expected_host:
            raise UnsupportedTensor(
                "Anima host-layer checkpoint mismatch: "
                f"missing={sorted(expected_host - host_keys)}, "
                f"unexpected={sorted(host_keys - expected_host)}"
            )
        for key in host_keys:
            if shapes[key].shape != host_state[key].shape:
                raise UnsupportedTensor(
                    f"Anima host-layer shape mismatch for {key}: "
                    f"{tuple(shapes[key].shape)} != {tuple(host_state[key].shape)}"
                )
        host_weights = {
            key: source.get_tensor(prefix + key).to(dtype=torch.bfloat16)
            for key in sorted(host_keys)
        }
        diffusion.load_state_dict(host_weights, strict=True)
    placeholders = [{} for _ in range(28)]
    for key in block_keys:
        _, index, name = key.split(".", 2)
        placeholders[int(index)][name] = torch.empty(
            shapes[key].shape, dtype=torch.bfloat16, device="meta"
        )
    diffusion.num_blocks = 28
    diffusion.blocks = torch.nn.ModuleList(CacheOnlyBlock(names) for names in placeholders)
    diffusion.xdna_cache_backed = True
    return model


def load_cache_backed_model(unet_name, rebuild_cache=False, cache_dir="", qkv_chaining=True):
    if comfy.model_management.get_torch_device().type != "cpu":
        raise RuntimeError("Load Anima XDNA (Cache-backed) requires ComfyUI --cpu")
    started = time.perf_counter()
    category, name = _anima_model_selector(unet_name)
    path = Path(folder_paths.get_full_path_or_raise(category, name)).resolve()
    identity = _file_identity_token(path)
    model = _load_host_model(path)
    loader_ms = (time.perf_counter() - started) * 1000
    runtime = SharedRuntime(
        path,
        Path(cache_dir) if cache_dir.strip() else None,
        rebuild_cache,
        qkv_chaining,
        source_loader_total_ms=loader_ms,
        source_loader_model_ms=loader_ms,
    )
    attachment = None
    try:
        runtime.prepare(model.diffusion_model, cache_backed=True)
        if _file_identity_token(path) != identity:
            raise RuntimeError("Anima checkpoint changed while loading; reload the MODEL")
        patcher = CacheBackedModelPatcher(
            model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
        )
        patcher.set_attachments(
            SOURCE_PROVENANCE_KEY,
            ModelSourceProvenance(unet_name, str(path), identity, loader_ms, loader_ms),
        )
        attachment = CacheRuntimeAttachment(runtime)
        patcher.set_attachments(ATTACHMENT_KEY, attachment)
        patcher.add_wrapper_with_key(
            WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, CacheBackedDiffusionWrapper()
        )
        runtime.diagnostics.attach_total_ms = (time.perf_counter() - started) * 1000
    except BaseException:
        if attachment is not None:
            attachment.cleanup()
        else:
            runtime.close()
        raise
    print(
        "[Anima XDNA] Cache-backed MODEL: no ComfyUI block parameters; "
        f"host parameters={runtime.diagnostics.model_nonblock_parameter_bytes} bytes; "
        f"prepare={runtime.diagnostics.attach_total_ms:.1f} ms"
    )
    return patcher
