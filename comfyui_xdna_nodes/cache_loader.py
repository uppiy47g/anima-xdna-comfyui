"""CPU host layers plus verified XDNA block weights, without a second block MODEL."""

from pathlib import Path
import time

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
    _file_identity_token,
)


class CacheOnlyBlock(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "Cache-backed Anima blocks require the XDNA runtime; CPU fallback is unavailable."
        )


class CacheBackedModelPatcher(ModelPatcher):
    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        # Reject even empty mappings: LoRA key discovery cannot see omitted blocks.
        raise RuntimeError(
            "Cache-backed Anima does not support live LoRA/weight patches. "
            "Use Load Anima (BF16) -> Load LoRA -> Load/Attach Anima XDNA Model."
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
        if (
            self.patches or self.hook_patches
            or self.weight_wrapper_patches or self.injections
        ):
            raise RuntimeError("Cache-backed Anima cannot execute live weight patches or injections")
        attachment = self.get_attachment(ATTACHMENT_KEY)
        if not isinstance(attachment, RuntimeAttachment) or not attachment._finalizer.alive:
            raise RuntimeError("Cache-backed Anima runtime was unloaded; reload the MODEL")

    def load(
        self, device_to=None, lowvram_model_memory=0,
        force_patch_weights=False, full_load=False,
    ):
        self._validate_cache_contract()
        if device_to is not None and device_to.type != "cpu":
            raise RuntimeError("Cache-backed Anima host layers must stay on CPU")
        return super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)

    def pre_run(self):
        self._validate_cache_contract()
        return super().pre_run()


def _load_host_model(path: Path):
    with safe_open(path, framework="pt", device="cpu") as source:
        schema = detect_checkpoint_schema(source.keys())
        prefixes = {
            "native-net": "net.",
            "native-model-diffusion-model": "model.diffusion_model.",
            "comfyui-anima-module": "",
            "comfyui-model-wrapper": "diffusion_model.",
        }
        if schema.name not in prefixes:
            raise UnsupportedTensor(
                "Cache-backed loader requires a native ComfyUI Anima safetensors checkpoint"
            )
        prefix = prefixes[schema.name]
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
    diffusion.num_blocks = 28
    diffusion.blocks = torch.nn.ModuleList(CacheOnlyBlock() for _ in range(28))
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
        attachment = RuntimeAttachment(runtime)
        patcher.set_attachments(ATTACHMENT_KEY, attachment)
        patcher.add_wrapper_with_key(
            WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, AnimaXDNADiffusionWrapper(runtime)
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
