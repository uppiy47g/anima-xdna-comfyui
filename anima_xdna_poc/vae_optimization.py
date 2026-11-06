"""Opt-in BF16/channels_last/clfix path for the ComfyUI Qwen Image VAE loader."""

import functools

import torch


VAE_NAME = "qwen_image_vae.safetensors"
_PATCHED = False


def _restrided_upsample_forward(forward):
    @functools.wraps(forward)
    def restrided(x, *args, **kwargs):
        if x.ndim == 4 and x.shape[0] == 1 and x.is_contiguous(memory_format=torch.channels_last):
            _, channels, height, width = x.shape
            x = x.as_strided(x.shape, (channels * height * width, 1, width * channels, channels))
        return forward(x, *args, **kwargs)

    return restrided


def optimize_vae(vae):
    """Convert one Qwen VAE to BF16 and install the measured channels_last decode path."""
    model = vae.first_stage_model
    if getattr(model, "_anima_xdna_bf16_clfix", False):
        return vae
    model.to(dtype=torch.bfloat16)
    vae.vae_dtype = torch.bfloat16

    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last_3d)
        elif isinstance(module, torch.nn.Conv2d):
            module.weight.data = module.weight.data.contiguous(memory_format=torch.channels_last)
        elif isinstance(module, torch.nn.Upsample):
            module.forward = _restrided_upsample_forward(module.forward)

    original_decode = model.decode

    @functools.wraps(original_decode)
    def decode(z, *args, **kwargs):
        return original_decode(z.contiguous(memory_format=torch.channels_last_3d), *args, **kwargs)

    model.decode = decode
    model._anima_xdna_bf16_clfix = True
    return vae


def install_vae_loader_patch():
    """Patch only ComfyUI's named Qwen Image VAE loader; leave every other VAE unchanged."""
    global _PATCHED
    if _PATCHED:
        return

    import nodes

    loader = getattr(nodes, "VAELoader", None)
    if loader is None or not callable(getattr(loader, "load_vae", None)):
        raise RuntimeError("ComfyUI VAELoader.load_vae is unavailable")
    original_load = loader.load_vae

    @functools.wraps(original_load)
    def load_vae(instance, vae_name):
        result = original_load(instance, vae_name)
        if vae_name == VAE_NAME:
            if not isinstance(result, tuple) or len(result) != 1:
                raise RuntimeError("ComfyUI VAELoader returned an unexpected VAE result")
            return (optimize_vae(result[0]),)
        return result

    loader.load_vae = load_vae
    _PATCHED = True
    print(f"[Anima XDNA] BF16+channels_last+clfix enabled for {VAE_NAME} only.")
