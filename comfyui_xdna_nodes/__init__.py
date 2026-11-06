"""ComfyUI nodes for the resident Anima XDNA 2 MODEL wrapper."""

import os

from .model_wrapper import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


def _install_opt_in_runtime():
    profile = os.environ.get("ANIMA_XDNA_OPTIMIZED_PROFILE", "").strip()
    if profile:
        from anima_xdna_poc.int8_runtime import install_profile

        install_profile(profile)

    vae_mode = os.environ.get("ANIMA_XDNA_BF16_VAE_CLFIX", "0").strip()
    if vae_mode not in ("0", "1"):
        raise ValueError("ANIMA_XDNA_BF16_VAE_CLFIX must be '0' or '1'")
    if vae_mode == "1":
        from anima_xdna_poc.vae_optimization import install_vae_loader_patch

        install_vae_loader_patch()


_install_opt_in_runtime()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
