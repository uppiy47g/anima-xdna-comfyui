"""Process-local runtime settings from an installer-written Anima XDNA config.

The Windows installer writes ``<ComfyUI root>\\xdna-runtime\\anima-xdna.json``
so that a plain ``main.py --cpu`` start (for example from Comfy Desktop) gets
the same settings as the PowerShell launchers. Values are applied with
``setdefault``: variables already present in the environment always win.
"""

import json
import os
import re
import sys
from pathlib import Path

CONFIG_ENV = "ANIMA_XDNA_CONFIG"
CONFIG_NAME = "anima-xdna.json"
RUNTIME_DIRECTORY = "xdna-runtime"
SCHEMA_VERSION = 1
STANDARD_PROFILE = "standard"
INT8_PROFILE = "int8-attn-l1k64-nq8pp"
PROFILES = (STANDARD_PROFILE, INT8_PROFILE)
_KEYS = {"schema_version", "profile", "xrt_dev_dir", "triton_cache_dir", "num_threads"}


def candidate_roots(module_file=__file__):
    """ComfyUI roots that may own an ``xdna-runtime`` directory, in priority order."""
    roots = []
    folder_paths = sys.modules.get("folder_paths")
    if folder_paths is not None:
        base_path = getattr(folder_paths, "base_path", None)
        if base_path:
            roots.append(Path(base_path))
        module_path = getattr(folder_paths, "__file__", None)
        if module_path:
            roots.append(Path(module_path).parent)
    node_root = Path(module_file).resolve().parent.parent
    if node_root.parent.name.lower() == "custom_nodes":
        roots.append(node_root.parent.parent)
    unique = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def find_config(environ=None, roots=None):
    environ = os.environ if environ is None else environ
    explicit = environ.get(CONFIG_ENV, "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{CONFIG_ENV} points to a missing file: {path}")
        return path
    for root in candidate_roots() if roots is None else roots:
        path = Path(root) / RUNTIME_DIRECTORY / CONFIG_NAME
        if path.is_file():
            return path.resolve()
    return None


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Anima XDNA config must be a JSON object: {path}")
    unknown = sorted(set(config) - _KEYS)
    if unknown:
        raise ValueError(f"Unknown Anima XDNA config keys {unknown}: {path}")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Anima XDNA config schema_version: {path}")
    if config.get("profile") not in PROFILES:
        raise ValueError(f"Anima XDNA config profile must be one of {PROFILES}: {path}")
    for key in ("xrt_dev_dir", "triton_cache_dir"):
        if config.get(key) is not None and not (isinstance(config[key], str) and config[key].strip()):
            raise ValueError(f"Anima XDNA config {key} must be a non-empty path: {path}")
    threads = config.get("num_threads")
    if threads is not None and (isinstance(threads, bool) or not isinstance(threads, int) or threads < 1):
        raise ValueError(f"Anima XDNA config num_threads must be a positive integer: {path}")
    return config


def _resolve(runtime_root, value):
    path = Path(os.path.expandvars(value)).expanduser()
    return str(path if path.is_absolute() else (runtime_root / path).resolve())


def config_environment(config, runtime_root):
    """Environment variables equivalent to the installer's launchers."""
    runtime_root = Path(runtime_root)
    environment = {}
    if config.get("xrt_dev_dir"):
        environment["XRT_DEV_DIR"] = _resolve(runtime_root, config["xrt_dev_dir"])
    if config.get("triton_cache_dir"):
        environment["TRITON_CACHE_DIR"] = _resolve(runtime_root, config["triton_cache_dir"])
    if config["profile"] == INT8_PROFILE:
        environment.update({
            "ANIMA_XDNA_OPTIMIZED_PROFILE": INT8_PROFILE,
            "ANIMA_XDNA_BF16_VAE_CLFIX": "1",
            "ANIMA_XDNA_INT8_ARTIFACT_DIR": str(runtime_root / INT8_PROFILE),
            "ANIMA_XDNA_AIR_ATTENTION_DIR": str(runtime_root / "air-attention-nq8pp"),
            "ANIMA_XDNA_ATTENTION_BACKEND": "air",
            "ANIMA_XDNA_ADALN_DEVICE": "host",
            "ANIMA_XDNA_GEMM_LAYOUT_DIR": str(runtime_root / "gemm-wide32-20261025" / "library"),
            "ANIMA_XDNA_NUM_THREADS": str(config.get("num_threads") or 10),
        })
    elif config.get("num_threads"):
        environment["ANIMA_XDNA_NUM_THREADS"] = str(config["num_threads"])
    return environment


def _is_xrt_sdk(path):
    return (path / "include" / "xrt").is_dir() and (path / "lib").is_dir()


def _version_key(path):
    return tuple(int(part) for part in re.findall(r"\d+", path.name))


def detect_xrt_sdk(environ=None):
    """Locate an XRT SDK that Triton-XDNA would not find on its own, or None."""
    environ = os.environ if environ is None else environ
    if os.name != "nt" or environ.get("XRT_DEV_DIR"):
        return None
    program_files = Path(environ.get("PROGRAMFILES", r"C:\Program Files"))
    if _is_xrt_sdk(program_files / "AMD" / "xrt"):
        return None
    local_app_data = environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    amd = Path(local_app_data) / "AMD"
    if not amd.is_dir():
        return None
    sdks = [path for path in amd.glob("xrt-*") if path.is_dir() and _is_xrt_sdk(path)]
    return str(max(sdks, key=_version_key)) if sdks else None


def apply_runtime_config(environ=None, roots=None, log=print):
    """Apply config defaults before the runtime reads its environment."""
    environ = os.environ if environ is None else environ
    path = find_config(environ, roots)
    if path is not None:
        config = load_config(path)
        overridden = []
        for name, value in config_environment(config, path.parent).items():
            if name in environ and environ[name] != value:
                overridden.append(name)
            environ.setdefault(name, value)
        log(f"[Anima XDNA] Runtime config: {path} (profile {config['profile']})")
        if overridden:
            log("[Anima XDNA] Explicit environment overrides config: " + ", ".join(sorted(overridden)))
    xrt = detect_xrt_sdk(environ)
    if xrt is not None:
        environ["XRT_DEV_DIR"] = xrt
        log(f"[Anima XDNA] Auto-detected XRT SDK: {xrt}")
    return path
