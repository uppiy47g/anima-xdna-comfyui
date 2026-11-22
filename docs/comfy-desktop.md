# Comfy Desktop with CPU + XDNA NPU

Use **Add Existing Instance** to run a working Anima XDNA installation inside
Comfy Desktop. Desktop supplies the window and process management; the existing
ComfyUI/Python environment still performs CPU + NPU inference. Do not replace
its CPU PyTorch, XRT or Triton-XDNA packages with a GPU installation.

The simplest source is an installation created by the
[one-shot Windows installer](../README.md#one-shot-windows-installer). It
already contains the runtime configuration, so Desktop only needs the
registration and startup arguments below. A manually provisioned installation
also needs the [INT8/AIR settings](#experimental-int8air-settings).

This integration was exercised with Comfy Desktop **1.1.9**, the existing
ComfyUI **0.38.0** backend and frontend **1.53.10** on Windows. Updating Desktop
does not require updating that backend. This is not a claim that every
Desktop-bundled ComfyUI/Python version is compatible.

## Register the existing environment

1. Stop any ComfyUI process started by a PowerShell launcher. Keep the
   installation, models and packed caches in place.
2. In Desktop, open **Menu > Add Existing Instance** and select
   `<COMFYUI_ROOT>`, the existing Git installation containing `main.py`
   (for the installer, its installation directory). Choose the detected
   **Git Clone** type.
3. Use **Virtual Environment > Browse** to select `<XDNA_VENV>`, the directory
   containing `pyvenv.cfg` and `Scripts\python.exe`; for the installer, this is
   `<COMFYUI_ROOT>\.venv`. An external virtual environment is supported; it
   need not be copied into ComfyUI or renamed `.venv`. Do not select the
   Python executable itself.
4. Name the instance, for example **Anima XDNA (CPU + NPU)**, and select
   **Track Install**. Configure the following settings **before launching**.

The tracked instance uses the existing installation without reinstalling its
packages. Leave any GPU/Standalone instances unchanged. Do not choose
**Migrate to Standalone** or change the tracked instance's PyTorch stack as
part of this setup.

## Startup settings

Open **Manage > Startup Args**. Choose **App window** and set:

```text
--cpu --listen 127.0.0.1 --port 8190 --disable-auto-launch --preview-method none --cache-classic
```

Choose a different unused port if needed. Prefer **If Port is Busy > Ask what
to do**: the command-line launcher and Desktop must not start two copies of
the same backend. Avoid simultaneous XDNA instances competing for the NPU.

`--cpu` is essential, including on machines where Desktop detects an AMD or
NVIDIA GPU. It controls ComfyUI's host tensors; XRT still dispatches the
transformer kernels to the NPU. The cache-backed loader's CPU-only guard
remains in place. Remove conflicting GPU/DirectML launch options rather than
removing this guard.

### Experimental INT8/AIR settings

An installer-created installation already has
`<COMFYUI_ROOT>\xdna-runtime\anima-xdna.json`; skip this section. The Console
shows `[Anima XDNA] Runtime config: ...` when the file is used.

Desktop starts Python directly. It does **not** execute `run-xdna-int8.ps1`
or inherit settings made in a separate PowerShell window. For a manually
provisioned installation, either create that configuration file or add
environment variables.

**Configuration file.** Save the following as
`<COMFYUI_ROOT>\xdna-runtime\anima-xdna.json`, encoded as UTF-8 without BOM:

```json
{
  "schema_version": 1,
  "profile": "int8-attn-l1k64-nq8pp",
  "xrt_dev_dir": "<XRT_SDK_DIR>",
  "triton_cache_dir": "<EXISTING_TRITON_CACHE_DIR>"
}
```

Write path separators as `/` or `\\`. Relative paths are resolved against the
`xdna-runtime` directory. The custom node derives the remaining INT8/AIR
settings from the runtime directories installed by `install_int8_runtime.ps1`
and uses 10 host threads unless `num_threads` is set. Use
`"profile": "standard"` for the standard path. Variables already present in
the environment take precedence, and `ANIMA_XDNA_CONFIG` can name a file in
another location.

**Environment variables.** Alternatively, add these entries under the
instance's **Environment Variables**:

| Name | Value |
|---|---|
| `XRT_DEV_DIR` | `<XRT_SDK_DIR>` |
| `TRITON_CACHE_DIR` | `<EXISTING_TRITON_CACHE_DIR>` |
| `PYTHONUTF8` | `1` |
| `ANIMA_XDNA_OPTIMIZED_PROFILE` | `int8-attn-l1k64-nq8pp` |
| `ANIMA_XDNA_BF16_VAE_CLFIX` | `1` |
| `ANIMA_XDNA_INT8_ARTIFACT_DIR` | `<COMFYUI_ROOT>\xdna-runtime\int8-attn-l1k64-nq8pp` |
| `ANIMA_XDNA_AIR_ATTENTION_DIR` | `<COMFYUI_ROOT>\xdna-runtime\air-attention-nq8pp` |
| `ANIMA_XDNA_ATTENTION_BACKEND` | `air` |
| `ANIMA_XDNA_ADALN_DEVICE` | `host` |
| `ANIMA_XDNA_GEMM_LAYOUT_DIR` | `<COMFYUI_ROOT>\xdna-runtime\gemm-wide32-20261025\library` |
| `ANIMA_XDNA_NUM_THREADS` | `10` |

Replace every placeholder with an absolute local path. Enter raw values:
no surrounding quotes, `$env:` assignments or references to other variables.
Do not set `XILINX_XRT` on Windows. Keep these settings instance-local.
`PYTHONUTF8` was part of the original Desktop validation; the
installer-created configuration compiled its Triton cache and ran without it.

This requires the already-provisioned
[INT8/AIR runtime](../README.md#enabling-the-experimental-int8-profile),
including the four host DLLs. For **1024x1024**, the AIR directory must contain
`self4096` as well as `self` and `cross`; the alpha.3 runtime bundle includes
it, while the alpha.2 bundle contains only `self` and `cross`. Missing
artifacts are errors, not a CPU fallback. Keep manifests UTF-8 without BOM.

These settings retain the existing experimental INT8 output, host/NPU overlap
and BF16 Qwen Image VAE optimization. They do not make INT8/AIR numerically
equivalent to the standard BF16 path. To use the standard path instead,
reproduce its launcher settings, omit the INT8 profile and VAE opt-in, and
select `baseline` attention (or separately provision experimental AIR).

## Models, input and output

In **Manage > Storage**, use the same model directories as the working
launcher. Shared directories are suitable when Desktop already lists those
models. Otherwise add them to this instance's **Model Directories**.
Desktop generates the instance's model-path configuration; do not depend
on a generated YAML file belonging to a different instance. ComfyUI still
reads `extra_model_paths.yaml` in the installation directory, such as the one
written by the installer's `-ModelsDirectory` option.

Set the shared input/output toggles, or the instance-specific directories,
to retain the launcher's existing locations. Configure these in Storage
rather than also passing competing `--input-directory`/`--output-directory`
arguments. No model or cache copy is necessary.

## Launch and use

Launch **Anima XDNA (CPU + NPU)** from Desktop. Existing workflows in this
ComfyUI installation remain available. Use **Load Anima XDNA (Cache-backed)**
with the existing Anima Turbo V1.1, Qwen text encoder and Qwen Image VAE;
connect its MODEL directly to KSampler. Ordinary supported LoRA nodes still
go after the loader, with the existing runtime restrictions.

For a smoke check, use batch 1, four steps, CFG 1, Euler/simple and either
512x512 or 1024x1024. Change the seed between prompts to actually exercise the
sampler. The first prompt prepares the runtime and checks INT8 boundaries;
it is expected to be slower.

The Console should show the optimized INT8/AIR hooks and cache-backed MODEL,
not a GPU backend. The following read-only checks verify that the running
instance is CPU-configured and that the loader registered:

```powershell
Invoke-RestMethod http://127.0.0.1:8190/system_stats
Invoke-RestMethod http://127.0.0.1:8190/object_info/LoadAnimaXDNACache
```

The device list reports **cpu**, not the NPU: it describes PyTorch's device.
NPU execution is verified separately by the runtime and successful XRT
dispatches. See the [Desktop validation record](anima-xdna-reproducibility.md#comfy-desktop-cpunpu-integration-2026-11-18).

## Troubleshooting and rollback

- **Loader requires `--cpu`:** edit this instance's Startup Args and restart
  it. Settings do not change an already-running Python process.
- **Node missing / XRT or DLL import failure:** verify the selected external
  virtual environment, `XRT_DEV_DIR`, installed host DLLs and instance-local
  environment values. Selecting CPU hardware alone does not install XDNA.
- **Configuration error at startup:** the message names the configuration
  file. Fix invalid JSON, unknown keys or a byte-order mark.
- **Models missing:** correct Storage paths; do not download duplicate
  checkpoints just to make the dropdown populate.
- **Port busy:** stop the old launcher/instance before starting Desktop.
- **Return to the launcher:** stop the Desktop instance, then use the
  unchanged `run-xdna-int8.ps1`. **Forget** removes tracking; do not use
  **Uninstall**, which can delete the underlying installation.

Official Desktop references:
[existing instances](https://docs.comfy.org/installation/desktop/usage/instance-management)
and [Startup Args / Storage](https://docs.comfy.org/installation/desktop/usage/manage).
