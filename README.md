# Experimental ComfyUI Image Generation on AMD XDNA 2

This project explores image generation in ComfyUI on AMD XDNA 2 NPUs through
Triton-XDNA and XRT, without conversion pipelines such as ONNX or Vitis AI EP.
Its long-term goal is to broaden native safetensors support so more models can
be loaded directly. Current support covers Anima Base v1.0, Turbo V1.1, the
structurally compatible WAI Nova Anima Turbo LoRA Ver V1.0 full checkpoint,
and the F16 Radiance Turbo Anima v2.0 checkpoint. Runtime failures are
explicit, with no silent CPU fallback.

Compatible F16 or F32 Anima sources are converted to the BF16 execution format
when the local packed cache is built. The original checkpoint remains
unchanged. Runtime status and the ComfyUI log state whether the verified BF16
cache was created or reused; MODEL/source matching uses the normalized BF16
execution weights while raw source identity still isolates and verifies each
cache entry.

This is an experimental **CPU-hybrid** implementation, not NPU-only image
generation. XDNA executes Linear and attention QK/AV GEMMs. Qwen, patch/time
embeddings, normalization, RoPE, softmax, GELU, residuals, sampling, final
projection, and VAE decoding remain on CPU.

## Development log

Development notes, experiment history, and progress reports are published on
https://note.com/loyal_owl5720/m/m79af97ed02e5

## Supported configuration

- Windows x64, AMD XDNA 2 / AIE2P (`npu2`), current AMD NPU driver.
- Python 3.13 for the validated Windows `pyxrt` ABI; CPU tests support Python
  3.10 and newer.
- XRT SDK 2.21.75 and Triton-XDNA `3.6.0.2026093004+75679dc` were validated.
- ComfyUI commit `170594057a22673349ddf0a3d88624b7fa5865bb`.
- Anima Base v1.0, Turbo V1.1, the validated WAI Nova Anima Turbo LoRA
  Ver V1.0 full checkpoint, or Radiance Turbo Anima v2.0; BF16 execution
  weights, batch 1, 512x512, 1024 image tokens and 512x1024 adapted Qwen
  context.

Model weights, packed caches, captured tensors and vendor binaries are not
distributed. Obtain models from their authorized sources and review their
separate licenses. Ordinary additive Anima LoRAs and the narrowly scoped
hybrid Turbo LoRA Q/K norm `.diff` patches described below are supported.
LLLite, LoHa/LoKr/OFT/DoRA, quantization, training, and other shapes or patch
formats are unsupported.

## Installation

Use an isolated ComfyUI installation and its Python environment. Clone with
the exact custom-node directory name:

```powershell
git clone https://github.com/uppiy47g/anima-xdna-comfyui.git `
  "<COMFYUI_ROOT>\custom_nodes\anima-xdna-comfyui"
cd "<COMFYUI_ROOT>\custom_nodes\anima-xdna-comfyui"
```

Install the ABI-matching Windows XRT `pyxrt`, official Triton-XDNA/MLIR/LLVM-AIE
wheels and CPU PyTorch first, following
[the runtime prerequisites](anima_xdna_poc/README.md#windows-prerequisites).
Then use the **same interpreter that launches ComfyUI**:

```powershell
& "<XDNA_VENV>\Scripts\python.exe" -m pip install -e .
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
& "<XDNA_VENV>\Scripts\anima-xdna-probe.exe"
```

The probe must report target `npu2` and runtime `xrt`; missing hardware or
dependencies are errors, not successful fallback. Do not set `XILINX_XRT`
on Windows. Copy the [model-path template](examples/extra_model_paths_anima_xdna.yaml)
outside the repository, replace placeholders locally, then start ComfyUI:

```powershell
.\examples\run_comfyui_xdna.ps1 `
  -ComfyUIRoot "<COMFYUI_ROOT>" `
  -XDNAVenv "<XDNA_VENV>" `
  -XRTDevDir "<XRT_SDK_DIR>" `
  -ModelPathsConfig "<USER_MODEL_PATHS_YAML>"
```

The launcher binds to localhost and uses `--cpu` for the host pipeline.
The repository root registers only XDNA nodes; it needs no LoRA Bridge
converter or activation-capture package.

## Usage and evidence

Load the native checkpoint with **Load Anima (BF16)**, connect its MODEL to
**Load/Attach Anima XDNA Model**, and leave `checkpoint` at
**Auto (from MODEL)**. The loader carries the exact read-only source into the
attach node, so it only needs to be selected once. Then connect the wrapped
MODEL to the normal sampler. **Anima XDNA Runtime
Status** reports residency, dispatches and transfers; **Unload Anima XDNA
Runtime** releases the attachment. See the
[node guide](comfyui_xdna_nodes/README.md) and
[API insertion fragment](examples/anima_xdna_512_api.json).

Standard ComfyUI **Load LoRA** or **Load LoRA Model Only** nodes may be placed
between **Load Anima (BF16)** and the XDNA attach node. Ordinary additive
Anima LoRAs are materialized non-destructively into exact BF16 effective
weights. Each distinct effective 560-tensor fingerprint gets a separately
verified packed cache; identical weights reuse it. Base checkpoints and LoRA
files remain read-only. The attach node uses ComfyUI's managed model-loading
lifecycle, so switching from a LoRA workflow back to the base MODEL restores
the correct shared weights before verification. Hybrid Turbo LoRAs may also
carry exact additive `.diff` tensors for the 28 blocks' self/cross-attention
Q/K norm weights; those narrowly scoped patches are supported with the same
shape, finite-value, fingerprint, and cache checks.

The default resident Q/K/V-chained path executes 19 XRT dispatches per block,
532 per 28-block denoising step. `qkv_chaining=false` selects the 23/644 control.
Historical measurements, metadata-free output images, exact gates, and
reproduction instructions are in the
[public report](docs/anima-xdna-reproducibility.md) and
[schema-backed evidence](docs/evidence/anima-xdna-validated.json).
These measurements were collected before this repository migration, not
rerun on migration hardware.

**Known numerical limitations:** the captured Base chain exceeds its strict
5% intermediate CPU-oracle gate (7.828% worst, 0.8407% final). Turbo uses a
separate 10% intermediate / 2% final captured-fixture gate. Its BF16+XDNA
8-step latent differs from the stock-FP32 CPU reference by about 25.55%
normalized RMS, despite reproducing the prior validated XDNA image exactly.
No blanket equivalence or NPU-only speed claim is made.

## Development checks

```powershell
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m compileall -q anima_xdna_poc comfyui_xdna_nodes
python -m build
git diff --check
```

The suite validates public-artifact privacy, PNG digests/metadata, the evidence
JSON schema, packaging identity and custom-node registration alongside CPU
unit tests. Hardware tests explicitly skip when Triton-XDNA or local
`ANIMA_XDNA_CHECKPOINT` / `ANIMA_XDNA_CHAIN_FIXTURE` inputs are unavailable;
skips are not hardware passes. Never commit fixtures, checkpoints or caches.

## License and provenance

MIT, with the original copyright notice preserved in [LICENSE](LICENSE).
This standalone project was extracted from the XDNA work in
`uppiy47g/anima-lora-style-bridge`, without its git history or LoRA conversion
functionality. See [provenance and migration scope](docs/provenance.md) for
source revisions, attribution, exclusions and the historical evidence boundary.
