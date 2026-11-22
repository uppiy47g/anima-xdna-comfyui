# Anima XDNA 2 ComfyUI MODEL wrapper

This custom-node package replaces only the 28 `transformer_blocks` in ComfyUI's
native Anima Base v1.0, Turbo V1.1, or validated WAI Nova Anima Turbo LoRA
Ver V1.0 diffusion model with the resident Triton-XDNA/XRT runtime. It does
**not** use ONNX or Vitis AI EP.

For an isolated Windows setup, sanitized launcher/model-path templates, the
full verification ladder, image evidence, and machine-readable measurements,
see the [public reproducibility report](../docs/anima-xdna-reproducibility.md).

ComfyUI still executes:

- Qwen 3 0.6B tokenization and the Anima LLM adapter
- latent patch embedding, timestep embedding, and RoPE setup
- LayerNorm/AdaLN, RMSNorm, softmax, GELU, gates, and residual operations
- final projection/unpatchify, sampler orchestration, and Qwen Image VAE

XDNA 2 executes 448 Linear/QK/AV GEMM dispatches in the default 28-block
chain when `ANIMA_XDNA_ADALN_DEVICE=xdna`. The self/cross Q/K/V projection groups each run as one `NPUChain`,
and each AdaLN `linear_1 -> linear_2` pair uses one externally bound BF16
activation BO instead of a D2H/H2D round trip. Shape/dtype-compatible
activation BOs are leased from a session-owned pool and closed with the session.
Unsupported pair layouts retain the existing host-visible two-dispatch path.
The optional `qkv_chaining=false` input disables Q/K/V grouping while retaining
AdaLN activation chaining, producing 560 dispatches. Disabling both optimizations
through the Python runtime restores the prior 644-dispatch path for comparison.

By default (`ANIMA_XDNA_ADALN_DEVICE=host`) the single-row AdaLN
`linear_1 -> linear_2` GEMVs instead run on the host in BF16 from the packed
(LoRA-effective) weights, removing their XDNA dispatches (three per block with
pair chaining, six without). These GEMVs are weight-bandwidth bound, so the
fixed dispatch cost dominated. Results are not bitwise identical to the XDNA
AdaLN path; set `ANIMA_XDNA_ADALN_DEVICE=xdna` before starting ComfyUI to
restore it.
Weights, kernel objects, contexts, and BOs
remain resident between denoising steps. The wrapper never silently falls
back to CPU.

## Supported contract

- ComfyUI API validated at commit
  `170594057a22673349ddf0a3d88624b7fa5865bb`
- native Base v1.0, Turbo V1.1, WAI Nova Anima Turbo LoRA Ver V1.0, or
  Radiance Turbo Anima v2.0 transformer loaded by
  **Load Anima (BF16)** (explicit ComfyUI `dtype=torch.bfloat16`)
- matching read-only XDNA source: validated Diffusers/native Base for Base,
  or the same native Turbo checkpoint for Turbo
- batch 1, latent `[1,16,1,64,64]` (512x512 output) or `[1,16,1,128,128]`
  (1024x1024 output: baseline attention, or experimental AIR attention with a
  `self4096` artifact; the experimental INT8 profile also accepts it, see
  below)
- adapted Qwen context `[1,512,1024]`
- BF16 model weights and block activations, FP32 sampler latent

At 1024x1024 the wrapper derives the 64x64 patch grid, 4096 image tokens and
RoPE from the latent. The runtime splits every XDNA dispatch into the
1024-token kernel shapes used at 512x512. Start ComfyUI with
`ANIMA_XDNA_ATTENTION_BACKEND` unset or set to `baseline` for the reference
path: a 4-step, CFG 1 Turbo V1.1 1024x1024 prompt completed in 251 s (about
55 s per step). Experimental AIR attention
(`ANIMA_XDNA_ATTENTION_BACKEND=air`) also accepts 4096 tokens when the AIR
artifact directory has a `self4096` artifact beside `self` and `cross`; the
same prompt then took 108 s on the first run and 82 s on a second run with
another seed (about 18 s per step). AIR is not quality-equivalent (see
[AIR attention at 1024x1024](../anima_xdna_poc/README.md#air-attention-at-1024x1024)),
and a missing `self4096` artifact stops the run with an explicit error. The
experimental INT8 profile also runs 4096 tokens with the `self4096` artifact
(see below). Expect roughly twice
the step time with CFG > 1, and much higher host memory use than at 512x512.

Preview/Aesthetic checkpoints, LoRA patches, LLLite, other
batch/resolution/context sizes, and training are rejected. The normal ComfyUI
MODEL is cloned; it is not modified. Before XRT opens, the wrapper computes a
canonical full digest over all 560 block tensors. Base/Turbo or other
MODEL/source mismatches are rejected before BO population or dispatch. The
default path remains BF16; the separate, experimental INT8 projection profile
below still requires the same BF16 Anima source.

## Experimental INT8/AIR profile

Install the versioned Windows runtime bundle from the root README with
`.\tools\install_int8_runtime.ps1 -ComfyUIRoot "<COMFYUI_ROOT>"`. It installs
the fixed-shape NPU artifacts, four host-kernel DLLs and the launcher. The
bundle does not install ComfyUI, Python dependencies, the AMD driver/XRT
stack, or model weights. To build the host DLLs yourself, use an x64 Visual
Studio developer PowerShell with `.\examples\build_fused_host.ps1`. The profile
requires OS-enabled AVX-512 F/DQ/BW/VL and the validated PyTorch reduction
behavior; it is not a portable CPU fallback.

`run-xdna-int8.ps1` is an explicit opt-in launcher for the fixed
`int8-attn-l1k64-nq8pp` profile. It requires the five tuned INT8 artifact
directories and the matching `air-attention-nq8pp` self/cross artifacts under
`xdna-runtime`; missing or incompatible artifacts stop startup rather than
falling back. It also enables host AdaLN, the 10-thread GEMM layout, and the
BF16 Qwen VAE channels-last/clfix path. The existing `run-xdna.ps1` and other
launchers remain unchanged; this profile launcher restores its environment
variables and working directory when ComfyUI exits.

The installer places this launcher beside an existing `run-xdna.ps1` in the
ComfyUI root. It is a deployment adapter for the validated local installation:
the base launcher and compatible Python/XRT toolchain must already be
provisioned. Artifact generation and experiment provenance are described in
the [reproducibility report](../docs/anima-xdna-reproducibility.md).

MLP and QKV/output projections use symmetric W8A8 with INT32 accumulation.
AIR QK, softmax, and AV remain BF16, and the VAE change is limited to
`qwen_image_vae.safetensors`. This is experimental and not quality-equivalent
to the standard BF16 pipeline, and this combined INT8 profile has no
quality-equivalence claim. The earlier standalone AIR comparison recorded a
29.035% fixed-seed latent NRMS versus control; that is not a measurement of the
combined INT8 profile. The BF16 VAE layout fix also changes decoded pixels.
Use the standard launcher for quality-sensitive work.

The deployed ComfyUI integration completed a 512x512, 4-step generation with
896 dispatches and passed 1,904 runtime boundary checks. The first checked
generation took 17.93 s; a same-process repeat reproduced the latent bitwise
and took 5.79 s for sampling plus 1.51 s for VAE decode (7.30 s total). These
are single-run smoke measurements, not a controlled benchmark or a quality
equivalence claim.

At 1024x1024 the profile runs its 512x512 INT8 kernels once per 1024-row tile
and needs the AIR `self4096` artifact beside `self` and `cross` in the AIR
directory. The alpha.3 runtime bundle installed by `install_int8_runtime.ps1`
includes it; an installation from the alpha.2 bundle needs a `self4096`
directory added to that AIR directory by hand. The
checked first prompt in a process runs sequentially, and later prompts overlap
host work with NPU jobs, including AIR attention (bitwise identical results,
about 18% faster warm sampling); set
`ANIMA_XDNA_INT8_OVERLAP=0` to disable the overlap. INT8 at 1024x1024 is not
quality-equivalent to BF16 AIR (one fixed-seed decoded image: MAE 24.3/255).
See the root README and the
[reproducibility report](../docs/anima-xdna-reproducibility.md#air-attention-overlap-at-1024x1024-2026-11-17).
For an F16 or F32 source, matching is performed after the same explicit BF16
normalization used by the XDNA packed cache. Raw source hashes and dtypes still
remain part of the cache identity.

## Installation

On Windows, the
[one-shot installer](../README.md#one-shot-windows-installer)
(`tools\install_windows.ps1`) creates a dedicated ComfyUI installation with
this node, its Python dependencies, the XRT SDK and the runtime
configuration. To install manually instead, place or clone this repository
below `ComfyUI\custom_nodes`, then install it in the same Python environment
that starts ComfyUI:

```powershell
git clone https://github.com/uppiy47g/anima-xdna-comfyui.git `
  ComfyUI\custom_nodes\anima-xdna-comfyui
cd ComfyUI\custom_nodes\anima-xdna-comfyui
python -m pip install -e .
```

That environment also needs the Windows XRT SDK/`pyxrt`, Triton-XDNA and its
matching MLIR wheels. Set `XRT_DEV_DIR` to the XRT SDK directory before
starting ComfyUI, or provide it in `xdna-runtime\anima-xdna.json` below the
ComfyUI root, which the node reads at startup. The packed cache defaults to
`%USERPROFILE%\.cache\anima-xdna\weights`.

For a Desktop window around this same CPU + NPU backend, follow the
[Comfy Desktop guide](../docs/comfy-desktop.md). Register the existing
environment rather than installing the node into a GPU Python environment;
the `--cpu` requirement remains unchanged.

## Nodes

### Cache-backed, execution-only loader

**Load Anima XDNA (Cache-backed)** (`LoadAnimaXDNACache`) is an opt-in
replacement for the **Load Anima (BF16)** + **Load/Attach Anima XDNA Model**
pair. Select a native Anima safetensors checkpoint and connect its `MODEL`
directly to the sampler. Do not attach it again. It requires ComfyUI `--cpu`;
attention selection and the existing runtime environment are unchanged.

The loader reads tensor shapes without materializing block tensors, constructs
only the host layers, and loads their BF16 weights. The 28 transformer blocks
execute from the existing verified packed-weight cache, built automatically
when absent. Input/time embeddings, the LLM adapter, and final projection stay
on the host, using ComfyUI's normal CPU compute-dtype policy. The original
checkpoint is still required and is never modified. The cache is not a
standalone checkpoint or a new quantization format.

This MODEL has no ComfyUI block Parameters, not a full MODEL whose weights
were deleted after loading. **Anima XDNA Runtime Status** reports
`model_block_parameters_omitted=true`, zero block Parameter bytes, and the
`cache_backed_no_comfy_block_parameters` memory policy. XRT weight buffers and
the mapped cache remain separate allocations; do not add their byte counts
to process memory counters.

Normal MODEL clones share the host module and reference-counted XDNA runtime.
**Unload Anima XDNA Runtime** releases one attachment; the final reference
closes the cache and device resources. Reload the MODEL before using an
explicitly unloaded attachment again.

Standard **Load LoRA** or **Load LoRA Model Only** nodes can follow this
loader, chained as usual. The MODEL exposes the block keys as BF16 meta
placeholders, so ComfyUI's normal LoRA key mapping accepts block patches
without allocating block weights. ComfyUI applies host-layer patches; block
patches are consumed by XDNA. Before a patched MODEL first samples, the loader
validates its patches, reads base block tensors from the original checkpoint
on demand, and builds or reuses the effective-weight packed cache exactly as
the original attach path does. The supported patch types and strengths are
the same as in the [original workflow](#original-loader-and-attach-workflow);
unsupported types raise before NPU dispatch.

Each distinct patch set gets one runtime, reused across prompts and clones.
When a different MODEL samples, for example switching from a LoRA back to the
base, the previously resident runtime is suspended: its XRT buffers close and
its cache mapping is released, then remapped with an identity check on next
use. The first prompt with a new LoRA therefore includes the effective-cache
build (about 25 s at 512x512) or verification (about 16 s); repeat prompts do
not. **Anima XDNA Runtime Status** shows the base runtime for a LoRA MODEL
until that MODEL has sampled once.

Weight hooks, bypass LoRA nodes (**Load LoRA (Bypass) (For debugging)** and
its Model Only variant), GPU placement and checkpoint saving or merging are
rejected rather than silently ignoring missing blocks. Full checkpoints that already
contain merged weights can be selected normally. This loader does not change
AIR's numerical limitations.

The [512x512 memory experiment](../docs/anima-xdna-reproducibility.md#cache-backed-model-loader-and-512x512-memory-2026-11-17)
measured a 3.60 GiB reduction in peak process working set, with identical
latents and pixels. A 1024x1024 memory comparison was not run. With one
LoRA, the [LoRA experiment](../docs/anima-xdna-reproducibility.md#cache-backed-loader-with-standard-lora-nodes-2026-11-17)
measured 11.92 GiB versus 16.52 GiB for the original LoRA workflow, with
identical latents, including after base/LoRA switching.

### Original loader and attach workflow

1. Load the native Anima MODEL with **Load Anima (BF16)**. Its selector
   includes both `diffusion_models:` transformer files and `checkpoints:`
   full checkpoints; unqualified legacy workflow values still resolve as
   `diffusion_models`.
   The normal `UNETLoader` does not expose BF16; on the validated CPU setup
   its default Anima policy chooses FP32. The custom loader passes BF16 through
   ComfyUI's supported `load_diffusion_model` options and refuses the result
   if any transformer-block Parameter expanded to another dtype.
2. Connect it to **Load/Attach Anima XDNA Model**. To use an Anima LoRA,
   insert one or more standard **Load LoRA** or **Load LoRA Model Only** nodes
   before the attach node. Strength changes and chained ordinary LoRAs produce
   separate exact effective-weight identities.
3. Leave `checkpoint` at **Auto (from MODEL)**. **Load Anima (BF16)** records
   its exact source on the MODEL and the attach node reuses it automatically,
   including through normal ModelPatcher clones. The file remains read-only
   and is the XDNA cache source of truth. A manual path remains available for
   MODELs loaded another way; it must contain the same block weights.
   The attach node activates the selected ModelPatcher through ComfyUI's
   model-management API before verification, so a LoRA-to-base workflow switch
   restores the correct shared weights.
   If its block weights are F16 or F32, the first attach creates a verified
   BF16 packed cache without modifying the checkpoint. The ComfyUI log and
   runtime status explain this conversion; later attaches report verified
   cache reuse.
4. Connect the returned MODEL to the normal sampler.
5. Use **Anima XDNA Runtime Status** to inspect cache state, first/warm timing,
   dispatches, transfers, BO population, allocations, and resident reuse.
6. Use **Unload Anima XDNA Runtime** before switching checkpoints when an
   immediate release is required. Garbage collection also releases the
   attachment.

   The validated WAI Nova file is a full ComfyUI checkpoint containing Anima,
   Qwen, and VAE tensors, not a standalone LoRA delta. The BF16 loader extracts
   its `model.diffusion_model.*` Anima component; use the normal Qwen and VAE
   workflow nodes for the remaining components. The XDNA source selector must
   reference the same full checkpoint so the 560-tensor canonical fingerprint
   check remains exact.

`qkv_chaining` defaults to enabled. It only fuses the three Q/K/V projection
launches within each attention module; LayerNorm/AdaLN, RMSNorm, RoPE,
scale/mask/softmax, GELU, gates, and residual operations remain on CPU.

`rebuild_cache` explicitly recreates the packed cache when the attach node
executes. ComfyUI caches the node result for unchanged inputs, so changing this
input from `false` to `true` forces one rebuild; repeated prompts with the same
`true` value reuse that attached result. Toggle it back before requesting a
later rebuild. Checkpoint file identity and attach options are part of the
cache key, while the MODEL dependency is tracked by ComfyUI. `cache_dir` may
point to another user-writable ASCII path. It must not point into the model
folder.

The attach status reports block Parameter logical/unique-storage bytes and
dtype counts after source identity matches. It warns when CPU block Parameters
are FP32. XRT BO population is reported separately. ModelPatcher clones share
the underlying module, so attach never deletes or replaces block Parameters.
For ordinary additive `LoRAAdapter` patches, and exact additive legacy `diff`
patches limited to self/cross-attention Q/K norm weights, it uses ComfyUI's
supported weight calculation path on CPU copies, validates all 560 effective
BF16 tensors, and builds or reuses a cache keyed by their exact values. This
covers hybrid Turbo LoRAs that combine matrix LoRA weights with norm deltas.
LoHa, LoKr, OFT, DoRA, reshape/mid-weight LoRA, other block diff/bias patches,
and runtime transformer patches stop before NPU dispatch rather than being
silently ignored.
Fresh-process measurements on the validated host found stock FP32 loading
used about 7.12 GB more process PrivateUsage than BF16 loading for either
checkpoint (46.5%); see the evidence for working-set and peak counters. A
separate BF16-plus-XRT fixture run verified matching Base/Turbo fingerprints
and resident BO reuse. The exact Turbo 8-step Euler/simple/CFG-1/seed-424242
BF16+XDNA workflow reproduced the previous validated XDNA image pixelwise;
the latent differs from the stock-FP32 CPU reference by 25.55% normalized
RMS. This supports decoded-image parity with the prior XDNA path, not latent
equivalence to CPU or a speedup claim.

## Verified result

On Ryzen AI 9 365 / Strix XDNA 2 (`npu2`), XRT 2.21.75 and Triton-XDNA
3.6.0.2026093004:

- native ComfyUI MODEL registration and loading succeeded
- one complete synthetic DiT comparison: normalized RMS 0.01528
- real Qwen prompt, Euler, two steps: latent normalized RMS 0.01819
- isolated cold wrapper call: 26.73 s, including 3,963,617,280 bytes of
  process-local weight BO population
- historical pre-Q/K/V second resident wrapper call: 7.73 s total / 7.57 s
  chain, 644 dispatches, 644 resident hits, zero weight population bytes and
  no recompile
- resident profiles count output bytes synchronized from XRT BOs to host
  visibility in `last_d2h_host_visible_bytes`; these are not additional
  staging-buffer allocations
- real Qwen prompt, CFG 3.5, one denoising step: normalized RMS 0.04068
  (the two CFG branches execute sequentially on the single-tenant NPU)
- decoded 512x512 images: PSNR 36.33 dB, SSIM 0.8458
- validated 28-block Q/K/V-only chained path: 532 dispatches and 8,085,045,248 H2D
  bytes per denoising step, versus 644 dispatches and 8,349,286,400 H2D bytes
  with chaining disabled; activation chaining reduces the expected dispatch
  count to 448 and reports avoided transfer bytes separately, pending hardware
  revalidation

Turbo V1.1 was additionally validated from its native 685-tensor BF16
checkpoint. A real Qwen/Euler/CFG 1 one-step sampler latent passed at
normalized RMS `0.02650`; an 8-step 512x512 run produced PSNR `19.855 dB`,
SSIM `0.82480`, and 8-bit MAE `10.884` against the CPU reference. The local
8-step setting is validated here, but is not claimed as an official V1.1
recommendation. Turbo's synthetic chain has a transient block-13 normalized
RMS peak of `0.08455` and final `0.01004`; use the documented separate
10% intermediate / 2% final gate rather than Base's uniform 5% gate.
On 2026-10-06, a same-server warm 8-step Euler/simple/CFG 1/seed 424242
comparison measured 147.56 s with `qkv_chaining=false` and 139.18 s with it
enabled. The final 512x512 RGB images were pixel-identical. A real Base
preprocessing fixture exposed 7.828% transient CPU-oracle NRMS at block 14
against the strict 5% Base gate; the chained and previous XDNA paths were
bitwise equal at every block, with 0.8407% final NRMS. This is recorded as a
Base oracle limitation, not a passing Base chain gate.

The first call additionally verifies the 3.96 GB packed cache and populates
process-local XRT BOs, so cold and warm timings must not be compared as the
same phase. Within one ComfyUI process, an unchanged payload that has already
passed full SHA-256 verification reuses an identity-bound verification lease;
model switching therefore skips the repeated 3.96 GB hash when returning to a
previously verified cache. XRT BO population remains runtime-local and is not
covered by this lease.

Measured BF16 loader details, Windows process-memory methodology, the
stock-FP32/BF16 memory pair, and the end-to-end image/latent comparison are
recorded in the [reproducibility report](../docs/anima-xdna-reproducibility.md).

If an applied LoRA/ModelPatcher patch, unsupported shape, incompatible
ComfyUI API, cache corruption, XRT failure, or numerical/runtime error is
encountered, execution raises an actionable error instead of calling the
original CPU block loop.
