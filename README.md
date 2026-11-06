Anima XDNA 2 for ComfyUI

An experimental ComfyUI custom node that runs Anima transformer blocks on
AMD XDNA 2 NPUs using Triton-XDNA and XRT. It works with native model files
and does not require ONNX or Vitis AI EP conversion.

The project has two distinct execution paths:

Standard BF16 path: the default resident XDNA path. It supports the
  validated Anima models and LoRA cases described below. Experimental AIR
  attention can be enabled separately.
Experimental INT8/AIR profile: an explicit opt-in, fixed-shape
  optimization for Anima Turbo V1.1. It combines INT8 projections, fused CPU
  boundaries, a retuned AIR attention schedule and an optional BF16 VAE path.
  It changes generated outputs and is not a quality-equivalent replacement
  for the standard path.

This is a CPU/NPU hybrid, not an all-NPU pipeline. Runtime failures are
explicit; the node does not silently fall back to CPU execution.

Important: quality and numerical limits

The experimental INT8 profile is not numerically or visually equivalent
to the original BF16/FP32 pipeline. Use it when throughput or latency is
more important than reproducing the standard-path output. Keep the standard
BF16 path for quality-sensitive work.

The full-projection INT8 comparison below is against a previous experimental
baseline that already used INT8 MLPs and BF16 attention projections. It is
not a comparison against the original BF16/FP32 model:

Prompt and seed
Final latent NRMSE
Decoded RGB MAE (0–255)
Cherry blossoms / 700755787614899
36.24%
18.240
Blue-haired portrait / 42
30.09%
16.302
Mountain village/lake / 314159265
30.92%
10.899

These are substantial differences, not negligible rounding noise. The tested
images were finite and reproducible, but visual inspection found changes in
details such as hair, clothing, flowers, buildings and foreground. NRMSE and
pixel MAE quantify difference from a reference; they are not perceptual
quality scores and do not mean perceived quality fell by the same percentage.
Three prompt/seed cases do not establish general image quality.

The VAE comparison is separate. For three fixed sampler latents, BF16+clfix
versus the FP32 VAE produced RGB MAE of about 0.30–0.36/255 and PSNR of
46.90–49.15 dB; about 22.50–23.49% of pixels differed by at least one 8-bit
level. This is not a measurement of the cumulative quality difference between
the complete optimized profile and the original pipeline.

Bitwise repeatability and boundary checks demonstrate consistency within the
tested optimized path. They do not establish equivalence to the original
BF16/FP32 pipeline. The combined profile has not been evaluated against that
pipeline across a broad prompt/seed set.

Performance and measurement scope

The changes below were measured in separate matched experiments with
different baselines and experiment series. Do not add the savings together
or interpret them as one cumulative A/B test. The sampler timings exclude VAE
decoding unless the table says otherwise.

Change tested
Four-step sampler median
Fuse Q/K RMSNorm, RoPE and head-major layout preparation
10.493 → 9.282 s
Add INT8 QKV/output projections and host-boundary integration
9.336 → 7.474 s
Retune INT8 GEMM schedules
7.392 → 6.994 s
Retune AIR K/V buffering and query-core schedule
7.086 → 6.056 s

These sampler comparisons retain 896 NPU dispatches per four-step sample.
The improvements therefore come from work per dispatch, CPU-side boundaries,
and data movement/layout as well as the NPU kernels—not simply from reducing
the number of launches.

In a separate eight-run VAE comparison, median decode time changed from
2.762 s to 1.219 s, and median sampler-plus-VAE time changed from 8.832 s to
7.322 s. These harness timings exclude model loading, text encoding and image
saving.

A production-node integration smoke run measured 5.787 s for sampling plus
1.512 s for VAE decode (7.299 s total). It completed 896 dispatches and 1,904
runtime boundary checks; a same-seed repeat latent was bitwise identical.
This was a single-run integration measurement, not a controlled benchmark
median. User-observed ComfyUI console runs include Prompt executed times in
the 6-second range; cache state was not controlled for those observations.

These results describe Anima on the tested AMD hardware. They are not a
direct comparison with Qualcomm measurements that use different models,
devices and timing scopes.

Detailed experiment methods, individual runs, caveats and machine-readable
evidence are in the reproducibility report
and docs/evidence/.

How the execution paths work

Standard BF16 path

The custom node replaces only the 28 Anima transformer_blocks with the
resident XDNA runtime. ComfyUI continues to run Qwen tokenization and the Anima
LLM adapter, latent and timestep embeddings, normalization, RoPE, softmax,
GELU, residual operations, sampling, final projection and VAE decoding on the
host.

With the default host AdaLN configuration, the standard path runs the
Q/K/V projection groups and supported Linear/attention GEMMs on XDNA. Weights,
contexts and XRT buffers remain resident across denoising steps. The regular
launcher uses baseline attention by default. Direct AIR attention is a
separate experimental opt-in.

Experimental INT8/AIR profile

The int8-attn-l1k64-nq8pp profile is deliberately separate from the standard
path and is selected only through explicit configuration. It includes:

Symmetric W8A8 INT8 MLP, QKV and attention output projections with INT32
  accumulation.
Fused CPU processing around quantization/dequantization, modulation,
  gated residuals, and Q/K RMSNorm, RoPE and head-major layout conversion.
Fixed-shape INT8 GEMM schedules tuned for the validated matrix dimensions.
A direct MLIR-AIR attention kernel using the tested 8-query-core,
  K/V ping-pong schedule. Attention QK, softmax and AV remain BF16; this
  attention kernel is authored in MLIR-AIR, not lowered by Triton.
An opt-in BF16/channels-last/clfix path for
  qwen_image_vae.safetensors.

The runtime validates required artifacts and settings, checks attention
boundaries on the first four forwards, and raises errors instead of silently
falling back on unsupported cases. This fixed-shape experimental profile is
not a general quantization mode.

Supported configuration

The validated configuration is:

Windows x64, AMD XDNA 2 / AIE2P (npu2) with a compatible AMD NPU driver.
XRT SDK 2.21.75 and Triton-XDNA
  3.6.0.2026093004+75679dc.
Python 3.13 for the validated Windows pyxrt ABI; CPU tests support Python
  3.10 and newer.
ComfyUI API validated at commit
  170594057a22673349ddf0a3d88624b7fa5865bb.
Anima Base v1.0, Turbo V1.1, the validated WAI Nova Anima Turbo LoRA
  Ver V1.0 full checkpoint, or Radiance Turbo Anima v2.0.
Batch 1, 512×512 output, 1024 image tokens and adapted Qwen context
  [1, 512, 1024].
BF16 execution weights and block activations; the sampler latent is FP32.

Compatible F16 or F32 Anima sources are normalized to BF16 when the local
packed cache is built. The source checkpoint remains unchanged. Cache identity
retains the raw source hash and dtype while the MODEL/source comparison uses
the normalized BF16 execution weights.

The standard path supports ordinary additive Anima LoRAs and the narrowly
scoped hybrid Turbo LoRA Q/K norm .diff patches described below. Preview and
Aesthetic checkpoints, LLLite, LoHa/LoKr/OFT/DoRA, other patch formats,
training and other shapes are unsupported. INT8 is supported only by the
separate experimental profile described above.

Model weights, packed caches, captured tensors and compiled vendor/device
artifacts are not distributed. Obtain model files from their authorized
sources and review their licenses separately.

Installation

Use an isolated ComfyUI installation and install the node into the same Python
environment that starts ComfyUI. Clone the repository using the expected
custom-node directory name:

git clone https://github.com/uppiy47g/anima-xdna-comfyui.git `
  "<COMFYUI_ROOT>\custom_nodes\anima-xdna-comfyui"
cd "<COMFYUI_ROOT>\custom_nodes\anima-xdna-comfyui"

Install the ABI-matching Windows XRT pyxrt, Triton-XDNA/MLIR/LLVM-AIE
wheels and CPU PyTorch first, following the
Windows runtime prerequisites.
Then install the package using the ComfyUI interpreter:

& "<XDNA_VENV>\Scripts\python.exe" -m pip install -e .
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
& "<XDNA_VENV>\Scripts\anima-xdna-probe.exe"

The probe must report target npu2 and runtime xrt. Missing hardware or
dependencies are errors, not successful fallback. Do not set XILINX_XRT on
Windows.

Copy the model-path template
outside the repository and replace its placeholders locally. Start the
standard ComfyUI path with:

.\examples\run_comfyui_xdna.ps1 `
  -ComfyUIRoot "<COMFYUI_ROOT>" `
  -XDNAVenv "<XDNA_VENV>" `
  -XRTDevDir "<XRT_SDK_DIR>" `
  -ModelPathsConfig "<USER_MODEL_PATHS_YAML>"

This launcher binds to localhost and uses --cpu for the host pipeline. The
repository registers only Anima XDNA nodes; no LoRA Bridge converter or
activation-capture package is needed.

Using the standard path

Load a native Anima model with Load Anima (BF16). The loader supports
   diffusion_models: transformer files and checkpoints: full checkpoints.
   The standard ComfyUI UNETLoader does not expose the same explicit BF16
   contract on the validated CPU setup.
Connect it to Load/Attach Anima XDNA Model. For a supported Anima LoRA,
   place standard Load LoRA or Load LoRA Model Only nodes before the
   attach node.
Keep checkpoint set to Auto (from MODEL). The BF16 loader records the
   exact source and the attach node reuses it. A manually selected source must
   contain matching block weights.
Connect the returned MODEL to the normal sampler.
Use Anima XDNA Runtime Status to inspect cache state, timings,
   dispatches, transfers, BO population, allocations and resident reuse.
Use Unload Anima XDNA Runtime when an immediate runtime release is
   needed.

The MODEL is cloned and not modified. Before XRT opens, the wrapper computes a
canonical digest over all 560 block tensors and rejects source mismatches
before buffer population or dispatch. If the source is F16 or F32, a verified
BF16 packed cache is created or reused without modifying the checkpoint.

The validated WAI Nova file is a full ComfyUI checkpoint containing Anima,
Qwen and VAE tensors, not a standalone LoRA delta. The BF16 loader extracts
its model.diffusion_model.* Anima component; use normal Qwen and VAE workflow
nodes for the remaining components, and reference the same full checkpoint as
the XDNA source.

qkv_chaining defaults to enabled and groups the three Q/K/V projection
launches within each attention module. It does not fuse LayerNorm/AdaLN,
RMSNorm, RoPE, softmax, GELU, gates or residual operations; those remain on
CPU in the standard path. rebuild_cache=true recreates the packed cache when
the attach node executes. ComfyUI caches node results for unchanged inputs;
toggle the option back before requesting a later rebuild. cache_dir may
point to another user-writable ASCII path, but not into the model directory.

Supported additive LoRAs are materialized non-destructively into exact BF16
effective weights, with a separately verified packed cache for each distinct
560-tensor fingerprint. The attach node follows ComfyUI's model-management
lifecycle so switching from a LoRA workflow back to the base model restores
the correct shared weights before verification. Hybrid Turbo LoRAs may also
contain exact additive .diff tensors for self/cross-attention Q/K norm
weights. LoHa, LoKr, OFT, DoRA, reshape/mid-weight LoRA, other block diff/bias
patches and runtime transformer patches stop before NPU dispatch rather than
being silently ignored.

Enabling the experimental INT8 profile

This profile requires the validated native host kernels and locally provisioned
NPU artifacts. It is not enabled by installing the custom node alone.

Build the native host DLLs

Use an x64 Visual Studio developer PowerShell with MSVC and OpenMP support:

cd "<COMFYUI_ROOT>\custom_nodes\anima-xdna-comfyui"
.\examples\build_fused_host.ps1

The included C++ sources produce four DLLs in anima_xdna_poc. Generated
DLLs are not committed. The host requires OS-enabled AVX-512 F/DQ/BW/VL and the
validated PyTorch reduction behavior; this is not a portable CPU fallback.

Provision artifacts and launch

The repository does not distribute compiled NPU artifacts. Provision the
matching artifacts from the documented experiment outputs so the ComfyUI
root contains:

<COMFYUI_ROOT>\
  run-xdna.ps1
  run-xdna-int8.ps1
  xdna-runtime\
    int8-attn-l1k64-nq8pp\
      i8-proj_in-l2k512-l1k64\
      i8-proj_out-kloop-l2k512-l1k64\
      i8-sq2048-l1k64\
      i8-qkv-shared-l1k64\
      i8-qkv-cross-l1k64\
    air-attention-nq8pp\
      self\
      cross\
    gemm-wide32-20261025\
      library\

Copy run-xdna-int8.ps1 from this repository beside the existing local
run-xdna.ps1. The profile launcher is a deployment adapter: it sets the
experimental environment for the child ComfyUI process, verifies required
directories and then invokes the existing launcher. It restores its own
PowerShell environment on exit. The standard launchers remain unchanged.

Start it from PowerShell with:

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File "<COMFYUI_ROOT>\run-xdna-int8.ps1"

Missing artifacts or unsupported configuration stop startup; there is no
silent baseline fallback. For quality-sensitive generation, use the standard
launcher instead. Artifact provenance and reproduction details are in the
reproducibility report.

Historical baseline and evidence

The default resident Q/K/V-chained path historically executed 19 XRT
dispatches per block (532 per 28-block denoising step). Disabling
qkv_chaining selected the 23-dispatch control (644 per step). These figures
describe the standard BF16-era path and should not be confused with the
experimental INT8 profile's 896 dispatches per four-step sample.

The validated BF16+XDNA Turbo 8-step Euler/simple/CFG-1/seed-424242 workflow
reproduced the previous validated XDNA image pixelwise. Its latent differed
from the stock-FP32 CPU reference by 25.55% normalized RMS. The captured Base
chain also exceeded its strict 5% intermediate CPU-oracle gate (7.828% worst,
0.8407% final); Turbo uses a separate 10% intermediate / 2% final
captured-fixture gate. These are specific fixture comparisons, not blanket
quality-equivalence claims.

Development notes and experiment history are published in the
reproducibility report, its
schema-backed evidence, and the
development log.

Development checks

python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m compileall -q anima_xdna_poc comfyui_xdna_nodes
python -m build
git diff --check

The tests cover CPU functionality, evidence schemas, package behavior and
public-artifact privacy. Hardware tests skip when Triton-XDNA or the local
ANIMA_XDNA_CHECKPOINT / ANIMA_XDNA_CHAIN_FIXTURE inputs are unavailable;
a skip is not a hardware pass. Never commit model files, captures, packed
caches or generated binaries.

License and provenance

MIT, with the original copyright notice preserved in LICENSE.
This standalone project was extracted from the XDNA work in
uppiy47g/anima-lora-style-bridge, without its git history or LoRA conversion
functionality. See provenance and migration scope for
source revisions, attribution, exclusions and the historical evidence
boundary.

