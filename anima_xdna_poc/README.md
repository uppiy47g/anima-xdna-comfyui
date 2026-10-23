# Anima resident transformer chain on AMD XDNA 2

This isolated proof of concept executes either one Anima Linear or the complete
Anima/Cosmos transformer chain (one block, a range, or all 28 blocks) directly
from its original safetensors on XDNA 2 through Triton-XDNA and XRT. It does
**not** use ONNX, Vitis AI EP, or a converted model format, and it does not
modify ComfyUI or model files.

## Supported model and exact block

The block adapter follows the local Diffusers `CosmosTransformer3DModel`
implementation used by Anima Base v1.0 (`diffusers 0.39.0.dev0`):

1. AdaLN-Zero: SiLU, 2048->256->6144, affine-free LayerNorm (`eps=1e-6`),
   shift/scale/gate, and the model-level `temb`.
2. Self-attention: 2048-wide Q/K/V, 16 heads of 128, learned Q/K RMSNorm
   (`eps=1e-6`), Cosmos image RoPE, non-causal attention, and output projection.
3. Cross-attention: 2048-wide Q, 1024-wide text K/V, 16x128 heads, Q/K
   RMSNorm, optional `[B,1,1,S]` mask, and output projection.
4. MLP: exact GELU (`approximate="none"`) with 2048->8192->2048 projections.
5. A gated residual follows each attention and MLP stage.

The loader validates all 20 block tensors and materializes only requested
tensors through `safetensors.safe_open`. The original checkpoint remains the
read-only source of truth. Native Triton cache entries contain compiled
kernels; the separate packed-weight cache described below is a verified,
regenerable local derivative and is never a replacement model.

Positive compatibility covers full BF16 **Anima Base v1.0** checkpoints in
Diffusers `transformer_blocks.*` or native `net.*` form and the validated full
native **Turbo V1.1** checkpoint in `model.diffusion_model.*` form. The loader
canonicalizes all 560 block tensors, while the cache manifest retains every
exact source key. LoRA adapters and quantized checkpoints remain unsupported.

## NPU, attention fusion, and residency

Every block Linear and every QK-transpose/attention-value GEMM runs on XDNA 2.
The current optimized path uses **19 XRT dispatches per block**:

- 15 standalone Linear dispatches (the 16 logical model Linear operations,
  with the 8192-wide MLP output reduction split into four FP32 partials,
  minus the six Q/K/V projections grouped below);
- two Q/K/V dispatches, each an `NPUChain` containing three BF16 Linear
  launches for one attention module;
- four attention dispatches: self QK, self AV, cross QK, and cross AV.

Each attention dispatch uses Triton-XDNA's official `NPUChain` multi-launch API
to stitch the 16 correct head-specific Triton kernels into one ELF/XRT run.
Q/K/V chaining removes four more XRT launches per block by sharing the self
attention input and staging cross query/context once each. Set the ComfyUI
`qkv_chaining` input to `false` to run the 23-dispatch comparison path. A
single 3D Triton grid was tested and rejected because the
installed AIE2P lowering corrupted heads at realistic multi-tile shapes;
folding heads into the program M dimension was also rejected by AIR tile-size
inference. Neither unsafe path is used by the package.

The resident session owns the XRT device, ELF objects, hardware contexts,
kernels, and mapped `ext.bo` buffers. Static weight BOs are populated once and
reused across blocks/runs; dynamic input and output BOs are reused by shape.
The Q/K/V chains own three `shared.empty` output buffers per signature and
bind them through Triton-XDNA's public `bound_buffers` interface; these
session-lifetime buffers are closed on normal exit or error. Outputs are
copied to BF16 CPU tensors because subsequent normalization/attention stages
are still host-side. This reuses output allocation, not block-to-block
activation values, and does not eliminate output synchronization. The profiler
counts device-to-host-visible bytes; no separate D2H staging tensor is
implied. Cache synchronization time is reported separately for ordinary
Linear dispatches; the head-chain API combines launch and output
synchronization time. After population, checkpoint tensors are replaced by
shape/dtype metadata (except the small Q/K RMSNorm vectors), avoiding a second
approximately 4 GB host copy while the resident BOs are live. The safetensors
checkpoint always remains the source of truth.

SiLU, LayerNorm/AdaLN modulation, Q/K RMSNorm, RoPE, scale/mask/softmax, exact
GELU, tensor reshapes, gates, and residual additions remain explicit PyTorch
CPU operations over shared/mapped memory. These operations and their timing
are reported; there is no silent CPU fallback. Parameterized GEMMs account for
137,451,536,384 logical FLOPs per block and are all on the NPU.

The supported 512x512 path is batch 1, 1024 image tokens, hidden size 2048,
512 context tokens, and context width 1024. The deterministic default input is
synthetic. The current Base and Turbo hardware chain fixtures were captured
read-only before block 0 during standalone ComfyUI prompt generation with the
Qwen encoder, model patch/time embeddings, and Cosmos RoPE; fixture files are
local-only and are not distributed.

## Windows prerequisites

- Windows 10/11 x64 and AMD XDNA 2/AIE2P (`npu2`) with a current AMD NPU
  driver.
- Python 3.13 is recommended because the official Windows `pyxrt.pyd` uses that
  ABI.
- Visual Studio 2022 Desktop development with C++, CMake, and Ninja.
- An official [Xilinx/XRT release](https://github.com/Xilinx/XRT/releases)
  Windows SDK. Extract the inner `xrt` directory to a user-writable location,
  set `XRT_DEV_DIR`, and copy only its ABI-matching `pyxrt.pyd` into the
  dedicated virtual environment.
- Official AMD Triton-XDNA, MLIR-AIR, MLIR-AIE, LLVM-AIE, and PyTorch CPU
  wheels.

Validated versions are Python 3.13.9, Triton-XDNA
`3.6.0.2026093004+75679dc`, and XRT 2.21.75. The XRT
`xrt_windows_sdk.zip` SHA-256 is
`ccc244c2c423588972ade76142cdc01049477aaa39a35be97e782b97eb7c5295`.
Do not set `XILINX_XRT` on Windows; use `XRT_DEV_DIR`.

```powershell
py -3.13 -m venv .venv-xdna
.\.venv-xdna\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install triton-xdna `
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels `
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 `
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly `
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

MLIR-AIE uses relative intermediates and upstream generated transforms contain
UTF-8 punctuation. The runtime stages compilation under the ASCII-only
`%USERPROFILE%\.cache\anima-xdna` path so Japanese repository paths work
without a global encoding change.

## Commands and output

```powershell
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
$checkpoint = "<MATCHING_BASE_OR_TURBO_SAFETENSORS>"
$fixture = "<OPTIONAL_CAPTURED_BLOCK_INPUT_PT>"
anima-xdna-probe
anima-xdna-linear $checkpoint
anima-xdna-block $checkpoint `
  --image-tokens 1024 --context-tokens 512 --runs 2
anima-xdna-chain $checkpoint `
  --fixture $fixture `
  --start-block 0 --end-block 28 --warmups 1 --runs 3 --profile
# Turbo's validated policy reports its bounded transient amplification while
# retaining a strict final-output gate:
anima-xdna-chain "<TURBO_V11_SAFETENSORS>" `
  --fixture "<CAPTURED_REAL_PREBLOCK_INPUT_PT>" `
  --start-block 0 --end-block 28 --max-block-nrms 0.10 --max-final-nrms 0.02
anima-xdna-cache build $checkpoint
anima-xdna-cache verify $checkpoint
anima-xdna-cache list
```

`anima-xdna-probe` must report `target: npu2` and `runtime: xrt`.
The block and chain CLIs report exact shape/range, NPU/host operation timing,
dispatches, H2D/D2H staging bytes, allocations/resident hits, host-copy,
cache-sync and kernel/wait time, logical NPU FLOPs, cold/populate/cached timing,
and blockwise numerical errors.

### Optional QK/scale epilogue experiment

`examples/qk_scale_epilogue_probe.py` is a model-free hardware experiment,
not a production attention replacement. It requires the opt-in epilogue
generator and transform library from
[Triton-XDNA PR 171](https://github.com/amd/Triton-XDNA/pull/171), with a
matching AIR/AIE/Peano toolchain. Use an isolated environment rather than
upgrading the environment serving ComfyUI.

From the repository root, using that environment's interpreter:

```powershell
$env:PYTHONUTF8 = "1"
python -m examples.qk_scale_epilogue_probe `
  --work-dir "<LOCAL_ASCII_EXPERIMENT_DIRECTORY>" --runs 7 --warmups 2
```

The default runs all 16 heads, head dimension 128 (padded to 256 as in the
current runtime), with 1024 queries and either 1024 or 512 keys. Seeded
synthetic BF16 inputs need no model checkpoint. It compares unfused QK plus
host scaling, a float32-output scale epilogue, and an epilogue requesting the existing
BF16-before-scale boundary with BF16 output. Compiler optimization and
lowering can still change rounding; the probe measures parity rather than
assuming it.

`results.json` records first-call and warmed timings, transfer byte counts,
CPU-oracle errors, and differences from the unfused control. Timings include
input staging, dispatch/wait, output synchronization, and host conversion or
scaling, but exclude input preparation, softmax, AV, and image generation.
Each head group remains one dispatch in every variant.

For separate BF16 direct-output and conversion-phase measurements, use:

```powershell
python -m examples.qk_scale_epilogue_probe `
  --work-dir "<NEW_LOCAL_ASCII_EXPERIMENT_DIRECTORY>" `
  --threads 10 --warmups 3 --runs 21 `
  --modes baseline qk_bf16 scale_fp32 direct_bf16 legacy_bf16
```

`qk_bf16` stores QK directly as BF16 but keeps the existing host scale.
`direct_bf16` requests scale followed by a BF16 store without an explicit
pre-scale BF16 cast. These are different experiments. Output-cast epilogues
use the epilogue transform even when no scale is fused. The report separates
chain, host numerical conversion, host bit-view wrapping, and host scale
times. See the [direct-output comparison](../docs/anima-xdna-reproducibility.md#separate-bf16-direct-output-comparison).

The fixed CPU elementwise gate is `rtol=0.02, atol=0.03`. The experiment
records gate failures without widening the threshold and continues to the
other variants; it exits nonzero if any case fails. A successful compile or
a speed improvement is not a production numerical-quality pass, especially
when the unfused control itself fails that gate.

### Fixed-seed real-model epilogue experiment

`examples/qk_scale_model_probe.py` uses the same isolated epilogue-capable
environment to call the validated ComfyUI sampler directly, without a web
server or loading third-party custom nodes. It temporarily instruments the
current checkout's runtime inside that process; production files and the
normal ComfyUI environment are not modified.

```powershell
$env:PYTHONUTF8 = "1"
python -m examples.qk_scale_model_probe `
  --comfy-root "<VALIDATED_COMFYUI_ROOT>" `
  --checkpoint "<TURBO_V11_SAFETENSORS>" `
  --text-encoder "<QWEN_3_06B_BASE_SAFETENSORS>" `
  --vae "<QWEN_IMAGE_VAE_SAFETENSORS>" `
  --work-dir "<NEW_LOCAL_ASCII_EXPERIMENT_DIRECTORY>" `
  --seed 700755787614899 --runs 3
```

Conditions are 512x512, batch 1, four Euler/simple steps, CFG 1, denoise 1,
BF16 loading, QKV chaining, and no LoRA. Positive and negative prompts are
the established cherry-blossom fixture prompts. This checkout uses the
19-dispatch/block control, not a separately installed optimized runtime.

The experiment saves all four actual pre-block inputs, compares all 28
block outputs for both same-input replays and independent sampler
trajectories, runs a CPU block-chain oracle on each captured input, and
compares sampler callbacks and final latents. CPU comparisons use the
existing BF16 CPU block oracle, not a separate stock-FP32 CPU sampler.
Turbo's fixed 10% intermediate / 2% final chain policy is reported without
relaxation. Independently generated images are decoded after sampler
profiling.

Warmed sampler measurements rotate the three variant orders across three
repeats. Stage spans and dispatch counts are recorded alongside individually
timed original XRT synchronization, start/wait, and NumPy copy calls.
Instrumentation is inserted into in-memory copies of the existing runner
methods; no package source is edited. Nested counters must not be added to
stage wall times. BO sync byte counts describe synchronized buffer sizes,
not physical bus traffic on this unified-memory device.

`report.json` separates cold setup, capture/comparison runs, and resident
timing runs. The run's status indicates execution completion; numerical
gate outcomes are separate and must be inspected before production use.
Local `.pt` files and PNGs are private experiment artifacts; never commit
model weights, conditioning, captured activations, or latent tensors.
The work directory may reuse a cache from a failed initial setup but refuses
to overwrite an existing real-input capture.

The completed fixed-seed experiment and all per-block errors are documented
in the [real-model results](../docs/anima-xdna-reproducibility.md#real-model-qkscale-experiment-turbo-v11-four-steps)
and [sanitized evidence](../docs/evidence/qk-scale-model-probe.json).
Warm sampler medians were 26.452 s (control), 26.353 s (FP32 output), and
25.083 s (BF16 output). Both fused final latents differed from control by
18.640% NRMS; the control and both fused variants failed the unchanged CPU
final-block gate on three of the four inputs. This is not production
enablement or a numerical-quality pass.

To test the separate direct-output paths with the same real-model command,
append `--modes baseline qk_bf16 direct_bf16` and choose a new work directory.
The completed comparison preserved all 112 block outputs, final latent,
and image exactly with QK-only BF16 output plus CPU scale, while halving QK
output sizes and removing the host numerical cast. Full-sampler medians were
25.821 s control / 25.487 s QK-only BF16; overlapping timing ranges prevent
claiming a robust generation speedup. Scale-then-direct-BF16 still changed
the final latent by 18.640% NRMS. See the
[real-model direct-output results](../docs/anima-xdna-reproducibility.md#direct-output-on-the-fixed-seed-real-model).

On the validated Ryzen AI 9 365 / NPU Strix system, Anima Base block 0 at
1024/512 captured inputs produced:

| Metric | Legacy per-head path | Resident stitched path |
|---|---:|---:|
| XRT dispatches/block | 83 | 23 |
| H2D staging/block | 612,237,312 B | 156,631,040 B |
| Device-to-host-visible output/block | 314,048,512 B | 314,048,512 B |
| cached wall/block | about 2.086 s | 0.236-0.362 s |

The historical resident D2H value was recorded as `0 B` because no separate
staging tensor was used; it did not mean the host could read output without an
XRT BO synchronization. The corrected counter reports 314,048,512 output
bytes per block while the number of separate D2H staging-buffer copies remains
zero. Current `anima-xdna-chain --profile` output counts these bytes as
`d2h_staging_bytes` for compatibility with the JSON profile field name.

On 2026-10-06, one warm resident block-0 run using deterministic synthetic
inputs measured Base at 334.47 ms wall (60.71 ms host stages, 269.62 ms NPU
stages) and Turbo at 342.37 ms (68.71 ms host, 268.66 ms NPU). Both used 23
dispatches, 156,631,040 H2D bytes, and 314,048,512 synchronized output bytes;
normalized RMS against each CPU block oracle was 0.0025100 (Base) and
0.0025191 (Turbo). These are single-block observations, not captured-chain or
end-to-end benchmarks. A trial that fused QK attention scaling into the
stitched Triton kernel was reverted: the validated AIR pipeline failed
compilation with `iterator_interchange` length 4 for a 2-loop operation. The
scale and softmax therefore remain on CPU, and this phase claims no generation
speedup.

One representative steady optimized block measured 252.4 ms wall, including
111.2 ms XRT kernel/wait, 35.7 ms explicit host operations, 2.5 ms mapped-BO
host copies, and 3.3 ms cache synchronization. The legacy synthetic benchmark
was 0.662 s and its CPU oracle was 0.717 s. The optimized path is both faster
than that CPU result and materially reduces dispatch and staging.

The first build of all four stitched 16-head artifacts took approximately
862 s on this toolchain; the native Triton cache reuses those shape-specific
artifacts for every block and later process. This compile cost is never
reported as steady execution.

The earlier pre-Q/K/V 23-dispatch full-chain profile measured:

- 644 dispatches (`23 x 28`);
- cached steady wall times of 7.02, 7.03, and 7.15 s;
- approximately 0.25 s median per block;
- 23.19 s for the CPU oracle (about 3.3x slower);
- 4,385,669,120 B H2D and 8,793,358,336 B of device-to-host-visible output;
- about 3.22 s total XRT kernel/wait, 73-75 ms host copy, and 86-89 ms cache
  synchronization;
- final normalized RMS error `0.007466678`;
- worst normalized RMS error `0.041646643` at block 13;
- final max/mean absolute error 1664 / 7.2158.

Absolute errors become large at sparse, very high-magnitude activations in the
deep chain. That earlier fixture passed its fixed 5% per-block gate; it is not
the current real ComfyUI Base fixture below. This does not replace strict
single-block coverage: the deterministic block integration still compares
elementwise with `rtol=0.02`, `atol=0.003`. The CPU adapter was also checked
against the installed Diffusers 0.39.0 authoritative 28-block chain and ended
at about 0.7% normalized RMS difference.

Turbo's separate 10% intermediate / 2% final chain gate is defined for the
captured real preprocessing fixture used by the ComfyUI validation. The
default synthetic seed-0 input is deliberately outside that claim and was
rejected at 9.61% final normalized RMS. Always pass a real captured fixture
when reproducing the Turbo chain gate; the real one-step ComfyUI sampler test
is the end-to-end oracle.

## Q/K/V chaining measurements

On 2026-10-06, the captured 1024-image-token / 512-context Base and Turbo
fixtures measured the 23-dispatch path against Q/K/V chaining on the same
resident runtime:

| Metric | Separate projections | Q/K/V chained |
|---|---:|---:|
| XRT dispatches/block | 23 | 19 |
| 28-block XRT dispatches | 644 | 532 |
| H2D bytes/block | 298,188,800 | 288,751,616 |
| H2D bytes/28-block chain (one denoising step) | 8,349,286,400 | 8,085,045,248 |
| D2H-visible bytes/block | 314,048,512 | 314,048,512 |
| D2H-visible bytes/28-block chain | 8,793,358,336 | 8,793,358,336 |
| Reported BO allocation events/28-block chain | 542 | 630 |

The 9,437,184-byte H2D reduction per block is one shared self-attention input
and one shared cross-attention context input; the Q/K/V weights and all
host-visible outputs are unchanged. The allocation-event counter includes
per-key weight BO population; three output BOs per chain signature are reused
across block keys after initial creation. A captured-fixture validation pair
measured Base at 18.397 s -> 17.313 s and Turbo at 17.418 s -> 17.843 s.
These are noisy chain-wall samples, not a stable speed claim or sampler
timing.

Every block output across the 28-block Base and Turbo fixtures was bitwise
equal between the separate-projection and chained XDNA paths. Turbo's CPU
oracle comparison passed its declared 10% intermediate / 2% final gate:
0.0599236 worst at block 12 and 0.0117299 final. The real Base fixture's
optimized and legacy XDNA outputs also match bitwise, but both show 0.0782801
worst transient normalized RMS at block 14 against the strict 5% Base
CPU-oracle gate; the final error is 0.00840710. This is reported as an
existing Base numerical limitation, not treated as a Base gate pass or hidden
by Turbo's wider intermediate threshold.

The standalone ComfyUI-XDNA 8-step Euler/simple/CFG 1, seed 424242, 512x512
Turbo workflow was repeated with only `qkv_chaining` changed. The paired warm
execution measured 147.56 s disabled and 139.18 s enabled (5.68% lower);
the RGB output pixels were identical (8-bit MAE 0). The first workflow after
server startup was not used for the paired timing because cold/warm cache
effects were visible. This remains CPU-hybrid: Qwen, all listed normalization,
RoPE, softmax, GELU, residual/gate operations, sampler, final projection, and
VAE remain on CPU. No NPU-only claim is made.

Failures use distinct categories and exit codes: dependency unavailable (2),
NPU unavailable (3), unsupported shape/config/dtype (4), compilation failure
(5), execution failure (6), and numerical mismatch (7).

## Packed-weight cache

The default cache is
`%USERPROFILE%\.cache\anima-xdna\weights\<cache-key>`. It stores the exact BF16
K-by-N tensors consumed by the AIE2P Linear kernels after transpose, K-chunking,
zero-padding, and layout preparation, plus the four small Q/K RMSNorm vectors
per block. It does not copy or bundle the source safetensors and is not a
portable or distributable model format. The cache may be deleted and rebuilt
at any time; model privacy and license obligations still apply to the original
checkpoint.

The human-readable `manifest.json` records:

- a full SHA-256 content fingerprint of every used tensor byte range, every
  safetensors header, and a shard index when present;
- model family, complete architecture, block range, and exact weight-key map;
- source/packed dtype and shape, offsets, padding, K-chunking, and layout ABI;
- `base_model_fingerprint`, future `adapter_fingerprints`, `merge_strength`,
  `quantization_scheme`, and `layout_version` namespace fields;
- Triton-XDNA, MLIR-AIE/AIR, LLVM-AIE, XRT SDK, `npu2`, and PoC cache ABI;
- one payload SHA-256 and a SHA-256 for every packed tensor record.

A warm hit verifies Windows file identity/change-time/size, all headers, and
three distributed byte ranges in every used source tensor. The full content
hash established at build remains in the cache key. It concurrently verifies
the complete 3.96 GB payload SHA-256 before mmap. Any guard, header, range,
manifest, size, or digest mismatch is a miss or explicit integrity failure;
corrupt entries are never loaded silently. Builds use an exclusive
per-key lock, write into a unique temporary directory, fsync the payload, and
atomically rename the completed directory. Failed builds clean only their own
temporary directory.

`anima-xdna-chain` enables the cache by default. Use `--no-weight-cache`,
`--cache-dir PATH`, or `--rebuild-cache` to disable, relocate, or explicitly
regenerate it. `anima-xdna-block` has the same switches.
`anima-xdna-linear --weight-cache` reuses the full entry for exact
`transformer_blocks.N.*` Linear keys. Management commands are:

```powershell
anima-xdna-cache inspect CHECKPOINT
anima-xdna-cache verify CHECKPOINT
anima-xdna-cache rebuild CHECKPOINT
anima-xdna-cache list
anima-xdna-cache prune CHECKPOINT
anima-xdna-cache prune --entry-key <64-hex-cache-key>
```

`prune` resolves and verifies one exact manifest entry; it never recursively
deletes a wildcard or the cache root. Payload pages are memory-mapped and
copied into process-local XRT BOs. BO handles/device pointers are never
persisted. CLI timings keep fingerprint/guard validation, packing, mmap,
payload verification, and BO population separate.

For Anima Base v1.0 blocks 0-27 the final entry contains 644 unique records:
532 Linear K-chunks and 112 RMSNorm tensors. It occupies 3,963,645,952 bytes
for 3,875,565,568 logical source bytes; zero padding adds 88,080,384 bytes
(2.27%). Measurements on the validated system were:

| Phase | Time |
|---|---:|
| cache disabled, cold prepare | 10.089 s |
| cache disabled, OS-warm prepare | 5.355-5.538 s |
| clean cold fingerprint | 5.528-5.758 s |
| clean cold pack/write | 13.409-13.671 s |
| clean cold total | 19.834-19.865 s |
| new-process warm source validation | 34-42 ms |
| new-process warm payload verification | 4.316-4.529 s |
| new-process warm total | 4.330-4.546 s |

The full first-run 28-block path, including verification, BO population,
kernel execution, and host stages, fell from 23.822 s without the cache to
21.843 s with a verified warm hit. The first resident upload copied
3,963,617,280 packed Linear bytes into weight BOs and measured 516.8 ms without
the cache versus 631.0 ms from mmap-backed payload pages in the compared runs;
the cache removes source transpose/padding, not BO upload. Once resident, both
paths reuse the same BOs and have identical transfer behavior. Cached and
uncached 28-block XDNA outputs were bitwise equal, and the existing Diffusers/
CPU normalized-RMS gate continued to pass. Times vary with OS file-cache state;
cold build time is intentionally not presented as warm-hit performance.

## Tests and limitations

### Opt-in direct MLIR-AIR attention

`examples.air_attention` is an experimental fixed-shape QK/scale/online-softmax/
AV implementation, not a production runtime switch. Its AIR schedule uses
32-query/64-key tiles, four query cores per head, and two concurrent heads.
Only Q, K, V and the final attention output are host-visible; no full score
or probability matrix is returned. K/V are streamed again for each 128-query
group, so smaller host synchronization buffers do not imply elimination of
internal DMA traffic.

The AIE microkernels are reused from AMD's MIT-licensed MLIR-AIR example at
revision `508ca731ef0384fd3729ac1688e0b5d9a0ef5efe`, not newly authored here.
Keep the upstream copyright/license notices when obtaining or redistributing
those sources. Download them to a **private scratch directory**, outside this
repository, using the isolated, validated AIR/AIE/Peano/XRT environment:

```powershell
$source = "<PRIVATE_UPSTREAM_SOURCE_DIRECTORY>"
New-Item -ItemType Directory -Path $source -Force | Out-Null
$revision = "508ca731ef0384fd3729ac1688e0b5d9a0ef5efe"
$base = "https://raw.githubusercontent.com/Xilinx/mlir-air/$revision/programming_examples/flash_attention/kernel_fusion_based"
foreach ($name in @("attn_npu2.cc", "zero.cc")) {
  Invoke-WebRequest "$base/$name" -OutFile (Join-Path $source $name)
}
python -m examples.air_attention --keys 1024 --parallel-heads 2 `
  --source-dir $source --work-dir "<PRIVATE_SELF_BUILD_DIRECTORY>"
python -m examples.air_attention --keys 512 --parallel-heads 2 `
  --source-dir $source --work-dir "<PRIVATE_CROSS_BUILD_DIRECTORY>"
```

Compilation verifies both source SHA-256 digests, writes direct AIR and ELF
artifacts plus a private manifest, and invokes the AIE compiler without
Triton JIT attention lowering. Four concurrent heads exhausted shim DMA
capacity on this device; the supported CLI schedules are one or two heads,
with two used for the measured prototype. Each core has 53,504 bytes of
explicit working buffers, excluding the 4 KiB stack and compiler padding.
K and V deliberately use separate memtile allocations: aliasing their
inbound staging caused device timeouts in an earlier schedule.

This is **not control-equivalent arithmetic**: BFP16-emulated BF16 matmuls,
BF16 online-softmax state, and scale inside the exponential path differ from
the control's full BF16 score rounding and FP32 host softmax. Numerical gates
must be inspected independently of successful execution.

Run the real-model command above with a new private work directory and:

```powershell
  --modes baseline air_attention `
  --air-self-manifest "<PRIVATE_SELF_BUILD_DIRECTORY>\manifest.json" `
  --air-cross-manifest "<PRIVATE_CROSS_BUILD_DIRECTORY>\manifest.json"
```

The process-local hooks replace only the attention hot path. Masks are
explicitly rejected; the production block code, normal environment, and
default probe modes are unchanged. This captures private Q/K/V/control
outputs at blocks 0, 13 and 27 of steps 0 and 3. Compare the **complete**
round-trip attention path, including host preparation, synchronization,
dispatch/wait and output conversion, rather than timing softmax alone:

```powershell
python -m examples.air_attention_probe `
  --self-manifest "<PRIVATE_SELF_BUILD_DIRECTORY>\manifest.json" `
  --cross-manifest "<PRIVATE_CROSS_BUILD_DIRECTORY>\manifest.json" `
  --input-dir "<PRIVATE_MODEL_WORK_DIRECTORY>\attention_inputs" `
  --report "<PRIVATE_PROBE_REPORT_JSON>" --runs 21 --warmups 3
```

The benchmark alternates control/AIR order, verifies the control reproduces
captured outputs, and records both the CPU BF16-chain reference and FP32
SDPA reference outside timed regions. Its explicit 2% attention-output CPU
NRMS gate is an additional prototype check, not a replacement or relaxation
of the existing 10% intermediate / 2% final block-chain policy.

The completed experiment reduced the warmed four-step sampler median from
27.095 s to 18.765 s and dispatches from 2,128 to 1,904. However, final latent
NRMS versus control was 29.035%, decoded RGB MAE was 15.975/255, and only
3/12 sampled attention cases passed the additional CPU gate. This remains
an opt-in prototype, not production enablement or a 12-second result. See
the [measured results and numerical limitations](../docs/anima-xdna-reproducibility.md#direct-mlir-air-tiled-attention-experiment).

### Native ComfyUI AIR execution

The measured AIR prototype is also wired into the normal chain runtime,
without process-global monkey patches or score-shaped placeholder tensors.
Set `ANIMA_XDNA_AIR_ATTENTION_DIR` to a persistent artifact directory containing
`self\manifest.json` and `cross\manifest.json` (and their ELFs). Each manifest
must include `elf_sha256`; compilation now writes this digest automatically.
ELF paths may be relative to their manifest. Loading verifies the ELF hash
before opening XRT.

```powershell
$env:ANIMA_XDNA_AIR_ATTENTION_DIR = "<PERSISTENT_AIR_ARTIFACT_DIRECTORY>"
$env:ANIMA_XDNA_ATTENTION_BACKEND = "air"
# Start ComfyUI with its usual XDNA launcher.
```

The repository default is always the baseline, even when an artifact directory
is configured. AIR requires an explicit `ANIMA_XDNA_ATTENTION_BACKEND="air"`.
The MODEL attachment and Runtime Status report
`attention_backend: "air_experimental"`. First use logs an
`EXPERIMENTAL AIR attention ACTIVE` warning including the known precision
difference. The normal Load Anima (BF16) -> Load/Attach XDNA -> sampler
workflow does not need replacement nodes. Both standard `run_range` and
CPU-versus-NPU chain validation use the selected NPU backend; the CPU oracle
never uses AIR.

AIR is fixed to batch 1, 1024 image tokens, 512 context tokens and 16 heads
of depth 128. Masks, unsupported shapes, missing/corrupt artifacts and device
failures raise errors; there is no silent baseline/CPU fallback. Runners
remain resident across sampler steps and close on runtime unload.

For a controlled comparison with identical seed, prompt and sampler settings,
stop ComfyUI, set the following and restart:

```powershell
$env:ANIMA_XDNA_ATTENTION_BACKEND = "baseline"
```

This disables AIR even if the artifact directory is still configured. Restart
with `"air"` to re-enable it. Merely changing an environment variable outside
an already-running server does not replace a cached MODEL runtime.

On the user's explicitly requested local deployment, AIR is the standard
route selected by its local configuration in the normal ComfyUI interpreter;
this does **not** change the
unconfigured repository default or make the failed numerical gates pass.

The repository launcher also supports explicit selection:

```powershell
.\examples\run_comfyui_xdna.ps1 `
  -ComfyUIRoot "<COMFYUI_ROOT>" -XDNAVenv "<XDNA_VENV>" `
  -XRTDevDir "<XRT_SDK_DIR>" -ModelPathsConfig "<USER_MODEL_PATHS_YAML>" `
  -AttentionBackend air -AIRAttentionDirectory "<PERSISTENT_AIR_ARTIFACT_DIRECTORY>"
```

Without `-AttentionBackend air`, this launcher explicitly selects baseline,
including when the parent shell has an AIR environment setting. The user's
previously configured local launchers are separate and are not changed by
this release.

### Runtime test coverage

CPU tests cover exact key/config discovery, lazy block loading, shape rejection,
layout/padding, deterministic inputs, RoPE, attention masks, the complete CPU
block residual path, block-index mapping, and CPU range execution. Real-NPU
tests cover Linear, stitched-versus-legacy block attention, the optimized
19-dispatch block, resident reuse, and the complete 28-block numerical chain
with old/new Q/K/V output parity:

```powershell
$env:ANIMA_XDNA_CHECKPOINT = "<DIFFUSERS_TRANSFORMER_SAFETENSORS>"
$env:ANIMA_XDNA_CHAIN_FIXTURE = "<CAPTURED_BLOCK_INPUT_PT>"
python -m unittest discover -s tests -v
```

Without those variables or the toolchain, hardware tests skip with the precise
missing boundary.

The low-level CLIs remain independently useful, while the current
[`comfyui_xdna_nodes`](../comfyui_xdna_nodes/README.md) package now integrates
the resident chain into ComfyUI. Host normalization, softmax, GELU, gates, and
residuals still expose mapped buffers to PyTorch and require cache
synchronization; the reported H2D figure is bytes copied into mapped BO pages
on a unified-memory APU, not necessarily physical DRAM traffic. LoRA and INT8
remain intentionally unsupported.
