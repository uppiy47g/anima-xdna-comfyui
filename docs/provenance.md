# Provenance and migration scope

This project was extracted on 2026-10-08 from the XDNA implementation in
[uppiy47g/anima-lora-style-bridge](https://github.com/uppiy47g/anima-lora-style-bridge),
branch `uppiy47g-anima-xdna-fusion`, source commit
`b31393286e9ad32ded72b619644fbb628b08a731`. The migration inventory compared
that revision with its active-work base
`4aad1f79c3ff71f373ad42c09a72281272555a6a`, then examined all tracked files
because foundational XDNA modules predate that base.

The original XDNA implementation and validation were authored by uppiy47g
with Copilot App contributions. The original MIT license and copyright
notice, "Anima LoRA Style Bridge contributors", are preserved verbatim;
that notice is attribution, not a runtime dependency or project identity.
AMD Triton-XDNA, Xilinx XRT/MLIR-AIE/MLIR-AIR/LLVM-AIE, PyTorch and ComfyUI
remain separately obtained dependencies under their own licenses.
No third-party implementation or model weights were added by this migration.

## Included

- `anima_xdna_poc`: checkpoint discovery/canonicalization, exact CPU oracle,
  Linear/block/chain kernels and runtime, packed-weight cache and CLIs.
- `comfyui_xdna_nodes`: native BF16 loader, MODEL attachment, status and unload;
  root `__init__.py` now registers only these XDNA nodes.
- XDNA CPU/hardware-gated tests and public-release checks.
- Runtime/node guides, reproducibility report, evidence JSON and its schema,
  four previously published metadata-free RGB PNG comparison outputs.
- Sanitized ComfyUI API fragment, model-path YAML and Windows launcher.
- Standalone metadata, README, CI and unchanged MIT license.

The four PNGs are generated comparison outputs, not checkpoint/binary runtime
artifacts. Their original bytes and recorded digests are retained. The
release tests require zero embedded text/EXIF/time metadata for every
distributed image. Local comparison captures mentioned in the historical
report are not included.

## Excluded

`anima_style_bridge.py`, `comfyui_nodes` activation-capture/converter nodes,
`tests/test_style_bridge.py`, the bridge README/changelog, bridge CLI and
combined package metadata. No XDNA module imports these excluded packages.
No old `.git` history, local/private model paths, credentials, checkpoints,
tensor fixtures, weight payloads, compiled kernels, SDK binaries, caches,
virtual environments, logs, local workflows or generated build output
are published.

## Historical evidence boundary

The source report and JSON contain original validation commit identifiers,
including wrapper `737338d76bf2a7dcf9806aaba5d6d936499b95ab` and attachment
cache fix `62dbd6063b69c0e20d327e991bfca8337fa24334`. These identify revisions
in the **source repository**, not ancestors in this clean repository.
Measurements and dates are preserved as historical observations; migration
CPU/package/privacy checks do not constitute a fresh hardware validation.

Known Base transient-gate failure, Turbo latent-vs-CPU difference, synthetic
fixture restrictions, CPU-hybrid boundary, and rejected compiler experiments
remain disclosed in the report. The fresh repository begins with one initial
commit and intentionally does not import the source commit graph.
