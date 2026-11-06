$ErrorActionPreference = "Stop"

if ($args -contains "-OriginalGemm") {
  throw "The optimized INT8 profile requires its tuned GEMM layout."
}

$baseLauncher = Join-Path $PSScriptRoot "run-xdna.ps1"
$runtimeRoot = Join-Path $PSScriptRoot "xdna-runtime"
$profileRoot = Join-Path $runtimeRoot "int8-attn-l1k64-nq8pp"
$requiredDirectories = @(
  (Join-Path $profileRoot "i8-proj_in-l2k512-l1k64"),
  (Join-Path $profileRoot "i8-proj_out-kloop-l2k512-l1k64"),
  (Join-Path $profileRoot "i8-sq2048-l1k64"),
  (Join-Path $profileRoot "i8-qkv-shared-l1k64"),
  (Join-Path $profileRoot "i8-qkv-cross-l1k64"),
  (Join-Path $runtimeRoot "air-attention-nq8pp"),
  (Join-Path $runtimeRoot "gemm-wide32-20261025\library")
)

if (-not (Test-Path -LiteralPath $baseLauncher -PathType Leaf)) {
  throw "The existing ComfyUI launcher is missing: $baseLauncher"
}

foreach ($path in $requiredDirectories) {
  if (-not (Test-Path -LiteralPath $path -PathType Container)) {
    throw "Required optimized XDNA profile path is missing: $path"
  }
}

$environmentNames = @(
  "ANIMA_XDNA_OPTIMIZED_PROFILE",
  "ANIMA_XDNA_BF16_VAE_CLFIX",
  "ANIMA_XDNA_INT8_ARTIFACT_DIR",
  "ANIMA_XDNA_AIR_ATTENTION_DIR",
  "ANIMA_XDNA_ATTENTION_BACKEND",
  "ANIMA_XDNA_ADALN_DEVICE",
  "ANIMA_XDNA_GEMM_LAYOUT_DIR",
  "ANIMA_XDNA_NUM_THREADS"
)
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
  $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$previousLocation = Get-Location
$exitCode = 0

try {
  $env:ANIMA_XDNA_OPTIMIZED_PROFILE = "int8-attn-l1k64-nq8pp"
  $env:ANIMA_XDNA_BF16_VAE_CLFIX = "1"
  $env:ANIMA_XDNA_INT8_ARTIFACT_DIR = $profileRoot
  $env:ANIMA_XDNA_AIR_ATTENTION_DIR = Join-Path $runtimeRoot "air-attention-nq8pp"
  $env:ANIMA_XDNA_ATTENTION_BACKEND = "air"
  $env:ANIMA_XDNA_ADALN_DEVICE = "host"
  $env:ANIMA_XDNA_GEMM_LAYOUT_DIR = Join-Path $runtimeRoot "gemm-wide32-20261025\library"
  $env:ANIMA_XDNA_NUM_THREADS = "10"

  & $baseLauncher @args
  if ($null -ne $LASTEXITCODE) {
    $exitCode = $LASTEXITCODE
  }
}
finally {
  Set-Location -LiteralPath $previousLocation
  foreach ($name in $environmentNames) {
    $value = $previousEnvironment[$name]
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
  }
}

if ($exitCode -ne 0) {
  throw "ComfyUI exited with status $exitCode."
}
