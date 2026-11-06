$ErrorActionPreference = "Stop"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  throw "Run from an x64 Visual Studio developer shell with MSVC and OpenMP support."
}

$sourceDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) "anima_xdna_poc"
Push-Location -LiteralPath $sourceDirectory
try {
  foreach ($name in @("fused_middle", "fused_io", "fused_qk", "fused_attention")) {
    & cl.exe /nologo /LD /O2 /MD /std:c++17 /EHsc /arch:AVX512 /fp:strict /openmp `
      "$name.cpp" "/Fe:$name.dll"
    if ($LASTEXITCODE -ne 0) {
      throw "Native host kernel build failed: $name"
    }
  }
}
finally {
  Pop-Location
}
