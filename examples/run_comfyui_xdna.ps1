param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,
    [Parameter(Mandatory = $true)]
    [string]$XDNAVenv,
    [Parameter(Mandatory = $true)]
    [string]$XRTDevDir,
    [Parameter(Mandatory = $true)]
    [string]$ModelPathsConfig,
    [string]$InputDirectory = "",
    [string]$OutputDirectory = "",
    [int]$Port = 8190,
    [ValidateSet("baseline", "air")]
    [string]$AttentionBackend = "baseline",
    [string]$AIRAttentionDirectory = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ComfyUIArgs
)

$ErrorActionPreference = "Stop"
$python = Join-Path $XDNAVenv "Scripts\python.exe"
$main = Join-Path $ComfyUIRoot "main.py"
foreach ($path in @($python, $main, $XRTDevDir, $ModelPathsConfig)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

$env:XRT_DEV_DIR = (Resolve-Path -LiteralPath $XRTDevDir).Path
$env:PYTHONUTF8 = "1"
$env:ANIMA_XDNA_ATTENTION_BACKEND = $AttentionBackend
if ($AttentionBackend -eq "air") {
    if (-not $AIRAttentionDirectory) {
        throw "-AttentionBackend air requires -AIRAttentionDirectory."
    }
    foreach ($name in @("self", "cross")) {
        $manifest = Join-Path (Join-Path $AIRAttentionDirectory $name) "manifest.json"
        if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
            throw "Required AIR manifest does not exist: $manifest"
        }
    }
    $env:ANIMA_XDNA_AIR_ATTENTION_DIR = (Resolve-Path -LiteralPath $AIRAttentionDirectory).Path
    Write-Warning "Experimental AIR attention is not quality-equivalent: fixed-seed latent NRMS 29.035%, RGB MAE 15.975/255. No silent fallback."
}
$arguments = @(
    $main,
    "--cpu",
    "--listen", "127.0.0.1",
    "--port", $Port,
    "--extra-model-paths-config", (Resolve-Path -LiteralPath $ModelPathsConfig).Path
)
if ($InputDirectory) {
    $arguments += @("--input-directory", $InputDirectory)
}
if ($OutputDirectory) {
    $arguments += @("--output-directory", $OutputDirectory)
}
$arguments += $ComfyUIArgs

& $python @arguments
exit $LASTEXITCODE
