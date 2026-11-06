param(
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot,
    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,
    [Parameter(Mandatory = $true)]
    [string]$NativeDllDirectory,
    [Parameter(Mandatory = $true)]
    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$runtimeRoot = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$repositoryRoot = (Resolve-Path -LiteralPath $RepositoryRoot).Path
$nativeDllDirectory = (Resolve-Path -LiteralPath $NativeDllDirectory).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputPath)

if (Test-Path -LiteralPath $outputPath) {
    throw "Refusing to overwrite an existing bundle: $outputPath"
}

$requiredFiles = [System.Collections.Generic.List[object]]::new()
function Add-BundleFile([string]$Source, [string]$Target) {
    $sourcePath = Join-Path $runtimeRoot $Source
    if ($Source.StartsWith("anima_xdna_poc\", [System.StringComparison]::Ordinal)) {
        $sourcePath = Join-Path $nativeDllDirectory $Source.Substring("anima_xdna_poc\".Length)
    }
    elseif ($Source.StartsWith("repository\", [System.StringComparison]::Ordinal)) {
        $sourcePath = Join-Path $repositoryRoot $Source.Substring("repository\".Length)
    }
    elseif ($Source -eq "run-xdna-int8.ps1") {
        $sourcePath = Join-Path $repositoryRoot $Source
    }
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Required bundle file is missing: $sourcePath"
    }
    $requiredFiles.Add([pscustomobject]@{
        Source = $sourcePath
        Target = $Target.Replace("\", "/")
    })
}

$profileNames = @(
    "i8-proj_in-l2k512-l1k64",
    "i8-proj_out-kloop-l2k512-l1k64",
    "i8-sq2048-l1k64"
)
foreach ($name in $profileNames) {
    foreach ($file in @("aie.elf", "manifest.json", "elf_kernel_name.txt")) {
        Add-BundleFile "int8-attn-l1k64-nq8pp\$name\$file" `
            "xdna-runtime\int8-attn-l1k64-nq8pp\$name\$file"
    }
}
foreach ($name in @("i8-qkv-shared-l1k64", "i8-qkv-cross-l1k64")) {
    foreach ($file in @("aie.elf", "manifest.json")) {
        Add-BundleFile "int8-attn-l1k64-nq8pp\$name\$file" `
            "xdna-runtime\int8-attn-l1k64-nq8pp\$name\$file"
    }
}
foreach ($name in @("self", "cross")) {
    foreach ($file in @("aie.elf", "manifest.json")) {
        Add-BundleFile "air-attention-nq8pp\$name\$file" `
            "xdna-runtime\air-attention-nq8pp\$name\$file"
    }
}
Add-BundleFile "gemm-wide32-20261025\library\library.json" `
    "xdna-runtime\gemm-wide32-20261025\library\library.json"
foreach ($name in @(
    "linear-1024,2048,2048",
    "linear-1024,8192,2048",
    "linear-512,2048,1024",
    "qkv-cross",
    "qkv-shared"
)) {
    foreach ($file in @("aie.elf", "manifest.json", "elf_kernel_name.txt")) {
        Add-BundleFile "gemm-wide32-20261025\library\$name\$file" `
            "xdna-runtime\gemm-wide32-20261025\library\$name\$file"
    }
}
foreach ($name in @("fused_attention", "fused_io", "fused_middle", "fused_qk")) {
    Add-BundleFile "anima_xdna_poc\$name.dll" "anima_xdna_poc\$name.dll"
}
Add-BundleFile "run-xdna-int8.ps1" "run-xdna-int8.ps1"
Add-BundleFile "repository\LICENSE" "xdna-runtime\LICENSE-Anima-XDNA.txt"
Add-BundleFile "repository\THIRD_PARTY_NOTICES.txt" "xdna-runtime\THIRD_PARTY_NOTICES.txt"

$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("anima-int8-package-" + [guid]::NewGuid().ToString("N"))
try {
    $payloadRoot = Join-Path $temporaryRoot "payload"
    New-Item -ItemType Directory -Path $payloadRoot -Force | Out-Null
    $manifestFiles = @()
    foreach ($file in $requiredFiles) {
        $destination = Join-Path $payloadRoot ($file.Target.Replace("/", "\"))
        $destinationParent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
        Copy-Item -LiteralPath $file.Source -Destination $destination
        $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        $manifestFiles += [pscustomobject]@{
            path = $file.Target
            sha256 = $hash
            size = (Get-Item -LiteralPath $destination).Length
        }
    }

    $manifest = [ordered]@{
        schema_version = 1
        profile = "int8-attn-l1k64-nq8pp"
        target = "windows-x64-amd-xdna2"
        files = $manifestFiles
    }
    $manifestPath = Join-Path $temporaryRoot "manifest.json"
    $manifest | ConvertTo-Json -Depth 8 |
        Set-Content -LiteralPath $manifestPath -Encoding UTF8

    $outputParent = Split-Path -Parent $outputPath
    if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) {
        New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    }
    Compress-Archive -Path (Join-Path $temporaryRoot "*") `
        -DestinationPath $outputPath -CompressionLevel Optimal
    $archiveHash = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Output "Bundle: $outputPath"
    Write-Output "Files: $($manifestFiles.Count)"
    Write-Output "SHA256: $archiveHash"
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
