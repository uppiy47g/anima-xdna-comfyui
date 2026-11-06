param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,
    [string]$BundlePath
)

$ErrorActionPreference = "Stop"
$releaseTag = "v0.2.0-alpha.2"
$assetName = "anima-xdna-int8-runtime-0.2.0-alpha.2-win-x64.zip"
$nodeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$comfyRoot = (Resolve-Path -LiteralPath $ComfyUIRoot).Path
$nodePackage = Join-Path $nodeRoot "anima_xdna_poc"

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "The INT8 runtime bundle requires 64-bit Windows."
}
foreach ($path in @(
    (Join-Path $comfyRoot "main.py"),
    (Join-Path $comfyRoot "run-xdna.ps1"),
    $nodePackage
)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required installation path is missing: $path"
    }
}

$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) `
    ("anima-int8-install-" + [guid]::NewGuid().ToString("N"))
$archivePath = $BundlePath
$downloadedArchive = $false
$backupRoot = $null
$operations = [System.Collections.Generic.List[object]]::new()

function Get-ExpectedBundlePaths {
    $paths = [System.Collections.Generic.List[string]]::new()
    foreach ($name in @(
        "i8-proj_in-l2k512-l1k64",
        "i8-proj_out-kloop-l2k512-l1k64",
        "i8-sq2048-l1k64"
    )) {
        foreach ($file in @("aie.elf", "manifest.json", "elf_kernel_name.txt")) {
            $paths.Add("xdna-runtime/int8-attn-l1k64-nq8pp/$name/$file")
        }
    }
    foreach ($name in @("i8-qkv-shared-l1k64", "i8-qkv-cross-l1k64")) {
        foreach ($file in @("aie.elf", "manifest.json")) {
            $paths.Add("xdna-runtime/int8-attn-l1k64-nq8pp/$name/$file")
        }
    }
    foreach ($name in @("self", "cross")) {
        foreach ($file in @("aie.elf", "manifest.json")) {
            $paths.Add("xdna-runtime/air-attention-nq8pp/$name/$file")
        }
    }
    $paths.Add("xdna-runtime/gemm-wide32-20261025/library/library.json")
    $paths.Add("xdna-runtime/LICENSE-Anima-XDNA.txt")
    $paths.Add("xdna-runtime/THIRD_PARTY_NOTICES.txt")
    foreach ($name in @(
        "linear-1024,2048,2048",
        "linear-1024,8192,2048",
        "linear-512,2048,1024",
        "qkv-cross",
        "qkv-shared"
    )) {
        foreach ($file in @("aie.elf", "manifest.json", "elf_kernel_name.txt")) {
            $paths.Add("xdna-runtime/gemm-wide32-20261025/library/$name/$file")
        }
    }
    foreach ($name in @("fused_attention", "fused_io", "fused_middle", "fused_qk")) {
        $paths.Add("anima_xdna_poc/$name.dll")
    }
    $paths.Add("run-xdna-int8.ps1")
    return $paths
}

function Get-InstallTarget([string]$RelativePath) {
    if ($RelativePath.StartsWith("anima_xdna_poc/", [System.StringComparison]::Ordinal)) {
        return Join-Path $nodePackage $RelativePath.Substring("anima_xdna_poc/".Length).Replace("/", "\")
    }
    return Join-Path $comfyRoot $RelativePath.Replace("/", "\")
}

function Add-InstallOperation([string]$Source, [string]$Destination) {
    $operations.Add([pscustomobject]@{
        Source = $Source
        Destination = $Destination
        Backup = $null
        HadOriginal = (Test-Path -LiteralPath $Destination -PathType Leaf)
        Changed = $false
    })
}

try {
    New-Item -ItemType Directory -Path $temporaryRoot -Force | Out-Null
    if (-not $archivePath) {
        $release = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/uppiy47g/anima-xdna-comfyui/releases/tags/$releaseTag" `
            -Headers @{ Accept = "application/vnd.github+json"; "X-GitHub-Api-Version" = "2022-11-28" }
        if ($release.tag_name -ne $releaseTag -or $release.draft -or -not $release.prerelease) {
            throw "Release $releaseTag is not a published prerelease."
        }
        $asset = @($release.assets | Where-Object { $_.name -eq $assetName }) | Select-Object -First 1
        if (-not $asset -or -not $asset.digest -or $asset.digest -notmatch "^sha256:[0-9a-fA-F]{64}$") {
            throw "Release $releaseTag does not provide a SHA-256-digested $assetName asset."
        }
        $archivePath = Join-Path $temporaryRoot $assetName
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archivePath
        $downloadedArchive = $true
        $downloadHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($downloadHash -ne $asset.digest.Substring(7).ToLowerInvariant()) {
            throw "Downloaded bundle SHA-256 does not match the GitHub release digest."
        }
    }
    elseif (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
        throw "Bundle file does not exist: $archivePath"
    }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $archivePath).Path)
    try {
        foreach ($entry in $zip.Entries) {
            $name = $entry.FullName.Replace("\", "/")
            if ($name.StartsWith("/") -or $name -match "(^|/)\.\.(/|$)" -or $name -match "^[A-Za-z]:") {
                throw "Unsafe path in bundle archive: $name"
            }
        }
    }
    finally {
        $zip.Dispose()
    }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $temporaryRoot -Force

    $manifestPath = Join-Path $temporaryRoot "manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Bundle is missing manifest.json."
    }
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.schema_version -ne 1 -or
        $manifest.profile -ne "int8-attn-l1k64-nq8pp" -or
        $manifest.target -ne "windows-x64-amd-xdna2") {
        throw "Unsupported INT8 runtime bundle manifest."
    }

    $expectedPaths = @(Get-ExpectedBundlePaths | Sort-Object)
    $manifestPaths = @($manifest.files | ForEach-Object { [string]$_.path } | Sort-Object)
    if (($expectedPaths -join "`n") -ne ($manifestPaths -join "`n")) {
        throw "Bundle manifest does not contain exactly the required INT8 runtime files."
    }

    $payloadRoot = Join-Path $temporaryRoot "payload"
    foreach ($file in $manifest.files) {
        $relativePath = [string]$file.path
        $source = Join-Path $payloadRoot $relativePath.Replace("/", "\")
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Bundle payload is missing: $relativePath"
        }
        if ((Get-Item -LiteralPath $source).Length -ne [long]$file.size) {
            throw "Bundle file size mismatch: $relativePath"
        }
        $hash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne ([string]$file.sha256).ToLowerInvariant()) {
            throw "Bundle file SHA-256 mismatch: $relativePath"
        }
        Add-InstallOperation $source (Get-InstallTarget $relativePath)
    }
    foreach ($operation in $operations) {
        if ($operation.HadOriginal) {
            $existingHash = (Get-FileHash -LiteralPath $operation.Destination -Algorithm SHA256).Hash
            $newHash = (Get-FileHash -LiteralPath $operation.Source -Algorithm SHA256).Hash
            if ($existingHash -eq $newHash) {
                continue
            }
        }
        if ($operation.HadOriginal -and -not $backupRoot) {
            $backupRoot = Join-Path $comfyRoot ("anima-xdna-int8-backup-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
            New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
        }
        if ($operation.HadOriginal) {
            $destinationPrefix = $comfyRoot.TrimEnd("\") + "\"
            if ($operation.Destination.StartsWith($nodeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                $relative = "custom-node\" + $operation.Destination.Substring($nodeRoot.Length).TrimStart("\")
            }
            elseif ($operation.Destination.StartsWith($destinationPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $relative = "comfy-root\" + $operation.Destination.Substring($destinationPrefix.Length)
            }
            else {
                throw "Refusing to back up an installation path outside the target roots: $($operation.Destination)"
            }
            $operation.Backup = Join-Path $backupRoot $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $operation.Backup) -Force | Out-Null
            Copy-Item -LiteralPath $operation.Destination -Destination $operation.Backup
        }
        else {
            $destinationParent = Split-Path -Parent $operation.Destination
            New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
        }
        Copy-Item -LiteralPath $operation.Source -Destination $operation.Destination -Force
        $operation.Changed = $true
    }
    Write-Output "Installed experimental INT8/AIR runtime into: $comfyRoot"
    Write-Output "Launch with: $comfyRoot\run-xdna-int8.ps1"
    if ($backupRoot) {
        Write-Output "Previous files were preserved in: $backupRoot"
    }
}
catch {
    for ($index = $operations.Count - 1; $index -ge 0; $index--) {
        $operation = $operations[$index]
        if (-not $operation.Changed) {
            continue
        }
        if ($operation.HadOriginal -and (Test-Path -LiteralPath $operation.Backup -PathType Leaf)) {
            Copy-Item -LiteralPath $operation.Backup -Destination $operation.Destination -Force
        }
        elseif (Test-Path -LiteralPath $operation.Destination -PathType Leaf) {
            Remove-Item -LiteralPath $operation.Destination -Force
        }
    }
    throw
}
finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
