param(
    [string]$DelphiBin = (Split-Path (Get-Command dcc64.exe -ErrorAction Stop).Source),
    [string]$Glslang,
    [switch]$RebuildShader
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force build | Out-Null
    if ($RebuildShader) {
        if (-not $Glslang) { $Glslang = (Get-Command glslang.exe -ErrorAction Stop).Source }
        & $Glslang -V dense.comp -o build/dense.spv
        if ($LASTEXITCODE) { throw 'Shader compilation failed' }
        & (Join-Path $DelphiBin 'brcc32.exe') '-foQwen35.res' Qwen35.rc
        if ($LASTEXITCODE) { throw 'Resource compilation failed' }
    }
    $lib = Join-Path (Split-Path $DelphiBin) 'lib/win64/release'
    $dcc = Join-Path $DelphiBin 'dcc64.exe'
    foreach ($project in @('Qwen35Smoke','Qwen35KernelTest','Qwen35ForwardTest','Qwen35StateTest')) {
        & $dcc "-U..;$lib" '-I..' '-NSSystem;Winapi' '-N0build' '-Ebuild' "$project.dpr"
        if ($LASTEXITCODE) { throw "Build failed: $project" }
    }
    New-Item -ItemType Directory -Force ../../VCL/Win64/Debug | Out-Null
    & $dcc "-U..;../../VCL;$lib" '-I..' '-NSSystem;Winapi;Vcl' '-N0build' '-E../../VCL/Win64/Debug' ../../VCL/Project1.dpr
    if ($LASTEXITCODE) { throw 'VCL build failed' }
} finally { Pop-Location }
