# Zip dist\Ravana for GitHub Releases (excludes models\ to keep the archive small).
# Run after scripts\build_exe.ps1
#   powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Src = Join-Path $Root "dist\Ravana"
$OutDir = Join-Path $Root "dist"
$Version = "0.3.4"
$Zip = Join-Path $OutDir "Ravana-$Version-windows-x64.zip"

$ExePath = Join-Path $Src "Ravana.exe"
if (-not (Test-Path $ExePath)) {
    throw "Missing Ravana.exe under dist\Ravana - run scripts\build_exe.ps1 first"
}

Copy-Item -Force (Join-Path $Root "packaging\README_DIST.txt") (Join-Path $Src "README.txt")

if (Test-Path $Zip) { Remove-Item -Force $Zip }

# Stage without models (junction/symlink or real folder)
$Stage = Join-Path $OutDir "_ravana_zip_stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory -Path $Stage | Out-Null
robocopy $Src $Stage /E /XD models /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
# robocopy exit codes 0-7 are success
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }

Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip -Force
Remove-Item -Recurse -Force $Stage

$mb = [math]::Round((Get-Item $Zip).Length / 1MB, 1)
Write-Host ("OK: {0} ({1} MB)" -f $Zip, $mb)
Write-Host "Upload this zip as a GitHub Release asset. Users download models on first run."
