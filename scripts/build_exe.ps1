# Build the Ravana desktop GUI as a Windows one-folder executable.
# Usage (from repo root, with .venv activated):
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Write-Host "Installing PyInstaller..."
& $Python -m pip install -q "pyinstaller>=6.0"

$Spec = Join-Path $Root "packaging\ravana_gui.spec"
$Dist = Join-Path $Root "dist\Ravana"
$Work = Join-Path $Root "build\pyinstaller"

Write-Host "Building one-folder bundle (this can take several minutes)..."
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath (Join-Path $Root "dist") `
    --workpath $Work `
    $Spec

$Exe = Join-Path $Dist "Ravana.exe"
if (-not (Test-Path $Exe)) {
    throw "Build finished but Ravana.exe was not found at $Exe"
}

Write-Host ""
Write-Host "OK: $Exe"
Write-Host "Models are NOT bundled. Place or download weights into:"
Write-Host "  $(Join-Path $Dist 'models')"
Write-Host "Or run once with an existing ./models folder next to the exe."
Write-Host ""
Write-Host "Launch:  & `"$Exe`""
