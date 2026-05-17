param(
    [string]$Python = "python",
    [string]$VenvPath = "venv",
    [switch]$Full
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LocalTemp = Join-Path $ProjectRoot ".tmp"
New-Item -ItemType Directory -Force -Path $LocalTemp | Out-Null
$env:TMP = $LocalTemp
$env:TEMP = $LocalTemp
$env:PIP_ONLY_BINARY = ":all:"
$env:PIP_PREFER_BINARY = "1"

Write-Host "Creating virtual environment at $VenvPath"
& $Python -m venv $VenvPath

$VenvPython = Join-Path $ProjectRoot "$VenvPath\Scripts\python.exe"

Write-Host "Ensuring pip is available"
& $VenvPython -m ensurepip --upgrade --default-pip
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing development/test dependencies"
& $VenvPython -m pip install -r requirements-dev.txt

if ($Full) {
    Write-Host "Installing full scientific stack from requirements.txt"
    & $VenvPython -m pip install -r requirements.txt
}

Write-Host "Running tests"
& $VenvPython -m pytest tests --basetemp=.tmp/pytest -o cache_dir=.tmp/pytest_cache

Write-Host "Verifying core imports"
& $VenvPython -c "import yaml, numpy; print('core imports ok')"

if ($Full) {
    Write-Host "Verifying full-stack imports"
    & $VenvPython -c "import yaml, torch, rasterio, xarray; print('full imports ok')"
}
