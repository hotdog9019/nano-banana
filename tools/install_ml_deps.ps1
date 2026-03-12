$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pyfixDir = (Join-Path $repoRoot "tools\\pyfix")
$tmpDir = (Join-Path $repoRoot ".tmp-pip")
$cacheDir = (Join-Path $repoRoot ".pip-cache")
$venvDir = (Join-Path $repoRoot ".venv")
$venvPython = (Join-Path $venvDir "Scripts\\python.exe")

New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

$env:NANO_BANANA_TEMPFILE_FIX = "1"
$env:PYTHONPATH = $pyfixDir
$env:TEMP = $tmpDir
$env:TMP = $tmpDir

if (-not (Test-Path $venvPython)) {
  python -m venv --without-pip $venvDir
  & $venvPython -m ensurepip --upgrade --default-pip
}

& $venvPython -m pip install --cache-dir $cacheDir torch torchvision matplotlib numpy
