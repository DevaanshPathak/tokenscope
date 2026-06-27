$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python -m pip install -r requirements.txt
pyinstaller --clean --noconfirm tokenscope.spec

if (Test-Path "dist\tokenscope.exe") {
    Move-Item -Force "dist\tokenscope.exe" "dist\tokenscope-windows-x86_64.exe"
}

Write-Host "Built dist\tokenscope-windows-x86_64.exe"
