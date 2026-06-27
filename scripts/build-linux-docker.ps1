$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Image = "tokenscope-linux-builder"

Set-Location $Root

function Invoke-Docker {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$DockerArgs
    )

    docker @DockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($DockerArgs -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Invoke-Docker info
Invoke-Docker build -f "packaging/Dockerfile.linux" -t $Image .

New-Item -ItemType Directory -Force -Path "dist" | Out-Null
$ContainerId = docker create $Image
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ContainerId)) {
    throw "docker create $Image failed with exit code $LASTEXITCODE"
}
try {
    Invoke-Docker cp "$ContainerId`:/out/tokenscope-linux-x86_64" "dist\tokenscope-linux-x86_64"
}
finally {
    docker rm -f $ContainerId | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "docker rm -f $ContainerId failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Built dist\tokenscope-linux-x86_64"
