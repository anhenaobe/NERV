[CmdletBinding()]
param(
    [string]$PythonExecutable
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$defaultPython = Join-Path $repoRoot '.venv-cuda\Scripts\python.exe'

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $PythonExecutable = $defaultPython
}

try {
    if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
        throw (
            'A prepared CUDA Python environment is required. Install the reviewed ' +
            'locked dependencies outside this script, then pass -PythonExecutable.'
        )
    }
    & $PythonExecutable (Join-Path $repoRoot 'scripts\verify_cuda_env.py') `
        --encoder-smoke `
        --local-files-only
    if ($LASTEXITCODE -ne 0) {
        throw 'CUDA verification smoke failed.'
    }
}
catch {
    Write-Error "[NERV CUDA SETUP] FAIL: $($_.Exception.Message)"
    exit 1
}
