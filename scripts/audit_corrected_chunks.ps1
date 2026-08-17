[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$RunId
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv-cuda\Scripts\python.exe'
$documents = Join-Path $repoRoot 'outputs\runs\20260816-142614-clean-upstream\artifacts\documentos.jsonl'
$runDirectory = Join-Path $repoRoot ("outputs\runs\{0}" -f $RunId)
$chunks = Join-Path $runDirectory 'artifacts\chunks.jsonl'
$validation = Join-Path $runDirectory 'post_chunk_validation.json'
$audit = Join-Path $runDirectory 'corrected_chunks_audit.json'

foreach ($requiredPath in @($python, $documents, $chunks)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required audit input is missing: $requiredPath"
    }
}

$originalPythonPath = $env:PYTHONPATH
$sourcePath = Join-Path $repoRoot 'src'
$env:PYTHONPATH = if ($originalPythonPath) {
    "$sourcePath$([IO.Path]::PathSeparator)$originalPythonPath"
}
else {
    $sourcePath
}

Push-Location $repoRoot
try {
    & $python -m nerv.chunking.real_corpus validate `
        --input $documents `
        --chunks $chunks `
        --metrics-output $validation `
        --validation-batch-size 256 `
        --local-files-only `
        --log-path (Join-Path $runDirectory 'post_chunk_validation.log')
    if ($LASTEXITCODE -ne 0) {
        throw "Canonical chunk validation failed with exit code $LASTEXITCODE."
    }

    & $python scripts\audit_corrected_chunks.py `
        --documents $documents `
        --chunks $chunks `
        --chunk-validation $validation `
        --queries (Join-Path $repoRoot 'corpus\queries\queries.jsonl') `
        --output $audit
    if ($LASTEXITCODE -ne 0) {
        throw "Corrected chunk audit did not pass; inspect $audit."
    }
}
finally {
    $env:PYTHONPATH = $originalPythonPath
    Pop-Location
}
