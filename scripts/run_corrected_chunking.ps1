[CmdletBinding()]
param(
    [string]$RunId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-stage1-corrected-chunks"
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv-cuda\Scripts\python.exe'
$documents = Join-Path $repoRoot 'outputs\runs\20260816-142614-clean-upstream\artifacts\documentos.jsonl'
$runDirectory = Join-Path $repoRoot ("outputs\runs\{0}" -f $RunId)
$artifacts = Join-Path $runDirectory 'artifacts'

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "CUDA environment Python is missing: $python"
}
if (-not (Test-Path -LiteralPath $documents -PathType Leaf)) {
    throw "Accepted documentos.jsonl is missing: $documents"
}
if (Test-Path -LiteralPath $runDirectory) {
    throw "Refusing to reuse an existing run directory: $runDirectory"
}

New-Item -ItemType Directory -Path $artifacts -ErrorAction Stop | Out-Null
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
    $chunks = Join-Path $artifacts 'chunks.jsonl'
    & $python -m nerv.pipeline `
        --mode custom `
        --run-id $RunId `
        --from-stage chunking `
        --to-stage chunking `
        --device cpu `
        --local-files-only `
        --documents $documents `
        --chunks $chunks `
        --chunk-config (Join-Path $artifacts 'chunking_config.json') `
        --chunk-validation (Join-Path $runDirectory 'canonical_chunk_validation.json') `
        --chunk-metrics (Join-Path $artifacts 'chunking_metrics.json') `
        --embeddings (Join-Path $artifacts 'NOT_RUN_embeddings.npy') `
        --embedding-manifest (Join-Path $artifacts 'NOT_RUN_embeddings.manifest.json') `
        --faiss-index (Join-Path $artifacts 'NOT_RUN_index.faiss') `
        --faiss-metadata $chunks `
        --queries (Join-Path $repoRoot 'corpus\queries\queries.jsonl') `
        --results (Join-Path $runDirectory 'NOT_RUN_official_results.jsonl') `
        --metrics (Join-Path $runDirectory 'run_metrics.json') `
        --run-manifest (Join-Path $runDirectory 'run_manifest.json') `
        --report (Join-Path $runDirectory 'chunking_execution_report.txt') `
        --pipeline-log (Join-Path $runDirectory 'pipeline.log') `
        --expected-document-count 1760
    if ($LASTEXITCODE -ne 0) {
        throw "Corrected chunking failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PYTHONPATH = $originalPythonPath
    Pop-Location
}

Write-Output "CORRECTED_CHUNK_RUN_ID=$RunId"
Write-Output "CORRECTED_CHUNK_RUN_DIR=$runDirectory"
