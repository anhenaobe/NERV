[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ChunkRunId,

    [string]$RunId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-stage1-corrected-downstream"
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repoRoot '.venv-cuda\Scripts\python.exe'
$documents = Join-Path $repoRoot 'outputs\runs\20260816-142614-clean-upstream\artifacts\documentos.jsonl'
$chunkRun = Join-Path $repoRoot ("outputs\runs\{0}" -f $ChunkRunId)
$chunks = Join-Path $chunkRun 'artifacts\chunks.jsonl'
$auditPath = Join-Path $chunkRun 'corrected_chunks_audit.json'
$runDirectory = Join-Path $repoRoot ("outputs\runs\{0}" -f $RunId)
$artifacts = Join-Path $runDirectory 'artifacts'

foreach ($requiredPath in @($python, $documents, $chunks, $auditPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required downstream input is missing: $requiredPath"
    }
}
$audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
if ($audit.status -ne 'CORRECTED_CHUNKS_PASS') {
    throw "Downstream regeneration requires CORRECTED_CHUNKS_PASS."
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
    & $python -m nerv.pipeline `
        --mode custom `
        --run-id $RunId `
        --from-stage embeddings `
        --to-stage validation `
        --device cuda `
        --embedding-batch-size 16 `
        --local-files-only `
        --documents $documents `
        --chunks $chunks `
        --chunk-config (Join-Path $chunkRun 'artifacts\chunking_config.json') `
        --chunk-validation (Join-Path $chunkRun 'post_chunk_validation.json') `
        --chunk-metrics (Join-Path $chunkRun 'artifacts\chunking_metrics.json') `
        --embeddings (Join-Path $artifacts 'embeddings.npy') `
        --embedding-manifest (Join-Path $artifacts 'embeddings.manifest.json') `
        --faiss-index (Join-Path $artifacts 'index.faiss') `
        --faiss-metadata $chunks `
        --queries (Join-Path $repoRoot 'corpus\queries\queries.jsonl') `
        --results (Join-Path $runDirectory 'official_results.jsonl') `
        --metrics (Join-Path $runDirectory 'run_metrics.json') `
        --run-manifest (Join-Path $runDirectory 'run_manifest.json') `
        --report (Join-Path $runDirectory 'downstream_execution_report.txt') `
        --pipeline-log (Join-Path $runDirectory 'pipeline.log') `
        --expected-document-count 1760
    if ($LASTEXITCODE -ne 0) {
        throw "Corrected downstream regeneration failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PYTHONPATH = $originalPythonPath
    Pop-Location
}

Write-Output "CORRECTED_DOWNSTREAM_RUN_ID=$RunId"
Write-Output "CORRECTED_DOWNSTREAM_RUN_DIR=$runDirectory"
