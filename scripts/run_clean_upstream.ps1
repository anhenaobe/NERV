[CmdletBinding()]
param(
    [string]$RunId = "$(Get-Date -Format 'yyyyMMdd-HHmmss')-clean-upstream"
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot 'outputs\runs'
$runDirectory = Join-Path $runRoot $RunId
$artifactDirectory = Join-Path $runDirectory 'artifacts'
$rawCorpus = Join-Path $repoRoot 'corpus\raw'

if (-not (Test-Path -LiteralPath $rawCorpus -PathType Container)) {
    throw "Raw corpus directory is missing: $rawCorpus"
}
if (Test-Path -LiteralPath $runDirectory) {
    throw "Refusing to reuse an existing run directory: $runDirectory"
}

$python = (Get-Command python -ErrorAction Stop).Source
$originalPythonPath = $env:PYTHONPATH
$currentSource = Join-Path $repoRoot 'src'
$env:PYTHONPATH = if ($originalPythonPath) {
    "$currentSource$([IO.Path]::PathSeparator)$originalPythonPath"
} else {
    $currentSource
}
New-Item -ItemType Directory -Path $artifactDirectory -ErrorAction Stop | Out-Null

Push-Location $repoRoot
try {
    & $python -m nerv.ingestion.lector_corpus `
        --corpus $rawCorpus `
        --salida $artifactDirectory `
        --trabajadores 1 `
        --procesos-pesados 1 `
        --sin-cache
    if ($LASTEXITCODE -ne 0) {
        throw "Clean ingestion failed with exit code $LASTEXITCODE."
    }

    $documents = Join-Path $artifactDirectory 'documentos.jsonl'
    $chunks = Join-Path $artifactDirectory 'chunks.jsonl'
    if (-not (Test-Path -LiteralPath $documents -PathType Leaf)) {
        throw "Clean ingestion did not publish documentos.jsonl."
    }

    & $python -m nerv.pipeline `
        --mode custom `
        --run-id $RunId `
        --from-stage chunking `
        --to-stage chunking `
        --device cpu `
        --local-files-only `
        --corpus $rawCorpus `
        --documents $documents `
        --chunks $chunks `
        --chunk-config (Join-Path $artifactDirectory 'chunking_config.json') `
        --chunk-validation (Join-Path $runDirectory 'canonical_chunk_validation.json') `
        --chunk-metrics (Join-Path $artifactDirectory 'chunking_metrics.json') `
        --embeddings (Join-Path $artifactDirectory 'embeddings.npy') `
        --embedding-manifest (Join-Path $artifactDirectory 'embeddings.manifest.json') `
        --faiss-index (Join-Path $artifactDirectory 'index.faiss') `
        --faiss-metadata $chunks `
        --queries (Join-Path $repoRoot 'corpus\queries\queries.jsonl') `
        --results (Join-Path $runDirectory 'official_results.jsonl') `
        --metrics (Join-Path $runDirectory 'run_metrics.json') `
        --run-manifest (Join-Path $runDirectory 'run_manifest.json') `
        --report (Join-Path $runDirectory 'upstream_execution_report.txt') `
        --pipeline-log (Join-Path $runDirectory 'pipeline.log')
    if ($LASTEXITCODE -ne 0) {
        throw "Clean chunking failed with exit code $LASTEXITCODE."
    }
}
finally {
    $env:PYTHONPATH = $originalPythonPath
    Pop-Location
}

Write-Output "CLEAN_UPSTREAM_RUN_ID=$RunId"
Write-Output "CLEAN_UPSTREAM_RUN_DIR=$runDirectory"
