[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('quick', 'full', 'fresh', 'resume-fresh')]
    [string]$Mode,

    [string]$ResumeRunId,

    [string]$ResumeApproval,

    [string]$CorpusPath,

    [ValidateSet('cuda')]
    [string]$Device = 'cuda',

    [ValidateRange(1, 1024)]
    [int]$EmbeddingBatchSize = 16,

    [switch]$AllowValidatedOverride,

    [string]$RecoverEmbeddingTemp,

    [string]$RecoverEmbeddingManifestTemp,

    [switch]$FixtureValidation,

    [string]$PythonExecutable
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot

if ($EmbeddingBatchSize -ne 16 -and -not $AllowValidatedOverride) {
    throw 'A non-default batch requires -AllowValidatedOverride.'
}

if ($Mode -eq 'resume-fresh') {
    if ([string]::IsNullOrWhiteSpace($ResumeRunId)) {
        throw 'resume-fresh requires -ResumeRunId.'
    }
    if ([string]::IsNullOrWhiteSpace($ResumeApproval)) {
        throw 'resume-fresh requires -ResumeApproval.'
    }
    $ResumeApproval = (Resolve-Path -LiteralPath $ResumeApproval).Path
    if (-not [string]::IsNullOrWhiteSpace($CorpusPath)) {
        throw '-CorpusPath is not available in resume-fresh mode.'
    }
}
elseif (-not [string]::IsNullOrWhiteSpace($ResumeRunId)) {
    throw '-ResumeRunId is available only in resume-fresh mode.'
}
elseif (-not [string]::IsNullOrWhiteSpace($ResumeApproval)) {
    throw '-ResumeApproval is available only in resume-fresh mode.'
}

if ($Mode -in @('fresh', 'resume-fresh')) {
    if ($Device -ne 'cuda') {
        throw "$Mode mode requires device=cuda and never falls back to CPU."
    }
    if ($EmbeddingBatchSize -ne 16) {
        throw "$Mode mode requires embedding batch size 16."
    }
    if (-not [string]::IsNullOrWhiteSpace($RecoverEmbeddingTemp) -or
        -not [string]::IsNullOrWhiteSpace($RecoverEmbeddingManifestTemp)) {
        throw "Embedding recovery is not allowed in $Mode mode."
    }
}
elseif (-not [string]::IsNullOrWhiteSpace($CorpusPath)) {
    throw '-CorpusPath is available only in fresh mode.'
}

if ($FixtureValidation -and $Mode -ne 'fresh') {
    throw '-FixtureValidation is available only in fresh mode.'
}

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    if (-not [string]::IsNullOrWhiteSpace($env:NERV_E2E_PYTHON)) {
        $PythonExecutable = $env:NERV_E2E_PYTHON
    }
    else {
        $PythonExecutable = Join-Path $repoRoot '.venv-cuda\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw (
        'CUDA Python is missing. Supply -PythonExecutable or NERV_E2E_PYTHON, ' +
        'then validate it with .\scripts\setup_cuda_env.ps1.'
    )
}

$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
$runId = "$timestamp-$Mode-$suffix"
$runDirectory = Join-Path $repoRoot ("outputs\runs\{0}" -f $runId)
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null

$manifest = Join-Path $runDirectory 'run_manifest.json'
$metrics = Join-Path $runDirectory 'run_metrics.json'
$report = Join-Path $runDirectory 'e2e_execution_report.txt'
$log = Join-Path $runDirectory 'pipeline.log'
$results = Join-Path $runDirectory 'results.jsonl'

$arguments = @(
    '-m', 'nerv.pipeline',
    '--mode', $Mode,
    '--run-id', $runId,
    '--device', $Device,
    '--embedding-batch-size', "$EmbeddingBatchSize",
    '--local-files-only',
    '--results', $results,
    '--metrics', $metrics,
    '--run-manifest', $manifest,
    '--report', $report,
    '--pipeline-log', $log
)

if ($Mode -eq 'quick') {
    if (-not [string]::IsNullOrWhiteSpace($RecoverEmbeddingTemp) -or
        -not [string]::IsNullOrWhiteSpace($RecoverEmbeddingManifestTemp)) {
        throw 'Embedding recovery is available only in full mode.'
    }
    $documents = Join-Path $runDirectory 'documentos.jsonl'
    $chunks = Join-Path $runDirectory 'chunks.jsonl'
    $arguments += @(
        '--corpus', (Join-Path $repoRoot 'tests\fixtures\e2e_tiny\raw'),
        '--documents', $documents,
        '--chunks', $chunks,
        '--chunk-config', (Join-Path $runDirectory 'chunking_config.json'),
        '--embeddings', (Join-Path $runDirectory 'embeddings.npy'),
        '--embedding-manifest', (Join-Path $runDirectory 'embeddings.manifest.json'),
        '--faiss-index', (Join-Path $runDirectory 'index.faiss'),
        '--faiss-metadata', $chunks,
        '--queries', (Join-Path $repoRoot 'tests\fixtures\e2e_tiny\queries.jsonl')
    )
}
elseif ($Mode -eq 'full') {
    $documents = Join-Path $repoRoot 'outputs\resultados\documentos.jsonl'
    $chunks = Join-Path $repoRoot 'outputs\resultados\chunks.phase1b_encoder_capacity_candidate.jsonl'
    $artifactRoot = Join-Path $repoRoot 'outputs\artifacts\production'
    New-Item -ItemType Directory -Path $artifactRoot -Force | Out-Null
    $productionEmbeddings = Join-Path $artifactRoot 'embeddings.npy'
    $productionManifest = Join-Path $artifactRoot 'embeddings.manifest.json'
    $hasRecoveryEmbedding = -not [string]::IsNullOrWhiteSpace($RecoverEmbeddingTemp)
    $hasRecoveryManifest = -not [string]::IsNullOrWhiteSpace($RecoverEmbeddingManifestTemp)
    if ($hasRecoveryEmbedding -ne $hasRecoveryManifest) {
        throw 'Both recovery temporary paths must be supplied together.'
    }
    if ($hasRecoveryEmbedding) {
        & $PythonExecutable -m nerv.embeddings.recover_artifact `
            --temporary-embeddings $RecoverEmbeddingTemp `
            --temporary-manifest $RecoverEmbeddingManifestTemp `
            --embeddings $productionEmbeddings `
            --manifest $productionManifest `
            --chunks $chunks
        if ($LASTEXITCODE -ne 0) {
            throw 'Validated embedding recovery failed; FULL was not started.'
        }
    }
    $officialQueries = Join-Path $repoRoot 'corpus\queries\queries.jsonl'
    if (Test-Path -LiteralPath $officialQueries -PathType Leaf) {
        $queries = $officialQueries
    }
    else {
        $queries = Join-Path $repoRoot 'tests\fixtures\e2e_tiny\queries.jsonl'
        Write-Warning 'Official queries are unavailable; using designated synthetic evaluation queries.'
    }
    $arguments += @(
        '--from-stage', 'embeddings',
        '--reuse-existing',
        '--documents', $documents,
        '--chunks', $chunks,
        '--chunk-config', (Join-Path $repoRoot 'outputs\resultados\chunking_phase1b_config.json'),
        '--chunk-validation', (Join-Path $repoRoot 'outputs\resultados\chunking_phase1b_validation_metrics.json'),
        '--embeddings', $productionEmbeddings,
        '--embedding-manifest', $productionManifest,
        '--faiss-index', (Join-Path $artifactRoot 'index.faiss'),
        '--faiss-metadata', $chunks,
        '--queries', $queries,
        '--expected-document-count', '1761',
        '--expected-document-sha256', 'FE4EB08FC22781B228EAB70E8FA5AF663EF34DF6D6FDAB32D2A5A2457472A982',
        '--expected-chunk-count', '336245',
        '--expected-chunk-sha256', 'D69A943E8069092D7F68AFB31EC2F5872EACE2743B20ABFFCF6BC4A0BDD835F3'
    )
}
elseif ($Mode -eq 'resume-fresh') {
    $parentRun = Join-Path $repoRoot ("outputs\runs\{0}" -f $ResumeRunId)
    $parentArtifacts = Join-Path $parentRun 'artifacts'
    $documents = Join-Path $parentArtifacts 'documentos.jsonl'
    $chunks = Join-Path $parentArtifacts 'chunks.jsonl'
    $artifacts = Join-Path $runDirectory 'artifacts'
    New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
    $arguments += @(
        '--resume-run-id', $ResumeRunId,
        '--resume-run-directory', $parentRun,
        '--resume-approval', $ResumeApproval,
        '--from-stage', 'embeddings',
        '--to-stage', 'validation',
        '--documents', $documents,
        '--chunks', $chunks,
        '--chunk-config', (Join-Path $parentArtifacts 'chunking_config.json'),
        '--chunk-metrics', (Join-Path $parentArtifacts 'chunking_metrics.json'),
        '--embeddings', (Join-Path $artifacts 'embeddings.npy'),
        '--embedding-manifest', (Join-Path $artifacts 'embeddings.manifest.json'),
        '--faiss-index', (Join-Path $artifacts 'index.faiss'),
        '--faiss-metadata', $chunks,
        '--queries', (Join-Path $repoRoot 'tests\fixtures\e2e_tiny\queries.jsonl')
    )
}
else {
    if ([string]::IsNullOrWhiteSpace($CorpusPath)) {
        $freshCorpus = Join-Path $repoRoot 'corpus\raw'
    }
    else {
        $freshCorpus = $CorpusPath
    }
    if ([System.IO.Path]::IsPathRooted($freshCorpus)) {
        $freshCorpus = [System.IO.Path]::GetFullPath($freshCorpus)
    }
    else {
        $freshCorpus = [System.IO.Path]::GetFullPath(
            (Join-Path $repoRoot $freshCorpus)
        )
    }

    if ($FixtureValidation) {
        $fixtureRoot = (Resolve-Path -LiteralPath (Join-Path $repoRoot 'tests\fixtures')).Path
        if (-not $freshCorpus.StartsWith($fixtureRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw '-FixtureValidation requires a corpus under tests\fixtures.'
        }
    }

    $artifacts = Join-Path $runDirectory 'artifacts'
    New-Item -ItemType Directory -Path $artifacts -Force | Out-Null
    $documents = Join-Path $artifacts 'documentos.jsonl'
    $chunks = Join-Path $artifacts 'chunks.jsonl'
    $arguments += @(
        '--from-stage', 'ingestion',
        '--to-stage', 'validation',
        '--corpus', $freshCorpus,
        '--documents', $documents,
        '--chunks', $chunks,
        '--chunk-config', (Join-Path $artifacts 'chunking_config.json'),
        '--chunk-metrics', (Join-Path $artifacts 'chunking_metrics.json'),
        '--embeddings', (Join-Path $artifacts 'embeddings.npy'),
        '--embedding-manifest', (Join-Path $artifacts 'embeddings.manifest.json'),
        '--faiss-index', (Join-Path $artifacts 'index.faiss'),
        '--faiss-metadata', $chunks,
        '--queries', (Join-Path $repoRoot 'tests\fixtures\e2e_tiny\queries.jsonl'),
        '--historical-document-count', '1761',
        '--historical-chunk-count', '336245',
        '--expected-document-sha256', 'FE4EB08FC22781B228EAB70E8FA5AF663EF34DF6D6FDAB32D2A5A2457472A982',
        '--expected-chunk-sha256', 'D69A943E8069092D7F68AFB31EC2F5872EACE2743B20ABFFCF6BC4A0BDD835F3'
    )
    if ($FixtureValidation) {
        $arguments += @('--fixture-validation')
    }
    else {
        $arguments += @(
            '--expected-document-count', '1761',
            '--expected-chunk-count', '336245',
            '--historical-documents', (Join-Path $repoRoot 'outputs\resultados\documentos.jsonl'),
            '--historical-chunks', (Join-Path $repoRoot 'outputs\resultados\chunks.phase1b_encoder_capacity_candidate.jsonl')
        )
    }
}

& $PythonExecutable @arguments
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[NERV] report: $report"
    exit $exitCode
}
