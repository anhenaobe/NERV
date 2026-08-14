# NERV Pipeline Contract

> Current implementation and operational flow: 2026-08-13. A Spanish plain
> text walkthrough is maintained in
> `docs/chunking/current_chunking_pipeline_flow.txt`.

## 1. Ingestion and preprocessing

Readers produce traceable UTF-8 documents and validate formats, decoding, and
I/O at the boundary. Preprocessing may normalize text but must retain source
metadata and information needed for traceability.

## 2. Chunking

The implemented chunking stage streams the established JSONL contract, detects
Spanish/English/Portuguese, splits complete sentences, and writes stable chunks.
It sizes candidates with the tokenizer for
`intfloat/multilingual-e5-small`: soft target 256 IDs, total encoder capacity
512 IDs, derived safe stored-input limit 510 IDs, whole-sentence overlap target
32, and `passage: ` included in counting but excluded from stored `texto`.
Units from 257 through 510 IDs are preserved; units above 510 are subdivided.
The hierarchy is format-aware,
then textual boundaries, then deterministic tokenizer offsets. No emitted
record may exceed 510 stored IDs (512 after the two model-required special
tokens) and truncation is forbidden.

Narrative input uses bounded PySBD segmentation: language/backend resolution
happens once, source text is traversed in contiguous blocks with a 32,768
character target and 65,536 character normal maximum, and the uncertain final
segment is carried into the next block. Oversized real sentences are preserved
whole through the encoder-safe range. The hard-limit layer runs after normal
chunking and before final record IDs; it does not alter PySBD. CSV, TSV, XLS,
and XLSX keep structural row boundaries and do not use PySBD.

The supported real-corpus workflow is split into independent commands:

```powershell
python -m nerv.chunking.real_corpus produce --input outputs/resultados/documentos.jsonl --output outputs/resultados/chunks.jsonl --work-dir C:\nerv_work --local-files-only
python -m nerv.chunking.real_corpus validate --input outputs/resultados/documentos.jsonl --chunks outputs/resultados/chunks.jsonl --local-files-only
python -m nerv.chunking.real_corpus benchmark --input outputs/resultados/documentos.jsonl --manifest tests/chunking/fixtures/real_corpus_benchmark_manifest.json --work-dir C:\nerv_work --local-files-only
```

Production streams records to a temporary file, flushes and synchronizes it,
and publishes `chunks.jsonl` atomically. Progress and metrics are checkpointed
atomically. Interruption preserves the previously published output and records
the incomplete temporary path for inspection; benchmark output is isolated and
never replaces production output.

### Windows atomic publication

All chunking JSON/JSONL publication paths use the same helper around
`os.replace`. The completed temporary file stays on the destination filesystem,
is flushed and synchronized before publication, and is reused unchanged if
Windows reports transient access denial (`WinError 5`) or sharing violation
(`WinError 32`). Retries are bounded to 0.05, 0.10, 0.25, 0.50, and 1.00
seconds; other errors and exhausted retries propagate immediately. Each retry
emits a warning with its count and WinError. A terminal publication failure
keeps a fully written JSON checkpoint temporary file for inspection, while
cleanup errors and failed best-effort error-metric writes cannot replace the
primary pipeline exception.

The controlled diagnostic can compare the repository output directory,
`C:\nerv_work`, and a fresh `%TEMP%` directory without running production:

```powershell
python tests/chunking/diagnose_atomic_replace.py --mode helper --iterations 1000
```

This hardening tolerates short-lived Windows file contention; it does not
identify or disable the external process holding a file handle. Antivirus,
indexing, synchronization, and other background processes remain hypotheses
unless independently traced.

## 3. Embeddings

The implemented artifact path uses the shared semantic configuration, prepends
`passage: ` to chunks, produces 384-dimensional `float32` vectors, applies L2
normalization, and binds the matrix to source-chunk SHA-256 plus ordered chunk
IDs. Publication closes the NumPy mapping, creates an exclusive hard link for
the matrix, and publishes the manifest last as the commit marker. It requires a
filesystem with hard-link support and never overwrites an existing destination.
Explicit recovery accepts only a fully validated temporary pair or its exact
matrix-first half-commit, preserving completed temporaries for a safe retry.

## 4. Vector index and retrieval

The implemented baseline is FAISS `IndexFlatIP` over normalized vectors. Its
sidecar binds index bytes, semantic configuration, embedding identity and
ordered chunk metadata. Query loading requires UTF-8 JSONL records with only
`query_id` and `text`; query encoding owns the `query: ` prefix, uses the same
model and L2 normalization, and searches by inner product before aggregation
and ranking.

## 5. Evaluation and delivery

`nerv.pipeline` implements ingestion through validation with isolated run
directories, atomic checkpoints, stage reuse/force policy and `resume-fresh`
lineage validation. The accepted fresh chain contains 1761 documents, 335393
chunks, a 335393 x 384 CUDA embedding matrix and an `IndexFlatIP` with matching
`ntotal`. Infrastructure retrieval and validation passed with the designated 50
synthetic queries.

Published embedding pairs are immutable. `--force-stage embeddings` can only be
used with unused `--embeddings` and `--embedding-manifest` paths; it never
authorizes overwriting an existing matrix or manifest.

The real 50-query PDF is not directly consumable because its embedded text
mapping corrupts Spanish punctuation and accents. It must be transcribed or
OCRed into human-verified UTF-8 JSONL before official retrieval. Synthetic
results must never be represented as official competition results.

Sanitized production evidence is recorded in
`docs/integration/final_production_status.txt`. The detailed query audit remains
local because it contains private source metadata. Chunking internals remain
described in `docs/chunking/current_chunking_pipeline_flow.txt`.

Delivery must preserve official filenames, locations, identifiers, scores, and
audit metadata.

The frozen cross-stage configuration is `config/encoder_config.json`; details
and change conditions are recorded in
`docs/decisions/ADR-encoder-hard-limit-rev-a.md`.
