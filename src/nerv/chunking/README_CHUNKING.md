# NERV Chunking Module

## Frozen Rev A contract

The source of truth is `config/encoder_config.json`.

| Field | Frozen value |
| --- | --- |
| Encoder/tokenizer | `intfloat/multilingual-e5-small` |
| Embedding dimension | 384 |
| Normal chunk limit | 256 tokenizer IDs |
| Encoder hard limit | 512 total tokenizer IDs |
| Whole-sentence overlap target | 32 tokenizer IDs |
| Document prefix | `passage: ` |
| Query prefix | `query: ` |
| Special tokens for chunk sizing | Disabled |
| Future vector normalization | L2 |
| Future similarity/index | Inner product / `IndexFlatIP` |
| Oversized unit | Preserve through 512; hierarchically subdivide above 512 |

The document prefix is included in every candidate and final token count, but
is never stored in `texto`. Tokenization always uses `truncation=False`. The
hard limit is checked against the frozen tokenizer's `model_max_length` and the
model's `max_position_embeddings` when the real counter is loaded.

The policy is deliberately two-level:

- at most 256 IDs: normal chunking;
- 257–512 IDs: preserve the complete semantic or structural unit;
- above 512 IDs: deterministic hierarchical subdivision before publication.

Hard subdivision runs after the existing semantic/structured chunker and
before record construction:

```text
semantic/structured units -> normal chunking -> possible oversized logical chunk
-> hard-limit enforcement -> final record construction
```

All logical chunks receive one exact batched recount at this seam. If an
additive estimate differs, the document falls back to the existing exact scalar
grouping before the hard-limit decision. This prevents an unsampled tokenizer
interaction from concealing a >512 chunk.

CSV/tabular text first uses cell delimiters; JSON text first uses reliable
member/list separators. If those are insufficient, the helper progressively
adds newline, semicolon, colon, comma, punctuation, and whitespace boundaries.
The last fallback uses tokenizer character offsets and exact recounting; it
never decodes token IDs back into text. Exact source coverage is retained.

Adjacent hard-split parts copy at most 32 content tokens from the preceding
source range. The overlap is reduced deterministically whenever the prefix plus
overlap would exceed 512. Coverage ranges remain non-overlapping, so overlap
does not hide loss or duplicate source accounting.

Only hard-split records receive trace fields: `hard_split`,
`parent_chunk_id`, `hard_split_strategy`, part index/count, actual internal
overlap, parent token count, and source/emitted offsets. Final `posicion` values
are enumerated once after expansion and `chunk_id` remains
`{doc_id}-chunk-{position:04d}`. Thus IDs and order are stable for identical
input/configuration, while later positions may change when a parent expands.

The validator independently recounts every chunk and reports the three soft /
hard ranges plus the maximum. Any actual count above 512 is invalid even when
the stored count claims otherwise. Production and benchmark metrics expose
aggregate hard-split source, output, format, strategy, and maximum-token data.
Silent truncation is forbidden.

Baseline evidence motivating Rev A (not a claim about future corpora): 447 of
335,722 existing chunks exceeded 512; 430 were CSV, 17 were JSON, and the
maximum observed count was 10,239.

## Boundary

This module validates UTF-8 JSONL, detects Spanish/English/Portuguese, splits
sentences, groups complete sentences, counts with the frozen tokenizer, and
atomically writes chunks. It does not load encoder weights or create
embeddings, metadata rows, a FAISS index, retrieval logic, or agents.

Portuguese currently uses Spanish PySBD rules and emits a warning. A sentence
that alone exceeds 256 IDs remains intact through 512 IDs. Above the encoder
hard limit it is subdivided by the Rev A policy and identified in the run
summary.

## CLI

```powershell
python -m nerv.chunking.pipeline `
  --input tests/chunking/fixtures/codefest_fictional_documents.jsonl `
  --output outputs/chunks.jsonl `
  --local-files-only `
  --config-output outputs/chunking_config.json `
  --metrics-output outputs/pipeline_summary.json
```

The frozen encoder, limit, and overlap are CLI defaults. Overrides remain
available for controlled experiments; an override is not a v1 contract change.
`--local-files-only` prevents network access. Logging defaults to
`outputs/logs/chunking_pipeline.log`; `--log-level` controls both handlers.

## Real Corpus Execution Workflow

Real-corpus production, validation, and benchmarking are separate commands.
This prevents validation or timing repetitions from rebuilding the production
output. The source corpus is `outputs/resultados/documentos.jsonl`.

Run one production pass:

```powershell
python -m nerv.chunking.real_corpus produce `
  --input outputs/resultados/documentos.jsonl `
  --output outputs/resultados/chunks.jsonl `
  --work-dir C:\nerv_work `
  --local-files-only
```

Validate the existing output without regenerating it:

```powershell
python -m nerv.chunking.real_corpus validate `
  --input outputs/resultados/documentos.jsonl `
  --chunks outputs/resultados/chunks.jsonl `
  --local-files-only
```

Benchmark the frozen representative manifest with one warm-up and five measured
repetitions by default:

```powershell
python -m nerv.chunking.real_corpus benchmark `
  --input outputs/resultados/documentos.jsonl `
  --manifest tests/chunking/fixtures/real_corpus_benchmark_manifest.json `
  --work-dir C:\nerv_work `
  --local-files-only
```

`--work-dir` keeps large incomplete files outside synchronized folders. A
completed work file is copied to a temporary file beside the destination,
flushed, and atomically published. It never replaces `chunks.jsonl` while it is
incomplete. Production writes incremental state to
`outputs/resultados/chunking_progress.json` and metrics to
`outputs/resultados/chunking_production_metrics.json`. Validation and benchmark
metrics are written to `chunking_validation_metrics.json` and
`chunking_benchmark_metrics.json` in the same directory. All progress and
metrics JSON writes are atomic.

Chunks and `chunking_config.json` are separate atomically written artifacts,
not one multi-file transaction. If configuration publication fails after a
completed chunks file is published, production metrics report
`output_publication_status: published_before_failure` and preserve both file
identities for inspection.

On `KeyboardInterrupt`, the CLI returns status 130, marks progress as
`interrupted`, records the last completed document and incomplete temporary
path, and preserves the previously published output. Inspect the recorded path
and verify that it is a `.tmp` file in the selected work directory before
removing it manually. The CLI does not delete incomplete files automatically.

The benchmark selects document IDs rather than copying giant documents into
the repository. It writes an isolated benchmark output in the work directory
and checkpoints after every repetition. A matching interrupted benchmark can
resume its measured repetitions. The work directory contains an exclusive
`.benchmark.lock`; if a process crashes, verify that no benchmark is active
before removing a stale lock manually. Compatibility includes the selected
input hash, frozen configuration, local workflow code, tokenizer/dependency
versions, environment, and work directory. A full-corpus benchmark is
available only by explicit request and is expensive:

```powershell
python -m nerv.chunking.real_corpus benchmark `
  --input outputs/resultados/documentos.jsonl `
  --full-corpus `
  --work-dir C:\nerv_work `
  --local-files-only
```

Pending optimization stages are tokenizer length-only counting,
intradocument chunk/record streaming, exact sequential sentence-integrity
validation, and measured multiprocessing. They are not implemented here.

## Bounded Narrative Sentence Segmentation

Production narrative documents no longer pass millions of characters through
one PySBD call. The language is detected once, the cached backend is resolved
once, and the text is segmented in deterministic execution blocks. CSV, TSV,
XLS, and XLSX remain row-based because their record boundaries are structural,
not linguistic.

The block target is 32,768 characters and the hard execution-block maximum is
65,536 characters. Starting at the target, selection prefers a blank-line
paragraph boundary, then a newline, then other whitespace; it searches back
before the target only if none is available before the maximum, and finally
uses a hard character boundary. Blocks are contiguous and retain every source
character.

For every non-final block, the last PySBD item is uncertain regardless of its
punctuation. Earlier items are emitted, while the last item is carried into the
next block and segmented again with additional context. The final block emits
all remaining items. If one sentence grows beyond 65,536 characters, its carry
is allowed to grow and one warning is recorded for that continuous oversized
carry; the sentence is never cut to satisfy an execution limit.

```text
detect language once; resolve backend once
carry = ""
for each bounded block:
    segments = PySBD(carry + block)
    if block is final: emit all segments
    else: emit segments except the last; carry = last segment
```

Preserved invariants are sentence order, normalized text reconstruction
(`" ".join(text.split())`), no empty output, complete oversized sentences, and
the existing JSONL/chunk/token contracts. Pipeline logs add block count,
largest raw block, maximum carry, duration, and carry warning count without
logging document content.

The focused evaluator is:

```powershell
python tests/chunking/evaluate_bounded_splitter.py
```

It reuses or generates 100 KiB and 1 MiB synthetic fixtures and evaluates six
real PDFs, ordered as two small, two medium, one large, and one extreme case,
selected by
`tests/chunking/fixtures/bounded_splitter_pdf_manifest.json`. It refuses to
replace completed metrics, atomically checkpoints each stage/document, resumes
partial runs, and writes evidence to
`tests/chunking/results/bounded_splitter_metrics.json`. In the initial
synthetic measurement, exact sentence lists and normalized reconstruction
matched at both sizes. The 100 KiB oversized-sentence case measured 0.61x
(bounded was slower), while the 1 MiB repeated-document case measured 2.22x.
This demonstrates both the expected large-input benefit and the limitation:
very long unresolved carry is repeatedly resegmented and can reduce throughput.

The focused real run completed five baselines; the 3,551,463-character extreme
PDF reached the 900-second baseline timeout. Bounded completed all six and took
86.37 seconds on that extreme PDF. The four small/medium sentence lists matched
exactly. The 1,381,071-character large PDF was 5.90x faster bounded, but its
sentence-list digest differed from the baseline. Both routes also failed
normalized reconstruction on that OCR input. The extreme bounded output failed
normalized reconstruction and has no exact comparison because its baseline
timed out. These are recorded as unresolved correctness evidence, not accepted
differences. Two medium PDFs had identical bounded/baseline digests despite the
same inherited PySBD reconstruction mismatch; those differences are explicitly
classified in the metrics.

This stage still returns a sentence list to the existing chunker. Full
intradocument chunk/record streaming is separate future work, as are tokenizer
length optimization and measured multiprocessing. No encoder, chunk budget,
overlap, embedding, FAISS, retrieval, or agent behavior changes here.

## Validation and evidence

```powershell
python -m pytest tests/chunking -q
python -m ruff check src tests
python -m mypy
python tests/chunking/evaluate_frozen_pipeline.py
```

The real-tokenizer integration test uses only a locally cached tokenizer and
skips with an explicit reason when it is unavailable. The evaluator performs
one warm-up plus five measured repetitions and publishes:

- `tests/chunking/results/frozen_pipeline_chunks.jsonl`
- `tests/chunking/results/frozen_pipeline_config.json`
- `tests/chunking/results/frozen_pipeline_metrics.json`
- `tests/chunking/results/frozen_pipeline_sample.md`

All evaluation documents are synthetic and do not represent the CODEFEST
corpus. The earlier detector/splitter and fake-tokenizer artifacts are retained
as separate historical baselines.
