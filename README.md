# NERV

NERV is a deterministic multilingual vector-retrieval system built for
CODEFEST AD ASTRA 2026 Stage 1. It turns a heterogeneous evidence corpus into
traceable documents and encoder-safe chunks, indexes their normalized vectors,
and returns ranked documents and source fragments for the 50 official queries.

Stage 1 contains no agent, large language model, or generative API in its data,
ranking, or serialization path. Experimental agents under `src/nerv/agents/`
are separate Stage-2 work and are not imported by the Stage-1 generator.

## Retrieval architecture

```text
corpus -> deterministic extraction -> documentos.jsonl
       -> language-aware chunking   -> chunks.jsonl
       -> passage embeddings        -> normalized float32 vectors
       -> FAISS IndexFlatIP         -> numeric retrieval
       -> max-score document pool   -> resultados.jsonl
```

- Supported source formats: PDF, JSON, CSV, XLSX, HTML, TXT, common image
  formats through optional OCR, and PBF through the configured adapter.
- Chunking: ES/EN sentence segmentation, documented Portuguese fallback,
  deterministic PDF-boundary reconciliation, 256-token soft target, 32-token
  complete-unit overlap, and a 510-token encoder-safe stored maximum.
- Structural CSV/JSON records and recognized extracted tables/lists may be
  subdivided with explicit trace metadata. Unclassified prose over the safe
  limit fails closed.
- Encoder: `intfloat/multilingual-e5-small`, 384 dimensions, exact `passage: `
  and `query: ` prefixes, and L2 normalization.
- Vector search: FAISS `IndexFlatIP`; inner product over normalized vectors is
  cosine-equivalent.
- Document aggregation: deterministic numeric max pooling. A document receives
  the highest similarity score among its candidate chunks.

No official NDCG@10 or F1@3 value is claimed because hidden relevance labels
are not available in the repository.

## Repository structure

```text
config/          Canonical encoder and execution configuration
corpus/          Local corpus/query locations; production data is not versioned
docs/            Architecture, decisions, and bounded validation evidence
entrega/         Official CODEFEST delivery layout and deterministic generator
outputs/         Generated run artifacts; production outputs are not versioned
scripts/         Guarded operator, audit, evaluation, and diagnostic tools
src/nerv/        Authoritative Python package
tests/           Unit, focused integration, and delivery-contract tests
```

Core modules are kept in `src/nerv/`; the competition folder is packaging, not
a second application architecture.

## Environment and setup

NERV requires Python 3.12 or newer. From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The public encoder must be available to SentenceTransformers. For a fully
offline run, cache it beforehand and pass `--local-files-only`.

## CODEFEST delivery

The mandatory submission package is:

```text
entrega/
├── resultados.jsonl
├── generador.py
├── informe_tecnico.pdf
└── base_vectorial/
    └── encoder_intfloat_multilingual-e5-small/
        ├── index.faiss
        └── metadata.jsonl
```

Once a corrected chunk lineage has passed `CORRECTED_CHUNKS_PASS` and its
downstream index/metadata have passed identity validation, reproduce the result
file without rebuilding the index:

```powershell
python .\entrega\generador.py --local-files-only
```

The generator reads `corpus/queries/queries.jsonl`, requires the exact ordered
IDs `q001` through `q050`, loads the delivered FAISS index and row-aligned
metadata directly, encodes queries with the canonical prefix, performs numeric
retrieval and max-pooling aggregation, and writes exactly 50 JSONL records.

## Reproducibility and release state

Run manifests bind semantic configuration, artifact hashes, counts, stage
parents, and ordered metadata identity. Production artifacts are generated in
new run directories and are never silently overwritten or inferred from file
timestamps.

The accepted input for corrected Stage-1 chunking contains 1,760 documents
(`sha256 96542c7af4245ba7eaf21b69dba7c1c84034b23931d3a938dd4a83cd8d8c2686`).
The bounded preflight examined 1,709,477 reconstructed units and found zero
unclassified prose candidates above 510 tokens. Final corrected chunk,
embedding, FAISS, and result counts remain release-gated until the independent
corrected-chunk audit and downstream validations pass.

Operator workflow and evidence are in `docs/integration/`. Generated corpus,
embeddings, indexes, results, logs, local environments, and model caches are
excluded from normal Git history; the actual validated runtime artifacts must
be copied into `entrega/` for the competition submission package.

## Quality checks

```powershell
python -m pytest -q
python -m compileall -q src scripts entrega\generador.py
ruff check src tests scripts entrega\generador.py
mypy src generador.py
```

Use the bounded checks documented by each release report before production.
Do not start full-corpus stages merely as a repository validation step.

## Documentation

- [Architecture](docs/architecture/architecture.md)
- [Pipeline contract](docs/architecture/pipeline.md)
- [Stage-1 technical source](docs/integration/informe_tecnico_stage1_source.md)
- [Delivery notes](entrega/README.md)
- [Contribution guide](CONTRIBUTING.md)
