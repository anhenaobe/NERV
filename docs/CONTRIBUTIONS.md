# Contribution and authorship boundaries

NERV is a collaborative Team NERV project. The current public tree was
consolidated for delivery, so project-level functionality must not be treated as
proof that the repository owner individually authored every subsystem. The
preserved pre-publication history is the evidence source for the boundaries
below.

## Team-level project functionality

Team NERV's repository implements heterogeneous document ingestion,
normalization and traceability; multilingual chunking and token limits;
passage/query embeddings; FAISS indexing and retrieval; document aggregation;
evaluation and contamination controls; and deterministic Stage-1 delivery and
validation infrastructure.

These are project claims. They are not sole-author claims for any contributor.

## Repository-owner contributions supported by history

Commits authored by `anhenaobe` support the following bounded contribution
areas:

- initial repository/project scaffolding and early chunking modules;
- expansion of chunking tests, fixtures, language handling, sentence splitting,
  token counting, and pipeline evaluation material;
- consolidation of the split historical repository into the current package and
  documentation layout;
- Stage-1 integration, linguistic/compliance remediation, release gating, and
  delivery/report publication work;
- the separate experimental Stage-2 agent orchestration scaffolding and tests.

These statements do not imply ownership of teammate-authored ingestion,
retrieval, FAISS, evaluation, or other downstream implementations.

## Other documented or unclear ownership

- Commits by Felipe Camacho document the substantive ingestion implementation,
  persistent ingestion cache, PBF handling, and source-provenance metadata.
- Commits by `CHECHO13` document FAISS/retrieval/evaluation work and
  contamination-filter changes.
- The consolidated release commit is attributed to `Team NERV`; authorship of
  changes represented only by that snapshot cannot safely be assigned to one
  individual.
- Embeddings, cross-stage integration, and later fixes include team and
  multi-contributor history. Claim individual ownership only when a specific
  commit or delivery record supports it.

For interviews and CV material, use “Team NERV developed...” for the complete
pipeline and describe personal work using only the bounded areas above.
