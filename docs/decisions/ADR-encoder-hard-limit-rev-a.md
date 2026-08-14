# ADR: Encoder hard-limit compatibility, Rev A

- Status: Accepted
- Date: 2026-08-09
- Supersedes: only the v1 unlimited oversized-unit policy

## Context

The frozen `intfloat/multilingual-e5-small` snapshot accepts at most 512 total
input IDs. Chunk counts already include `passage: ` and use
`add_special_tokens=false`. A real-corpus baseline found 447 of 335,722 chunks
above that limit (430 CSV and 17 JSON), with a maximum of 10,239. Those figures
motivate this correction and are not forecasts for future corpora.

## Decision

Version `encoder_max_input_tokens=512` in the canonical encoder configuration.
Keep 256 as the soft target and 32 as the overlap target. Preserve complete
units from 257 through 512 IDs; subdivide only logical chunks above 512.

Enforcement occurs after normal semantic/structured chunking and before final
record construction. It first prefers tabular or JSON separators, then newline,
semicolon, colon, comma, punctuation, and whitespace. Tokenizer offsets with
exact recounting are the final fallback. Text is sliced from the original
string, never reconstructed by decoding token IDs.

Internal hard-split overlap is at most 32 content tokens and shrinks whenever
needed to keep the complete encoder input at or below 512. Non-overlapping
coverage offsets prove complete source coverage. Final records are recounted;
the pipeline refuses publication of an oversized part and the independent
validator marks one invalid. Silent truncation is forbidden.

Trace metadata is conditional on hard splitting. Final positions are sequential
and IDs remain derived from final position. Parent logical IDs and ordered part
metadata make the expansion recoverable.

## Consequences

Encoder-safe output remains unchanged. Regeneration will increase chunk count
only where a logical chunk previously exceeded 512 and will shift subsequent
positions deterministically. Every logical chunk receives one exact batched
recount before the hard-limit decision; an estimate mismatch falls back to the
existing scalar grouping. Exceptional units also pay boundary-search and
subdivision recount costs. This favors the hard safety guarantee over the prior
estimated-count throughput.

The real tokenizer and model configuration are checked so the configured hard
limit cannot exceed either reported capacity. Embeddings, vector dimensions,
normalization, FAISS, retrieval, language detection, and PySBD are unchanged.

## Validation

Focused tests cover both boundaries, structured and textual hierarchy,
token-offset fallback, multilingual text, overlap, coverage, deterministic IDs,
trace metadata, independent hard failure, and an input above 5,000 tokens.
Controlled real-tokenizer synthetic cases precede any full production run.
