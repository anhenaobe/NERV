"""Typed contracts for the grounded NERV agent workflow."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _require_nonempty_entries(values: tuple[str, ...], field_name: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must not contain empty values.")


class AgentState(str, Enum):
    """States recorded by the bounded agent orchestrator."""

    QUERY = "QUERY"
    RETRIEVE = "RETRIEVE"
    ANALYZE = "ANALYZE"
    VERIFY = "VERIFY"
    FINAL = "FINAL"


class VerificationStatus(str, Enum):
    """Bounded decisions available to the verifier."""

    PASS = "PASS"
    RETRY_RETRIEVAL = "RETRY_RETRIEVAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class AgentQuery:
    """One user query entering the agent workflow."""

    query_id: str
    text: str

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.text, "text")


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """One traceable chunk returned by a retrieval backend."""

    chunk_id: str
    doc_id: str
    source: str
    rank: int
    text: str
    score: float | None = None
    retrieval_query: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")
        _require_text(self.doc_id, "doc_id")
        _require_text(self.source, "source")
        _require_text(self.text, "text")
        if self.rank <= 0:
            raise ValueError("rank must be positive.")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite when provided.")
        if self.retrieval_query is not None:
            _require_text(self.retrieval_query, "retrieval_query")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """One bounded search request issued for an agent query."""

    query_id: str
    text: str
    top_k: int
    cycle: int

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.text, "text")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.cycle <= 0:
            raise ValueError("cycle must be positive.")


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Deduplicated evidence produced for one retrieval request."""

    request: RetrievalRequest
    evidence: tuple[RetrievedEvidence, ...]

    def __post_init__(self) -> None:
        identifiers = [item.chunk_id for item in self.evidence]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("retrieval evidence must have unique chunk IDs.")


@dataclass(frozen=True, slots=True)
class EvidenceCitation:
    """A draft's explicit reference to one retrieved chunk."""

    chunk_id: str

    def __post_init__(self) -> None:
        _require_text(self.chunk_id, "chunk_id")


@dataclass(frozen=True, slots=True)
class DraftAnswer:
    """A structured analyst draft grounded by explicit chunk citations."""

    text: str
    citations: tuple[EvidenceCitation, ...]
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.text, "text")
        if not self.citations:
            raise ValueError("a draft answer must cite at least one evidence chunk.")
        identifiers = [citation.chunk_id for citation in self.citations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("draft citations must have unique chunk IDs.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0.")

    @property
    def cited_chunk_ids(self) -> tuple[str, ...]:
        """Return cited chunk IDs in analyst-provided order."""
        return tuple(citation.chunk_id for citation in self.citations)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """A verifier decision that never rewrites the analyst draft."""

    status: VerificationStatus
    unsupported_claims: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    suggested_retrieval_focus: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_entries(self.unsupported_claims, "unsupported_claims")
        _require_nonempty_entries(self.missing_evidence, "missing_evidence")
        if self.suggested_retrieval_focus is not None:
            _require_text(
                self.suggested_retrieval_focus,
                "suggested_retrieval_focus",
            )
        if self.status is VerificationStatus.PASS and (
            self.unsupported_claims
            or self.missing_evidence
            or self.suggested_retrieval_focus is not None
        ):
            raise ValueError("PASS cannot include unsupported or missing evidence.")


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Final grounded answer or explicit insufficient-evidence response."""

    query_id: str
    answer: str
    cited_evidence: tuple[RetrievedEvidence, ...]
    verification_status: VerificationStatus
    cycles_used: int
    insufficient_evidence: bool

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        _require_text(self.answer, "answer")
        if self.cycles_used <= 0:
            raise ValueError("cycles_used must be positive.")
        if self.verification_status is VerificationStatus.RETRY_RETRIEVAL:
            raise ValueError("a final answer cannot retain a retry status.")
        if self.verification_status is VerificationStatus.PASS:
            if self.insufficient_evidence or not self.cited_evidence:
                raise ValueError("a passing answer must include cited evidence.")
        elif not self.insufficient_evidence or self.cited_evidence:
            raise ValueError(
                "an insufficient-evidence answer cannot cite accepted evidence."
            )


@dataclass(frozen=True, slots=True)
class AgentRunTrace:
    """Deterministic trace from query through retrieval and verification."""

    query_id: str
    state_transitions: tuple[AgentState, ...]
    retrieval_requests: tuple[RetrievalRequest, ...]
    evidence_ids: tuple[tuple[str, ...], ...]
    verifier_outcomes: tuple[VerificationResult, ...]

    def __post_init__(self) -> None:
        _require_text(self.query_id, "query_id")
        if not self.state_transitions:
            raise ValueError("state_transitions must not be empty.")
        if self.state_transitions[0] is not AgentState.QUERY:
            raise ValueError("a run trace must start in QUERY.")
        if self.state_transitions[-1] is not AgentState.FINAL:
            raise ValueError("a run trace must end in FINAL.")
        if len(self.retrieval_requests) != len(self.evidence_ids):
            raise ValueError(
                "each retrieval request must have one evidence ID snapshot."
            )


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    """Final answer paired with its complete deterministic run trace."""

    answer: AgentAnswer
    trace: AgentRunTrace

    def __post_init__(self) -> None:
        if self.answer.query_id != self.trace.query_id:
            raise ValueError("answer and trace query IDs must match.")
