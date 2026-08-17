"""The three focused logical agents used by NERV orchestration."""

from __future__ import annotations

from dataclasses import replace

from .backends import AnalysisBackend, RetrievalBackend, VerificationBackend
from .contracts import (
    AgentQuery,
    DraftAnswer,
    RetrievedEvidence,
    RetrievalRequest,
    RetrievalResult,
    VerificationResult,
)


class GroundingError(ValueError):
    """Raised when an agent references evidence it was not supplied."""


class RetrieverAgent:
    """Issue conservative searches and deduplicate returned chunks."""

    def __init__(self, backend: RetrievalBackend, *, top_k: int = 10) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self._backend = backend
        self._top_k = top_k

    def retrieve(
        self,
        query: AgentQuery,
        *,
        cycle: int,
        retrieval_text: str | None = None,
    ) -> RetrievalResult:
        """Search once, preserve first-ranked metadata, and deduplicate IDs."""
        request = RetrievalRequest(
            query_id=query.query_id,
            text=retrieval_text or query.text,
            top_k=self._top_k,
            cycle=cycle,
        )
        unique: list[RetrievedEvidence] = []
        seen: set[str] = set()
        for item in self._backend.search(request.text, request.top_k):
            if not isinstance(item, RetrievedEvidence):
                raise TypeError(
                    "retrieval backend must return RetrievedEvidence values."
                )
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            if item.retrieval_query is None:
                item = replace(item, retrieval_query=request.text)
            unique.append(item)
        return RetrievalResult(request=request, evidence=tuple(unique))


class AnalystAgent:
    """Generate a structured draft and enforce citation membership."""

    def __init__(self, backend: AnalysisBackend) -> None:
        self._backend = backend

    def analyze(
        self,
        query: AgentQuery,
        evidence: tuple[RetrievedEvidence, ...],
        feedback: VerificationResult | None = None,
    ) -> DraftAnswer:
        """Generate a draft using evidence and optional verifier feedback."""
        draft = self._backend.generate_analysis(query, evidence, feedback)
        if not isinstance(draft, DraftAnswer):
            raise TypeError("analysis backend must return a DraftAnswer.")
        available = {item.chunk_id for item in evidence}
        unknown = sorted(set(draft.cited_chunk_ids) - available)
        if unknown:
            raise GroundingError(
                "analyst cited unknown evidence chunk IDs: " + ", ".join(unknown)
            )
        return draft


class VerifierAgent:
    """Classify support for a draft without changing its contents."""

    def __init__(self, backend: VerificationBackend) -> None:
        self._backend = backend

    def verify(
        self,
        query: AgentQuery,
        evidence: tuple[RetrievedEvidence, ...],
        draft: DraftAnswer,
    ) -> VerificationResult:
        """Return the backend's bounded verification decision."""
        available = {item.chunk_id for item in evidence}
        unknown = sorted(set(draft.cited_chunk_ids) - available)
        if unknown:
            raise GroundingError(
                "verifier received unknown evidence chunk IDs: "
                + ", ".join(unknown)
            )
        result = self._backend.verify_answer(query, evidence, draft)
        if not isinstance(result, VerificationResult):
            raise TypeError("verification backend must return VerificationResult.")
        return result
