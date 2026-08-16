"""Provider-neutral backend protocols for the NERV agent layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .contracts import (
    AgentQuery,
    DraftAnswer,
    RetrievedEvidence,
    VerificationResult,
)


class RetrievalBackend(Protocol):
    """Search an evidence store without generating a final answer."""

    def search(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        """Return ranked evidence for one retrieval query."""
        ...


class AnalysisBackend(Protocol):
    """Generate a structured draft using only supplied evidence."""

    def generate_analysis(
        self,
        query: AgentQuery,
        evidence: Sequence[RetrievedEvidence],
        feedback: VerificationResult | None,
    ) -> DraftAnswer:
        """Return a draft with explicit evidence citations."""
        ...


class VerificationBackend(Protocol):
    """Evaluate a draft against supplied evidence without rewriting it."""

    def verify_answer(
        self,
        query: AgentQuery,
        evidence: Sequence[RetrievedEvidence],
        draft: DraftAnswer,
    ) -> VerificationResult:
        """Return one bounded verification decision."""
        ...
