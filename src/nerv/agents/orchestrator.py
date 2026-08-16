"""Bounded state machine for grounded three-agent execution."""

from __future__ import annotations

from .agents import AnalystAgent, RetrieverAgent, VerifierAgent
from .contracts import (
    AgentAnswer,
    AgentQuery,
    AgentRunResult,
    AgentRunTrace,
    AgentState,
    RetrievedEvidence,
    RetrievalRequest,
    VerificationResult,
    VerificationStatus,
)

_INSUFFICIENT_ANSWER = "Insufficient evidence is available to answer this query."


class AgentOrchestrator:
    """Run Retriever, Analyst, and Verifier with a strict cycle limit."""

    def __init__(
        self,
        retriever: RetrieverAgent,
        analyst: AnalystAgent,
        verifier: VerifierAgent,
        *,
        max_cycles: int = 2,
    ) -> None:
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive.")
        self._retriever = retriever
        self._analyst = analyst
        self._verifier = verifier
        self._max_cycles = max_cycles

    def run(self, query: AgentQuery) -> AgentRunResult:
        """Execute the bounded state machine and return answer plus trace."""
        states = [AgentState.QUERY]
        requests: list[RetrievalRequest] = []
        evidence_snapshots: list[tuple[str, ...]] = []
        verifier_outcomes: list[VerificationResult] = []
        accumulated: list[RetrievedEvidence] = []
        accumulated_ids: set[str] = set()
        feedback: VerificationResult | None = None
        retrieval_text: str | None = None

        for cycle in range(1, self._max_cycles + 1):
            states.append(AgentState.RETRIEVE)
            retrieval = self._retriever.retrieve(
                query,
                cycle=cycle,
                retrieval_text=retrieval_text,
            )
            requests.append(retrieval.request)
            evidence_snapshots.append(
                tuple(item.chunk_id for item in retrieval.evidence)
            )
            for item in retrieval.evidence:
                if item.chunk_id not in accumulated_ids:
                    accumulated_ids.add(item.chunk_id)
                    accumulated.append(item)

            if not accumulated:
                return self._insufficient_result(
                    query,
                    cycle,
                    states,
                    requests,
                    evidence_snapshots,
                    verifier_outcomes,
                )

            evidence = tuple(accumulated)
            states.append(AgentState.ANALYZE)
            draft = self._analyst.analyze(query, evidence, feedback)
            states.append(AgentState.VERIFY)
            verification = self._verifier.verify(query, evidence, draft)
            verifier_outcomes.append(verification)

            if verification.status is VerificationStatus.PASS:
                cited_by_id = {item.chunk_id: item for item in evidence}
                cited_evidence = tuple(
                    cited_by_id[chunk_id] for chunk_id in draft.cited_chunk_ids
                )
                states.append(AgentState.FINAL)
                answer = AgentAnswer(
                    query_id=query.query_id,
                    answer=draft.text,
                    cited_evidence=cited_evidence,
                    verification_status=VerificationStatus.PASS,
                    cycles_used=cycle,
                    insufficient_evidence=False,
                )
                return AgentRunResult(
                    answer=answer,
                    trace=self._trace(
                        query,
                        states,
                        requests,
                        evidence_snapshots,
                        verifier_outcomes,
                    ),
                )

            if (
                verification.status is VerificationStatus.RETRY_RETRIEVAL
                and cycle < self._max_cycles
            ):
                feedback = verification
                retrieval_text = (
                    verification.suggested_retrieval_focus or query.text
                )
                continue

            return self._insufficient_result(
                query,
                cycle,
                states,
                requests,
                evidence_snapshots,
                verifier_outcomes,
            )

        raise AssertionError("bounded agent loop terminated unexpectedly.")

    @staticmethod
    def _trace(
        query: AgentQuery,
        states: list[AgentState],
        requests: list[RetrievalRequest],
        evidence_snapshots: list[tuple[str, ...]],
        verifier_outcomes: list[VerificationResult],
    ) -> AgentRunTrace:
        return AgentRunTrace(
            query_id=query.query_id,
            state_transitions=tuple(states),
            retrieval_requests=tuple(requests),
            evidence_ids=tuple(evidence_snapshots),
            verifier_outcomes=tuple(verifier_outcomes),
        )

    def _insufficient_result(
        self,
        query: AgentQuery,
        cycle: int,
        states: list[AgentState],
        requests: list[RetrievalRequest],
        evidence_snapshots: list[tuple[str, ...]],
        verifier_outcomes: list[VerificationResult],
    ) -> AgentRunResult:
        states.append(AgentState.FINAL)
        answer = AgentAnswer(
            query_id=query.query_id,
            answer=_INSUFFICIENT_ANSWER,
            cited_evidence=(),
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            cycles_used=cycle,
            insufficient_evidence=True,
        )
        return AgentRunResult(
            answer=answer,
            trace=self._trace(
                query,
                states,
                requests,
                evidence_snapshots,
                verifier_outcomes,
            ),
        )
