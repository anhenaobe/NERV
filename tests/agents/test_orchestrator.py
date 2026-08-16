"""Deterministic unit tests for grounded three-agent orchestration."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

from nerv.agents import (
    AgentOrchestrator,
    AgentQuery,
    AgentState,
    AnalystAgent,
    DraftAnswer,
    EvidenceCitation,
    GroundingError,
    RetrievedEvidence,
    RetrieverAgent,
    VerificationResult,
    VerificationStatus,
    VerifierAgent,
)


class QueueRetrievalBackend:
    """Return one synthetic evidence sequence per recorded call."""

    def __init__(
        self,
        responses: Sequence[Sequence[RetrievedEvidence]],
    ) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> Sequence[RetrievedEvidence]:
        self.calls.append((query, top_k))
        return self._responses[len(self.calls) - 1]


class QueueAnalysisBackend:
    """Return configured drafts while recording supplied evidence."""

    def __init__(self, drafts: Sequence[DraftAnswer]) -> None:
        self._drafts = list(drafts)
        self.evidence_seen: list[tuple[RetrievedEvidence, ...]] = []
        self.feedback_seen: list[VerificationResult | None] = []

    def generate_analysis(
        self,
        query: AgentQuery,
        evidence: Sequence[RetrievedEvidence],
        feedback: VerificationResult | None,
    ) -> DraftAnswer:
        self.evidence_seen.append(tuple(evidence))
        self.feedback_seen.append(feedback)
        return self._drafts[len(self.evidence_seen) - 1]


class QueueVerificationBackend:
    """Return configured bounded decisions in call order."""

    def __init__(self, results: Sequence[VerificationResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def verify_answer(
        self,
        query: AgentQuery,
        evidence: Sequence[RetrievedEvidence],
        draft: DraftAnswer,
    ) -> VerificationResult:
        result = self._results[self.calls]
        self.calls += 1
        return result


def _evidence(
    chunk_id: str,
    *,
    rank: int = 1,
    doc_id: str = "doc-1",
    source: str = "source.txt",
    text: str = "Grounded fact.",
    score: float | None = 0.75,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=source,
        rank=rank,
        text=text,
        score=score,
    )


def _draft(text: str, *chunk_ids: str) -> DraftAnswer:
    return DraftAnswer(
        text=text,
        citations=tuple(EvidenceCitation(chunk_id) for chunk_id in chunk_ids),
        confidence=0.8,
    )


def _orchestrator(
    retrieval: QueueRetrievalBackend,
    analysis: QueueAnalysisBackend,
    verification: QueueVerificationBackend,
    *,
    max_cycles: int = 2,
) -> AgentOrchestrator:
    return AgentOrchestrator(
        RetrieverAgent(retrieval, top_k=5),
        AnalystAgent(analysis),
        VerifierAgent(verification),
        max_cycles=max_cycles,
    )


class AgentOrchestratorTests(unittest.TestCase):
    """Exercise the deterministic state machine without external I/O."""

    def test_successful_retrieval_analysis_and_verification_pass(self) -> None:
        evidence = _evidence("chunk-1")
        retrieval = QueueRetrievalBackend([[evidence]])
        analysis = QueueAnalysisBackend([_draft("Supported answer.", "chunk-1")])
        verification = QueueVerificationBackend(
            [VerificationResult(VerificationStatus.PASS)]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q001", "What is supported?")
        )

        self.assertEqual(result.answer.answer, "Supported answer.")
        self.assertIs(result.answer.verification_status, VerificationStatus.PASS)
        self.assertEqual(result.answer.cycles_used, 1)
        self.assertEqual(result.answer.cited_evidence[0].chunk_id, "chunk-1")

    def test_empty_retrieval_returns_insufficient_evidence(self) -> None:
        retrieval = QueueRetrievalBackend([[]])
        analysis = QueueAnalysisBackend([])
        verification = QueueVerificationBackend([])

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q002", "Unknown topic")
        )

        self.assertTrue(result.answer.insufficient_evidence)
        self.assertIs(
            result.answer.verification_status,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
        )
        self.assertFalse(analysis.evidence_seen)
        self.assertEqual(verification.calls, 0)
        self.assertEqual(
            result.trace.state_transitions,
            (AgentState.QUERY, AgentState.RETRIEVE, AgentState.FINAL),
        )

    def test_analyst_reference_to_unknown_chunk_is_rejected(self) -> None:
        retrieval = QueueRetrievalBackend([[_evidence("known")]])
        analysis = QueueAnalysisBackend([_draft("Unsupported.", "invented")])
        verification = QueueVerificationBackend([])

        with self.assertRaisesRegex(GroundingError, "invented"):
            _orchestrator(retrieval, analysis, verification).run(
                AgentQuery("q003", "Question")
            )

        self.assertEqual(verification.calls, 0)

    def test_verifier_retry_causes_second_focused_retrieval(self) -> None:
        first = _evidence("chunk-1")
        second = _evidence("chunk-2", rank=1, doc_id="doc-2")
        retrieval = QueueRetrievalBackend([[first], [second]])
        analysis = QueueAnalysisBackend(
            [
                _draft("First draft.", "chunk-1"),
                _draft("Grounded after retry.", "chunk-2"),
            ]
        )
        retry = VerificationResult(
            VerificationStatus.RETRY_RETRIEVAL,
            missing_evidence=("A second source is needed.",),
            suggested_retrieval_focus="second source",
        )
        verification = QueueVerificationBackend(
            [retry, VerificationResult(VerificationStatus.PASS)]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q004", "Original question")
        )

        self.assertEqual(
            retrieval.calls,
            [("Original question", 5), ("second source", 5)],
        )
        self.assertEqual(result.answer.cycles_used, 2)
        self.assertEqual(analysis.feedback_seen, [None, retry])
        self.assertEqual(
            [item.chunk_id for item in analysis.evidence_seen[1]],
            ["chunk-1", "chunk-2"],
        )

    def test_retry_limit_prevents_an_infinite_loop(self) -> None:
        retrieval = QueueRetrievalBackend(
            [[_evidence("chunk-1")], [_evidence("chunk-2")]]
        )
        analysis = QueueAnalysisBackend(
            [_draft("Draft one.", "chunk-1"), _draft("Draft two.", "chunk-2")]
        )
        retry = VerificationResult(
            VerificationStatus.RETRY_RETRIEVAL,
            missing_evidence=("More support is required.",),
        )
        verification = QueueVerificationBackend([retry, retry])

        result = _orchestrator(
            retrieval,
            analysis,
            verification,
            max_cycles=2,
        ).run(AgentQuery("q005", "Bound this query"))

        self.assertEqual(len(retrieval.calls), 2)
        self.assertEqual(verification.calls, 2)
        self.assertEqual(result.answer.cycles_used, 2)
        self.assertTrue(result.answer.insufficient_evidence)
        self.assertEqual(
            [item.status for item in result.trace.verifier_outcomes],
            [
                VerificationStatus.RETRY_RETRIEVAL,
                VerificationStatus.RETRY_RETRIEVAL,
            ],
        )

    def test_verifier_can_return_insufficient_evidence(self) -> None:
        retrieval = QueueRetrievalBackend([[_evidence("chunk-1")]])
        analysis = QueueAnalysisBackend([_draft("Weak draft.", "chunk-1")])
        verification = QueueVerificationBackend(
            [
                VerificationResult(
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    unsupported_claims=("The central claim lacks support.",),
                )
            ]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q006", "Can this be answered?")
        )

        self.assertTrue(result.answer.insufficient_evidence)
        self.assertEqual(result.answer.cited_evidence, ())
        self.assertEqual(result.answer.cycles_used, 1)

    def test_duplicate_chunks_keep_first_ranked_metadata(self) -> None:
        first = _evidence("chunk-1", rank=1, source="first.txt", score=0.9)
        duplicate = _evidence(
            "chunk-1",
            rank=4,
            source="duplicate.txt",
            score=0.2,
        )
        retrieval = QueueRetrievalBackend([[first, duplicate]])
        analysis = QueueAnalysisBackend([_draft("Answer.", "chunk-1")])
        verification = QueueVerificationBackend(
            [VerificationResult(VerificationStatus.PASS)]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q007", "Deduplicate")
        )

        self.assertEqual(len(analysis.evidence_seen[0]), 1)
        self.assertEqual(analysis.evidence_seen[0][0].rank, 1)
        self.assertEqual(analysis.evidence_seen[0][0].source, "first.txt")
        self.assertEqual(result.trace.evidence_ids, (("chunk-1",),))

    def test_evidence_metadata_survives_orchestration(self) -> None:
        evidence = _evidence(
            "chunk-meta",
            rank=3,
            doc_id="doc-meta",
            source="nested/source.pdf",
            text="Specific support.",
            score=None,
        )
        retrieval = QueueRetrievalBackend([[evidence]])
        analysis = QueueAnalysisBackend([_draft("Answer.", "chunk-meta")])
        verification = QueueVerificationBackend(
            [VerificationResult(VerificationStatus.PASS)]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q008", "Metadata")
        )

        cited = result.answer.cited_evidence[0]
        self.assertEqual(
            (cited.chunk_id, cited.doc_id, cited.source, cited.rank),
            ("chunk-meta", "doc-meta", "nested/source.pdf", 3),
        )
        self.assertEqual(cited.text, "Specific support.")
        self.assertIsNone(cited.score)
        self.assertEqual(cited.retrieval_query, "Metadata")

    def test_final_cited_evidence_is_subset_of_retrieved(self) -> None:
        retrieval = QueueRetrievalBackend(
            [[_evidence("chunk-1"), _evidence("chunk-2", rank=2)]]
        )
        analysis = QueueAnalysisBackend([_draft("Narrow answer.", "chunk-2")])
        verification = QueueVerificationBackend(
            [VerificationResult(VerificationStatus.PASS)]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q009", "Use a subset")
        )

        retrieved_ids = set(result.trace.evidence_ids[0])
        cited_ids = {item.chunk_id for item in result.answer.cited_evidence}
        self.assertEqual(cited_ids, {"chunk-2"})
        self.assertLess(cited_ids, retrieved_ids)

    def test_run_trace_records_deterministic_state_transitions(self) -> None:
        retrieval = QueueRetrievalBackend(
            [[_evidence("chunk-1")], [_evidence("chunk-2")]]
        )
        analysis = QueueAnalysisBackend(
            [_draft("First.", "chunk-1"), _draft("Second.", "chunk-2")]
        )
        verification = QueueVerificationBackend(
            [
                VerificationResult(
                    VerificationStatus.RETRY_RETRIEVAL,
                    missing_evidence=("More evidence.",),
                ),
                VerificationResult(VerificationStatus.PASS),
            ]
        )

        result = _orchestrator(retrieval, analysis, verification).run(
            AgentQuery("q010", "Trace this")
        )

        self.assertEqual(
            result.trace.state_transitions,
            (
                AgentState.QUERY,
                AgentState.RETRIEVE,
                AgentState.ANALYZE,
                AgentState.VERIFY,
                AgentState.RETRIEVE,
                AgentState.ANALYZE,
                AgentState.VERIFY,
                AgentState.FINAL,
            ),
        )
        self.assertEqual(
            [request.cycle for request in result.trace.retrieval_requests],
            [1, 2],
        )
        self.assertEqual(
            result.trace.evidence_ids,
            (("chunk-1",), ("chunk-2",)),
        )


if __name__ == "__main__":
    unittest.main()
