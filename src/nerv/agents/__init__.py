"""Provider-neutral, grounded three-agent orchestration for NERV."""

from .agents import AnalystAgent, GroundingError, RetrieverAgent, VerifierAgent
from .backends import AnalysisBackend, RetrievalBackend, VerificationBackend
from .contracts import (
    AgentAnswer,
    AgentQuery,
    AgentRunResult,
    AgentRunTrace,
    AgentState,
    DraftAnswer,
    EvidenceCitation,
    RetrievedEvidence,
    RetrievalRequest,
    RetrievalResult,
    VerificationResult,
    VerificationStatus,
)
from .orchestrator import AgentOrchestrator

__all__ = [
    "AgentAnswer",
    "AgentOrchestrator",
    "AgentQuery",
    "AgentRunResult",
    "AgentRunTrace",
    "AgentState",
    "AnalysisBackend",
    "AnalystAgent",
    "DraftAnswer",
    "EvidenceCitation",
    "GroundingError",
    "RetrievedEvidence",
    "RetrievalBackend",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrieverAgent",
    "VerificationBackend",
    "VerificationResult",
    "VerificationStatus",
    "VerifierAgent",
]
