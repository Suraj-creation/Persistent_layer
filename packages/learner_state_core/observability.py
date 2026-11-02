"""Public contracts for saved learning turns and their decision provenance."""
from typing import Any, Literal
from pydantic import BaseModel
from .diagnosis.prober import DiagnosticItem
from .evidence.models import CognitiveEvidence, Interaction
from .state.models import LearnerStateSnapshot, StateChangeEvent


class DecisionTrace(BaseModel):
    trace_id: str
    request_id: str
    timestamp: str
    retrieval_source: Literal["sqlite"]
    snapshot_before_id: str
    snapshot_after_id: str
    extraction_context_snapshot_id: str | None
    evidence_ids: list[str]
    retrieved_evidence_ids: list[str]
    context_sha256: str
    decision_context: dict[str, Any]
    extractor_model: str
    fallback_reason: str | None
    reply_mode: Literal["state_policy"]
    policy_version: str
    question_id: str
    question_text: str
    decision_reason: str
    applied_rules: list[str]
    timings_ms: dict[str, float]
    quote_validated: Literal[True]


class LearningTurn(BaseModel):
    interaction: Interaction
    reply: str
    evidence: CognitiveEvidence
    new_state: LearnerStateSnapshot
    events: list[StateChangeEvent]
    recommended_diagnostic_probe: DiagnosticItem
    diagnostic_rationale: str
    trace: DecisionTrace
