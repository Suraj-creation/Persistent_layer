from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MisconceptionBelief(BaseModel):
    """Belief state for a specific candidate misconception."""
    misconception_id: str = Field(..., description="Unique taxonomy ID for misconception")
    misconception_name: str = Field(..., description="Short descriptive title of misconception")
    posterior_probability: float = Field(..., ge=0.0, le=1.0, description="Current estimated probability P(M)")
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of evidence supporting this bug")
    contradicting_evidence_ids: List[str] = Field(default_factory=list, description="IDs of evidence refuting this bug")
    status: Literal["suspected", "confirmed", "resolved", "dismissed"] = Field(
        default="suspected", description="Lifecycle state of this candidate misconception"
    )
    last_updated: datetime = Field(default_factory=utc_now)


class ConceptState(BaseModel):
    """Current probabilistic belief over a single curriculum concept."""
    concept_id: str
    concept_name: str
    alpha: float = Field(default=1.0, ge=0.01, description="Beta distribution positive evidence mass")
    beta: float = Field(default=1.0, ge=0.01, description="Beta distribution negative/misconception evidence mass")
    mastery_estimate: float = Field(default=0.5, ge=0.0, le=1.0, description="Mean mastery belief E[L] = alpha / (alpha + beta)")
    uncertainty: float = Field(default=0.2887, ge=0.0, le=1.0, description="Epistemic uncertainty sd = sqrt(alpha*beta / ((alpha+beta)^2*(alpha+beta+1)))")
    evidence_sufficiency: Literal["insufficient", "sparse", "moderate", "sufficient"] = "insufficient"
    candidate_misconceptions: Dict[str, MisconceptionBelief] = Field(
        default_factory=dict, description="Active misconception hypotheses for this concept"
    )
    last_interaction_timestamp: Optional[datetime] = None
    interaction_count: int = 0


class LearnerStateSnapshot(BaseModel):
    """
    Immutable snapshot of a learner's cognitive state as of a specific interaction.
    Allows exact time-travel replay and provenance tracking.
    """
    snapshot_id: str = Field(default_factory=lambda: f"S_{uuid.uuid4().hex[:8]}")
    learner_id: str
    as_of_interaction_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    concept_states: Dict[str, ConceptState] = Field(default_factory=dict)
    active_prerequisite_gaps: List[str] = Field(default_factory=list)
    policy_version: str = "beta-bayesian-v2.0"


class StateChangeEvent(BaseModel):
    """
    Structured change event capturing 'Why did this state change?'
    """
    event_id: str = Field(default_factory=lambda: f"EV_{uuid.uuid4().hex[:8]}")
    learner_id: str
    concept_id: str
    prior_snapshot_id: Optional[str]
    new_snapshot_id: str
    triggering_evidence_id: str
    delta_mastery: float
    prior_uncertainty: Optional[float] = None
    new_uncertainty: Optional[float] = None
    prior_misconception_prob: Optional[float] = None
    new_misconception_prob: Optional[float] = None
    change_summary: str
    applied_rule: str
    math_trace: Dict[str, Any] = Field(default_factory=dict, description="Exact step-by-step mathematical computation trace")
    timestamp: datetime = Field(default_factory=utc_now)
