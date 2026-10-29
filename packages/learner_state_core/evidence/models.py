from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConceptLink(BaseModel):
    """Grounding link connecting evidence to a curriculum Knowledge Component."""
    concept_id: str = Field(..., description="Unique KC identifier in curriculum taxonomy")
    concept_name: str = Field(..., description="Human-readable concept name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Semantic grounding confidence (0.0 to 1.0)")
    grounding_rationale: str = Field(..., description="Explanation of why text maps to this concept")


class CandidateInterpretation(BaseModel):
    """Inferred hypothesis regarding learner's underlying cognitive state."""
    hypothesis_id: str = Field(..., description="Identifier for candidate misconception or mastery hypothesis")
    category: Literal[
        "mastery",
        "misconception",
        "prerequisite_gap",
        "calculation_slip",
        "guessing",
        "ambiguous",
    ] = Field(..., description="Qualitative category of the cognitive phenomenon")
    description: str = Field(..., description="Detailed description of the hypothesized mental model")
    hypothesis_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this hypothesis (0.0 to 1.0)")


class CognitiveEvidence(BaseModel):
    """
    Core Cognitive Evidence contract.
    Decouples raw learner observation from system interpretation and state inference.
    """
    evidence_id: str = Field(default_factory=lambda: f"E_{uuid.uuid4().hex[:8]}", description="Unique identifier")
    interaction_id: str = Field(..., description="Source interaction reference")
    learner_id: str = Field(..., description="Pseudonymous learner identifier")
    timestamp: datetime = Field(default_factory=utc_now)

    # 1. Observation
    source_type: Literal[
        "student_explanation",
        "student_question",
        "chat_dialogue",
        "hint_request",
        "scratchpad",
        "answer_choice",
    ] = Field(..., description="Modal origin of evidence")
    source_quote: str = Field(..., description="Exact textual span from student interaction")
    is_verbatim_quote: bool = Field(default=True, description="Whether quote is an exact span or paraphrase")

    # 2. Extracted Claim
    extracted_claim: str = Field(..., description="Normalized atomic claim articulated by the learner")
    polarity: Literal["supports", "contradicts", "neutral", "ambiguous"] = Field(
        ..., description="Polarity toward understanding or target hypothesis"
    )
    expressed_uncertainty: Literal[
        "none",
        "low",
        "medium",
        "high",
        "explicitly_stated_doubt",
    ] = Field(default="none", description="Learner-expressed uncertainty")

    # 3. Grounding & Inferences
    concept_links: List[ConceptLink] = Field(..., min_length=1, description="Mapped curriculum concepts")
    candidate_interpretation: CandidateInterpretation = Field(..., description="Inferred cognitive hypothesis")

    # 4. Provenance & Audit
    extractor_model: str = Field(default="pretrained-instructor-v2", description="Extractor model identifier")
    prompt_version: str = Field(default="v2.0", description="Extraction prompt version")
    schema_version: str = Field(default="2.0")
    review_status: Literal["unreviewed", "accepted", "corrected", "rejected", "adjudicated"] = "unreviewed"
    reviewer_notes: Optional[str] = None

    def validate_against_raw_text(self, raw_text: str) -> bool:
        """Verifies that source_quote actually exists within the raw text if verbatim."""
        if not self.is_verbatim_quote or not self.source_quote.strip():
            return True
        clean_quote = re.sub(r"[^\w\s]", "", self.source_quote.strip().lower())
        clean_raw = re.sub(r"[^\w\s]", "", raw_text.strip().lower())
        return clean_quote in clean_raw


class Interaction(BaseModel):
    """Raw student interaction record."""
    interaction_id: str = Field(default_factory=lambda: f"I_{uuid.uuid4().hex[:8]}")
    learner_id: str
    item_id: str
    item_text: str
    item_concepts: List[str] = Field(default_factory=list, description="Primary concept IDs for this item")
    response_text: Optional[str] = None
    selected_option: Optional[str] = None
    correctness: Optional[int] = Field(None, ge=0, le=1, description="Binary correctness (0=incorrect, 1=correct)")
    latency_seconds: Optional[float] = None
    attempt_number: int = 1
    hint_used: bool = False
    timestamp: datetime = Field(default_factory=utc_now)
