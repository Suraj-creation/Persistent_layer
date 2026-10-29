from __future__ import annotations

import pytest
from pydantic import ValidationError
from packages.learner_state_core.evidence.models import (
    CognitiveEvidence,
    ConceptLink,
    CandidateInterpretation,
    Interaction,
)


def test_evidence_model_schema_validation():
    """Verify that CognitiveEvidence validates required fields and types."""
    concept = ConceptLink(
        concept_id="fraction_addition",
        concept_name="Fraction Addition",
        confidence=0.95,
        grounding_rationale="Explicit mention of adding fractions.",
    )

    interp = CandidateInterpretation(
        hypothesis_id="denominator_addition",
        category="misconception",
        description="Adding numerators and denominators independently.",
        hypothesis_confidence=0.88,
    )

    evidence = CognitiveEvidence(
        evidence_id="E_001",
        interaction_id="I_001",
        learner_id="L_101",
        source_type="student_explanation",
        source_quote="I added the top and bottom numbers.",
        extracted_claim="Student added numerators and denominators independently.",
        polarity="contradicts",
        expressed_uncertainty="none",
        concept_links=[concept],
        candidate_interpretation=interp,
    )

    assert evidence.evidence_id == "E_001"
    assert evidence.validate_against_raw_text("To solve this, I added the top and bottom numbers because they are separate.")
    assert not evidence.validate_against_raw_text("I used a completely different technique.")


def test_invalid_evidence_confidence_range():
    """Verify that confidence out of [0, 1] raises a ValidationError."""
    with pytest.raises(ValidationError):
        ConceptLink(
            concept_id="c1",
            concept_name="Test Concept",
            confidence=1.5,  # Invalid: > 1.0
            grounding_rationale="Test",
        )
