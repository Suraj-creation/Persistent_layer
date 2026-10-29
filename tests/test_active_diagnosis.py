from __future__ import annotations

import pytest
from packages.learner_state_core.diagnosis.prober import (
    ActiveDiagnosisProber,
    DiagnosticItem,
    DiagnosticOption,
    FRACTION_DIAGNOSTIC_BANK,
)
from packages.learner_state_core.state.models import ConceptState, MisconceptionBelief


def test_active_diagnosis_selection():
    """
    Test that when a student has a suspected misconception,
    the active diagnosis prober calculates EIG and selects the probe
    that maximally resolves the uncertainty between mastery and the misconception.
    """
    prober = ActiveDiagnosisProber(item_bank=FRACTION_DIAGNOSTIC_BANK)
    concept_id = "fraction_addition_unlike_denominators"

    # State with suspected denominator_addition_rule misconception
    cstate = ConceptState(
        concept_id=concept_id,
        concept_name="Adding Fractions with Unlike Denominators",
        mastery_estimate=0.45,
        uncertainty=0.40,
        evidence_sufficiency="sparse",
        candidate_misconceptions={
            "denominator_addition_rule": MisconceptionBelief(
                misconception_id="denominator_addition_rule",
                misconception_name="Independent Numerator and Denominator Addition",
                posterior_probability=0.65,
                status="suspected",
            )
        },
    )

    result = prober.select_best_diagnostic_probe(concept_id, cstate)
    assert result is not None

    selected_item, eig, rationale = result
    assert isinstance(selected_item, DiagnosticItem)
    assert selected_item.concept_id == concept_id
    assert eig > 0.05  # Significant information gain
    assert "Expected Information Gain" in rationale
    assert any(
        opt.targeted_misconception_id == "denominator_addition_rule"
        for opt in selected_item.options
    )
