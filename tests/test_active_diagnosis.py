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


def test_empty_item_bank_and_excluded_items_are_respected():
    state = ConceptState(concept_id="c", concept_name="c", mastery_estimate=0)
    assert ActiveDiagnosisProber(item_bank=[]).select_best_diagnostic_probe("c", state) is None
    prober = ActiveDiagnosisProber()
    state.concept_id = "fraction_addition_unlike_denominators"
    assert prober.select_best_diagnostic_probe(state.concept_id, state, [item.item_id for item in FRACTION_DIAGNOSTIC_BANK]) is None


def test_information_gain_is_bounded_by_prior_entropy_with_normalized_likelihoods():
    state = ConceptState(concept_id="fraction_addition_unlike_denominators", concept_name="addition", mastery_estimate=0.5)
    result = ActiveDiagnosisProber().select_best_diagnostic_probe(state.concept_id, state)
    assert result is not None
    assert 0 <= result[1] <= 1
