from __future__ import annotations

import pytest
from packages.learner_state_core.evidence.models import Interaction
from packages.learner_state_core.evidence.extractor import RuleBasedEvidenceExtractor
from packages.learner_state_core.state.engine import BayesianLearnerStateEngine


def test_student_a_misconception_flow():
    """
    Test Case Student A:
    Problem: 1/2 + 1/3
    Answer: 2/5 (Incorrect)
    Explanation: "I added the top numbers and the bottom numbers because we add both parts."
    Outcome: System must extract a systematic misconception (denominator_addition_rule),
             and update state from neutral to 'suspected'.
             On repeated error, update state to 'confirmed'.
    """
    extractor = RuleBasedEvidenceExtractor()
    state_engine = BayesianLearnerStateEngine()

    learner_id = "student_A"
    concept_id = "fraction_addition_unlike_denominators"

    # 1. Initialize learner state
    initial_state = state_engine.initialize_learner_state(learner_id, [concept_id])
    assert initial_state.concept_states[concept_id].mastery_estimate == 0.50
    assert len(initial_state.concept_states[concept_id].candidate_misconceptions) == 0

    # 2. First interaction
    interaction_1 = Interaction(
        interaction_id="I_A_01",
        learner_id=learner_id,
        item_id="ITEM_FRAC_01",
        item_text="Solve 1/2 + 1/3",
        item_concepts=[concept_id],
        response_text="I added the top numbers and the bottom numbers because we add both parts of a fraction.",
        selected_option="2/5",
        correctness=0,
    )

    evidence_1 = extractor.extract(interaction_1)
    assert evidence_1.candidate_interpretation.category == "misconception"
    assert evidence_1.candidate_interpretation.hypothesis_id == "denominator_addition_rule"
    assert "added the top numbers and the bottom numbers" in evidence_1.source_quote
    assert evidence_1.validate_against_raw_text(interaction_1.response_text)

    # 3. Update state with Evidence 1
    state_1, events_1 = state_engine.update_state(initial_state, evidence_1, interaction_1)
    cstate_1 = state_1.concept_states[concept_id]

    # Verify mastery decreased and candidate misconception is suspected
    assert cstate_1.mastery_estimate < 0.50
    assert "denominator_addition_rule" in cstate_1.candidate_misconceptions
    belief_1 = cstate_1.candidate_misconceptions["denominator_addition_rule"]
    assert belief_1.posterior_probability >= 0.45
    assert belief_1.status == "suspected"
    assert evidence_1.evidence_id in belief_1.supporting_evidence_ids
    assert len(events_1) == 1
    assert events_1[0].applied_rule == "RULE_MISCONCEPTION_REINFORCEMENT"

    # 4. Second interaction (Repeated misconception on 1/4 + 2/3)
    interaction_2 = Interaction(
        interaction_id="I_A_02",
        learner_id=learner_id,
        item_id="ITEM_FRAC_02",
        item_text="Solve 1/4 + 2/3",
        item_concepts=[concept_id],
        response_text="I added the top and bottom numbers again to get 3/7.",
        selected_option="3/7",
        correctness=0,
    )
    evidence_2 = extractor.extract(interaction_2)
    state_2, events_2 = state_engine.update_state(state_1, evidence_2, interaction_2)
    cstate_2 = state_2.concept_states[concept_id]

    belief_2 = cstate_2.candidate_misconceptions["denominator_addition_rule"]
    # Posterior probability should now be confirmed (>= 0.70)
    assert belief_2.posterior_probability >= 0.70
    assert belief_2.status == "confirmed"
    assert len(belief_2.supporting_evidence_ids) == 2


def test_student_b_calculation_slip_flow():
    """
    Test Case Student B:
    Problem: 1/2 + 1/3
    Answer: 4/6 (Incorrect)
    Explanation: "I know the denominators must match, so I tried to find a common denominator, but I made a calculation mistake."
    Outcome: System must extract category 'calculation_slip',
             must NOT classify this as denominator_addition_rule,
             mastery penalty must be minimal (<= 0.05).
    """
    extractor = RuleBasedEvidenceExtractor()
    state_engine = BayesianLearnerStateEngine()

    learner_id = "student_B"
    concept_id = "fraction_addition_unlike_denominators"

    initial_state = state_engine.initialize_learner_state(learner_id, [concept_id])

    interaction = Interaction(
        interaction_id="I_B_01",
        learner_id=learner_id,
        item_id="ITEM_FRAC_01",
        item_text="Solve 1/2 + 1/3",
        item_concepts=[concept_id],
        response_text="I know the denominators must match, but I made a calculation mistake while multiplying.",
        selected_option="4/6",
        correctness=0,
    )

    evidence = extractor.extract(interaction)
    assert evidence.candidate_interpretation.category == "calculation_slip"
    assert evidence.candidate_interpretation.hypothesis_id == "calculation_slip"

    state_after, events = state_engine.update_state(initial_state, evidence, interaction)
    cstate = state_after.concept_states[concept_id]

    # Mastery should drop only slightly
    delta_mastery = initial_state.concept_states[concept_id].mastery_estimate - cstate.mastery_estimate
    assert delta_mastery <= 0.05
    # Misconception must NOT be added
    assert "denominator_addition_rule" not in cstate.candidate_misconceptions
    assert events[0].applied_rule == "RULE_SLIP_ATTENUATION"


def test_contradiction_handling_and_reversal():
    """
    Test Contradiction Flow:
    Student A initially has denominator_addition_rule.
    In a subsequent interaction, Student A gives a correct explanation:
    "I found a common denominator of 6, converted fractions to 3/6 and 2/6, and got 5/6."
    Outcome: The misconception posterior must drop significantly (discounted),
             status marked 'resolved', while preserving evidence history.
    """
    extractor = RuleBasedEvidenceExtractor()
    state_engine = BayesianLearnerStateEngine()

    learner_id = "student_A"
    concept_id = "fraction_addition_unlike_denominators"

    initial_state = state_engine.initialize_learner_state(learner_id, [concept_id])

    # 1. Establish misconception
    interaction_1 = Interaction(
        interaction_id="I_A_01",
        learner_id=learner_id,
        item_id="ITEM_FRAC_01",
        item_text="Solve 1/2 + 1/3",
        item_concepts=[concept_id],
        response_text="I added the top and bottom numbers.",
        selected_option="2/5",
        correctness=0,
    )
    evidence_1 = extractor.extract(interaction_1)
    state_1, _ = state_engine.update_state(initial_state, evidence_1, interaction_1)
    assert state_1.concept_states[concept_id].candidate_misconceptions["denominator_addition_rule"].status == "suspected"

    # 2. Student demonstrates mastery
    interaction_2 = Interaction(
        interaction_id="I_A_02",
        learner_id=learner_id,
        item_id="ITEM_FRAC_02",
        item_text="Solve 1/2 + 1/3",
        item_concepts=[concept_id],
        response_text="I found a common denominator of 6, converted to 3/6 and 2/6, and added numerators to get 5/6.",
        selected_option="5/6",
        correctness=1,
    )
    evidence_2 = extractor.extract(interaction_2)
    state_2, events_2 = state_engine.update_state(state_1, evidence_2, interaction_2)

    cstate_2 = state_2.concept_states[concept_id]
    m_belief_2 = cstate_2.candidate_misconceptions["denominator_addition_rule"]

    # Misconception probability should be sharply discounted
    assert m_belief_2.posterior_probability <= 0.25
    assert m_belief_2.status == "resolved"
    assert evidence_2.evidence_id in m_belief_2.contradicting_evidence_ids
    # Both supporting and contradicting evidence IDs must be preserved
    assert len(m_belief_2.supporting_evidence_ids) == 1
    assert len(m_belief_2.contradicting_evidence_ids) == 1

    # Check state change events
    discount_event = next(e for e in events_2 if e.applied_rule == "RULE_CONTRADICTION_DISCOUNT")
    assert discount_event.prior_misconception_prob > discount_event.new_misconception_prob


def test_deterministic_replay():
    """
    Test that replaying a list of evidence records from scratch produces identical state snapshots.
    """
    extractor = RuleBasedEvidenceExtractor()
    state_engine = BayesianLearnerStateEngine()

    learner_id = "student_replay_test"
    concept_id = "fraction_addition_unlike_denominators"

    interactions = [
        Interaction(
            interaction_id="I_01",
            learner_id=learner_id,
            item_id="ITEM_1",
            item_text="1/2 + 1/3",
            item_concepts=[concept_id],
            response_text="I added the top and bottom numbers.",
            correctness=0,
        ),
        Interaction(
            interaction_id="I_02",
            learner_id=learner_id,
            item_id="ITEM_2",
            item_text="1/4 + 1/2",
            item_concepts=[concept_id],
            response_text="I know the denominators must match, but I made a calculation mistake.",
            correctness=0,
        ),
        Interaction(
            interaction_id="I_03",
            learner_id=learner_id,
            item_text="2/3 + 1/6",
            item_id="ITEM_3",
            item_concepts=[concept_id],
            response_text="I found a common denominator of 6, converted 2/3 to 4/6, got 5/6.",
            correctness=1,
        ),
    ]

    evidence_ledger = [extractor.extract(i) for i in interactions]

    # Replay 1
    traj_1 = state_engine.replay_trajectory(learner_id, [concept_id], evidence_ledger)
    final_state_1 = traj_1[-1][0]

    # Replay 2
    traj_2 = state_engine.replay_trajectory(learner_id, [concept_id], evidence_ledger)
    final_state_2 = traj_2[-1][0]

    # Verify identical concept state values
    c1 = final_state_1.concept_states[concept_id]
    c2 = final_state_2.concept_states[concept_id]

    assert round(c1.mastery_estimate, 4) == round(c2.mastery_estimate, 4)
    assert round(c1.uncertainty, 4) == round(c2.uncertainty, 4)
    assert c1.evidence_sufficiency == c2.evidence_sufficiency
    assert c1.interaction_count == c2.interaction_count

    m1 = {k: round(v.posterior_probability, 4) for k, v in c1.candidate_misconceptions.items()}
    m2 = {k: round(v.posterior_probability, 4) for k, v in c2.candidate_misconceptions.items()}
    assert m1 == m2
