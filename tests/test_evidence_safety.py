import copy

import pytest
from packages.learner_state_core.evidence.extractor import RuleBasedEvidenceExtractor
from packages.learner_state_core.evidence.models import Interaction
from packages.learner_state_core.state.engine import BayesianLearnerStateEngine

CID = "fraction_addition_unlike_denominators"


def interaction(text, correctness=None, key="1"):
    return Interaction(interaction_id=key, learner_id="L", item_id="item", item_text="1/2 + 1/3", item_concepts=[CID], response_text=text, correctness=correctness)


@pytest.mark.parametrize("text,correctness,expected", [("I don't add both parts.", None, "ambiguous"), ("I just guessed.", 1, "guessing"), ("What is a denominator?", None, "prerequisite_gap"), ("", 1, "ambiguous"), ("I added the top and bottom numbers", 1, "ambiguous"), ("I found a common denominator but my answer was wrong", 0, "ambiguous")])
def test_rules_abstain_on_negation_guesses_missing_reasoning_and_conflicts(text, correctness, expected):
    evidence = RuleBasedEvidenceExtractor().extract(interaction(text, correctness))
    assert evidence.candidate_interpretation.category == expected


def test_rules_produce_identical_evidence_for_identical_interaction():
    raw = interaction("I added the top and bottom numbers", 0)
    extractor = RuleBasedEvidenceExtractor()
    assert extractor.extract(raw).model_dump() == extractor.extract(raw).model_dump()
    evidence = extractor.extract(raw)
    assert evidence.timestamp == raw.timestamp
    assert evidence.validate_against_raw_text(raw.response_text)
    evidence.source_quote = "I added the top and bottom numbers!"
    assert not evidence.validate_against_raw_text(raw.response_text)


def test_counterevidence_only_discounts_explicitly_targeted_candidate():
    engine = BayesianLearnerStateEngine()
    extractor = RuleBasedEvidenceExtractor()
    state = engine.initialize_learner_state("L", [CID])
    state, _ = engine.update_state(state, extractor.extract(interaction("I added the top and bottom numbers", 0, "a")))
    belief = state.concept_states[CID].candidate_misconceptions["denominator_addition_rule"]
    other = belief.model_copy(deep=True)
    other.misconception_id = "cross_multiplication_addition_confusion"
    state.concept_states[CID].candidate_misconceptions[other.misconception_id] = other
    next_state, _ = engine.update_state(state, extractor.extract(interaction("I found a common denominator of 6", 1, "b")))
    beliefs = next_state.concept_states[CID].candidate_misconceptions
    assert beliefs["denominator_addition_rule"].posterior_probability < belief.posterior_probability
    assert beliefs[other.misconception_id].model_dump() == other.model_dump()


def test_repeated_guessing_does_not_create_sufficient_reasoning_evidence():
    engine = BayesianLearnerStateEngine()
    state = engine.initialize_learner_state("L", [CID])
    for index in range(8):
        evidence = RuleBasedEvidenceExtractor().extract(interaction("I guessed", 1, str(index)))
        state, _ = engine.update_state(state, evidence)
    concept = state.concept_states[CID]
    assert concept.evidence_sufficiency == "insufficient"
    assert concept.mastery_estimate == 0.5
    assert concept.informative_evidence_count == 0


@pytest.mark.parametrize("concept,text,correctness,category,hypothesis", [
    ("fraction_equivalence", "I add one to both parts to get an equivalent fraction", 0, "misconception", "additive_equivalence"),
    ("fraction_equivalence", "I multiply both parts by the same factor", 1, "mastery", "fraction_equivalence_mastery"),
    ("fraction_addition_like_denominators", "I add numerators and keep the denominator", 1, "mastery", "like_denominator_mastery"),
    ("fraction_basics", "The denominator gives equal parts and the numerator counts those parts", 1, "mastery", "fraction_representation_mastery"),
])
def test_each_curriculum_concept_uses_its_own_interpretations(concept, text, correctness, category, hypothesis):
    raw = interaction(text, correctness)
    raw.item_concepts = [concept]
    evidence = RuleBasedEvidenceExtractor().extract(raw)
    assert evidence.concept_links[0].concept_id == concept
    assert evidence.candidate_interpretation.category == category
    assert evidence.candidate_interpretation.hypothesis_id == hypothesis


@pytest.mark.parametrize("invalid", ["invented_quote", "unknown_concept", "duplicate_concept"])
def test_optional_provider_evidence_is_validated_before_state_update(monkeypatch, invalid):
    import sys
    from types import SimpleNamespace
    from packages.learner_state_core.evidence.extractor import InstructorEvidenceExtractor
    raw = interaction("I added the top and bottom numbers", 0)
    generated = RuleBasedEvidenceExtractor().extract(raw)
    if invalid == "invented_quote":
        generated.source_quote = "The learner never said these words."
    elif invalid == "unknown_concept":
        generated.concept_links[0].concept_id = "invented_curriculum"
    else:
        generated.concept_links.append(generated.concept_links[0].model_copy())
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: generated)))
    monkeypatch.setitem(sys.modules, "instructor", SimpleNamespace(from_openai=lambda client: fake_client))
    evidence = InstructorEvidenceExtractor(client=object()).extract(raw, "trusted curriculum")
    assert evidence.model_dump() == RuleBasedEvidenceExtractor().extract(raw).model_dump()
