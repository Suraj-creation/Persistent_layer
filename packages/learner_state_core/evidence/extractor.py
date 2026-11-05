from __future__ import annotations

import re
import uuid
from .models import CandidateInterpretation, CognitiveEvidence, ConceptLink, Interaction
from ..grounding.taxonomy import FRACTION_TAXONOMY


class RuleBasedEvidenceExtractor:
    """Conservative deterministic fraction evidence rules; abstain when reasoning is absent."""

    def extract(self, interaction: Interaction) -> CognitiveEvidence:
        raw = (interaction.response_text or "").strip()
        lower = raw.lower()
        negated = bool(re.search(r"\b(?:not|never|no longer|cannot|can't|don't|didn't|did not|do not|shouldn't|should not|used to)\b", lower))
        cid = interaction.item_concepts[0] if interaction.item_concepts else "fraction_addition_unlike_denominators"
        if cid not in FRACTION_TAXONOMY:
            raise ValueError(f"Unknown curriculum concept: {cid}")
        kc = FRACTION_TAXONOMY[cid]
        category, hypothesis, confidence, polarity = "ambiguous", "ambiguous_unspecified", 0.4, "ambiguous"
        claim = "The response provides insufficient verified reasoning to characterize conceptual understanding."
        targets = []
        common = any(phrase in lower for phrase in ("found a common denominator", "converted to common denominator", "common denominator is", "equivalent fractions first", "multiply top and bottom by"))
        misc_patterns = ("added the top and bottom", "add the top numbers and the bottom numbers", "added the top numbers and the bottom numbers", "add both parts", "added numerators and denominators", "add numerators and add denominators")
        matches = [phrase for phrase in misc_patterns if phrase in lower]
        # A negated rule or discussion of someone else's rule is not evidence of its use.
        component = any(not re.search(r"(?:not|never|don't|shouldn't|incorrect to)\s+(?:just\s+)?$", lower[:lower.find(phrase)]) for phrase in matches)
        guess = any(phrase in lower for phrase in ("guess", "not sure"))
        gap = any(phrase in lower for phrase in ("what is a denominator", "what is a common denominator", "what does denominator mean", "don't understand fractions", "what is a numerator"))
        slip = any(phrase in lower for phrase in ("calculation mistake", "calculation error", "multiplied incorrectly", "arithmetic error", "math mistake"))
        if gap:
            cid = "fraction_equivalence" if "common denominator" in lower else "fraction_basics"
            kc = FRACTION_TAXONOMY[cid]
            category, hypothesis, confidence, polarity = "prerequisite_gap", "fraction_basics", 0.8, "neutral"
            claim = "The learner explicitly asks about a prerequisite fraction definition."
        elif guess:
            category, hypothesis, confidence = "guessing", "unverified_guess", 0.6
            claim = "The learner expresses guessing or uncertainty; reasoning remains unverified."
        elif negated:
            # A negated or historical description is not evidence of a current rule.
            pass
        elif cid in {"fraction_addition_like_denominators", "fraction_addition_unlike_denominators"} and component and interaction.correctness != 1:
            category, hypothesis, confidence, polarity = "misconception", "denominator_addition_rule" if cid.endswith("unlike_denominators") else "like_denominator_addition", 0.91, "contradicts"
            claim = "The learner describes independently adding numerators and denominators."
        elif slip and ("denominator" in lower or "multip" in lower):
            category, hypothesis, confidence, polarity = "calculation_slip", "calculation_slip", 0.88, "neutral"
            claim = "The learner describes the common-denominator method and an arithmetic slip."
        elif common and interaction.correctness != 0 and not component:
            category, hypothesis, confidence, polarity = "mastery", "fraction_common_denominator_mastery", 0.89, "supports"
            claim = "The learner explains equivalent fractions or common-denominator reasoning."
            if cid == "fraction_addition_unlike_denominators":
                targets = ["denominator_addition_rule"]
        elif cid == "fraction_equivalence" and any(phrase in lower for phrase in ("add one to both", "added 1 to both", "adding the same number")) and interaction.correctness != 1:
            category, hypothesis, confidence, polarity = "misconception", "additive_equivalence", 0.85, "contradicts"
            claim = "The learner describes additive changes as preserving fraction equivalence."
        elif cid == "fraction_addition_like_denominators" and "keep the denominator" in lower and "numerator" in lower and interaction.correctness != 0:
            category, hypothesis, confidence, polarity = "mastery", "like_denominator_mastery", 0.85, "supports"
            claim = "The learner explains adding numerators while keeping the common denominator."
            targets = ["like_denominator_addition"]
        elif cid == "fraction_basics" and "equal parts" in lower and "denominator" in lower and ("numerator" in lower or "whole" in lower) and interaction.correctness != 0:
            category, hypothesis, confidence, polarity = "mastery", "fraction_representation_mastery", 0.8, "supports"
            claim = "The learner describes numerator and denominator using equal parts of a whole."
            targets = ["numerator_denominator_inversion"]
        elif cid == "fraction_equivalence" and "same factor" in lower and interaction.correctness != 0:
            category, hypothesis, confidence, polarity = "mastery", "fraction_equivalence_mastery", 0.85, "supports"
            claim = "The learner explains scaling both fraction parts by the same factor."
            targets = ["additive_equivalence"]
        # A selected option alone is an observation; it does not establish reasoning.
        quote = raw or interaction.selected_option or ""
        return CognitiveEvidence(
            evidence_id="E_" + uuid.uuid5(uuid.NAMESPACE_URL, interaction.interaction_id + ":rules-v3").hex[:24],
            interaction_id=interaction.interaction_id, learner_id=interaction.learner_id, timestamp=interaction.timestamp,
            source_type="student_explanation" if raw else "answer_choice",
            source_quote=quote, is_verbatim_quote=bool(quote), extracted_claim=claim,
            polarity=polarity, expressed_uncertainty="high" if category in {"ambiguous", "guessing"} else "none",
            concept_links=[ConceptLink(concept_id=cid, concept_name=kc.concept_name, confidence=0.95, grounding_rationale="Canonical concept from validated item metadata.")],
            candidate_interpretation=CandidateInterpretation(hypothesis_id=hypothesis, category=category, description=claim, hypothesis_confidence=confidence),
            contradicted_hypothesis_ids=targets, extractor_model="rule-based-baseline-v3", prompt_version="rules-v3", schema_version="3.0",
        )


class InstructorEvidenceExtractor:
    """
    Schema-constrained extractor wrapping an LLM client via Instructor.
    Validates generated schema, taxonomy IDs, and exact source quotes before use.
    """

    def __init__(self, client=None, model_name: str = "gpt-4o-mini"):
        self.client = client
        self.model_name = model_name
        self.rule_fallback = RuleBasedEvidenceExtractor()

    def extract(self, interaction: Interaction, kc_taxonomy_context: str = "") -> CognitiveEvidence:
        if self.client is None:
            # Seamless fallback to rule-based extractor when LLM client is not connected
            return self.rule_fallback.extract(interaction)

        import instructor
        from ..grounding.taxonomy import FRACTION_TAXONOMY
        from ..grounding.matcher import SemanticTaxonomyMatcher

        instructor_client = instructor.from_openai(self.client)

        if not kc_taxonomy_context:
            kc_taxonomy_context = """Curriculum Knowledge Components:
- fraction_addition_unlike_denominators: Adding Fractions with Unlike Denominators (Finding common denominators using equivalent fractions and then adding numerators).
  Known Misconceptions:
  - 'denominator_addition_rule': Independent Numerator and Denominator Addition (e.g. 1/2 + 1/3 = 2/5).
  - 'cross_multiplication_addition_confusion': Confusing cross multiplication with addition.
  - 'calculation_slip': Arithmetic calculation mistake despite understanding common denominators.
- fraction_equivalence: Equivalent Fractions.
- fraction_addition_like_denominators: Adding Fractions with Like Denominators.
- fraction_basics: Fraction Representation & Parts.
"""

        prompt = f"""
You are an expert cognitive evidence extraction system in education.
Analyze the following student interaction on the problem:
Problem: {interaction.item_text}
Student's raw explanation: "{interaction.response_text or ''}"
Selected option: {interaction.selected_option or 'None'}
Student answer correctness: {interaction.correctness} (1=correct, 0=incorrect)

Relevant Knowledge Component Context:
{kc_taxonomy_context}

Taxonomy Rules:
1. Always link to one of the canonical concept_ids: 'fraction_addition_unlike_denominators', 'fraction_equivalence', 'fraction_addition_like_denominators', or 'fraction_basics'.
2. If the student added numerators and denominators independently (e.g., adding 1+1=2 and 2+3=5), set category='misconception', hypothesis_id='denominator_addition_rule', and description='Independent Numerator and Denominator Addition'.
3. If the student understood common denominators but made an arithmetic slip, set category='calculation_slip', hypothesis_id='calculation_slip', and description='Arithmetic Calculation Slip'.
4. If the student correctly found common denominators and added accurately (especially when correctness=1), set category='mastery', hypothesis_id='fraction_common_denominator_mastery'.
5. If the student asks fundamental questions revealing uncertainty or missing foundational definitions (e.g., 'what is a common denominator?'), set category='prerequisite_gap', hypothesis_id='prerequisite_gap', and description='Foundational definition or prerequisite gap'.

Task:
1. Extract the exact source quote from the student explanation (do NOT invent words).
2. Extract the atomic claim articulated by the student.
3. Use any schema category, including ambiguous or guessing when reasoning is insufficient.
4. Ground to the canonical curriculum concept_id.
5. Provide an uncalibrated confidence estimate. Never infer mastery from an answer choice alone; use ambiguous or guessing when evidence is insufficient.
"""

        evidence = instructor_client.chat.completions.create(
            model=self.model_name,
            response_model=CognitiveEvidence,
            messages=[
                {"role": "system", "content": "Extract structured Cognitive Evidence strictly conforming to schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        # Validate observations before permitting any state update. Invalid generated
        # evidence is replaced by deterministic rules, not relabeled as a paraphrase.
        evidence.learner_id = interaction.learner_id
        evidence.interaction_id = interaction.interaction_id
        evidence.timestamp = interaction.timestamp
        evidence.review_status = "unreviewed"
        evidence.reviewer_notes = None
        evidence.evidence_id = "E_" + uuid.uuid5(uuid.NAMESPACE_URL, interaction.interaction_id + ":" + self.model_name).hex[:24]
        raw = interaction.response_text or interaction.selected_option or ""
        ids = [link.concept_id for link in evidence.concept_links]
        if not evidence.is_verbatim_quote or not evidence.validate_against_raw_text(raw) or len(ids) != len(set(ids)) or any(cid not in FRACTION_TAXONOMY for cid in ids):
            return self.rule_fallback.extract(interaction)
        for link in evidence.concept_links:
            link.concept_name = FRACTION_TAXONOMY[link.concept_id].concept_name
        known = {m.misconception_id for cid in ids for m in FRACTION_TAXONOMY[cid].known_misconceptions}
        if evidence.candidate_interpretation.category == "misconception" and evidence.candidate_interpretation.hypothesis_id not in known:
            return self.rule_fallback.extract(interaction)
        evidence.contradicted_hypothesis_ids = [hid for hid in evidence.contradicted_hypothesis_ids if hid in known]
        return evidence
