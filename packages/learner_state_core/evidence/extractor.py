from __future__ import annotations

import re
import uuid
from .models import CandidateInterpretation, CognitiveEvidence, ConceptLink, Interaction
from ..grounding.taxonomy import FRACTION_TAXONOMY


# Interpret a fraction explanation with deterministic cue-based rules.
class RuleBasedEvidenceExtractor:
    """Conservative deterministic fraction evidence rules; abstain when reasoning is absent."""

    # Produce candidate cognitive evidence from an actual fraction interaction.
    def extract(self, interaction: Interaction) -> CognitiveEvidence:
        # Read the written explanation, keeping the trimmed substring available as a quote.
        raw = (interaction.response_text or "").strip()
        # Create a lowercase view for comparing phrase cues.
        lower = raw.lower()
        # Detect negation and language that may describe a past belief.
        negated = bool(re.search(r"\b(?:not|never|no longer|cannot|can't|don't|didn't|did not|do not|shouldn't|should not|used to)\b", lower))
        # Take the task's concept from its known metadata or use the primary fraction concept.
        cid = interaction.item_concepts[0] if interaction.item_concepts else "fraction_addition_unlike_denominators"
        # Verify that the concept belongs to the curated fraction curriculum.
        if cid not in FRACTION_TAXONOMY:
            # Reject a concept the fraction rules cannot ground.
            raise ValueError(f"Unknown curriculum concept: {cid}")
        # Load the taxonomy record for the assessed concept.
        kc = FRACTION_TAXONOMY[cid]
        # Start conservatively with an ambiguous observation.
        category, hypothesis, confidence, polarity = "ambiguous", "ambiguous_unspecified", 0.4, "ambiguous"
        # Use a generic insufficient-evidence claim until a stronger rule matches.
        claim = "The response provides insufficient verified reasoning to characterize conceptual understanding."
        # Start without an explicit contradiction of an earlier misconception.
        targets = []
        # Detect phrases indicating a common-denominator or equivalent-fraction method.
        common = any(phrase in lower for phrase in ("found a common denominator", "converted to common denominator", "common denominator is", "equivalent fractions first", "multiply top and bottom by"))
        # List common ways learners express independent addition of numerator and denominator.
        misc_patterns = ("added the top and bottom", "add the top numbers and the bottom numbers", "added the top numbers and the bottom numbers", "add both parts", "added numerators and denominators", "add numerators and add denominators")
        # Find the expressions present in the response.
        matches = [phrase for phrase in misc_patterns if phrase in lower]
        # A negated rule or discussion of someone else's rule is not evidence of its use.
        # Require a matching phrase without an immediately preceding negation.
        component = any(not re.search(r"(?:not|never|don't|shouldn't|incorrect to)\s+(?:just\s+)?$", lower[:lower.find(phrase)]) for phrase in matches)
        # Detect explicit uncertainty or guessing cues.
        guess = any(phrase in lower for phrase in ("guess", "not sure"))
        # Detect questions about foundational fraction vocabulary in this sandbox rule set.
        gap = any(phrase in lower for phrase in ("what is a denominator", "what is a common denominator", "what does denominator mean", "don't understand fractions", "what is a numerator"))
        # Detect explicit references to arithmetic slips.
        slip = any(phrase in lower for phrase in ("calculation mistake", "calculation error", "multiplied incorrectly", "arithmetic error", "math mistake"))
        # Treat a rule-matched foundation question as a prerequisite check in the sandbox.
        if gap:
            # Choose the corresponding foundation concept.
            cid = "fraction_equivalence" if "common denominator" in lower else "fraction_basics"
            # Reload the taxonomy entry for that foundation.
            kc = FRACTION_TAXONOMY[cid]
            # Record the prerequisite-gap interpretation with neutral direction.
            category, hypothesis, confidence, polarity = "prerequisite_gap", "fraction_basics", 0.8, "neutral"
            # Explain the reason for this classification.
            claim = "The learner explicitly asks about a prerequisite fraction definition."
        # Treat explicit guessing as unverified reasoning.
        elif guess:
            # Set the guessing label and its rule-based confidence.
            category, hypothesis, confidence = "guessing", "unverified_guess", 0.6
            # Describe the reason for keeping the estimate unsupported.
            claim = "The learner expresses guessing or uncertainty; reasoning remains unverified."
        # Hold the default interpretation on negated or historical phrasing.
        elif negated:
            # A negated or historical description is not evidence of a current rule.
            pass
        # Detect independent addition as a possible misconception in fraction addition.
        elif cid in {"fraction_addition_like_denominators", "fraction_addition_unlike_denominators"} and component and interaction.correctness != 1:
            # Assign the fraction misconception that matches the task and negative direction.
            category, hypothesis, confidence, polarity = "misconception", "denominator_addition_rule" if cid.endswith("unlike_denominators") else "like_denominator_addition", 0.91, "contradicts"
            # Summarize the incorrect rule expressed by the learner.
            claim = "The learner describes independently adding numerators and denominators."
        # Identify an arithmetic slip when supporting method wording is present.
        elif slip and ("denominator" in lower or "multip" in lower):
            # Record a calculation-slip category with neutral evidence polarity.
            category, hypothesis, confidence, polarity = "calculation_slip", "calculation_slip", 0.88, "neutral"
            # Describe the method and separately acknowledge the error.
            claim = "The learner describes the common-denominator method and an arithmetic slip."
        # Recognize affirmative common-denominator reasoning with no conflicting cues.
        elif common and interaction.correctness != 0 and not component:
            # Record support for the concept using the corresponding understanding rule.
            category, hypothesis, confidence, polarity = "mastery", "fraction_common_denominator_mastery", 0.89, "supports"
            # Explain the method actually mentioned in the answer.
            claim = "The learner explains equivalent fractions or common-denominator reasoning."
            # Check whether the explanation challenges the primary unlike-denominator
            # misconception.
            if cid == "fraction_addition_unlike_denominators":
                # Link that specific misconception as a contradiction target.
                targets = ["denominator_addition_rule"]
        # Detect additive changes being used as a proposed equivalence rule.
        elif cid == "fraction_equivalence" and any(phrase in lower for phrase in ("add one to both", "added 1 to both", "adding the same number")) and interaction.correctness != 1:
            # Record the matching equivalence misconception.
            category, hypothesis, confidence, polarity = "misconception", "additive_equivalence", 0.85, "contradicts"
            # Summarize the incorrect equivalence rule.
            claim = "The learner describes additive changes as preserving fraction equivalence."
        # Recognize sound reasoning when adding fractions with equal denominators.
        elif cid == "fraction_addition_like_denominators" and "keep the denominator" in lower and "numerator" in lower and interaction.correctness != 0:
            # Credit the matching like-denominator method.
            category, hypothesis, confidence, polarity = "mastery", "like_denominator_mastery", 0.85, "supports"
            # Explain the actual credited reasoning.
            claim = "The learner explains adding numerators while keeping the common denominator."
            # Mark the specific competing misconception that the explanation contradicts.
            targets = ["like_denominator_addition"]
        # Recognize a correct explanation of fraction parts and the whole.
        elif cid == "fraction_basics" and "equal parts" in lower and "denominator" in lower and ("numerator" in lower or "whole" in lower) and interaction.correctness != 0:
            # Record positive evidence for the fraction representation concept.
            category, hypothesis, confidence, polarity = "mastery", "fraction_representation_mastery", 0.8, "supports"
            # Summarize the part-whole reasoning.
            claim = "The learner describes numerator and denominator using equal parts of a whole."
            # Record the specific fraction representation misconception contradicted.
            targets = ["numerator_denominator_inversion"]
        # Recognize explanations of equivalent fractions that scale by one factor.
        elif cid == "fraction_equivalence" and "same factor" in lower and interaction.correctness != 0:
            # Record support for the equivalence concept.
            category, hypothesis, confidence, polarity = "mastery", "fraction_equivalence_mastery", 0.85, "supports"
            # Explain the feature of the answer that supports equivalence.
            claim = "The learner explains scaling both fraction parts by the same factor."
            # Link its explicit contrast with the additive equivalence misconception.
            targets = ["additive_equivalence"]
        # A selected option alone is an observation; it does not establish reasoning.
        # Keep exact response text if available, or the selected option as the source artifact.
        quote = raw or interaction.selected_option or ""
        # Return structured evidence with a quote, claim, concept and interpretation.
        return CognitiveEvidence(
            # Derive a repeatable rule evidence ID from the interaction and rule version.
            evidence_id="E_" + uuid.uuid5(uuid.NAMESPACE_URL, interaction.interaction_id + ":rules-v3").hex[:24],
            # Preserve the learner, interaction and original time for provenance.
            interaction_id=interaction.interaction_id, learner_id=interaction.learner_id, timestamp=interaction.timestamp,
            # Distinguish a written explanation from an option selection.
            source_type="student_explanation" if raw else "answer_choice",
            # Separate the source quote and extracted claim in the persistent record.
            source_quote=quote, is_verbatim_quote=bool(quote), extracted_claim=claim,
            # Report polarity along with uncertainty inferred by these cues.
            polarity=polarity, expressed_uncertainty="high" if category in {"ambiguous", "guessing"} else "none",
            # Ground the interpretation to the known curriculum concept.
            concept_links=[ConceptLink(concept_id=cid, concept_name=kc.concept_name, confidence=0.95, grounding_rationale="Canonical concept from validated item metadata.")],
            # Save the tentative hypothesis, category, wording and confidence together.
            candidate_interpretation=CandidateInterpretation(hypothesis_id=hypothesis, category=category, description=claim, hypothesis_confidence=confidence),
            # Preserve contradiction targets and the versions of the extractor and schema.
            contradicted_hypothesis_ids=targets, extractor_model="rule-based-baseline-v3", prompt_version="rules-v3", schema_version="3.0",
        )


# Wrap the optional pretrained fraction extractor around the same evidence contract.
class InstructorEvidenceExtractor:
    """
    Schema-constrained extractor wrapping an LLM client via Instructor.
    Validates generated schema, taxonomy IDs, and exact source quotes before use.
    """

    # Accept a provider client and deployment name for optional inference.
    def __init__(self, client=None, model_name: str = "gpt-4o-mini"):
        self.client = client
        self.model_name = model_name
        self.rule_fallback = RuleBasedEvidenceExtractor()

    # Interpret the fraction response using the structured provider adapter.
    def extract(self, interaction: Interaction, kc_taxonomy_context: str = "") -> CognitiveEvidence:
        # Use deterministic extraction immediately when the model client is unavailable.
        if self.client is None:
            # Seamless fallback to rule-based extractor when LLM client is not connected
            return self.rule_fallback.extract(interaction)

        import instructor
        from ..grounding.taxonomy import FRACTION_TAXONOMY
        from ..grounding.matcher import SemanticTaxonomyMatcher

        # Adapt the model client to return a schema-validated object.
        instructor_client = instructor.from_openai(self.client)

        # Supply the built-in taxonomy context when the caller did not provide context.
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

        # Form the model prompt from the current item, explanation and available fraction context.
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
        # Reject non-verbatim evidence, duplicate links and concepts outside the fraction taxonomy.
        if not evidence.is_verbatim_quote or not evidence.validate_against_raw_text(raw) or len(ids) != len(set(ids)) or any(cid not in FRACTION_TAXONOMY for cid in ids):
            # Return conservative rule-based evidence when those checks fail.
            return self.rule_fallback.extract(interaction)
        # Ground the accepted concept links to known curriculum labels.
        for link in evidence.concept_links:
            link.concept_name = FRACTION_TAXONOMY[link.concept_id].concept_name
        known = {m.misconception_id for cid in ids for m in FRACTION_TAXONOMY[cid].known_misconceptions}
        # Verify that a named misconception is supported by the known taxonomy.
        if evidence.candidate_interpretation.category == "misconception" and evidence.candidate_interpretation.hypothesis_id not in known:
            # Fall back if the model chose an unknown misconception.
            return self.rule_fallback.extract(interaction)
        evidence.contradicted_hypothesis_ids = [hid for hid in evidence.contradicted_hypothesis_ids if hid in known]
        # Return the candidate only after all adapter checks passed.
        return evidence
