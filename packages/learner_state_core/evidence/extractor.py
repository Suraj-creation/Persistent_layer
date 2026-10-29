from __future__ import annotations

import re
import uuid
from typing import Optional
from .models import (
    CandidateInterpretation,
    CognitiveEvidence,
    ConceptLink,
    Interaction,
)
from ..grounding.taxonomy import FRACTION_TAXONOMY


class RuleBasedEvidenceExtractor:
    """
    Deterministic rule-based extractor for baseline benchmarking, unit tests,
    and fallback operation when offline.
    """

    def extract(self, interaction: Interaction) -> CognitiveEvidence:
        raw_text = (interaction.response_text or "").strip()
        lowered = raw_text.lower()

        # Default concept
        primary_cid = (
            interaction.item_concepts[0]
            if interaction.item_concepts
            else "fraction_addition_unlike_denominators"
        )
        kc = FRACTION_TAXONOMY.get(
            primary_cid,
            list(FRACTION_TAXONOMY.values())[-1],
        )

        # Check for calculation slip patterns (Student B)
        # e.g., "calculation mistake", "multiplied incorrectly", "arithmetic error", "know common denominator"
        slip_keywords = [
            "calculation mistake",
            "calculation error",
            "multiplied incorrectly",
            "arithmetic error",
            "math mistake",
            "know the denominators must match",
            "need a common denominator, but multiplied",
        ]
        is_slip = any(k in lowered for k in slip_keywords)

        # Check for component addition misconception (Student A)
        # e.g., "added the top and bottom", "added numerators and denominators", "add both parts"
        misc_keywords = [
            "added the top and bottom",
            "add the top numbers and the bottom numbers",
            "added the top numbers and the bottom numbers",
            "add both parts",
            "added numerators and denominators",
            "add numerators and add denominators",
        ]
        is_misc = any(k in lowered for k in misc_keywords)

        # Check for explicit mastery explanation
        mastery_keywords = [
            "found a common denominator",
            "converted to common denominator",
            "common denominator is",
            "equivalent fractions first",
            "multiply top and bottom by",
        ]
        is_mastery = any(k in lowered for k in mastery_keywords) or interaction.correctness == 1

        if is_slip:
            quote_match = next((k for k in slip_keywords if k in lowered), raw_text[:50])
            # Find actual casing from raw text
            start_idx = lowered.find(quote_match)
            actual_quote = raw_text[start_idx : start_idx + len(quote_match)] if start_idx != -1 else raw_text

            return CognitiveEvidence(
                evidence_id=f"E_SLIP_{uuid.uuid4().hex[:6]}",
                interaction_id=interaction.interaction_id,
                learner_id=interaction.learner_id,
                source_type="student_explanation",
                source_quote=actual_quote,
                is_verbatim_quote=True,
                extracted_claim="Learner understands the common denominator requirement but committed an arithmetic calculation slip.",
                polarity="neutral",
                expressed_uncertainty="low",
                concept_links=[
                    ConceptLink(
                        concept_id=kc.concept_id,
                        concept_name=kc.concept_name,
                        confidence=0.92,
                        grounding_rationale="Interaction explicitly concerns fraction addition with unlike denominators.",
                    )
                ],
                candidate_interpretation=CandidateInterpretation(
                    hypothesis_id="calculation_slip",
                    category="calculation_slip",
                    description="Arithmetic calculation slip during common denominator multiplication.",
                    hypothesis_confidence=0.88,
                ),
                extractor_model="rule-based-baseline-v2",
            )

        elif is_misc:
            quote_match = next((k for k in misc_keywords if k in lowered), raw_text[:50])
            start_idx = lowered.find(quote_match)
            actual_quote = raw_text[start_idx : start_idx + len(quote_match)] if start_idx != -1 else raw_text

            return CognitiveEvidence(
                evidence_id=f"E_MISC_{uuid.uuid4().hex[:6]}",
                interaction_id=interaction.interaction_id,
                learner_id=interaction.learner_id,
                source_type="student_explanation",
                source_quote=actual_quote,
                is_verbatim_quote=True,
                extracted_claim="Learner describes adding numerators together and denominators together as separate components.",
                polarity="contradicts",  # Contradicts correct conceptual understanding
                expressed_uncertainty="none",
                concept_links=[
                    ConceptLink(
                        concept_id=kc.concept_id,
                        concept_name=kc.concept_name,
                        confidence=0.95,
                        grounding_rationale="Directly applies invalid component-addition rule to unlike fractions.",
                    )
                ],
                candidate_interpretation=CandidateInterpretation(
                    hypothesis_id="denominator_addition_rule",
                    category="misconception",
                    description="Systematic rule error: adding fractions by adding both numerators and denominators independently.",
                    hypothesis_confidence=0.91,
                ),
                extractor_model="rule-based-baseline-v2",
            )

        elif is_mastery:
            actual_quote = raw_text if raw_text else f"Option {interaction.selected_option} (Correct answer)"
            return CognitiveEvidence(
                evidence_id=f"E_MAST_{uuid.uuid4().hex[:6]}",
                interaction_id=interaction.interaction_id,
                learner_id=interaction.learner_id,
                source_type="student_explanation" if raw_text else "answer_choice",
                source_quote=actual_quote,
                is_verbatim_quote=bool(raw_text),
                extracted_claim="Learner demonstrated understanding of common denominators in fraction addition.",
                polarity="supports",
                expressed_uncertainty="none",
                concept_links=[
                    ConceptLink(
                        concept_id=kc.concept_id,
                        concept_name=kc.concept_name,
                        confidence=0.90,
                        grounding_rationale="Correct explanation/response to fraction addition item.",
                    )
                ],
                candidate_interpretation=CandidateInterpretation(
                    hypothesis_id="fraction_common_denominator_mastery",
                    category="mastery",
                    description="Valid conceptual execution of finding equivalent fractions with common denominators.",
                    hypothesis_confidence=0.89,
                ),
                extractor_model="rule-based-baseline-v2",
            )

        else:
            # Ambiguous or guessing
            return CognitiveEvidence(
                evidence_id=f"E_AMBIG_{uuid.uuid4().hex[:6]}",
                interaction_id=interaction.interaction_id,
                learner_id=interaction.learner_id,
                source_type="student_explanation" if raw_text else "answer_choice",
                source_quote=raw_text if raw_text else "No explanation provided",
                is_verbatim_quote=bool(raw_text),
                extracted_claim="Insufficient detail to definitively characterize conceptual reasoning.",
                polarity="ambiguous",
                expressed_uncertainty="high",
                concept_links=[
                    ConceptLink(
                        concept_id=kc.concept_id,
                        concept_name=kc.concept_name,
                        confidence=0.75,
                        grounding_rationale="Mapped from item taxonomy metadata.",
                    )
                ],
                candidate_interpretation=CandidateInterpretation(
                    hypothesis_id="ambiguous_unspecified",
                    category="ambiguous",
                    description="Ambiguous explanation or unassisted guess.",
                    hypothesis_confidence=0.40,
                ),
                extractor_model="rule-based-baseline-v2",
            )


class InstructorEvidenceExtractor:
    """
    Schema-constrained extractor wrapping an LLM client via Instructor.
    Guarantees that the LLM generation conforms 100% to CognitiveEvidence schema.
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
3. Classify category strictly as 'misconception', 'calculation_slip', 'mastery', or 'prerequisite_gap'.
4. Ground to the canonical curriculum concept_id.
5. Estimate calibrated confidence.
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

        # Enforce exact provenance linkage from source interaction
        evidence.learner_id = interaction.learner_id
        evidence.interaction_id = interaction.interaction_id

        # Grounding post-processing: enforce valid taxonomy concepts
        valid_concepts = set(FRACTION_TAXONOMY.keys())
        matcher = SemanticTaxonomyMatcher(FRACTION_TAXONOMY, use_embeddings=False)

        grounded_links = []
        for cl in evidence.concept_links:
            if cl.concept_id in valid_concepts:
                grounded_links.append(cl)
            else:
                # Map using semantic matcher
                matched = matcher.match(cl.concept_name or cl.grounding_rationale or interaction.item_text, top_k=1)
                if matched:
                    grounded_links.append(matched[0])
                elif interaction.item_concepts:
                    grounded_links.append(
                        ConceptLink(
                            concept_id=interaction.item_concepts[0],
                            concept_name=FRACTION_TAXONOMY.get(interaction.item_concepts[0], KnowledgeComponent(concept_id=interaction.item_concepts[0], concept_name=interaction.item_concepts[0], description="")).concept_name,
                            confidence=0.90,
                            grounding_rationale="Mapped from item concepts taxonomy fallback.",
                        )
                    )

        if not grounded_links and interaction.item_concepts:
            grounded_links.append(
                ConceptLink(
                    concept_id=interaction.item_concepts[0],
                    concept_name="Adding Fractions with Unlike Denominators",
                    confidence=0.90,
                    grounding_rationale="Fallback item taxonomy mapping.",
                )
            )

        evidence.concept_links = grounded_links

        # Misconception canonicalization
        if evidence.candidate_interpretation.category == "misconception":
            if "denominator_addition" in evidence.candidate_interpretation.hypothesis_id or "independent" in evidence.candidate_interpretation.description.lower() or "add" in evidence.extracted_claim.lower():
                evidence.candidate_interpretation.hypothesis_id = "denominator_addition_rule"
                evidence.candidate_interpretation.description = "Independent Numerator and Denominator Addition"
        elif evidence.candidate_interpretation.category == "calculation_slip":
            evidence.candidate_interpretation.hypothesis_id = "calculation_slip"
            evidence.candidate_interpretation.description = "Arithmetic Calculation Slip"

        # Enforce quote veracity check
        if interaction.response_text and not evidence.validate_against_raw_text(interaction.response_text):
            evidence.is_verbatim_quote = False

        return evidence
