from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class MisconceptionDefinition(BaseModel):
    misconception_id: str
    name: str
    description: str
    typical_distractor_pattern: str
    remediation_hint: str


class KnowledgeComponent(BaseModel):
    concept_id: str
    concept_name: str
    domain: str = "Mathematics"
    subdomain: str = "Fractions"
    grade_level: str = "Grade 4-6"
    description: str
    prerequisite_concept_ids: List[str] = Field(default_factory=list)
    known_misconceptions: List[MisconceptionDefinition] = Field(default_factory=list)


# Reference curriculum taxonomy for fraction operations
FRACTION_TAXONOMY: Dict[str, KnowledgeComponent] = {
    "fraction_basics": KnowledgeComponent(
        concept_id="fraction_basics",
        concept_name="Fraction Representation & Parts",
        description="Understanding fractions as part of a whole, numerators as count, denominators as unit size.",
        prerequisite_concept_ids=[],
        known_misconceptions=[
            MisconceptionDefinition(
                misconception_id="numerator_denominator_inversion",
                name="Numerator Denominator Inversion",
                description="Confusing the role of numerator and denominator.",
                typical_distractor_pattern="Treating denominator as count.",
                remediation_hint="Remember the denominator defines the number of equal slices.",
            )
        ],
    ),
    "fraction_equivalence": KnowledgeComponent(
        concept_id="fraction_equivalence",
        concept_name="Equivalent Fractions",
        description="Recognizing and generating equivalent fractions by multiplying or dividing numerator and denominator by the same non-zero number.",
        prerequisite_concept_ids=["fraction_basics"],
        known_misconceptions=[
            MisconceptionDefinition(
                misconception_id="additive_equivalence",
                name="Additive Equivalence Fallacy",
                description="Believing adding the same number to numerator and denominator creates an equivalent fraction (e.g. 1/2 = 2/3).",
                typical_distractor_pattern="Adding 1 to both parts.",
                remediation_hint="Equivalence requires multiplying or dividing both parts by the same factor.",
            )
        ],
    ),
    "fraction_addition_like_denominators": KnowledgeComponent(
        concept_id="fraction_addition_like_denominators",
        concept_name="Adding Fractions with Like Denominators",
        description="Adding fractions with the same denominator by adding numerators and keeping the common denominator.",
        prerequisite_concept_ids=["fraction_basics"],
        known_misconceptions=[
            MisconceptionDefinition(
                misconception_id="like_denominator_addition",
                name="Adding Denominators in Like Fractions",
                description="Adding denominators even when they are identical (e.g. 1/4 + 2/4 = 3/8).",
                typical_distractor_pattern="Adding denominators together.",
                remediation_hint="When denominators are the same, the slice size does not change.",
            )
        ],
    ),
    "fraction_addition_unlike_denominators": KnowledgeComponent(
        concept_id="fraction_addition_unlike_denominators",
        concept_name="Adding Fractions with Unlike Denominators",
        description="Finding common denominators using equivalent fractions and then adding the numerators.",
        prerequisite_concept_ids=["fraction_equivalence", "fraction_addition_like_denominators"],
        known_misconceptions=[
            MisconceptionDefinition(
                misconception_id="denominator_addition_rule",
                name="Independent Numerator and Denominator Addition",
                description="Adding numerators and denominators independently (e.g. a/b + c/d = (a+c)/(b+d), such as 1/2 + 1/3 = 2/5).",
                typical_distractor_pattern="Direct component addition.",
                remediation_hint="Fractions cannot be added until they have the same unit denominator.",
            ),
            MisconceptionDefinition(
                misconception_id="cross_multiplication_addition_confusion",
                name="Cross Multiplication Confusion in Addition",
                description="Attempting to cross-multiply when adding fractions without using common denominators.",
                typical_distractor_pattern="Confusing fraction division/proportion rules with addition.",
                remediation_hint="Cross-multiplication is for solving proportions or dividing, not adding.",
            ),
            MisconceptionDefinition(
                misconception_id="calculation_slip",
                name="Arithmetic Calculation Slip",
                description="Correctly identifying the need for common denominators, but committing an arithmetic error.",
                typical_distractor_pattern="Off-by-one or small multiplication error in numerator.",
                remediation_hint="Double-check your multiplication when finding the common numerator.",
            ),
        ],
    ),
}
