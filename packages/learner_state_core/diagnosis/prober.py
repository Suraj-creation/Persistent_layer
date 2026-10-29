from __future__ import annotations

import math
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from ..state.models import ConceptState, LearnerStateSnapshot


class DiagnosticOption(BaseModel):
    option_key: str  # "A", "B", "C", "D"
    option_text: str
    is_correct: bool
    targeted_misconception_id: Optional[str] = None
    expected_probability_given_misconception: float = 0.85
    expected_probability_given_mastery: float = 0.05


class DiagnosticItem(BaseModel):
    item_id: str
    concept_id: str
    question_text: str
    options: List[DiagnosticOption]
    diagnostic_rationale: str


# Sample curated diagnostic item bank for fractions
FRACTION_DIAGNOSTIC_BANK: List[DiagnosticItem] = [
    DiagnosticItem(
        item_id="DIAG_FRAC_01",
        concept_id="fraction_addition_unlike_denominators",
        question_text="What is 1/4 + 1/2?",
        diagnostic_rationale="Tests whether student adds denominators (yielding 2/6 = 1/3) vs. using common denominator (yielding 3/4).",
        options=[
            DiagnosticOption(
                option_key="A",
                option_text="2/6 (or 1/3)",
                is_correct=False,
                targeted_misconception_id="denominator_addition_rule",
                expected_probability_given_misconception=0.90,
                expected_probability_given_mastery=0.03,
            ),
            DiagnosticOption(
                option_key="B",
                option_text="3/4",
                is_correct=True,
                targeted_misconception_id=None,
                expected_probability_given_misconception=0.05,
                expected_probability_given_mastery=0.92,
            ),
            DiagnosticOption(
                option_key="C",
                option_text="2/4 (or 1/2)",
                is_correct=False,
                targeted_misconception_id="calculation_slip",
                expected_probability_given_misconception=0.03,
                expected_probability_given_mastery=0.04,
            ),
            DiagnosticOption(
                option_key="D",
                option_text="1/6",
                is_correct=False,
                targeted_misconception_id="cross_multiplication_addition_confusion",
                expected_probability_given_misconception=0.02,
                expected_probability_given_mastery=0.01,
            ),
        ],
    ),
    DiagnosticItem(
        item_id="DIAG_FRAC_02",
        concept_id="fraction_addition_unlike_denominators",
        question_text="Which step is required to solve 2/3 + 1/5?",
        diagnostic_rationale="Tests conceptual awareness of finding a common unit denominator before performing addition.",
        options=[
            DiagnosticOption(
                option_key="A",
                option_text="Add 2+1 on top and 3+5 on bottom to get 3/8.",
                is_correct=False,
                targeted_misconception_id="denominator_addition_rule",
                expected_probability_given_misconception=0.88,
                expected_probability_given_mastery=0.02,
            ),
            DiagnosticOption(
                option_key="B",
                option_text="Find a common denominator of 15, converting fractions to 10/15 and 3/15.",
                is_correct=True,
                targeted_misconception_id=None,
                expected_probability_given_misconception=0.06,
                expected_probability_given_mastery=0.94,
            ),
            DiagnosticOption(
                option_key="C",
                option_text="Multiply 2 by 5 and 3 by 1 and leave the denominators as 15.",
                is_correct=False,
                targeted_misconception_id="cross_multiplication_addition_confusion",
                expected_probability_given_misconception=0.04,
                expected_probability_given_mastery=0.03,
            ),
        ],
    ),
]


class ActiveDiagnosisProber:
    """
    Evaluates candidate diagnostic items using Expected Information Gain (EIG).
    Selects the probe that maximally reduces entropy over competing cognitive hypotheses.
    """

    def __init__(self, item_bank: Optional[List[DiagnosticItem]] = None):
        self.item_bank = item_bank or FRACTION_DIAGNOSTIC_BANK

    @staticmethod
    def _entropy(probs: List[float]) -> float:
        """Computes Shannon entropy in bits for a discrete distribution."""
        h = 0.0
        for p in probs:
            if p > 1e-6:
                h -= p * math.log2(p)
        return h

    def select_best_diagnostic_probe(
        self,
        concept_id: str,
        concept_state: ConceptState,
    ) -> Optional[Tuple[DiagnosticItem, float, str]]:
        """
        Finds the item in the bank with the highest Expected Information Gain for this concept.
        Returns: (DiagnosticItem, EIG_score, rationale) or None if no items exist.
        """
        candidate_items = [item for item in self.item_bank if item.concept_id == concept_id]
        if not candidate_items:
            return None

        # Prior distribution over states: P(Mastery) vs. P(Misconception_i) vs. P(Other)
        p_mastery = concept_state.mastery_estimate
        active_misconceptions = list(concept_state.candidate_misconceptions.values())

        if active_misconceptions:
            p_misc = sum(m.posterior_probability for m in active_misconceptions) / len(active_misconceptions)
        else:
            p_misc = 0.20  # Default background prior

        # Normalize prior
        total = p_mastery + p_misc
        prior_dist = [p_mastery / total, p_misc / total]
        h_prior = self._entropy(prior_dist)

        best_item: Optional[DiagnosticItem] = None
        best_eig = -1.0
        best_rationale = ""

        for item in candidate_items:
            # Estimate expected posterior entropy across outcomes
            # Outcome 1: Student chooses distractor matching target misconception
            # Outcome 2: Student chooses correct answer
            # Outcome 3: Student chooses other error
            exp_post_entropy = 0.0

            for opt in item.options:
                # Probability of choosing this option: P(opt) = P(opt|mastery)*P(mastery) + P(opt|misc)*P(misc)
                p_opt = (
                    opt.expected_probability_given_mastery * prior_dist[0]
                    + opt.expected_probability_given_misconception * prior_dist[1]
                )

                if p_opt < 1e-4:
                    continue

                # Posterior P(misc | opt) via Bayes rule
                post_misc = (opt.expected_probability_given_misconception * prior_dist[1]) / p_opt
                post_mastery = (opt.expected_probability_given_mastery * prior_dist[0]) / p_opt
                post_dist = [max(0.01, min(0.99, post_mastery)), max(0.01, min(0.99, post_misc))]
                norm = sum(post_dist)
                post_dist = [p / norm for p in post_dist]

                h_post = self._entropy(post_dist)
                exp_post_entropy += p_opt * h_post

            eig = max(0.0, h_prior - exp_post_entropy)

            if eig > best_eig:
                best_eig = eig
                best_item = item
                best_rationale = (
                    f"Selected item '{item.item_id}' with Expected Information Gain of {eig:.3f} bits. "
                    f"Maximally discriminates between mastery and '{item.options[0].targeted_misconception_id}'."
                )

        if best_item is None:
            return None

        return best_item, best_eig, best_rationale
