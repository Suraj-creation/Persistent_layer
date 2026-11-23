from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from ..state.models import ConceptState, LearnerStateSnapshot


# Describe a possible learner selection and its two hand-set likelihoods.
class DiagnosticOption(BaseModel):
    option_key: str  # "A", "B", "C", "D"
    option_text: str
    is_correct: bool
    targeted_misconception_id: Optional[str] = None
    expected_probability_given_misconception: float = Field(0.85, ge=0, le=1)
    expected_probability_given_mastery: float = Field(0.05, ge=0, le=1)


# Describe a curated diagnostic question, options and intended concept.
class DiagnosticItem(BaseModel):
    item_id: str
    concept_id: str
    question_text: str
    options: List[DiagnosticOption]
    diagnostic_rationale: str

    @model_validator(mode="after")
    # Check that multiple-choice items can be compared by the diagnosis policy.
    def validate_choices(self):
        if not self.options:
            # Free-response clarification prompts have no categorical likelihoods.
            return self
        if len(self.options) < 2:
            raise ValueError("Multiple-choice diagnostics require at least two options")
        if len({option.option_key for option in self.options}) != len(self.options):
            raise ValueError("Diagnostic option keys must be unique")
        if sum(option.is_correct for option in self.options) != 1:
            raise ValueError("Diagnostic items require exactly one correct option")
        if sum(option.expected_probability_given_mastery for option in self.options) <= 0 or sum(option.expected_probability_given_misconception for option in self.options) <= 0:
            raise ValueError("Diagnostic likelihoods require positive mass")
        return self


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


# Rank the fraction sandbox's curated probes by Expected Information Gain.
class ActiveDiagnosisProber:
    """
    Evaluates candidate diagnostic items using Expected Information Gain (EIG).
    Selects the probe that maximally reduces entropy over competing cognitive hypotheses.
    """

    # Accept a test-specific item bank or use the curated fraction bank.
    def __init__(self, item_bank: Optional[List[DiagnosticItem]] = None):
        self.item_bank = item_bank if item_bank is not None else FRACTION_DIAGNOSTIC_BANK

    @staticmethod
    # Measure how mixed a probability distribution is in bits.
    def _entropy(probs: List[float]) -> float:
        """Computes Shannon entropy in bits for a discrete distribution."""
        # Start accumulating the entropy at zero.
        h = 0.0
        # Inspect each probability in the supplied distribution.
        for p in probs:
            # Ignore entries too small to contribute reliably to the logarithm.
            if p > 1e-6:
                # Add the Shannon entropy contribution for this probability.
                h -= p * math.log2(p)
        # Return the total uncertainty in bits.
        return h

    # Choose the unused multiple-choice task that best distinguishes the two hypotheses.
    def select_best_diagnostic_probe(
        self,
        concept_id: str,
        concept_state: ConceptState,
        excluded_item_ids: Optional[List[str]] = None,
    ) -> Optional[Tuple[DiagnosticItem, float, str]]:
        """
        Finds the item in the bank with the highest Expected Information Gain for this concept.
        Returns: (DiagnosticItem, EIG_score, rationale) or None if no items exist.
        """
        # Restrict candidates to the selected concept, available options and unanswered item IDs.
        candidate_items = sorted([item for item in self.item_bank if item.options and item.concept_id == concept_id and item.item_id not in (excluded_item_ids or [])], key=lambda item: item.item_id)
        # Check whether the learner has any eligible probe left.
        if not candidate_items:
            # Return no recommendation if the filtered bank is empty.
            return None

        # Prior distribution over states: P(Mastery) vs. P(Misconception_i) vs. P(Other)
        # Use the saved concept mastery estimate as one prior weight.
        p_mastery = concept_state.mastery_estimate
        # Find misconception hypotheses that still have active support.
        active_misconceptions = [belief for belief in concept_state.candidate_misconceptions.values() if belief.status in {"suspected", "confirmed"}]

        # Use the saved hypothesis weights when the concept has active beliefs.
        if active_misconceptions:
            # Average the active belief scores for the other prior weight.
            p_misc = sum(m.posterior_probability for m in active_misconceptions) / len(active_misconceptions)
        else:
            # Use a hand-set background weight when there is no active candidate.
            p_misc = 0.20  # Default background prior

        # Normalize prior
        # Add the two weights before normalizing.
        total = p_mastery + p_misc
        # Turn the mastery and misconception weights into a two-way distribution.
        prior_dist = [p_mastery / total, p_misc / total] if total > 0 else [0.5, 0.5]
        # Measure current uncertainty before asking a question.
        h_prior = self._entropy(prior_dist)

        # Start with no recommended item.
        best_item: Optional[DiagnosticItem] = None
        # Permit even a zero-gain item to become the first candidate.
        best_eig = -1.0
        # Save an explanation alongside the best candidate.
        best_rationale = ""

        # Simulate each eligible item's possible answers.
        for item in candidate_items:
            # Estimate expected posterior entropy across outcomes
            # Outcome 1: Student chooses distractor matching target misconception
            # Outcome 2: Student chooses correct answer
            # Outcome 3: Student chooses other error
            # Start its average remaining uncertainty at zero.
            exp_post_entropy = 0.0

            # Sum choice weights under the mastery hypothesis.
            mastery_mass = sum(option.expected_probability_given_mastery for option in item.options)
            # Sum choice weights under the misconception hypothesis separately.
            misc_mass = sum(option.expected_probability_given_misconception for option in item.options)
            # Evaluate what could be inferred if the learner chose each option.
            for opt in item.options:
                # Normalize the option's likelihood under mastery.
                likelihood_mastery = opt.expected_probability_given_mastery / mastery_mass
                # Normalize the option's likelihood under a misconception.
                likelihood_misc = opt.expected_probability_given_misconception / misc_mass
                # Probability of choosing this option: P(opt) = P(opt|mastery)*P(mastery) + P(opt|misc)*P(misc)
                # Compute how likely the option is across both prior hypotheses.
                p_opt = (
                    # Add the probability that a learner with mastery would choose it.
                    likelihood_mastery * prior_dist[0]
                    # Add the probability that a learner with the misconception would choose it.
                    + likelihood_misc * prior_dist[1]
                )

                # Ignore outcomes too unlikely for a stable update.
                if p_opt < 1e-4:
                    # Continue to the next choice without dividing by a near-zero probability.
                    continue

                # Posterior P(misc | opt) via Bayes rule
                # Apply Bayes' rule to estimate the misconception weight after this possible
                # answer.
                post_misc = (likelihood_misc * prior_dist[1]) / p_opt
                # Calculate the corresponding mastery weight.
                post_mastery = (likelihood_mastery * prior_dist[0]) / p_opt
                # Form the updated two-way hypothesis distribution for this option.
                post_dist = [post_mastery, post_misc]

                # Measure how much uncertainty would remain after this outcome.
                h_post = self._entropy(post_dist)
                # Weight that uncertainty by how often this option is predicted to occur.
                exp_post_entropy += p_opt * h_post

            # Expected Information Gain is current entropy minus average remaining entropy.
            eig = max(0.0, h_prior - exp_post_entropy)

            # Keep the option bank item that would reduce uncertainty the most.
            if eig > best_eig:
                # Record the winning expected gain in bits.
                best_eig = eig
                # Save the selected question.
                best_item = item
                # Explain the expected gain and the intended cognitive distinction.
                best_rationale = (
                    f"Selected item '{item.item_id}' with Expected Information Gain of {eig:.3f} bits. "
                    f"Maximally discriminates between mastery and '{item.options[0].targeted_misconception_id}'."
                )

        # Handle the case where no eligible item was scored.
        if best_item is None:
            # Return no recommendation in that case.
            return None

        # Return the best candidate together with its score and rationale.
        return best_item, best_eig, best_rationale
