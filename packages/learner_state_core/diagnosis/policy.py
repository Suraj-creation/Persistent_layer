"""Versioned, deterministic question policy. No generative tutor call is used."""
from .prober import ActiveDiagnosisProber, DiagnosticItem, DiagnosticOption, FRACTION_DIAGNOSTIC_BANK

# Set the main concept for the compatibility fraction conversation.
PRIMARY_CONCEPT = "fraction_addition_unlike_denominators"
# Label the deterministic follow-up rules in the turn trace.
POLICY_VERSION = "grounded-question-policy-v1"


def option(key, text, correct=False):
    return DiagnosticOption(option_key=key, option_text=text, is_correct=correct)


# Define the initial fraction problem used in the older workspace.
INITIAL_ITEM = DiagnosticItem(
    item_id="ITEM_FRAC_01", concept_id=PRIMARY_CONCEPT,
    question_text="What is 1/2 + 1/3? Explain how you worked it out.",
    options=[option("A", "2/5"), option("B", "3/5"), option("C", "4/6"), option("D", "5/6", True)],
    diagnostic_rationale="Initial observation; an answer alone does not establish conceptual mastery.",
)

# Define curated prerequisite checks, a simple addition task and a transfer question.
EXTRA_ITEMS = [
    DiagnosticItem(item_id="DIAG_BASICS_01", concept_id="fraction_basics",
                   question_text="In 3/4, what does the 4 tell us about the whole? Explain in your own words.",
                   options=[], diagnostic_rationale="Clarify the meaning of a denominator before inferring an addition-rule difficulty."),
    DiagnosticItem(item_id="DIAG_EQUIV_01", concept_id="fraction_equivalence",
                   question_text="How would you rewrite 1/2 in sixths without changing its value? Explain what happens to both numbers.",
                   options=[option("A", "1/6"), option("B", "3/6", True), option("C", "2/6")],
                   diagnostic_rationale="Check fraction equivalence, a prerequisite for unlike-denominator addition."),
    DiagnosticItem(item_id="DIAG_LIKE_01", concept_id="fraction_addition_like_denominators",
                   question_text="What is 1/6 + 2/6? Explain whether the size of the parts changes.",
                   options=[option("A", "3/12"), option("B", "3/6", True), option("C", "2/6")],
                   diagnostic_rationale="Check whether equal units remain equal when combining fractions."),
    DiagnosticItem(item_id="DIAG_TRANSFER_01", concept_id=PRIMARY_CONCEPT,
                   question_text="What is 1/3 + 1/6? Show your equivalent fractions before adding.",
                   options=[option("A", "2/9"), option("B", "1/2", True), option("C", "1/9")],
                   diagnostic_rationale="Collect a separate observation to check whether the learner's approach transfers."),
]
# Index all known compatibility questions by ID.
ITEM_BANK = {i.item_id: i for i in [INITIAL_ITEM, *FRACTION_DIAGNOSTIC_BANK, *EXTRA_ITEMS]}


# Choose a follow-up from persisted fraction state, accepted evidence and attempted tasks.
def choose_followup(snapshot, evidence, attempted_ids):
    """Inputs must be reloaded from SQLite; returns a curated item and rationale."""
    # Read the latest evidence category.
    category = evidence.candidate_interpretation.category
    # Find the main concept's state and uncertainty.
    primary = snapshot.concept_states[PRIMARY_CONCEPT]
    # Keep only suspected or confirmed misconceptions when deciding whether to probe.
    active = [m for m in primary.candidate_misconceptions.values() if m.status in ("suspected", "confirmed")]
    # Filter out attempted items and the default starting task.
    remaining = [i for i in ITEM_BANK.values() if i.item_id not in attempted_ids and i.item_id != INITIAL_ITEM.item_id]

    # Give prerequisite checks first priority when the saved evidence flags a foundation.
    if snapshot.active_prerequisite_gaps or category == "prerequisite_gap":
        # Try flagged prerequisites before the two common foundations.
        for cid in [*snapshot.active_prerequisite_gaps, "fraction_basics", "fraction_equivalence"]:
            # Find the next unused question on that prerequisite.
            match = next((i for i in remaining if i.concept_id == cid), None)
            # Use it immediately when one remains.
            if match:
                # Return the chosen item with a reason tied to the stored evidence.
                return match, "Stored evidence flags a candidate prerequisite gap; check its foundation before proceeding."

    # Consider a diagnostic when an active belief or high uncertainty remains.
    if active or primary.uncertainty >= .20:
        # Take unused tasks from the curated diagnostic bank.
        diagnostic = [i for i in remaining if i.item_id in {x.item_id for x in FRACTION_DIAGNOSTIC_BANK}]
        # Check that an eligible diagnostic is available.
        if diagnostic:
            # Select among them by expected reduction in uncertainty.
            result = ActiveDiagnosisProber(diagnostic).select_best_diagnostic_probe(PRIMARY_CONCEPT, primary)
            # Return the probe and explain where its likelihoods came from.
            return result[0], result[2] + " Likelihoods are hand-authored research assumptions."

    # Use the first remaining item when no targeted probe is needed.
    if remaining:
        # Take the next curated task.
        chosen = remaining[0]
        # Include the task's rationale and the use of saved history.
        return chosen, chosen.diagnostic_rationale + " Selected using saved state and previously attempted item IDs."

    # No unseen assessment items: ask for more language instead of inventing a problem.
    # Ask for clarification when all curated questions have been used.
    question = DiagnosticItem(
        item_id="CLARIFY_REASONING", concept_id=evidence.concept_links[0].concept_id,
        question_text="Explain one step of your last approach and why it preserves the value of the fraction. What part still feels uncertain?",
        options=[], diagnostic_rationale="The curated question bank is exhausted. Collect further explanation without claiming mastery.",
    )
    # Return the clarification prompt and its rationale.
    return question, question.diagnostic_rationale


# Compose a deterministic learner reply from the observation category and stored follow-up.
def grounded_reply(snapshot, evidence, question):
    # Choose the matching feedback message for this interpretation.
    categories = {
        "misconception": "Your explanation suggests a possible rule to check; one response is not enough to confirm it.",
        "calculation_slip": "Your explanation suggests a possible calculation slip. Let's check the reasoning in another example.",
        "mastery": "Your explanation provides evidence of understanding. Let's check it with another question.",
        "prerequisite_gap": "Your question suggests a foundation we should clarify before moving on.",
        "ambiguous": "I don't yet have enough evidence to judge your reasoning.",
        "guessing": "An answer alone doesn't tell me how you reasoned. Let's explore your approach.",
    }
    # Text is fixed policy copy; the exact question is stored in the decision trace.
    # Append the already selected question to that feedback.
    return categories[evidence.candidate_interpretation.category] + "\n\n" + question.question_text
