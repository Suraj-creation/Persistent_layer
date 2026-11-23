from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from ..evidence.models import CognitiveEvidence, Interaction
from .models import ConceptState, LearnerStateSnapshot, MisconceptionBelief, StateChangeEvent


# Calculate the fraction sandbox's Beta mean and standard deviation.
def compute_beta_stats(alpha: float, beta: float) -> Tuple[float, float]:
    """Computes mean mastery and uncertainty standard deviation for Beta(alpha, beta)."""
    # Divide the positive mass by the sum of both evidence masses.
    mean = alpha / (alpha + beta)
    # Compute the Beta distribution variance from those masses.
    var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    # Use a small variance floor before taking the square root.
    sd = math.sqrt(max(1e-6, var))
    # Return both statistics rounded to five decimal places.
    return round(mean, 5), round(sd, 5)


# Derive document-independent IDs used in deterministic replay.
def _stable_identifier(prefix: str, *parts: str) -> str:
    """Create repeatable IDs for replayed state transitions and audit events."""
    # Join the identifying fields in a stable order with a separator.
    name = "\x1f".join(parts)
    # Generate a repeatable namespaced UUID and shorten it for display.
    return f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, name).hex[:16]}"


# Fingerprint the prior state so a transition can identify its exact input.
def _snapshot_fingerprint(snapshot: LearnerStateSnapshot) -> str:
    """Hash the state content, excluding presentation IDs and wall-clock creation time."""
    # Exclude the snapshot identifier and runtime timestamp from the fingerprint.
    payload = snapshot.model_dump(mode="json", exclude={"snapshot_id", "timestamp"})
    # Serialize with sorted keys and a stable layout.
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    # Hash the canonical state content.
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Apply the hand-weighted evidence rules for the fraction sandbox and compatibility paths.
class BayesianLearnerStateEngine:
    """
    Deterministic heuristic state engine with Beta-distribution evidence masses.
    Updates learner conceptual beliefs based on structured Cognitive Evidence.
    
    Guarantees:
    - Deterministic transitions for the same state, evidence, and policy.
    - Beta-distribution summaries use E[L] = alpha / (alpha + beta).
    - Differentiates slips from systematic misconceptions (Student A vs. Student B).
    - Resolves contradictory evidence without discarding history.
    - Preserves evidence provenance and records auditable heuristic update traces.

    The evidence weights and misconception scores are hand-set policy heuristics.
    They are not learned parameters or calibrated Bayesian posteriors.
    """

    # Give this scoring policy a version that replay can record.
    def __init__(self, policy_version: str = "evidence-heuristic-v3.0"):
        # Store the version on the engine.
        self.policy_version = policy_version

    # Create one Beta prior per fraction curriculum concept for a new learner.
    def initialize_learner_state(self, learner_id: str, concept_ids: List[str]) -> LearnerStateSnapshot:
        """Creates an initial prior state for a new learner."""
        # Begin with an empty concept map for this learner.
        concept_states: Dict[str, ConceptState] = {}
        # Add a prior for each supplied curriculum concept.
        for cid in concept_ids:
            # Start with one positive unit of evidence mass.
            alpha = 1.0
            # Start with one negative unit of evidence mass.
            beta = 1.0
            # Compute the starting 0.5 estimate and its uncertainty.
            mean_m, sd_u = compute_beta_stats(alpha, beta)
            # Build the stored concept record with its display name and initial statistics.
            concept_states[cid] = ConceptState(
                concept_id=cid,
                concept_name=cid.replace("_", " ").title(),
                alpha=alpha,
                beta=beta,
                mastery_estimate=mean_m,
                uncertainty=sd_u,
                # Mark the new concept as having insufficient observations.
                evidence_sufficiency="insufficient",
                # Start with no misconception beliefs.
                candidate_misconceptions={},
                # Record that there has been no prior interaction.
                last_interaction_timestamp=None,
                # Start the interaction counter at zero.
                interaction_count=0,
            )

        # Assemble the complete initial fraction snapshot.
        return LearnerStateSnapshot(
            # Produce a stable starting ID from the same learner, policy and concepts.
            snapshot_id=_stable_identifier(
                "S", learner_id, self.policy_version, *sorted(concept_ids)
            ),
            learner_id=learner_id,
            # Use a fixed start timestamp for reproducible replay.
            timestamp=datetime(1970, 1, 1, tzinfo=timezone.utc),
            as_of_interaction_id="INIT",
            concept_states=concept_states,
            # Start with no candidate prerequisite gaps.
            active_prerequisite_gaps=[],
            policy_version=self.policy_version,
        )

    # Update a fraction snapshot using one structured item of cognitive evidence.
    def update_state(
        self,
        prior_snapshot: LearnerStateSnapshot,
        evidence: CognitiveEvidence,
        interaction: Optional[Interaction] = None,
    ) -> Tuple[LearnerStateSnapshot, List[StateChangeEvent]]:
        """
        Integrates a single CognitiveEvidence record into the prior state snapshot,
        producing a new LearnerStateSnapshot and corresponding StateChangeEvents with mathematical traces.
        """
        # Prevent one learner's evidence from modifying another learner's state.
        if evidence.learner_id != prior_snapshot.learner_id:
            raise ValueError("Evidence learner_id must match the prior snapshot learner_id")
        # Get the concepts named by the extractor's links.
        concept_ids = [link.concept_id for link in evidence.concept_links]
        # Require each named concept to appear only once.
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("Evidence must contain at most one concept link per concept_id")

        # Verify the optional original interaction agrees with the evidence provenance.
        if interaction and (interaction.interaction_id != evidence.interaction_id or interaction.learner_id != evidence.learner_id):
            raise ValueError("Interaction and evidence provenance must match")
        # Catch another application of the current snapshot's most recent interaction.
        if prior_snapshot.as_of_interaction_id == evidence.interaction_id:
            raise ValueError("Interaction already applied to this snapshot")
        # Copy the saved prior before calculating changes.
        new_snapshot = copy.deepcopy(prior_snapshot)
        # Label the new snapshot with the policy executing now.
        new_snapshot.policy_version = self.policy_version
        # Build a deterministic snapshot ID for this exact transition.
        new_snapshot.snapshot_id = _stable_identifier(
            "S",
            self.policy_version,
            evidence.learner_id,
            evidence.evidence_id,
            # Include the prior state's fingerprint so different starting states stay
            # distinguishable.
            _snapshot_fingerprint(prior_snapshot),
        )
        # Link the new snapshot back to the interaction just applied.
        new_snapshot.as_of_interaction_id = evidence.interaction_id
        # Use the evidence timestamp for reproducible replay.
        new_snapshot.timestamp = evidence.timestamp
        # Collect audit events that describe the policy's decisions.
        events: List[StateChangeEvent] = []

        # Retrieve the model or rule-based interpretation to classify the state update.
        interp = evidence.candidate_interpretation

        # Apply the evidence independently to each validated concept link.
        for link in evidence.concept_links:
            cid = link.concept_id
            cname = link.concept_name

            # Ensure concept exists in state
            # Add a Beta prior if a linked concept is absent from an older snapshot.
            if cid not in new_snapshot.concept_states:
                alpha_init = 1.0
                beta_init = 1.0
                mean_init, sd_init = compute_beta_stats(alpha_init, beta_init)
                new_snapshot.concept_states[cid] = ConceptState(
                    concept_id=cid,
                    concept_name=cname,
                    alpha=alpha_init,
                    beta=beta_init,
                    mastery_estimate=mean_init,
                    uncertainty=sd_init,
                    evidence_sufficiency="insufficient",
                    candidate_misconceptions={},
                    last_interaction_timestamp=evidence.timestamp,
                    interaction_count=0,
                )

            # Load the concept state that will be updated.
            cstate = new_snapshot.concept_states[cid]
            # Capture the earlier point estimate for comparison.
            prior_mastery = cstate.mastery_estimate
            # Capture its previous uncertainty for the audit event.
            prior_uncertainty = cstate.uncertainty
            # Recover the previous positive mass with a default for older records.
            prior_alpha = getattr(cstate, "alpha", 1.0)
            # Recover the previous negative mass in the same way.
            prior_beta = getattr(cstate, "beta", 1.0)

            # Count the interaction even if the rule later holds its numerical estimate.
            cstate.interaction_count += 1
            # Mark when the learner last provided evidence for this concept.
            cstate.last_interaction_timestamp = evidence.timestamp

            # Repeated guesses must not make reasoning evidence sufficient.
            # Count evidence as informative only with a substantive category and enough confidence.
            if interp.category not in {"ambiguous", "guessing"} and interp.hypothesis_confidence >= 0.5:
                cstate.informative_evidence_count += 1
            # A concept with at least five informative inputs reaches the highest evidence
            # sufficiency label.
            if cstate.informative_evidence_count >= 5:
                cstate.evidence_sufficiency = "sufficient"
            # Three or four substantive observations give moderate coverage.
            elif cstate.informative_evidence_count >= 3:
                cstate.evidence_sufficiency = "moderate"
            # One or two such observations give sparse coverage.
            elif cstate.informative_evidence_count > 0:
                cstate.evidence_sufficiency = "sparse"
            else:
                # Leave the label insufficient if there is no qualifying observation.
                cstate.evidence_sufficiency = "insufficient"

            # Start with zero numerical change on the positive side.
            delta_alpha = 0.0
            # Start with zero numerical change on the negative side.
            delta_beta = 0.0
            rule_applied = ""
            summary = ""
            likelihood_ratio = None
            prior_m_prob = None
            new_m_prob = None

            # ----------------------------------------------------
            # Case 1: Pure Calculation Slip (e.g. Student B)
            # ----------------------------------------------------
            # Handle a small arithmetic or transcription error as a light penalty.
            if interp.category == "calculation_slip":
                # Arithmetic or typographical slip: concept rule is understood
                # Very light penalty on beta, concept mastery remains largely intact
                # Multiply by the concept-link and interpretation confidences to scale the slip
                # penalty.
                delta_beta = 0.15 * link.confidence * interp.hypothesis_confidence
                # Preserve the positive mass for an arithmetic slip.
                new_alpha = prior_alpha
                # Add only the small penalty to the negative mass.
                new_beta = prior_beta + delta_beta
                # Summarize the new estimate and uncertainty for the slip.
                new_mastery, new_uncertainty = compute_beta_stats(new_alpha, new_beta)

                # Apply its positive evidence total to the returned state.
                cstate.alpha = new_alpha
                # Apply the updated negative mass.
                cstate.beta = new_beta
                # Store the recomputed sandbox mastery estimate.
                cstate.mastery_estimate = new_mastery
                # Store the resulting standard deviation.
                cstate.uncertainty = new_uncertainty
                # Name the rule so the event can identify the scoring decision.
                rule_applied = "RULE_SLIP_ATTENUATION"
                # Explain the light adjustment and its effect to the inspector.
                summary = (
                    f"Calculation slip detected. Conceptual rule preserved with minimal attenuation "
                    f"({prior_mastery:.3f} -> {new_mastery:.3f})."
                )

                # Save the formula, applied increment and before and after statistics for the slip.
                math_trace = {
                    "distribution": "Beta evidence-mass heuristic; not a calibrated posterior",
                    "prior_alpha": round(prior_alpha, 4),
                    "prior_beta": round(prior_beta, 4),
                    "delta_alpha": 0.0,
                    "delta_beta": round(delta_beta, 4),
                    "posterior_alpha": round(new_alpha, 4),
                    "posterior_beta": round(new_beta, 4),
                    "prior_mean_mastery": round(prior_mastery, 4),
                    "posterior_mean_mastery": round(new_mastery, 4),
                    "prior_uncertainty_sd": round(prior_uncertainty, 4),
                    "posterior_uncertainty_sd": round(new_uncertainty, 4),
                    "formula": "Beta(alpha, beta + delta_beta_slip); E[L]=alpha/(alpha+beta)",
                }

                # Create an event linking the slip calculation to its evidence and snapshots.
                event = StateChangeEvent(
                    learner_id=evidence.learner_id,
                    concept_id=cid,
                    prior_snapshot_id=prior_snapshot.snapshot_id,
                    new_snapshot_id=new_snapshot.snapshot_id,
                    triggering_evidence_id=evidence.evidence_id,
                    delta_mastery=new_mastery - prior_mastery,
                    prior_uncertainty=prior_uncertainty,
                    new_uncertainty=new_uncertainty,
                    change_summary=summary,
                    applied_rule=rule_applied,
                    math_trace=math_trace,
                )
                # Add the slip decision to the returned audit trail.
                events.append(event)

            # ----------------------------------------------------
            # Case 2: Systematic Misconception (e.g. Student A)
            # ----------------------------------------------------
            # Handle an expressed incorrect rule as a larger negative contribution.
            elif interp.category == "misconception":
                # Significant increment on beta based on evidence confidence
                # Apply the misconception weight of 1.25 to the two confidence factors.
                delta_beta = 1.25 * link.confidence * interp.hypothesis_confidence
                # Keep the positive evidence mass unchanged for this rule.
                new_alpha = prior_alpha
                # Add the penalty to negative mass.
                new_beta = prior_beta + delta_beta
                # Recompute the estimate and uncertainty from the updated totals.
                new_mastery, new_uncertainty = compute_beta_stats(new_alpha, new_beta)

                # Save the two revised Beta masses for the misconception case.
                cstate.alpha = new_alpha
                cstate.beta = new_beta
                # Save the resulting mastery estimate and uncertainty.
                cstate.mastery_estimate = new_mastery
                cstate.uncertainty = new_uncertainty

                # Read the taxonomy hypothesis ID assigned by extraction.
                hyp_id = interp.hypothesis_id
                # Keep the supporting claim's description as the candidate name.
                hyp_name = interp.description
                # Create a candidate misconception when it first appears.
                if hyp_id not in cstate.candidate_misconceptions:
                    # New candidate misconception
                    # Treat a new misconception as having no earlier evidence score.
                    prior_m_prob = 0.0
                    # Initialize the score at 55 percent of interpretation confidence with a cap.
                    new_m_prob = min(0.95, 0.55 * interp.hypothesis_confidence)
                    # Assign the label from the score threshold.
                    status = "confirmed" if new_m_prob >= 0.70 else "suspected"
                    # Save the candidate with its supporting evidence and time.
                    cstate.candidate_misconceptions[hyp_id] = MisconceptionBelief(
                        misconception_id=hyp_id,
                        misconception_name=hyp_name,
                        posterior_probability=new_m_prob,
                        # Link the first source evidence supporting the candidate belief.
                        supporting_evidence_ids=[evidence.evidence_id],
                        # Start with no direct contradiction records.
                        contradicting_evidence_ids=[],
                        status=status,
                        last_updated=evidence.timestamp,
                    )
                # When evidence matches an existing misconception, reinforce its current record.
                else:
                    # Reinforce existing misconception
                    # Read the existing misconception record.
                    m_belief = cstate.candidate_misconceptions[hyp_id]
                    # Preserve its prior score for the event.
                    prior_m_prob = m_belief.posterior_probability
                    # Move the score toward one in proportion to confidence, with a cap.
                    new_m_prob = min(0.98, prior_m_prob + (1.0 - prior_m_prob) * 0.55 * interp.hypothesis_confidence)
                    # Save the updated heuristic score to this existing belief.
                    m_belief.posterior_probability = new_m_prob
                    # Append the evidence that supports this belief.
                    m_belief.supporting_evidence_ids.append(evidence.evidence_id)
                    # Refresh the status using the updated score threshold.
                    m_belief.status = "confirmed" if new_m_prob >= 0.70 else "suspected"
                    # Update when the candidate was last supported.
                    m_belief.last_updated = evidence.timestamp

                # Identify the rule that treated this response as supporting the misconception.
                rule_applied = "RULE_MISCONCEPTION_REINFORCEMENT"
                # Summarize the candidate change and effect on mastery for inspection.
                summary = (
                    f"Misconception '{hyp_name}' probability updated ({prior_m_prob:.2f} -> {new_m_prob:.2f}) "
                    f"from evidence quote '{evidence.source_quote}'. Mastery adjusted {prior_mastery:.3f} -> {new_mastery:.3f}."
                )

                # Save the weights, before and after state and misconception score for audit.
                math_trace = {
                    "distribution": "Beta evidence-mass heuristic; not a calibrated posterior",
                    "prior_alpha": round(prior_alpha, 4),
                    "prior_beta": round(prior_beta, 4),
                    "delta_alpha": 0.0,
                    "delta_beta": round(delta_beta, 4),
                    "posterior_alpha": round(new_alpha, 4),
                    "posterior_beta": round(new_beta, 4),
                    "prior_mean_mastery": round(prior_mastery, 4),
                    "posterior_mean_mastery": round(new_mastery, 4),
                    "prior_uncertainty_sd": round(prior_uncertainty, 4),
                    "posterior_uncertainty_sd": round(new_uncertainty, 4),
                    "prior_misconception_prob": round(prior_m_prob, 4),
                    "new_misconception_prob": round(new_m_prob, 4),
                    "misconception_score_update": (
                        "p_new = 0.55 * confidence for first support; "
                        "otherwise p_new = p_old + (1 - p_old) * 0.55 * confidence"
                    ),
                    "formula": "Beta evidence mass plus a separate hand-set misconception score update",
                }

                # Build an event linking the update to the source evidence and applied rule.
                event = StateChangeEvent(
                    learner_id=evidence.learner_id,
                    concept_id=cid,
                    prior_snapshot_id=prior_snapshot.snapshot_id,
                    new_snapshot_id=new_snapshot.snapshot_id,
                    triggering_evidence_id=evidence.evidence_id,
                    delta_mastery=new_mastery - prior_mastery,
                    prior_uncertainty=prior_uncertainty,
                    new_uncertainty=new_uncertainty,
                    prior_misconception_prob=prior_m_prob,
                    new_misconception_prob=new_m_prob,
                    change_summary=summary,
                    applied_rule=rule_applied,
                    math_trace=math_trace,
                )
                # Append the misconception update to the returned audit trail.
                events.append(event)

            # ----------------------------------------------------
            # Case 3: Mastery / Contradiction of Previous Misconception
            # ----------------------------------------------------
            # Treat affirmative conceptual evidence as support when its polarity also agrees.
            elif interp.category == "mastery" and evidence.polarity == "supports":
                # Check whether the supported concept was previously flagged as a missing
                # foundation.
                if cid in new_snapshot.active_prerequisite_gaps:
                    # Clear that prerequisite flag for this concept.
                    new_snapshot.active_prerequisite_gaps.remove(cid)
                # Add twice the product of link and interpretation confidence to positive mass.
                delta_alpha = 2.0 * link.confidence * interp.hypothesis_confidence
                # Accumulate the new support on top of earlier positive evidence.
                new_alpha = prior_alpha + delta_alpha
                # Leave negative evidence history in place.
                new_beta = prior_beta
                # Recompute the estimate and uncertainty.
                new_mastery, new_uncertainty = compute_beta_stats(new_alpha, new_beta)

                # Store the accumulated positive support and preserved negative evidence.
                cstate.alpha = new_alpha
                cstate.beta = new_beta
                # Store the new estimate and uncertainty on the concept.
                cstate.mastery_estimate = new_mastery
                cstate.uncertainty = new_uncertainty

                # Only explicitly targeted candidate hypotheses receive counterevidence.
                # Review saved candidate beliefs for explicit counterevidence.
                for m_belief in list(cstate.candidate_misconceptions.values()):
                    # Change only active beliefs named by the evidence's contradiction IDs.
                    if m_belief.status in {"suspected", "confirmed"} and m_belief.misconception_id in evidence.contradicted_hypothesis_ids:
                        # Read the prior belief score for the trace.
                        prior_prob = m_belief.posterior_probability
                        # Scale the contradiction discount by both confidence factors.
                        discount = 0.75 * link.confidence * interp.hypothesis_confidence
                        # Apply the discount with a lower bound for the candidate score.
                        new_prob = max(0.05, prior_prob * (1.0 - discount))
                        # Save the reduced score to the stored belief.
                        m_belief.posterior_probability = new_prob
                        # Keep each piece of contradicting evidence once.
                        if evidence.evidence_id not in m_belief.contradicting_evidence_ids:
                            # Link the new quote through its evidence ID.
                            m_belief.contradicting_evidence_ids.append(evidence.evidence_id)
                        # Update the belief's last-evidence time.
                        m_belief.last_updated = evidence.timestamp
                        # Check whether the score is low enough to consider the belief resolved.
                        if new_prob <= 0.25:
                            # Apply the resolved status while preserving its history.
                            m_belief.status = "resolved"

                        # Record the calculation that reduced the misconception score.
                        contradiction_trace = {
                            "type": "Heuristic contradiction discount",
                            "prior_misconception_prob": round(prior_prob, 4),
                            "discount_factor": round(discount, 4),
                            "posterior_misconception_prob": round(new_prob, 4),
                            "trigger": f"Demonstrated mastery quote: '{evidence.source_quote}'",
                        }

                        # Add a distinct audit event for each affected belief.
                        event = StateChangeEvent(
                            learner_id=evidence.learner_id,
                            concept_id=cid,
                            prior_snapshot_id=prior_snapshot.snapshot_id,
                            new_snapshot_id=new_snapshot.snapshot_id,
                            triggering_evidence_id=evidence.evidence_id,
                            delta_mastery=new_mastery - prior_mastery,
                            prior_uncertainty=prior_uncertainty,
                            new_uncertainty=new_uncertainty,
                            prior_misconception_prob=prior_prob,
                            new_misconception_prob=new_prob,
                            change_summary=f"Misconception '{m_belief.misconception_name}' discounted ({prior_prob:.2f} -> {new_prob:.2f}) by contradictory mastery evidence.",
                            applied_rule="RULE_CONTRADICTION_DISCOUNT",
                            math_trace=contradiction_trace,
                        )
                        # Append the explicit contradiction event before the overall mastery event.
                        events.append(event)

                # Save the positive evidence update formula and both snapshots' statistics.
                math_trace = {
                    "distribution": "Beta evidence-mass heuristic; not a calibrated posterior",
                    "prior_alpha": round(prior_alpha, 4),
                    "prior_beta": round(prior_beta, 4),
                    "delta_alpha": round(delta_alpha, 4),
                    "delta_beta": 0.0,
                    "posterior_alpha": round(new_alpha, 4),
                    "posterior_beta": round(new_beta, 4),
                    "prior_mean_mastery": round(prior_mastery, 4),
                    "posterior_mean_mastery": round(new_mastery, 4),
                    "prior_uncertainty_sd": round(prior_uncertainty, 4),
                    "posterior_uncertainty_sd": round(new_uncertainty, 4),
                    "formula": "Beta(alpha + delta_alpha_mastery, beta); E[L]=alpha/(alpha+beta)",
                }

                # Record one audit event for the underlying mastery increase as well.
                event = StateChangeEvent(
                    learner_id=evidence.learner_id,
                    concept_id=cid,
                    prior_snapshot_id=prior_snapshot.snapshot_id,
                    new_snapshot_id=new_snapshot.snapshot_id,
                    triggering_evidence_id=evidence.evidence_id,
                    delta_mastery=new_mastery - prior_mastery,
                    prior_uncertainty=prior_uncertainty,
                    new_uncertainty=new_uncertainty,
                    change_summary=f"Mastery increased ({prior_mastery:.2f} -> {new_mastery:.2f}) based on demonstrated understanding.",
                    applied_rule="RULE_MASTERY_REINFORCEMENT",
                    math_trace=math_trace,
                )
                # Append the mastery event to the complete set of changes.
                events.append(event)

            # ----------------------------------------------------
            # Case 4: Prerequisite Gap
            # ----------------------------------------------------
            # Flag a concern about the foundation without changing the Beta estimate.
            elif interp.category == "prerequisite_gap":
                # Add an active prerequisite only when it was not already present.
                if cid not in new_snapshot.active_prerequisite_gaps:
                    # Add the concept so a later diagnostic can check the foundation.
                    new_snapshot.active_prerequisite_gaps.append(cid)

                # Record that a prerequisite flag leaves positive and negative evidence masses
                # unchanged.
                math_trace = {
                    "distribution": "Beta evidence-mass heuristic; not a calibrated posterior",
                    "prior_alpha": round(prior_alpha, 4),
                    "prior_beta": round(prior_beta, 4),
                    "delta_alpha": 0.0,
                    "delta_beta": 0.0,
                    "posterior_alpha": round(prior_alpha, 4),
                    "posterior_beta": round(prior_beta, 4),
                    "prior_mean_mastery": round(prior_mastery, 4),
                    "posterior_mean_mastery": round(prior_mastery, 4),
                    "prior_uncertainty_sd": round(prior_uncertainty, 4),
                    "posterior_uncertainty_sd": round(prior_uncertainty, 4),
                    "status": "Prerequisite gap detected on foundational fraction representation.",
                }

                # Explain the flag as an audit event tied to the inquiry that triggered it.
                event = StateChangeEvent(
                    learner_id=evidence.learner_id,
                    concept_id=cid,
                    prior_snapshot_id=prior_snapshot.snapshot_id,
                    new_snapshot_id=new_snapshot.snapshot_id,
                    triggering_evidence_id=evidence.evidence_id,
                    delta_mastery=0.0,
                    prior_uncertainty=prior_uncertainty,
                    new_uncertainty=prior_uncertainty,
                    change_summary=f"Prerequisite gap flagged from student inquiry: '{evidence.source_quote}'.",
                    applied_rule="RULE_PREREQUISITE_GAP_DETECTED",
                    math_trace=math_trace,
                )
                # Append the prerequisite event without changing the estimate.
                events.append(event)

            # ----------------------------------------------------
            # Case 5: Ambiguous or Guessing
            # ----------------------------------------------------
            # Handle remaining ambiguous or guessing interpretations with a numerical hold.
            else:
                # Show zero increments and identical before and after values in the calculation.
                math_trace = {
                    "distribution": "Beta evidence-mass heuristic; not a calibrated posterior",
                    "prior_alpha": round(prior_alpha, 4),
                    "prior_beta": round(prior_beta, 4),
                    "delta_alpha": 0.0,
                    "delta_beta": 0.0,
                    "posterior_alpha": round(prior_alpha, 4),
                    "posterior_beta": round(prior_beta, 4),
                    "prior_mean_mastery": round(prior_mastery, 4),
                    "posterior_mean_mastery": round(prior_mastery, 4),
                    "prior_uncertainty_sd": round(prior_uncertainty, 4),
                    "posterior_uncertainty_sd": round(prior_uncertainty, 4),
                    "status": "Ambiguity hold - no change to alpha or beta.",
                }

                # Retain the uncertainty and quote linkage even when the estimate holds.
                event = StateChangeEvent(
                    learner_id=evidence.learner_id,
                    concept_id=cid,
                    prior_snapshot_id=prior_snapshot.snapshot_id,
                    new_snapshot_id=new_snapshot.snapshot_id,
                    triggering_evidence_id=evidence.evidence_id,
                    delta_mastery=0.0,
                    prior_uncertainty=prior_uncertainty,
                    new_uncertainty=prior_uncertainty,
                    change_summary="Evidence marked ambiguous or guessing. State held constant.",
                    applied_rule="RULE_AMBIGUITY_HOLD",
                    math_trace=math_trace,
                )
                # Include the hold in the audit trail for replay and inspection.
                events.append(event)

        # Finalize stable IDs and timestamps for all rule events produced by the update.
        for index, event in enumerate(events):
            # Give each event the same timestamp as its source evidence.
            event.timestamp = evidence.timestamp
            # Use contextual fields to make the event ID reproducible during replay.
            event.event_id = _stable_identifier(
                "EV",
                new_snapshot.snapshot_id,
                evidence.evidence_id,
                str(index),
                event.concept_id,
                event.applied_rule,
                json.dumps(event.math_trace, sort_keys=True, separators=(",", ":")),
            )

        # Return the calculated state plus the complete audit trail.
        return new_snapshot, events

    # Recap the fraction trajectory from a saved ordered evidence ledger.
    def replay_trajectory(
        self,
        learner_id: str,
        concept_ids: List[str],
        evidence_ledger: List[CognitiveEvidence],
    ) -> List[Tuple[LearnerStateSnapshot, List[StateChangeEvent]]]:
        """
        Recomputes state snapshots in evidence order using the current update policy.
        Replaying the same initial conditions and evidence yields stable snapshot
        and audit-event IDs. Duplicate evidence IDs are rejected to avoid silently
        applying one ledger event twice.
        """
        # Start replay at the deterministic prior.
        current_state = self.initialize_learner_state(learner_id, concept_ids)
        # Collect the state and events produced after each observation.
        trajectory = []
        # Track seen evidence so a bad ledger cannot apply a record twice.
        seen_evidence_ids: set[str] = set()

        # Apply evidence in the supplied order.
        for evidence in evidence_ledger:
            # Check for a duplicate evidence ID before updating.
            if evidence.evidence_id in seen_evidence_ids:
                # Refuse an invalid replay that would overcount evidence.
                raise ValueError(f"Duplicate evidence_id in replay ledger: {evidence.evidence_id}")
            # Mark this ID as processed.
            seen_evidence_ids.add(evidence.evidence_id)
            # Reuse the normal state update rules.
            next_state, events = self.update_state(current_state, evidence)
            # Save the reconstructed snapshot and audit events.
            trajectory.append((next_state, events))
            # Carry the computed state forward into the next observation.
            current_state = next_state

        # Return the complete reconstructed trajectory.
        return trajectory
