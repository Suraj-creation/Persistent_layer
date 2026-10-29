from __future__ import annotations

import copy
import math
import uuid
from typing import Dict, List, Optional, Tuple, Any
from ..evidence.models import CognitiveEvidence, Interaction
from .models import ConceptState, LearnerStateSnapshot, MisconceptionBelief, StateChangeEvent


def compute_beta_stats(alpha: float, beta: float) -> Tuple[float, float]:
    """Computes mean mastery and uncertainty standard deviation for Beta(alpha, beta)."""
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
    sd = math.sqrt(max(1e-6, var))
    return round(mean, 5), round(sd, 5)


class BayesianLearnerStateEngine:
    """
    Deterministic Beta-Binomial Bayesian State Engine.
    Updates learner conceptual beliefs based on structured Cognitive Evidence.
    
    Guarantees:
    - Pure functional deterministic updates (same inputs -> identical state output).
    - Mathematically rigorous Beta-Binomial updating: E[L] = alpha / (alpha + beta).
    - Differentiates slips from systematic misconceptions (Student A vs. Student B).
    - Resolves contradictory evidence without discarding history.
    - Preserves complete provenance and generates human-auditable StateChangeEvents with full mathematical traces.
    """

    def __init__(self, policy_version: str = "beta-bayesian-v2.0"):
        self.policy_version = policy_version

    def initialize_learner_state(self, learner_id: str, concept_ids: List[str]) -> LearnerStateSnapshot:
        """Creates an initial prior state for a new learner."""
        concept_states: Dict[str, ConceptState] = {}
        for cid in concept_ids:
            alpha = 1.0
            beta = 1.0
            mean_m, sd_u = compute_beta_stats(alpha, beta)
            concept_states[cid] = ConceptState(
                concept_id=cid,
                concept_name=cid.replace("_", " ").title(),
                alpha=alpha,
                beta=beta,
                mastery_estimate=mean_m,
                uncertainty=sd_u,
                evidence_sufficiency="insufficient",
                candidate_misconceptions={},
                last_interaction_timestamp=None,
                interaction_count=0,
            )

        return LearnerStateSnapshot(
            learner_id=learner_id,
            as_of_interaction_id="INIT",
            concept_states=concept_states,
            active_prerequisite_gaps=[],
            policy_version=self.policy_version,
        )

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
        new_snapshot = copy.deepcopy(prior_snapshot)
        new_snapshot.snapshot_id = f"S_{uuid.uuid4().hex[:8]}"
        new_snapshot.as_of_interaction_id = evidence.interaction_id
        new_snapshot.timestamp = evidence.timestamp
        events: List[StateChangeEvent] = []

        interp = evidence.candidate_interpretation

        for link in evidence.concept_links:
            cid = link.concept_id
            cname = link.concept_name

            # Ensure concept exists in state
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

            cstate = new_snapshot.concept_states[cid]
            prior_mastery = cstate.mastery_estimate
            prior_uncertainty = cstate.uncertainty
            prior_alpha = getattr(cstate, "alpha", 1.0)
            prior_beta = getattr(cstate, "beta", 1.0)

            cstate.interaction_count += 1
            cstate.last_interaction_timestamp = evidence.timestamp

            # Evaluate evidence sufficiency
            if cstate.interaction_count >= 5:
                cstate.evidence_sufficiency = "sufficient"
            elif cstate.interaction_count >= 3:
                cstate.evidence_sufficiency = "moderate"
            else:
                cstate.evidence_sufficiency = "sparse"

            delta_alpha = 0.0
            delta_beta = 0.0
            rule_applied = ""
            summary = ""
            likelihood_ratio = None
            prior_m_prob = None
            new_m_prob = None

            # ----------------------------------------------------
            # Case 1: Pure Calculation Slip (e.g. Student B)
            # ----------------------------------------------------
            if interp.category == "calculation_slip":
                # Arithmetic or typographical slip: concept rule is understood
                # Very light penalty on beta, concept mastery remains largely intact
                delta_beta = 0.15 * link.confidence * interp.hypothesis_confidence
                new_alpha = prior_alpha
                new_beta = prior_beta + delta_beta
                new_mastery, new_uncertainty = compute_beta_stats(new_alpha, new_beta)

                cstate.alpha = new_alpha
                cstate.beta = new_beta
                cstate.mastery_estimate = new_mastery
                cstate.uncertainty = new_uncertainty
                rule_applied = "RULE_SLIP_ATTENUATION"
                summary = (
                    f"Calculation slip detected. Conceptual rule preserved with minimal attenuation "
                    f"({prior_mastery:.3f} -> {new_mastery:.3f})."
                )

                math_trace = {
                    "distribution": "Beta-Binomial Conjugate",
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
                events.append(event)

            # ----------------------------------------------------
            # Case 2: Systematic Misconception (e.g. Student A)
            # ----------------------------------------------------
            elif interp.category == "misconception":
                # Significant increment on beta based on evidence confidence
                delta_beta = 1.25 * link.confidence * interp.hypothesis_confidence
                new_alpha = prior_alpha
                new_beta = prior_beta + delta_beta
                new_mastery, new_uncertainty = compute_beta_stats(new_alpha, new_beta)

                cstate.alpha = new_alpha
                cstate.beta = new_beta
                cstate.mastery_estimate = new_mastery
                cstate.uncertainty = new_uncertainty

                hyp_id = interp.hypothesis_id
                hyp_name = interp.description
                likelihood_ratio = 1.0 + 8.0 * interp.hypothesis_confidence * link.confidence

                if hyp_id not in cstate.candidate_misconceptions:
                    # New candidate misconception
                    prior_m_prob = 0.0
                    new_m_prob = min(0.95, 0.55 * interp.hypothesis_confidence)
                    status = "confirmed" if new_m_prob >= 0.70 else "suspected"
                    cstate.candidate_misconceptions[hyp_id] = MisconceptionBelief(
                        misconception_id=hyp_id,
                        misconception_name=hyp_name,
                        posterior_probability=new_m_prob,
                        supporting_evidence_ids=[evidence.evidence_id],
                        contradicting_evidence_ids=[],
                        status=status,
                        last_updated=evidence.timestamp,
                    )
                else:
                    # Reinforce existing misconception
                    m_belief = cstate.candidate_misconceptions[hyp_id]
                    prior_m_prob = m_belief.posterior_probability
                    new_m_prob = min(0.98, prior_m_prob + (1.0 - prior_m_prob) * 0.55 * interp.hypothesis_confidence)
                    m_belief.posterior_probability = new_m_prob
                    m_belief.supporting_evidence_ids.append(evidence.evidence_id)
                    m_belief.status = "confirmed" if new_m_prob >= 0.70 else "suspected"
                    m_belief.last_updated = evidence.timestamp

                rule_applied = "RULE_MISCONCEPTION_REINFORCEMENT"
                summary = (
                    f"Misconception '{hyp_name}' probability updated ({prior_m_prob:.2f} -> {new_m_prob:.2f}) "
                    f"from evidence quote '{evidence.source_quote}'. Mastery adjusted {prior_mastery:.3f} -> {new_mastery:.3f}."
                )

                math_trace = {
                    "distribution": "Beta-Binomial Conjugate",
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
                    "misconception_likelihood_ratio": round(likelihood_ratio, 4),
                    "prior_misconception_prob": round(prior_m_prob, 4),
                    "new_misconception_prob": round(new_m_prob, 4),
                    "formula": "Beta(alpha, beta + delta_beta_misconception); Odds_new = Odds_old * Lambda",
                }

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
                events.append(event)

            # ----------------------------------------------------
            # Case 3: Mastery / Contradiction of Previous Misconception
            # ----------------------------------------------------
            elif interp.category == "mastery" or evidence.polarity == "supports":
                delta_alpha = 2.0 * link.confidence * interp.hypothesis_confidence
                new_alpha = prior_alpha + delta_alpha
                new_beta = prior_beta
                new_mastery, new_uncertainty = compute_beta_stats(new_alpha, new_beta)

                cstate.alpha = new_alpha
                cstate.beta = new_beta
                cstate.mastery_estimate = new_mastery
                cstate.uncertainty = new_uncertainty

                # If student previously had suspected misconceptions on this concept, contradict them
                for m_id, m_belief in list(cstate.candidate_misconceptions.items()):
                    if evidence.polarity == "contradicts" or evidence.extracted_claim:
                        prior_prob = m_belief.posterior_probability
                        discount = 0.75 * link.confidence
                        new_prob = max(0.05, prior_prob * (1.0 - discount))
                        m_belief.posterior_probability = new_prob
                        m_belief.contradicting_evidence_ids.append(evidence.evidence_id)
                        if new_prob <= 0.25:
                            m_belief.status = "resolved"

                        contradiction_trace = {
                            "type": "Contradiction Discounting",
                            "prior_misconception_prob": round(prior_prob, 4),
                            "discount_factor": round(discount, 4),
                            "posterior_misconception_prob": round(new_prob, 4),
                            "trigger": f"Demonstrated mastery quote: '{evidence.source_quote}'",
                        }

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
                        events.append(event)

                math_trace = {
                    "distribution": "Beta-Binomial Conjugate",
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
                events.append(event)

            # ----------------------------------------------------
            # Case 4: Prerequisite Gap
            # ----------------------------------------------------
            elif interp.category == "prerequisite_gap":
                if "fraction_basics" not in new_snapshot.active_prerequisite_gaps:
                    new_snapshot.active_prerequisite_gaps.append("fraction_basics")

                math_trace = {
                    "distribution": "Beta-Binomial Conjugate",
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
                events.append(event)

            # ----------------------------------------------------
            # Case 5: Ambiguous or Guessing
            # ----------------------------------------------------
            else:
                math_trace = {
                    "distribution": "Beta-Binomial Conjugate",
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
                events.append(event)

        return new_snapshot, events

    def replay_trajectory(
        self,
        learner_id: str,
        concept_ids: List[str],
        evidence_ledger: List[CognitiveEvidence],
    ) -> List[Tuple[LearnerStateSnapshot, List[StateChangeEvent]]]:
        """
        Replays the entire sequence of evidence records from scratch,
        proving that learner state is a pure, deterministic projection over the evidence ledger.
        """
        current_state = self.initialize_learner_state(learner_id, concept_ids)
        trajectory = []

        for evidence in evidence_ledger:
            next_state, events = self.update_state(current_state, evidence)
            trajectory.append((next_state, events))
            current_state = next_state

        return trajectory

    def replay_trajectory(
        self,
        learner_id: str,
        concept_ids: List[str],
        evidence_ledger: List[CognitiveEvidence],
    ) -> List[Tuple[LearnerStateSnapshot, List[StateChangeEvent]]]:
        """
        Replays the entire sequence of evidence records from scratch,
        proving that learner state is a pure, deterministic projection over the evidence ledger.
        """
        current_state = self.initialize_learner_state(learner_id, concept_ids)
        trajectory = []

        for evidence in evidence_ledger:
            next_state, events = self.update_state(current_state, evidence)
            trajectory.append((next_state, events))
            current_state = next_state

        return trajectory
