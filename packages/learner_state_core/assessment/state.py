"""Deterministic evidence integration and adaptive planning, independent of the LLM."""
# Copy the prior snapshot so each update leaves the saved before-state available for inspection.
import copy
# Use the square root when turning Beta variance into standard deviation.
import math
# Hash criterion descriptions and newly proposed beliefs to give them stable identifiers.
import hashlib
# Use the shared server-side ID generator for each snapshot.
from .models import new_id
# Use the same UTC timestamp format as the assessment database.
from .storage import now

# Record the scoring policy on each snapshot so results can be traced to the rules that produced
# them.
POLICY_VERSION = "criterion-weighted-state-v1"
# Version the question planner separately because choosing a task and scoring evidence are
# different steps.
PLANNER_VERSION = "focused-adaptive-planner-v2"


# Give the same skill description the same gap ID even when questions use different local rubric
# IDs.
def criterion_key(description):
    # Hash the exact description and shorten the hex digest for storage.
    return "gap_" + hashlib.sha256(description.encode()).hexdigest()[:20]


# Summarize the two accumulated evidence totals as an estimate and its spread.
def statistics(alpha, beta):
    # The first value is the Beta mean. The second is its standard deviation. Both are rounded to
    # five places.
    return round(alpha / (alpha + beta), 5), round(math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))), 5)


# Create the starting topic memory when there is no earlier state for this learner and topic.
def initialize(definition):
    # Give this starting snapshot its own ID, timestamp and policy label.
    return {"snapshot_id": new_id("state"), "timestamp": now(), "policy_version": POLICY_VERSION,
            # Start every concept with one unit of positive mass and one unit of negative mass.
            "concepts": {c["concept_id"]: {"concept_id": c["concept_id"], "name": c["name"], "alpha": 1.0, "beta": 1.0,
                # A Beta(1,1) prior has mean 0.5 and standard deviation about 0.28868. No answers
                # have been observed yet.
                "estimate": .5, "uncertainty": .28868, "observation_count": 0, "informative_count": 0,
                # Keep empty lists for evidence and fingerprints, and empty records for beliefs and
                # skill gaps.
                "evidence_ids": [], "response_hashes": [], "hypotheses": {}, "criterion_gaps": {}, "missing_criteria": [], "last_category": None}
                         # Repeat the initialization for each concept in the saved topic map.
                         for c in definition["concepts"]}}


# Combine one validated answer with the prior state and return both the new snapshot and the
# calculation.
def integrate(prior, evidence, question, nlp):
    # Deep-copy the snapshot because the nested concept records will be edited.
    result = copy.deepcopy(prior)
    # Stamp the new copy so later questions can reference this exact revision of memory.
    result.update(snapshot_id=new_id("state"), timestamp=now(), policy_version=POLICY_VERSION)
    # Update only the concept this question assessed.
    concept = result["concepts"][evidence["concept_id"]]
    # Keep its previous record separately for the before/after calculation.
    before = copy.deepcopy(concept)
    # Count every accepted observation, including an ambiguous one that holds the estimate.
    concept["observation_count"] += 1
    # Link the observation to the concept by its saved evidence ID.
    concept["evidence_ids"].append(evidence["evidence_id"])
    # Read the model's uncalibrated confidence as a weight input.
    confidence = evidence["confidence"]
    # Check whether this exact response fingerprint already appeared for this concept.
    duplicate = nlp["response_hash"] in concept["response_hashes"]
    # Remember the fingerprint even when this observation has zero weight.
    concept["response_hashes"].append(nlp["response_hash"])
    # Work out the fraction of this question's rubric criteria that the response met.
    coverage = len(evidence["met_criteria"]) / len(question["criteria"])
    # Require a non-ambiguous category, confidence of at least 0.55 and at least four tokens to
    # move the estimate.
    meaningful = evidence["category"] != "ambiguous" and confidence >= .55 and nlp["token_count"] >= 4
    # A repeated fingerprint receives 15 percent of the usual weight.
    novelty = .15 if duplicate else 1.0
    # Give short explanations less weight, with full length credit at 16 tokens.
    informativeness = min(1.0, nlp["token_count"] / 16)
    # Multiply the three factors and round to six places. A weak or ambiguous observation
    # contributes zero.
    weight = round(confidence * novelty * informativeness, 6) if meaningful else 0.0
    # Add the covered share of the weight to positive evidence mass.
    da = round(weight * coverage, 6)
    # Add the uncovered share to negative mass. A slip counts only 15 percent as much on this side.
    db = round(weight * (1 - coverage) * (.15 if evidence["category"] == "slip" else 1.0), 6)
    # Accumulate support on top of the topic's earlier positive evidence.
    concept["alpha"] += da
    # Accumulate gaps on top of the earlier negative evidence.
    concept["beta"] += db
    # Recompute the Beta mean and spread from the updated totals.
    concept["estimate"], concept["uncertainty"] = statistics(concept["alpha"], concept["beta"])
    # Count a fresh qualifying answer separately from a repeated qualifying answer.
    if meaningful and not duplicate:
        # Increase the informative count used by the planner to identify concepts that have been
        # assessed.
        concept["informative_count"] += 1
    # Save the latest diagnostic category for this concept, even on a numerical hold.
    concept["last_category"] = evidence["category"]
    # Re-key older gap records by what the rubric skill actually says.
    gaps = {criterion_key(gap["description"]): {**gap, "criterion_id": criterion_key(gap["description"])} for gap in concept.get("criterion_gaps", {}).values()}
    # Replace the stored lookup with the normalized gap IDs.
    concept["criterion_gaps"] = gaps
    # Preserve an earlier v1 projection if it had not yet stored criterion identities.
    # Walk the older text-only list of open gaps so saved learners can still resume.
    for description in concept["missing_criteria"]:
        # Give each previously saved skill description the same stable ID used by new gaps.
        gid = criterion_key(description)
        # Restore any missing gap record with open status and empty evidence history.
        gaps.setdefault(gid, {"criterion_id": gid, "description": description, "status": "open", "evidence_ids": [], "resolved_by_evidence_ids": []})
    # Collect the skills addressed by this answer for the trace and report.
    resolved_criteria = []
    # Change the gap records only when the response passes the same evidence quality gate.
    if meaningful:
        # Turn current rubric IDs into sets for quick membership checks.
        missing, met = set(evidence["missing_criteria"]), set(evidence["met_criteria"])
        # Check each skill on the current question. Other saved skills stay as they were.
        for criterion in question["criteria"]:
            # Match the current skill to persistent memory by its description.
            gid = criterion_key(criterion["description"])
            # If the response missed this rubric item, register or reopen its gap.
            if criterion["criterion_id"] in missing:
                # Create the gap on first sight with empty support and resolution lists.
                gap = gaps.setdefault(gid, {"criterion_id": gid, "description": criterion["description"], "status": "open", "evidence_ids": [], "resolved_by_evidence_ids": []})
                # Mark the skill open even if it had previously been addressed.
                gap["status"] = "open"
                # Record the evidence that showed this criterion was missing.
                gap["evidence_ids"].append(evidence["evidence_id"])
            # Close a gap only if the current rubric explicitly tested and met that same skill.
            elif criterion["criterion_id"] in met and gid in gaps and gaps[gid]["status"] == "open":
                # Mark the matching stored gap as addressed.
                gaps[gid]["status"] = "addressed"
                # Save the evidence that addressed it so it can be inspected later.
                gaps[gid]["resolved_by_evidence_ids"].append(evidence["evidence_id"])
                # Include the resolved skill in this turn's list of changes.
                resolved_criteria.append(gaps[gid]["description"])
        # Rebuild the short list of open skills the planner and UI read.
        concept["missing_criteria"] = [gap["description"] for gap in gaps.values() if gap["status"] == "open"]
    # Collect a history of score changes for the misconception hypotheses touched by this answer.
    hypothesis_changes = []
    # Increase a candidate belief only when weighted evidence supports a specific misconception.
    if weight and evidence["category"] == "misconception" and evidence.get("misconception"):
        # The extractor reuses a matching prior description. IDs are server-owned.
        # Reuse a belief with the same wording or assign a stable ID from the proposed wording.
        hid = next((key for key, value in concept["hypotheses"].items() if value["description"] == evidence["misconception"]), "hyp_" + hashlib.sha256(evidence["misconception"].encode()).hexdigest()[:20])
        # On first support, start the candidate belief at 0.1.
        hypothesis = concept["hypotheses"].setdefault(hid, {"hypothesis_id": hid, "description": evidence["misconception"], "score": .1,
             # Track the evidence supporting and contradicting it separately.
             "status": "suspected", "supporting_evidence_ids": [], "contradicting_evidence_ids": []})
        # Capture the old score for the calculation trace.
        old = hypothesis["score"]
        # Move the score partway toward one, in proportion to the evidence weight, with a cap of
        # 0.98.
        hypothesis["score"] = round(min(.98, old + (1-old) * .5 * weight), 5)
        # Link the original explanation that supported this candidate belief.
        hypothesis["supporting_evidence_ids"].append(evidence["evidence_id"])
        # Call the belief recurring only after at least two pieces of support and a score of at
        # least 0.6.
        hypothesis["status"] = "recurring" if len(hypothesis["supporting_evidence_ids"]) >= 2 and hypothesis["score"] >= .6 else "suspected"
        # Record why the score rose along with its before and after values.
        hypothesis_changes.append({"hypothesis_id": hid, "before": old, "after": hypothesis["score"], "reason": "supporting quoted evidence"})
    # Allow only weighted understanding or partial evidence to weaken a belief.
    if weight and evidence["category"] in ("understanding", "partial"):
        # Follow the specific prior hypothesis IDs that the new explanation explicitly challenged.
        for hid in evidence["contradicts_hypothesis_ids"]:
            # Load the already validated belief record from that concept.
            hypothesis = concept["hypotheses"][hid]
            # Save the old score for comparison.
            old = hypothesis["score"]
            # Reduce that score by a fraction proportional to the weight of the contradiction.
            hypothesis["score"] = round(old * (1 - .7 * weight), 5)
            # Preserve the evidence that contradicted the belief.
            hypothesis["contradicting_evidence_ids"].append(evidence["evidence_id"])
            # Label the belief weakened below 0.25, or suspected if it remains above that
            # threshold.
            hypothesis["status"] = "weakened" if hypothesis["score"] < .25 else "suspected"
            # Explain the decrease by linking it to the explicit contradiction.
            hypothesis_changes.append({"hypothesis_id": hid, "before": old, "after": hypothesis["score"], "reason": "explicit contradictory quoted evidence"})
    # Record whether this observation held the estimate or applied the rubric-weighted calculation.
    change = {"concept_id": evidence["concept_id"], "rule": "evidence_hold" if not weight else "weighted_criterion_update",
              # Expose the confidence, rubric coverage, novelty and input length factors behind the
              # weight.
              "confidence": confidence, "coverage": coverage, "novelty_factor": novelty, "informativeness": informativeness,
              # Save the duplicate flag and the exact applied increments.
              "duplicate_response": duplicate, "weight": weight, "delta_alpha": da, "delta_beta": db,
              # Expose the before and after evidence totals.
              "before_alpha": before["alpha"], "before_beta": before["beta"], "after_alpha": concept["alpha"], "after_beta": concept["beta"],
              # Record the estimated change so the report can sum a conversation's own updates.
              "before_estimate": before["estimate"], "after_estimate": concept["estimate"], "delta_estimate": round(concept["estimate"] - before["estimate"], 5),
              # Show how uncertainty and any specific hypotheses or gaps changed.
              "before_uncertainty": before["uncertainty"], "after_uncertainty": concept["uncertainty"], "hypothesis_changes": hypothesis_changes, "resolved_criteria": resolved_criteria,
              # Save the readable formula next to the numeric trace for the inspector.
              "formula": "w=confidence×novelty×min(tokens/16,1); α′=α+w×rubric_coverage; β′=β+w×(1−coverage)×slip_factor; estimate=α′/(α′+β′)",
              # Describe the estimate as a heuristic and keep coaching outside the scoring rule.
              "limitation": "Heuristic evidence estimate, not a calibrated measurement of understanding. Tutor coaching never updates these parameters."}
    # Return the new state separately from the calculation used to explain it.
    return result, change


# Choose the next assessment objective from saved memory and recent question history.
def plan_next(state, definition, previous_questions, prior_turns):
    # Read the latest per-concept state.
    concepts = state["concepts"]
    # Use the latest saved answer in this conversation if there is one.
    last = prior_turns[-1] if prior_turns else None
    # Start with an initial diagnostic intent before considering follow-ups.
    target, intent, reason = None, "diagnose", "Collect an initial explanation; there is no evidence for this concept yet."
    # Check for a repair opportunity when an answer is available.
    if last:
        # Identify the concept assessed by the latest answer.
        cid = last["evidence"]["concept_id"]
        # Read its current estimate, gaps and belief history.
        current = concepts[cid]
        # Count appearances of this concept among the three latest questions.
        repeated = sum(q["concept_id"] == cid for q in previous_questions[-3:])
        # Treat incomplete, mistaken, slipped or ambiguous reasoning as a reason to consider
        # repair.
        needs_repair = last["evidence"]["category"] in ("partial", "misconception", "prerequisite_gap", "slip", "ambiguous")
        # Repair the same concept if it has appeared fewer than three times recently.
        if needs_repair and repeated < 3:
            # Assign that concept as the next target and set the repair objective.
            target, intent = cid, "repair"
            # Explain the repair using saved gaps or the fact that the reasoning remains unclear.
            reason = "The last saved explanation leaves these criteria unverified: " + "; ".join(current["missing_criteria"] or ["reasoning is still ambiguous"])
        # If recent questions keep returning to this difficulty, consider a foundation.
        elif needs_repair:
            # Read the assessed concept's prerequisite IDs from the topic map.
            parents = next(c["prerequisites"] for c in definition["concepts"] if c["concept_id"] == cid)
            # Only choose a prerequisite when there is one to assess.
            if parents:
                # Target the prerequisite with the lowest saved estimate.
                target, intent = min(parents, key=lambda p: concepts[p]["estimate"]), "repair"
                # Frame this as a check of the foundation based on observed difficulty.
                reason = "Repeated difficulty in the saved evidence warrants checking a prerequisite, without assuming it is missing."
    # If the earlier steps chose no target, look for a candidate misconception worth testing.
    if target is None:
        # Find concepts with no open gaps that still carry a belief that has not been weakened.
        pending_beliefs = [c for c in concepts.values() if not c["missing_criteria"] and any(h["status"] != "weakened" for h in c["hypotheses"].values())]
        # Continue only when an active belief fits that test.
        if pending_beliefs:
            # Prioritize the eligible concept with the lowest estimate.
            candidate = min(pending_beliefs, key=lambda c: c["estimate"])
            # Set the task to verify the earlier belief with new reasoning.
            target, intent = candidate["concept_id"], "verify_hypothesis"
            # Explain that the question should seek fresh confirming or contradictory evidence.
            reason = "An earlier candidate belief has not been explicitly contradicted. Use a fresh contrast or counterexample to check it rather than silently dismissing it."
    # If there is still no target, look for an open criterion anywhere in saved topic memory.
    if target is None:
        # Gather the concepts carrying unresolved skills from this or earlier conversations.
        unresolved = [c for c in concepts.values() if c["missing_criteria"]]
        # Pick a repair when an unresolved skill is available.
        if unresolved:
            # Prioritize the lower estimate, with fewer informative observations breaking a tie.
            candidate = min(unresolved, key=lambda c: (c["estimate"], c["informative_count"]))
            # Assign the selected concept to the repair plan.
            target, intent = candidate["concept_id"], "repair"
            # Carry forward the saved skill wording as the reason for selection.
            reason = "Previously saved criteria remain unverified across conversations: " + "; ".join(candidate["missing_criteria"])
    # Use this fallback when no repair or belief check was chosen.
    if target is None:
        # Unassessed foundations first; afterwards revisit low estimates with uncertainty.
        # Sort unassessed concepts ahead of assessed ones.
        target = min(definition["concepts"], key=lambda c: (concepts[c["concept_id"]]["informative_count"] > 0,
                     # Use observation count, number of prerequisites and estimate to break ties in
                     # that order.
                     concepts[c["concept_id"]]["informative_count"], len(c["prerequisites"]), concepts[c["concept_id"]]["estimate"]))["concept_id"]
        # A concept with informative evidence can be checked for transfer to a new situation.
        if concepts[target]["informative_count"]:
            # Switch from initial diagnosis to a transfer objective.
            intent = "verify_transfer"
            # Ask for new evidence that the concept applies in a different setting.
            reason = "Use a fresh situation to check whether saved evidence of understanding transfers; do not assume improvement."
    # Count earlier tasks on the chosen concept to vary the format.
    count = sum(q["concept_id"] == target for q in previous_questions)
    # Keep the available formats for diagnosis and transfer in a fixed rotation.
    kinds = ["explain", "apply", "contrast", "debug", "transfer"]
    # Start repairs with guided practice and rotate formats for other intents.
    kind = "guided_practice" if intent == "repair" else ("contrast" if count % 2 else "debug") if intent == "verify_hypothesis" else kinds[count % len(kinds)]
    # Pass along the target's current, non-weakened hypotheses.
    active = [h for h in concepts[target]["hypotheses"].values() if h["status"] != "weakened"]
    # Start with no rubric skills that the generator must copy.
    required = []
    # Fill in those requirements only for a repair plan.
    if intent == "repair":
        # Include up to three saved skills with stable IDs and their original descriptions.
        required = [{"criterion_id": criterion_key(description), "description": description} for description in concepts[target]["missing_criteria"][:3]]
        # If there are no saved gaps, a recent answer on this same concept may identify skills to
        # recheck.
        if not required and last and last["evidence"]["concept_id"] == target:
            # Take up to three missing items from the latest question's actual rubric.
            required = [{"criterion_id": criterion_key(c["description"]), "description": c["description"]} for c in last["question"]["criteria"] if c["criterion_id"] in last["evidence"]["missing_criteria"]][:3]
        # Find earlier questions that used a repair intent for this concept.
        prior_repairs = [q for q in previous_questions if q["concept_id"] == target and q.get("generation_trace", {}).get("plan", {}).get("intent") == "repair"]
        # After the first repair, make later checks more focused and vary how the learner is asked.
        if prior_repairs:
            # Isolate one unresolved skill and change the task format rather than repeat
            # the same compound guided-practice question indefinitely.
            # Keep a rotation of repair formats that ask for reasoning in different ways.
            formats = ["contrast", "debug", "apply", "guided_practice"]
            # Choose the format by the number of previous repairs on the target.
            kind = formats[(len(prior_repairs) - 1) % len(formats)]
            # Limit the required recheck to one skill for the new question.
            required = required[:1]
            # Add that change in task design to the explanation of the plan.
            reason += " Focus on one remaining skill in a different question format."
    # Return the target, task format, reason and intent for question generation.
    return {"concept_id": target, "kind": kind, "intent": intent, "reason": reason,
            # Prefer foundational work for repairs and unassessed concepts. Request transfer-level
            # work otherwise.
            "difficulty": "foundational" if intent == "repair" or not concepts[target]["informative_count"] else "transfer",
            # Return both the broader open issues and the specific rubric skills the next task must
            # recheck.
            "missing_criteria": concepts[target]["missing_criteria"], "required_criteria": required, "hypotheses": active,
            # Attach recent evidence IDs, the exact state snapshot ID and the planner version for
            # inspection.
            "evidence_ids": concepts[target]["evidence_ids"][-10:], "state_snapshot_id": state["snapshot_id"], "policy_version": PLANNER_VERSION}
