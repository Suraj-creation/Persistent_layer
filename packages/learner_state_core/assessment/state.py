"""Deterministic evidence integration and adaptive planning, independent of the LLM."""
import copy
import math
import hashlib
from .models import new_id
from .storage import now

POLICY_VERSION = "criterion-weighted-state-v1"
PLANNER_VERSION = "focused-adaptive-planner-v2"


def criterion_key(description):
    return "gap_" + hashlib.sha256(description.encode()).hexdigest()[:20]


def statistics(alpha, beta):
    return round(alpha / (alpha + beta), 5), round(math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))), 5)


def initialize(definition):
    return {"snapshot_id": new_id("state"), "timestamp": now(), "policy_version": POLICY_VERSION,
            "concepts": {c["concept_id"]: {"concept_id": c["concept_id"], "name": c["name"], "alpha": 1.0, "beta": 1.0,
                "estimate": .5, "uncertainty": .28868, "observation_count": 0, "informative_count": 0,
                "evidence_ids": [], "response_hashes": [], "hypotheses": {}, "criterion_gaps": {}, "missing_criteria": [], "last_category": None}
                         for c in definition["concepts"]}}


def integrate(prior, evidence, question, nlp):
    result = copy.deepcopy(prior)
    result.update(snapshot_id=new_id("state"), timestamp=now(), policy_version=POLICY_VERSION)
    concept = result["concepts"][evidence["concept_id"]]
    before = copy.deepcopy(concept)
    concept["observation_count"] += 1
    concept["evidence_ids"].append(evidence["evidence_id"])
    confidence = evidence["confidence"]
    duplicate = nlp["response_hash"] in concept["response_hashes"]
    concept["response_hashes"].append(nlp["response_hash"])
    coverage = len(evidence["met_criteria"]) / len(question["criteria"])
    meaningful = evidence["category"] != "ambiguous" and confidence >= .55 and nlp["token_count"] >= 4
    novelty = .15 if duplicate else 1.0
    informativeness = min(1.0, nlp["token_count"] / 16)
    weight = round(confidence * novelty * informativeness, 6) if meaningful else 0.0
    da = round(weight * coverage, 6)
    db = round(weight * (1 - coverage) * (.15 if evidence["category"] == "slip" else 1.0), 6)
    concept["alpha"] += da
    concept["beta"] += db
    concept["estimate"], concept["uncertainty"] = statistics(concept["alpha"], concept["beta"])
    if meaningful and not duplicate:
        concept["informative_count"] += 1
    concept["last_category"] = evidence["category"]
    gaps = {criterion_key(gap["description"]): {**gap, "criterion_id": criterion_key(gap["description"])} for gap in concept.get("criterion_gaps", {}).values()}
    concept["criterion_gaps"] = gaps
    # Preserve an earlier v1 projection if it had not yet stored criterion identities.
    for description in concept["missing_criteria"]:
        gid = criterion_key(description)
        gaps.setdefault(gid, {"criterion_id": gid, "description": description, "status": "open", "evidence_ids": [], "resolved_by_evidence_ids": []})
    resolved_criteria = []
    if meaningful:
        missing, met = set(evidence["missing_criteria"]), set(evidence["met_criteria"])
        for criterion in question["criteria"]:
            gid = criterion_key(criterion["description"])
            if criterion["criterion_id"] in missing:
                gap = gaps.setdefault(gid, {"criterion_id": gid, "description": criterion["description"], "status": "open", "evidence_ids": [], "resolved_by_evidence_ids": []})
                gap["status"] = "open"
                gap["evidence_ids"].append(evidence["evidence_id"])
            elif criterion["criterion_id"] in met and gid in gaps and gaps[gid]["status"] == "open":
                gaps[gid]["status"] = "addressed"
                gaps[gid]["resolved_by_evidence_ids"].append(evidence["evidence_id"])
                resolved_criteria.append(gaps[gid]["description"])
        concept["missing_criteria"] = [gap["description"] for gap in gaps.values() if gap["status"] == "open"]
    hypothesis_changes = []
    if weight and evidence["category"] == "misconception" and evidence.get("misconception"):
        # The extractor reuses a matching prior description. IDs are server-owned.
        hid = next((key for key, value in concept["hypotheses"].items() if value["description"] == evidence["misconception"]), "hyp_" + hashlib.sha256(evidence["misconception"].encode()).hexdigest()[:20])
        hypothesis = concept["hypotheses"].setdefault(hid, {"hypothesis_id": hid, "description": evidence["misconception"], "score": .1,
             "status": "suspected", "supporting_evidence_ids": [], "contradicting_evidence_ids": []})
        old = hypothesis["score"]
        hypothesis["score"] = round(min(.98, old + (1-old) * .5 * weight), 5)
        hypothesis["supporting_evidence_ids"].append(evidence["evidence_id"])
        hypothesis["status"] = "recurring" if len(hypothesis["supporting_evidence_ids"]) >= 2 and hypothesis["score"] >= .6 else "suspected"
        hypothesis_changes.append({"hypothesis_id": hid, "before": old, "after": hypothesis["score"], "reason": "supporting quoted evidence"})
    if weight and evidence["category"] in ("understanding", "partial"):
        for hid in evidence["contradicts_hypothesis_ids"]:
            hypothesis = concept["hypotheses"][hid]
            old = hypothesis["score"]
            hypothesis["score"] = round(old * (1 - .7 * weight), 5)
            hypothesis["contradicting_evidence_ids"].append(evidence["evidence_id"])
            hypothesis["status"] = "weakened" if hypothesis["score"] < .25 else "suspected"
            hypothesis_changes.append({"hypothesis_id": hid, "before": old, "after": hypothesis["score"], "reason": "explicit contradictory quoted evidence"})
    change = {"concept_id": evidence["concept_id"], "rule": "evidence_hold" if not weight else "weighted_criterion_update",
              "confidence": confidence, "coverage": coverage, "novelty_factor": novelty, "informativeness": informativeness,
              "duplicate_response": duplicate, "weight": weight, "delta_alpha": da, "delta_beta": db,
              "before_alpha": before["alpha"], "before_beta": before["beta"], "after_alpha": concept["alpha"], "after_beta": concept["beta"],
              "before_estimate": before["estimate"], "after_estimate": concept["estimate"], "delta_estimate": round(concept["estimate"] - before["estimate"], 5),
              "before_uncertainty": before["uncertainty"], "after_uncertainty": concept["uncertainty"], "hypothesis_changes": hypothesis_changes, "resolved_criteria": resolved_criteria,
              "formula": "w=confidence×novelty×min(tokens/16,1); α′=α+w×rubric_coverage; β′=β+w×(1−coverage)×slip_factor; estimate=α′/(α′+β′)",
              "limitation": "Heuristic evidence estimate, not a calibrated measurement of understanding. Tutor coaching never updates these parameters."}
    return result, change


def plan_next(state, definition, previous_questions, prior_turns):
    concepts = state["concepts"]
    last = prior_turns[-1] if prior_turns else None
    target, intent, reason = None, "diagnose", "Collect an initial explanation; there is no evidence for this concept yet."
    if last:
        cid = last["evidence"]["concept_id"]
        current = concepts[cid]
        repeated = sum(q["concept_id"] == cid for q in previous_questions[-3:])
        needs_repair = last["evidence"]["category"] in ("partial", "misconception", "prerequisite_gap", "slip", "ambiguous")
        if needs_repair and repeated < 3:
            target, intent = cid, "repair"
            reason = "The last saved explanation leaves these criteria unverified: " + "; ".join(current["missing_criteria"] or ["reasoning is still ambiguous"])
        elif needs_repair:
            parents = next(c["prerequisites"] for c in definition["concepts"] if c["concept_id"] == cid)
            if parents:
                target, intent = min(parents, key=lambda p: concepts[p]["estimate"]), "repair"
                reason = "Repeated difficulty in the saved evidence warrants checking a prerequisite, without assuming it is missing."
    if target is None:
        pending_beliefs = [c for c in concepts.values() if not c["missing_criteria"] and any(h["status"] != "weakened" for h in c["hypotheses"].values())]
        if pending_beliefs:
            candidate = min(pending_beliefs, key=lambda c: c["estimate"])
            target, intent = candidate["concept_id"], "verify_hypothesis"
            reason = "An earlier candidate belief has not been explicitly contradicted. Use a fresh contrast or counterexample to check it rather than silently dismissing it."
    if target is None:
        unresolved = [c for c in concepts.values() if c["missing_criteria"]]
        if unresolved:
            candidate = min(unresolved, key=lambda c: (c["estimate"], c["informative_count"]))
            target, intent = candidate["concept_id"], "repair"
            reason = "Previously saved criteria remain unverified across conversations: " + "; ".join(candidate["missing_criteria"])
    if target is None:
        # Unassessed foundations first; afterwards revisit low estimates with uncertainty.
        target = min(definition["concepts"], key=lambda c: (concepts[c["concept_id"]]["informative_count"] > 0,
                     concepts[c["concept_id"]]["informative_count"], len(c["prerequisites"]), concepts[c["concept_id"]]["estimate"]))["concept_id"]
        if concepts[target]["informative_count"]:
            intent = "verify_transfer"
            reason = "Use a fresh situation to check whether saved evidence of understanding transfers; do not assume improvement."
    count = sum(q["concept_id"] == target for q in previous_questions)
    kinds = ["explain", "apply", "contrast", "debug", "transfer"]
    kind = "guided_practice" if intent == "repair" else ("contrast" if count % 2 else "debug") if intent == "verify_hypothesis" else kinds[count % len(kinds)]
    active = [h for h in concepts[target]["hypotheses"].values() if h["status"] != "weakened"]
    required = []
    if intent == "repair":
        required = [{"criterion_id": criterion_key(description), "description": description} for description in concepts[target]["missing_criteria"][:3]]
        if not required and last and last["evidence"]["concept_id"] == target:
            required = [{"criterion_id": criterion_key(c["description"]), "description": c["description"]} for c in last["question"]["criteria"] if c["criterion_id"] in last["evidence"]["missing_criteria"]][:3]
        prior_repairs = [q for q in previous_questions if q["concept_id"] == target and q.get("generation_trace", {}).get("plan", {}).get("intent") == "repair"]
        if prior_repairs:
            # Isolate one unresolved skill and change the task format rather than repeat
            # the same compound guided-practice question indefinitely.
            formats = ["contrast", "debug", "apply", "guided_practice"]
            kind = formats[(len(prior_repairs) - 1) % len(formats)]
            required = required[:1]
            reason += " Focus on one remaining skill in a different question format."
    return {"concept_id": target, "kind": kind, "intent": intent, "reason": reason,
            "difficulty": "foundational" if intent == "repair" or not concepts[target]["informative_count"] else "transfer",
            "missing_criteria": concepts[target]["missing_criteria"], "required_criteria": required, "hypotheses": active,
            "evidence_ids": concepts[target]["evidence_ids"][-10:], "state_snapshot_id": state["snapshot_id"], "policy_version": PLANNER_VERSION}
