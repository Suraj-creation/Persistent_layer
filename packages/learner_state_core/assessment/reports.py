"""Reports are deterministic projections of real conversations, never generated summaries."""
from .storage import now
from .service import public_question


def build_report(service, cid, learner_id):
    session = service.session(cid, learner_id)
    conversation = service.require_conversation(cid, learner_id)
    turns = session["turns"]
    baseline = conversation["baseline_state"]
    ending = turns[-1]["state_after"] if turns else baseline
    progress, remaining, unassessed = [], [], []
    for concept in ending["concepts"].values():
        cturns = [t for t in turns if t["evidence"]["concept_id"] == concept["concept_id"]]
        original = baseline["concepts"][concept["concept_id"]]
        delta = round(sum(t["state_changes"]["delta_estimate"] for t in cturns), 5)
        rechecked = [t for t in cturns if t["state_changes"].get("resolved_criteria") and not t["state_changes"]["duplicate_response"]]
        evidence_improved = bool(rechecked)
        relevant_evidence = {eid for gap in concept.get("criterion_gaps", {}).values() if gap["status"] == "open" for eid in gap["evidence_ids"]}
        relevant_evidence.update(eid for hypothesis in concept["hypotheses"].values() if hypothesis["status"] != "weakened" for eid in hypothesis["supporting_evidence_ids"])
        row = {"concept_id": concept["concept_id"], "name": concept["name"], "before_estimate": original["estimate"], "after_estimate": concept["estimate"],
               "conversation_delta": delta, "uncertainty": concept["uncertainty"], "observations": len(cturns), "informative_observations": sum(bool(t["state_changes"]["weight"]) for t in cturns),
               "evidence_of_progress": evidence_improved, "missing_criteria": concept["missing_criteria"], "hypotheses": list(concept["hypotheses"].values()),
               "supporting_quotes": [{"quote": t["evidence"]["source_quote"], "evidence_id": t["evidence"]["evidence_id"], "category": t["evidence"]["category"]} for t in cturns if t["evidence"]["evidence_id"] in relevant_evidence or (not relevant_evidence and t["evidence"]["category"] == "ambiguous")]}
        progress.append(row)
        if not cturns:
            unassessed.append({"concept_id": concept["concept_id"], "name": concept["name"], "reason": "Not checked in this conversation; prior topic evidence may exist."})
        elif concept["missing_criteria"] or any(h["status"] != "weakened" for h in concept["hypotheses"].values()) or not any(t["state_changes"]["weight"] for t in cturns):
            remaining.append(row)
    questions = service.store.questions(cid)
    source_ids = {chunk["source_id"] for question in questions for chunk in question["generation_trace"]["context"]["retrieval"]["materials"]}
    source_ids.update(chunk["source_id"] for turn in turns for chunk in turn["extraction_trace"]["retrieval"]["materials"])
    next_by_source = {q["generation_trace"]["state_snapshot_id"]: q for q in questions}
    timeline = []
    for turn in turns:
        following = next_by_source.get(turn["state_after"]["snapshot_id"])
        timeline.append({**turn, "next_question": public_question(following),
                         "next_question_reason": following["generation_trace"]["plan"]["reason"] if following else "Next question has not been generated or this conversation was ended."})
    return {"report_id": "report_" + cid, "generated_at": now(), "conversation": session["conversation"],
            "source_documents": [source for source in session["sources"] if source["document_id"] in source_ids], "baseline_state": baseline, "ending_state": ending,
            "answered_count": len(turns), "concept_progress": progress, "remaining_gaps": remaining, "unassessed_concepts": unassessed,
            "evidence_of_progress": any(c["evidence_of_progress"] for c in progress), "timeline": timeline,
            "interpretation": "Evidence of progress means a fresh question rechecked an earlier missing criterion and the later explanation addressed it. Different questions are not psychometrically equated; this is evidence to investigate, not a measured causal learning gain.",
            "state_method": {"version": ending["policy_version"], "prior": "Beta(1,1) for an unassessed concept; the 50% prior is not a test score.",
                             "weight": "extraction confidence × response informativeness × novelty; guesses and low-confidence evidence hold state",
                             "update": "alpha += weight × met-criterion proportion; beta += weight × missing-criterion proportion × slip factor",
                             "uncertainty": "Standard deviation of the weighted Beta distribution, not calibrated real-world confidence.",
                             "hypotheses": "Support and explicit contradiction retain source evidence IDs. Coaching never raises an estimate by itself.",
                             "cross_conversation_memory": "Topic state persists across conversations. Conversation deltas include only this conversation's updates."}}
