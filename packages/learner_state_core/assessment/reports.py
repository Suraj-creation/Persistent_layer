"""Reports are deterministic projections of real conversations, never generated summaries."""
# Use the assessment's shared UTC timestamp for the generated report.
from .storage import now
# Apply the same safety filter used for an unanswered workspace question.
from .service import public_question


# Build a report from persisted snapshots and turns.
def build_report(service, cid, learner_id):
    # Read the same saved session data used by the workspace.
    session = service.session(cid, learner_id)
    # Retrieve the full conversation record to access its baseline.
    conversation = service.require_conversation(cid, learner_id)
    # Use only accepted answers in the conversation report.
    turns = session["turns"]
    # Begin with the state saved when the conversation started.
    baseline = conversation["baseline_state"]
    # Use the last accepted turn as the ending state, or the baseline if there were no answers.
    ending = turns[-1]["state_after"] if turns else baseline
    # Collect rows for concept progress, remaining issues and skills not checked here.
    progress, remaining, unassessed = [], [], []
    # Examine every concept in the ending snapshot.
    for concept in ending["concepts"].values():
        # Keep only answers in this conversation that assessed the current concept.
        cturns = [t for t in turns if t["evidence"]["concept_id"] == concept["concept_id"]]
        # Read how the concept looked when this conversation started.
        original = baseline["concepts"][concept["concept_id"]]
        # Sum the changes attributable to this conversation, even if other sessions shared the
        # topic.
        delta = round(sum(t["state_changes"]["delta_estimate"] for t in cturns), 5)
        # Look for a turn that addressed a prior gap with a fresh response fingerprint.
        rechecked = [t for t in cturns if t["state_changes"].get("resolved_criteria") and not t["state_changes"]["duplicate_response"]]
        # Use the existence of such a recheck as the report's progress flag.
        evidence_improved = bool(rechecked)
        # Find evidence IDs that support still-open rubric gaps.
        relevant_evidence = {eid for gap in concept.get("criterion_gaps", {}).values() if gap["status"] == "open" for eid in gap["evidence_ids"]}
        # Add IDs that support candidate misconceptions that remain active.
        relevant_evidence.update(eid for hypothesis in concept["hypotheses"].values() if hypothesis["status"] != "weakened" for eid in hypothesis["supporting_evidence_ids"])
        # Start the concept row with its identity and before/after estimates.
        row = {"concept_id": concept["concept_id"], "name": concept["name"], "before_estimate": original["estimate"], "after_estimate": concept["estimate"],
               # Include the session's own delta, final uncertainty and counts of saved and
               # weighted observations.
               "conversation_delta": delta, "uncertainty": concept["uncertainty"], "observations": len(cturns), "informative_observations": sum(bool(t["state_changes"]["weight"]) for t in cturns),
               # Show progress evidence, skills still open and the saved hypothesis records.
               "evidence_of_progress": evidence_improved, "missing_criteria": concept["missing_criteria"], "hypotheses": list(concept["hypotheses"].values()),
               # Attach relevant quotes, or ambiguous quotes when there is no other supporting
               # evidence.
               "supporting_quotes": [{"quote": t["evidence"]["source_quote"], "evidence_id": t["evidence"]["evidence_id"], "category": t["evidence"]["category"]} for t in cturns if t["evidence"]["evidence_id"] in relevant_evidence or (not relevant_evidence and t["evidence"]["category"] == "ambiguous")]}
        # Include each concept in the complete progress table.
        progress.append(row)
        # Check whether the concept had any accepted turn in this conversation.
        if not cturns:
            # Report it as unassessed here, even if earlier sessions have evidence.
            unassessed.append({"concept_id": concept["concept_id"], "name": concept["name"], "reason": "Not checked in this conversation; prior topic evidence may exist."})
        # Flag assessed concepts with open skills, active beliefs or no positively weighted turn.
        elif concept["missing_criteria"] or any(h["status"] != "weakened" for h in concept["hypotheses"].values()) or not any(t["state_changes"]["weight"] for t in cturns):
            # Add their rows to the remaining gaps section.
            remaining.append(row)
    # Load the stored question history to reconstruct sources and transitions.
    questions = service.store.questions(cid)
    # Find the documents used to formulate the saved questions.
    source_ids = {chunk["source_id"] for question in questions for chunk in question["generation_trace"]["context"]["retrieval"]["materials"]}
    # Add the reference documents used when assessing the saved explanations.
    source_ids.update(chunk["source_id"] for turn in turns for chunk in turn["extraction_trace"]["retrieval"]["materials"])
    # Map the state snapshot that led to each subsequent question.
    next_by_source = {q["generation_trace"]["state_snapshot_id"]: q for q in questions}
    # Start a chronological view for the learner and JSON export.
    timeline = []
    # Walk the accepted answers in their saved order.
    for turn in turns:
        # Find the question generated using the state committed by this answer.
        following = next_by_source.get(turn["state_after"]["snapshot_id"])
        # Add the following question with its private rubric removed.
        timeline.append({**turn, "next_question": public_question(following),
                         # Explain the next task using the saved planning reason or the absence of
                         # a next question.
                         "next_question_reason": following["generation_trace"]["plan"]["reason"] if following else "Next question has not been generated or this conversation was ended."})
    # Identify the report, timestamp its construction and include the session header.
    return {"report_id": "report_" + cid, "generated_at": now(), "conversation": session["conversation"],
            # Return only the source summaries that participated, plus the starting and ending
            # states.
            "source_documents": [source for source in session["sources"] if source["document_id"] in source_ids], "baseline_state": baseline, "ending_state": ending,
            # Include the accepted answer count and the three groups of concept findings.
            "answered_count": len(turns), "concept_progress": progress, "remaining_gaps": remaining, "unassessed_concepts": unassessed,
            # Include the overall progress signal and the recorded timeline.
            "evidence_of_progress": any(c["evidence_of_progress"] for c in progress), "timeline": timeline,
            # Explain the difference between an addressed rubric gap and a measured learning gain.
            "interpretation": "Evidence of progress means a fresh question rechecked an earlier missing criterion and the later explanation addressed it. Different questions are not psychometrically equated; this is evidence to investigate, not a measured causal learning gain.",
            # Label the scoring policy and explain the starting Beta prior.
            "state_method": {"version": ending["policy_version"], "prior": "Beta(1,1) for an unassessed concept; the 50% prior is not a test score.",
                             # Describe how confidence, response length and novelty form the
                             # evidence weight.
                             "weight": "extraction confidence × response informativeness × novelty; guesses and low-confidence evidence hold state",
                             # Describe how rubric coverage and slips affect the two evidence
                             # totals.
                             "update": "alpha += weight × met-criterion proportion; beta += weight × missing-criterion proportion × slip factor",
                             # Explain the uncertainty shown by the report.
                             "uncertainty": "Standard deviation of the weighted Beta distribution, not calibrated real-world confidence.",
                             # Link beliefs to supporting or contradictory learner evidence and
                             # keep coaching outside scoring.
                             "hypotheses": "Support and explicit contradiction retain source evidence IDs. Coaching never raises an estimate by itself.",
                             # Describe how shared topic memory and conversation-specific changes
                             # relate.
                             "cross_conversation_memory": "Topic state persists across conversations. Conversation deltas include only this conversation's updates."}}
