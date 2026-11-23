"""One atomic, observable pipeline for assessment and conversational turns."""
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from .diagnosis.policy import ITEM_BANK, INITIAL_ITEM, POLICY_VERSION, choose_followup, grounded_reply
from .evidence.models import Interaction
from .grounding.taxonomy import FRACTION_TAXONOMY

logger = logging.getLogger("learner.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# Fingerprint compatibility requests for safe retries.
def digest(value):
    # Turn the structured payload into ordered JSON before hashing it.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Recover a saved fraction learner snapshot or initialize its history.
def ensure_snapshot(db, engine, learner_id):
    # Load the existing current snapshot.
    state = db.get_learner_snapshot(learner_id)
    # Initialize only if the learner has no saved state.
    if state is None:
        # Create a prior over the curated fraction curriculum.
        state = engine.initialize_learner_state(learner_id, list(FRACTION_TAXONOMY))
        # Save the starting state for later retrieval and replay.
        db.save_snapshot(state)
    # Check whether a legacy learner needs a history baseline.
    elif not db.get_snapshot_history(learner_id):
        # Preserve the legacy projection as a baseline instead of resetting real data.
        # Preserve their current projection as that first historical snapshot.
        db.save_snapshot(state)
    # Return the authoritative existing or initialized state.
    return state


# Check that compatibility extraction is supportable from the current interaction.
def validate_evidence(evidence, interaction):
    # Select the learner's text or chosen option as the source evidence.
    raw = interaction.response_text or interaction.selected_option or ""
    # Require a nonempty quote present in that source.
    if not evidence.source_quote.strip() or evidence.source_quote not in raw:
        # Reject the result if there is no verifiable quote.
        return False
    # Collect all concept links supplied by the extractor.
    ids = [link.concept_id for link in evidence.concept_links]
    # Require unique concept IDs from the known fraction taxonomy.
    if len(ids) != len(set(ids)) or any(cid not in FRACTION_TAXONOMY for cid in ids):
        # Reject duplicate or outside-taxonomy claims.
        return False
    # Read the extractor's diagnostic category.
    category = evidence.candidate_interpretation.category
    # A mastery claim requires some explanation text.
    if category == "mastery" and not (interaction.response_text or "").strip():
        # Refuse mastery based only on a chosen option.
        return False
    # Reject supportive polarity for guesses, slips, ambiguity or a foundation gap.
    if category in ("ambiguous", "guessing", "prerequisite_gap", "calculation_slip") and evidence.polarity == "supports":
        # Prevent those categories from increasing the concept estimate as mastery.
        return False
    # Also block mastery for a known incorrect answer or explicit uncertainty cues.
    if category == "mastery" and (interaction.correctness == 0 or any(term in raw.lower() for term in ("guess", "not sure", "don't understand", "do not understand"))):
        # Declare that result incompatible with the available evidence.
        return False
    # A named misconception must match a curated candidate for every linked concept.
    if evidence.candidate_interpretation.category == "misconception":
        # Check the taxonomy entry for each linked concept.
        for cid in ids:
            # Read that concept's known misconception IDs.
            known = {m.misconception_id for m in FRACTION_TAXONOMY[cid].known_misconceptions}
            # Compare the proposed ID with that list.
            if evidence.candidate_interpretation.hypothesis_id not in known:
                # Reject an unknown misconception assignment.
                return False
    # Return acceptance once every compatibility check passed.
    return True


# Run the older fraction-workspace path from response to saved evidence and follow-up.
def run_turn(req, db, engine, extractor, rule_extractor, conversational=False):
    # Start the timer for the full compatibility turn.
    started = time.perf_counter()
    # Read the caller's idempotency key.
    request_id = req.request_id
    # Fingerprint the actual operation fields, excluding untrusted client history.
    request_hash = digest(req.model_dump(exclude={"request_id", "chat_history"}))
    # Use the requested local learner identity for all reads and writes.
    learner_id = req.learner_id

    # Resolve the question from persisted conversation, never browser-supplied history.
    # Open a short transaction to resolve any prior result and current question.
    with db.transaction():
        # Look up this request ID within the learner.
        cached = db.get_turn(learner_id, request_id)
        # Handle a previously committed request.
        if cached:
            # Require the same body fingerprint.
            if cached[1] != request_hash:
                # Reject reuse for different input.
                raise HTTPException(409, "This request ID was already used for different input.")
            # Return the earlier accepted result immediately.
            return cached[0]
        # Load state to provide context for interpretation.
        context_state = ensure_snapshot(db, engine, learner_id)
        # Read the conversation history stored by the server.
        turns = db.get_learning_turns(learner_id)
        # Find the recommended unanswered question from the latest turn.
        pending = turns[-1]["recommended_diagnostic_probe"] if turns else None

    # Resolve the assigned item when using the compatibility chat route.
    if conversational:
        # Prefer the requested item or saved pending item, with a curated initial task on a fresh
        # session.
        item_id = req.item_id or (pending["item_id"] if pending else INITIAL_ITEM.item_id)
        # Check whether that explicit ID is stale compared with persisted history.
        if pending and req.item_id and req.item_id != pending["item_id"]:
            # Ask the learner to reload instead of answering a stale problem.
            raise HTTPException(409, "The active question changed. Reload the saved session before answering.")
        # Look for the item in the curated compatibility bank.
        item = ITEM_BANK.get(item_id)
        # A dynamically saved pending item may not be present in the static bank.
        if not item and pending and item_id == pending["item_id"]:
            # Get the structured model used for a diagnostic item.
            from .diagnosis.prober import DiagnosticItem
            # Recover the pending item from its stored data.
            item = DiagnosticItem.model_validate(pending)
        # Check that this question is known.
        if item is None:
            # Return a request error for an unknown item.
            raise HTTPException(422, "Unknown question ID.")
        # Use the saved question text and concept when creating the interaction.
        text, concepts, response = item.question_text, [item.concept_id], req.message
    # For a structured compatibility request, read the supplied assessment fields.
    else:
        # Read the item identity and the submitted explanation.
        item_id, response = req.item_id, req.response_text
        # Check whether it is already a known item.
        item = ITEM_BANK.get(item_id)
        # Prefer the known question wording where available.
        text = item.question_text if item else req.item_text
        # Prefer the known concept, or validate the supplied concepts below.
        concepts = [item.concept_id] if item else req.item_concepts
        # Require a nonempty set of canonical curriculum concepts.
        if not concepts or any(c not in FRACTION_TAXONOMY for c in concepts):
            # Reject an unsupported target.
            raise HTTPException(422, "Provide canonical curriculum concept IDs.")

    # Start without an objective correctness score.
    correctness = None
    # Score a selected option when one was provided.
    if req.selected_option:
        # Use curated option metadata when the item is known.
        if item:
            # Match the submitted value to an option key or its display text.
            selected = next((o for o in item.options if req.selected_option in (o.option_key, o.option_text)), None)
            # Check whether the requested option exists.
            if selected is None:
                # Ask the learner to select a valid option.
                raise HTTPException(422, "Choose an option from the active question.")
            # Convert the known answer's correctness to a numeric value.
            correctness = int(selected.is_correct)
        # For a custom question, try the supported fraction-addition pattern.
        else:
            # Use exact rational arithmetic for the comparison.
            from fractions import Fraction
            # Load the pattern matcher for the custom task format.
            import re
            # Recognize the two numeric fraction operands in the task.
            parts = re.fullmatch(r"\s*(?:Solve:?\s*)?(\d+\s*/\s*\d+)\s*\+\s*(\d+\s*/\s*\d+)\s*\??\s*", text, re.I)
            # Guard the optional computation against invalid fraction values.
            try:
                # Only compare when the question fits this form.
                if parts:
                    # Check the selected answer against the exact fraction sum.
                    correctness = int(Fraction(req.selected_option) == Fraction(parts[1].replace(" ", "")) + Fraction(parts[2].replace(" ", "")))
            # Ignore an unusable numeric value for this optional score.
            except (ValueError, ZeroDivisionError):
                # Leave correctness unknown and continue to language evidence.
                pass
    # Build the original interaction record around the learner and item.
    interaction = Interaction(learner_id=learner_id, item_id=item_id, item_text=text,
                              # Attach the task's skill IDs and the original explanation.
                              item_concepts=concepts, response_text=response,
                              # Preserve the chosen option and any known correctness signal.
                              selected_option=req.selected_option, correctness=correctness,
                              # Include response latency only when the request supplied it.
                              latency_seconds=getattr(req, "latency_seconds", None))
    # Start measuring extraction separately from the total operation.
    extraction_started = time.perf_counter()
    # Assume the primary extractor will succeed unless fallback is needed.
    fallback_reason = None
    # Handle model and evidence validation failures before opening the write transaction.
    try:
        # Prior state is context for interpretation only. It cannot supply a source quote.
        # Prepare the saved context for the extractor.
        context = json.dumps({"prior_state": context_state.model_dump(mode="json") if context_state else None,
                              # Include the definitions of the relevant concepts.
                              "concepts": {cid: FRACTION_TAXONOMY[cid].model_dump() for cid in concepts}})
        # Ask the selected extractor for candidate evidence.
        evidence = extractor.extract(interaction, kc_taxonomy_context=context)
        # Check that the result is supported by the source and taxonomy.
        if not validate_evidence(evidence, interaction):
            # Record that an invalid candidate required fallback.
            fallback_reason = "evidence_validation_failed"
            # Run deterministic extraction for the same interaction.
            evidence = rule_extractor.extract(interaction)
    # Handle an unavailable primary extractor.
    except Exception as error:
        # Persist a sanitized reason, never credentials or provider response bodies.
        # Record the safe fallback reason.
        fallback_reason = "extractor_unavailable"
        # Log the key and exception type for diagnosis.
        logger.warning(json.dumps({"event": "extraction_fallback", "request_id": request_id, "error_type": type(error).__name__}))
        # Use deterministic extraction when the provider is unavailable.
        evidence = rule_extractor.extract(interaction)
    # Link the resulting evidence to the scoped learner.
    evidence.learner_id = learner_id
    # Link it to the exact original interaction.
    evidence.interaction_id = interaction.interaction_id
    # Reuse the interaction timestamp for the evidence.
    evidence.timestamp = interaction.timestamp
    # Mark whether the quote is directly present in the learner's response.
    evidence.is_verbatim_quote = bool(evidence.source_quote and evidence.source_quote in (response or ""))
    # Apply the same acceptance checks after any fallback.
    if not validate_evidence(evidence, interaction):
        # Stop before writing if there is still no verifiable evidence.
        raise HTTPException(422, "No verifiable evidence could be extracted from this response.")
    # Record how long extraction and validation took.
    extraction_ms = round((time.perf_counter() - extraction_started) * 1000, 2)

    # Open the write transaction only after interpretation has finished.
    with db.transaction():
        # Recheck after model latency, including retries received by another worker.
        # Look again for a duplicate attempt after model latency.
        cached = db.get_turn(learner_id, request_id)
        # Use the already accepted result when a competing request saved it.
        if cached:
            # Check that the existing request matches this payload.
            if cached[1] != request_hash:
                # Reject a conflicting reuse.
                raise HTTPException(409, "This request ID was already used for different input.")
            # Return the previously committed result.
            return cached[0]
        # Reload the learner's saved compatibility conversation.
        current_turns = db.get_learning_turns(learner_id)
        # Check whether another chat response advanced the pending task.
        if conversational and len(current_turns) != len(turns):
            # Reject the stale response safely so the learner can reload.
            raise HTTPException(409, "Another response changed this session. Reload before answering.")
        # Read the current snapshot to use for the update.
        prior = ensure_snapshot(db, engine, learner_id)
        # Apply the fraction sandbox state policy and collect the resulting audit events.
        new_state, events = engine.update_state(prior, evidence, interaction)
        # Persist the original interaction.
        db.save_interaction(interaction)
        # Store the accepted evidence in its ledger.
        db.save_evidence(evidence)
        # Save the new current snapshot and its history.
        db.save_snapshot(new_state)
        # Inspect each policy decision emitted by the update.
        for event in events:
            # Append each event to the audit history.
            db.save_audit_event(event)

        # Policy uses the exact saved snapshot and evidence, reloaded inside the transaction.
        # Read the saved post-update snapshot.
        saved = db.get_learner_snapshot(learner_id)
        # Reload the evidence ledger for traceable context.
        ledger = db.get_learner_evidence_ledger(learner_id)
        # Find the exact saved observation used for the follow-up.
        stored_evidence = next(e for e in ledger if e.evidence_id == evidence.evidence_id)
        # Exclude earlier attempted item IDs during question selection.
        attempted = {i["item_id"] for i in db.get_learner_interactions(learner_id)}
        # Choose the next fraction task and record its reason.
        question, rationale = choose_followup(saved, stored_evidence, attempted)
        # Compose deterministic feedback from the stored state and evidence.
        reply = grounded_reply(saved, stored_evidence, question)
        # Record the decision context including the full snapshot and evidence IDs.
        context_payload = {"state": saved.model_dump(mode="json"), "evidence_ids": [e.evidence_id for e in ledger],
                           # Sort attempted item IDs for a stable fingerprint.
                           "attempted_item_ids": sorted(attempted)}
        # Build a trace for the accepted turn and follow-up decision.
        trace = {
            # Assign the trace its own ID and link the caller's retry key.
            "trace_id": f"T_{uuid.uuid4().hex}", "request_id": request_id,
            # Record the UTC time and saved-state source.
            "timestamp": datetime.now(timezone.utc).isoformat(), "retrieval_source": "sqlite",
            # Identify the snapshots before and after the scoring rule.
            "snapshot_before_id": prior.snapshot_id, "snapshot_after_id": saved.snapshot_id,
            # Record the potentially earlier snapshot used when extracting evidence.
            "extraction_context_snapshot_id": context_state.snapshot_id if context_state else None,
            # List evidence for this turn separately from the ledger context.
            "evidence_ids": [evidence.evidence_id], "retrieved_evidence_ids": [e.evidence_id for e in ledger],
            # Hash and save the complete decision context.
            "context_sha256": digest(context_payload), "decision_context": context_payload,
            # Track extractor provenance and any reason it fell back.
            "extractor_model": evidence.extractor_model, "fallback_reason": fallback_reason,
            # Identify the reply mode and follow-up policy.
            "reply_mode": "state_policy", "policy_version": POLICY_VERSION,
            # Save the exact next task that the policy chose.
            "question_id": question.item_id, "question_text": question.question_text,
            # Explain its choice and list the applied state rules.
            "decision_reason": rationale, "applied_rules": [e.applied_rule for e in events],
            # Save extraction timing and elapsed time measured before commit.
            "timings_ms": {"extraction": extraction_ms, "total_before_commit": round((time.perf_counter() - started) * 1000, 2)},
            # Record that the evidence passed the source check.
            "quote_validated": True,
        }
        # Assemble the API-compatible turn with the original interaction and reply.
        result = {"interaction": interaction.model_dump(mode="json"), "reply": reply,
                  # Include serialized evidence and the newly saved state.
                  "evidence": evidence.model_dump(mode="json"), "new_state": saved.model_dump(mode="json"),
                  # Include the audit events emitted for this turn.
                  "events": [e.model_dump(mode="json") for e in events],
                  # Include the recommended question and rationale.
                  "recommended_diagnostic_probe": question.model_dump(), "diagnostic_rationale": rationale,
                  # Attach the same persisted decision trace.
                  "trace": trace}
        # Write the complete turn and retry fingerprint inside the transaction.
        db.save_learning_turn(learner_id, request_id, request_hash, result)
    # Log the committed turn after the write context exits.
    logger.info(json.dumps({"event": "learning_turn_committed", "trace_id": trace["trace_id"], "request_id": request_id,
                           # Include extractor and fallback provenance in the log.
                           "extractor": trace["extractor_model"], "fallback_reason": fallback_reason,
                           # Include the pre-commit duration recorded by the trace.
                           "duration_ms": trace["timings_ms"]["total_before_commit"]}))
    # Return the committed compatibility result.
    return result
