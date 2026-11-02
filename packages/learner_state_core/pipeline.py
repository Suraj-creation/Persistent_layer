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


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_snapshot(db, engine, learner_id):
    state = db.get_learner_snapshot(learner_id)
    if state is None:
        state = engine.initialize_learner_state(learner_id, list(FRACTION_TAXONOMY))
        db.save_snapshot(state)
    elif not db.get_snapshot_history(learner_id):
        # Preserve the legacy projection as a baseline instead of resetting real data.
        db.save_snapshot(state)
    return state


def validate_evidence(evidence, interaction):
    raw = interaction.response_text or interaction.selected_option or ""
    if not evidence.source_quote.strip() or evidence.source_quote not in raw:
        return False
    ids = [link.concept_id for link in evidence.concept_links]
    if len(ids) != len(set(ids)) or any(cid not in FRACTION_TAXONOMY for cid in ids):
        return False
    category = evidence.candidate_interpretation.category
    if category == "mastery" and not (interaction.response_text or "").strip():
        return False
    if category in ("ambiguous", "guessing", "prerequisite_gap", "calculation_slip") and evidence.polarity == "supports":
        return False
    if category == "mastery" and (interaction.correctness == 0 or any(term in raw.lower() for term in ("guess", "not sure", "don't understand", "do not understand"))):
        return False
    if evidence.candidate_interpretation.category == "misconception":
        for cid in ids:
            known = {m.misconception_id for m in FRACTION_TAXONOMY[cid].known_misconceptions}
            if evidence.candidate_interpretation.hypothesis_id not in known:
                return False
    return True


def run_turn(req, db, engine, extractor, rule_extractor, conversational=False):
    started = time.perf_counter()
    request_id = req.request_id
    request_hash = digest(req.model_dump(exclude={"request_id", "chat_history"}))
    learner_id = req.learner_id

    # Resolve the question from persisted conversation, never browser-supplied history.
    with db.transaction():
        cached = db.get_turn(learner_id, request_id)
        if cached:
            if cached[1] != request_hash:
                raise HTTPException(409, "This request ID was already used for different input.")
            return cached[0]
        context_state = ensure_snapshot(db, engine, learner_id)
        turns = db.get_turns(learner_id)
        pending = turns[-1]["recommended_diagnostic_probe"] if turns else None

    if conversational:
        item_id = req.item_id or (pending["item_id"] if pending else INITIAL_ITEM.item_id)
        if pending and req.item_id and req.item_id != pending["item_id"]:
            raise HTTPException(409, "The active question changed. Reload the saved session before answering.")
        item = ITEM_BANK.get(item_id)
        if not item and pending and item_id == pending["item_id"]:
            from .diagnosis.prober import DiagnosticItem
            item = DiagnosticItem.model_validate(pending)
        if item is None:
            raise HTTPException(422, "Unknown question ID.")
        text, concepts, response = item.question_text, [item.concept_id], req.message
    else:
        item_id, response = req.item_id, req.response_text
        item = ITEM_BANK.get(item_id)
        text = item.question_text if item else req.item_text
        concepts = [item.concept_id] if item else req.item_concepts
        if not concepts or any(c not in FRACTION_TAXONOMY for c in concepts):
            raise HTTPException(422, "Provide canonical curriculum concept IDs.")

    correctness = None
    if req.selected_option:
        if item:
            selected = next((o for o in item.options if req.selected_option in (o.option_key, o.option_text)), None)
            if selected is None:
                raise HTTPException(422, "Choose an option from the active question.")
            correctness = int(selected.is_correct)
        else:
            from fractions import Fraction
            import re
            parts = re.fullmatch(r"\s*(?:Solve:?\s*)?(\d+\s*/\s*\d+)\s*\+\s*(\d+\s*/\s*\d+)\s*\??\s*", text, re.I)
            try:
                if parts:
                    correctness = int(Fraction(req.selected_option) == Fraction(parts[1].replace(" ", "")) + Fraction(parts[2].replace(" ", "")))
            except (ValueError, ZeroDivisionError):
                pass
    interaction = Interaction(learner_id=learner_id, item_id=item_id, item_text=text,
                              item_concepts=concepts, response_text=response,
                              selected_option=req.selected_option, correctness=correctness,
                              latency_seconds=getattr(req, "latency_seconds", None))
    extraction_started = time.perf_counter()
    fallback_reason = None
    try:
        # Prior state is context for interpretation only. It cannot supply a source quote.
        context = json.dumps({"prior_state": context_state.model_dump(mode="json") if context_state else None,
                              "concepts": {cid: FRACTION_TAXONOMY[cid].model_dump() for cid in concepts}})
        evidence = extractor.extract(interaction, kc_taxonomy_context=context)
        if not validate_evidence(evidence, interaction):
            fallback_reason = "evidence_validation_failed"
            evidence = rule_extractor.extract(interaction)
    except Exception as error:
        # Persist a sanitized reason, never credentials or provider response bodies.
        fallback_reason = "extractor_unavailable"
        logger.warning(json.dumps({"event": "extraction_fallback", "request_id": request_id, "error_type": type(error).__name__}))
        evidence = rule_extractor.extract(interaction)
    evidence.learner_id = learner_id
    evidence.interaction_id = interaction.interaction_id
    evidence.timestamp = interaction.timestamp
    evidence.is_verbatim_quote = bool(evidence.source_quote and evidence.source_quote in (response or ""))
    if not validate_evidence(evidence, interaction):
        raise HTTPException(422, "No verifiable evidence could be extracted from this response.")
    extraction_ms = round((time.perf_counter() - extraction_started) * 1000, 2)

    with db.transaction():
        # Recheck after model latency, including retries received by another worker.
        cached = db.get_turn(learner_id, request_id)
        if cached:
            if cached[1] != request_hash:
                raise HTTPException(409, "This request ID was already used for different input.")
            return cached[0]
        current_turns = db.get_turns(learner_id)
        if conversational and len(current_turns) != len(turns):
            raise HTTPException(409, "Another response changed this session. Reload before answering.")
        prior = ensure_snapshot(db, engine, learner_id)
        new_state, events = engine.update_state(prior, evidence, interaction)
        db.save_interaction(interaction)
        db.save_evidence(evidence)
        db.save_snapshot(new_state)
        for event in events:
            db.save_audit_event(event)

        # Policy uses the exact saved snapshot and evidence, reloaded inside the transaction.
        saved = db.get_learner_snapshot(learner_id)
        ledger = db.get_learner_evidence_ledger(learner_id)
        stored_evidence = next(e for e in ledger if e.evidence_id == evidence.evidence_id)
        attempted = {i["item_id"] for i in db.get_learner_interactions(learner_id)}
        question, rationale = choose_followup(saved, stored_evidence, attempted)
        reply = grounded_reply(saved, stored_evidence, question)
        context_payload = {"state": saved.model_dump(mode="json"), "evidence_ids": [e.evidence_id for e in ledger],
                           "attempted_item_ids": sorted(attempted)}
        trace = {
            "trace_id": f"T_{uuid.uuid4().hex}", "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "retrieval_source": "sqlite",
            "snapshot_before_id": prior.snapshot_id, "snapshot_after_id": saved.snapshot_id,
            "extraction_context_snapshot_id": context_state.snapshot_id if context_state else None,
            "evidence_ids": [evidence.evidence_id], "retrieved_evidence_ids": [e.evidence_id for e in ledger],
            "context_sha256": digest(context_payload), "decision_context": context_payload,
            "extractor_model": evidence.extractor_model, "fallback_reason": fallback_reason,
            "reply_mode": "state_policy", "policy_version": POLICY_VERSION,
            "question_id": question.item_id, "question_text": question.question_text,
            "decision_reason": rationale, "applied_rules": [e.applied_rule for e in events],
            "timings_ms": {"extraction": extraction_ms, "total_before_commit": round((time.perf_counter() - started) * 1000, 2)},
            "quote_validated": True,
        }
        result = {"interaction": interaction.model_dump(mode="json"), "reply": reply,
                  "evidence": evidence.model_dump(mode="json"), "new_state": saved.model_dump(mode="json"),
                  "events": [e.model_dump(mode="json") for e in events],
                  "recommended_diagnostic_probe": question.model_dump(), "diagnostic_rationale": rationale,
                  "trace": trace}
        db.save_turn(learner_id, request_id, request_hash, result)
    logger.info(json.dumps({"event": "learning_turn_committed", "trace_id": trace["trace_id"], "request_id": request_id,
                           "extractor": trace["extractor_model"], "fallback_reason": fallback_reason,
                           "duration_ms": trace["timings_ms"]["total_before_commit"]}))
    return result
