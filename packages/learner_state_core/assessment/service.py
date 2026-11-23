"""Database-first assessment loop. Model calls never hold a SQLite write lock."""
# Hash structured requests to detect retries and conflicting reuse of a request ID.
import hashlib
# Serialize request context and log records in a stable format.
import json
# Record safe diagnostic details for backend failures.
import logging
# Measure the duration of model calls and the full answer flow.
import time
# Compare new question wording with prior questions to reject near duplicates.
from difflib import SequenceMatcher
# Return errors with HTTP statuses the frontend can handle.
from fastapi import HTTPException
# Validate model outputs against the expected topic, question and evidence shapes.
from .models import ExtractedObservation, GeneratedQuestion, TopicDefinition, new_id
# Use the same preprocessing rules for topic keys, response spans and retrieval context.
from .nlp import PREPROCESSING_VERSION, analyze, normalize, text_hash
# Keep scoring and planning in separate deterministic functions.
from .state import initialize, integrate, plan_next
# Use the assessment store for SQL records and consistent UTC timestamps.
from .storage import AssessmentStore, now
# Wrap important work in inspectable stages and track the current execution.
from .execution import CURRENT, VALIDATION_CODES, call, stage, traced
# Record which question and evidence prompt contracts were used.
from .provider import EVIDENCE_PROMPT_VERSION, QUESTION_PROMPT_VERSION

# Route assessment lifecycle messages through the learner pipeline logger.
logger = logging.getLogger("learner.pipeline")


# Turn a request or model context into a reproducible fingerprint.
def digest(payload):
    # Sort dictionary keys before hashing so key order cannot change the result.
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# Prepare a question that is safe to show the learner before answering.
def public_question(question):
    # An assessment may be waiting for a question.
    if question is None:
        # Preserve the absence of a question.
        return None
    # Hide the grading rubric, reference answer and internal provider payload field.
    return {k: v for k, v in question.items() if k not in ("criteria", "reference_answer", "_usage")}


# Coordinate the full live assessment loop around saved conversation and topic records.
class AssessmentService:
    # Receive the database and provider as dependencies so tests can use isolated storage and a
    # fixture model.
    def __init__(self, db, provider):
        # Keep these dependencies for the operations that need them.
        self.db, self.provider = db, provider
        # Use a single store for the assessment-specific database tables.
        self.store = AssessmentStore(db)

    # Check availability before an operation that needs model inference.
    def require_provider(self):
        # Only a configured provider can build questions or interpret a new explanation.
        if not self.provider.configured:
            # Return a recoverable availability error for inference-dependent requests.
            raise HTTPException(503, "A question-generation model is not configured. Connect the Azure model in .env; saved conversations and reports remain available.")

    # This check depends only on the supplied notes.
    @staticmethod
    # Apply the minimum material check before indexing study notes.
    def validate_material(notes):
        # A minimum-size guard, not a claim that the material is factual or relevant.
        # Compare the normalized note word count with the eight-word minimum.
        if len(normalize(notes).split()) < 8:
            # Give the learner a concrete way to supply enough material or switch reference mode.
            raise HTTPException(422, 'Add at least 8 words of study material about the topic. A class or grade label alone cannot ground a question. Leave notes empty to use a model-authored reference.')

    # Find usable earlier study notes for this learner and topic.
    def usable_note_sources(self, learner_id, topic_key):
        # Use the current connection so reads can share the surrounding transaction.
        with self.db.connection() as conn:
            # Read only study-note documents in the requested learner/topic scope, in insertion
            # order.
            rows = conn.execute("SELECT document_id, raw_text FROM assessment_documents WHERE learner_id=? AND topic_key=? AND kind='study_notes' ORDER BY rowid", (learner_id, topic_key)).fetchall()
            # Exclude old records that fail the same minimum word check used for new notes.
            return [r['document_id'] for r in rows if len(normalize(r['raw_text']).split()) >= 8]

    # Load a conversation belonging to the learner ID supplied by the request.
    def require_conversation(self, cid, learner_id):
        # Look up the conversation within that learner's scope.
        conversation = self.store.conversation(cid, learner_id)
        # Treat an unknown ID and one outside this scope the same way.
        if not conversation:
            # Return a scoped not-found error for the API.
            raise HTTPException(404, "Conversation not found for this learner.")
        # Send back the saved conversation so callers can use its status, topic and current
        # question.
        return conversation

    # Build the workspace response from saved data.
    def session(self, cid, learner_id):
        # Keep the session reads within one connection.
        with self.db.connection() as conn:
            # Open a read transaction only if the caller has not already done so.
            if not conn.in_transaction:
                # Use one database view for the conversation, state and answers.
                conn.execute("BEGIN")
            # Load the conversation with the learner scope check.
            conversation = self.require_conversation(cid, learner_id)
            # Load the current topic definition and state from the shared topic record.
            topic = self.store.topic(learner_id, conversation["topic_key"])
            # Show the frozen ending state for a completed session and current topic state
            # otherwise.
            state = conversation.get("ending_state") if conversation["status"] == "completed" else topic["state"]
            # Strip embedded state copies from the conversation header because state has its own
            # response field.
            return {"conversation": {k: v for k, v in conversation.items() if k not in ("baseline_state", "ending_state")},
                    # Return the topic structure and the appropriate state view.
                    "definition": topic["definition"], "state": state,
                    # Supply the already accepted answer history in order.
                    "turns": self.store.answers(cid),
                    # Recover the latest received or failed text for the still-active question.
                    "pending_submission": self.store.pending_submission(cid, conversation.get('current_question_id')) if conversation['status'] == 'active' else None,
                    # Show the learner-safe active question only while the session is open.
                    "active_question": public_question(self.store.question(conversation.get("current_question_id"))) if conversation["status"] != "completed" else None,
                    # Include document summaries for the learner and topic.
                    "sources": self.store.documents(learner_id, conversation["topic_key"])}

    # Attach an execution trace to the create operation.
    @traced('create')
    # Start a conversation from a topic and optional notes, reusing persistent memory where
    # available.
    def create(self, req):
        # Check the model before accepting a start operation that may need a new question.
        self.require_provider()
        # Treat whitespace-only notes as absent.
        if req.notes.strip():
            # Reject notes that are too short to ground an assessment.
            self.validate_material(req.notes)
        # Fingerprint the content separately from the caller's retry ID.
        request_hash = digest(req.model_dump(exclude={"request_id"}))
        # Normalize the topic into a stable key so later conversations can reuse its data.
        topic_key = text_hash(req.topic)[:40]
        # Use a read connection for the request and topic lookups.
        with self.db.connection():
            # Check whether this start ID already created a conversation.
            cached = self.store.creation(req.learner_id, req.request_id)
            # When it has, compare the payload to the original.
            if cached:
                # An identical request can be retried safely.
                if cached[1] != request_hash:
                    # Return a conflict for a changed payload using the same ID.
                    raise HTTPException(409, "This start request was already used with different details.")
            # Find the topic map and learner memory, if any, for this normalized topic.
            topic = self.store.topic(req.learner_id, topic_key)
        # Generation may write; release the owning read connection before retrying it.
        # For a retry, use the already saved conversation.
        if cached:
            # Make the reused result visible in the trace.
            CURRENT.get().metadata['cache_hit'] = True
            # Reopen the same session or continue its question generation.
            return self.ensure_question(cached[0]['conversation_id'], req.learner_id)
        # Reuse the existing concept map across sessions.
        definition = topic["definition"] if topic else None
        # Generate a map only when this learner has no topic definition yet.
        if definition is None:
            # Handle provider or schema failures before writing any assessment records.
            try:
                # Trace construction and validation of the structured topic map.
                definition = call('topic', lambda: TopicDefinition.model_validate(self.provider.build_topic(req.topic, req.notes)).model_dump(),
                                  # Preserve the topic and note length as trace input, with the
                                  # model and schema name.
                                  {'topic': req.topic, 'notes_character_count': len(req.notes)}, {'model': self.provider.model_name, 'contract': 'TopicDefinition'})
            # Convert provider and returned-data failures into one actionable start error.
            except Exception as error:
                # Record the error type without logging the entire provider exception body.
                logger.warning(json.dumps({"event": "topic_generation_failed", "error_type": type(error).__name__}))
                # Tell the learner the map failed before their notes were saved.
                raise HTTPException(502, "The model could not build a valid topic map. Your notes have not been saved; retry or add clearer source material.") from None
        # Start a short transaction for topic initialization, sources and the conversation record.
        with stage('persistence', {'kind': 'topic_and_conversation'}) as persisted, self.db.transaction():
            # Check the request again under the write lock in case another request won the race.
            cached = self.store.creation(req.learner_id, req.request_id)
            # Reuse an already saved conversation when a concurrent call used the same request ID.
            if cached:
                # Compare the original fingerprint again before reuse.
                if cached[1] != request_hash:
                    # Reject reuse with different details.
                    raise HTTPException(409, "This start request was already used with different details.")
                # Keep the saved conversation identity.
                cid = cached[0]["conversation_id"]
            # Create records only when no matching start request has committed.
            else:
                # Re-read topic memory under the lock to see if another session initialized it.
                topic = self.store.topic(req.learner_id, topic_key)
                # Initialize only on the very first conversation for this learner and topic.
                if topic is None:
                    # Build the default per-concept Beta prior and empty evidence records.
                    state = initialize(definition)
                    # Store both the current topic projection and the first history snapshot.
                    self.store.put_topic(req.learner_id, topic_key, definition, state)
                # Otherwise preserve the shared concept map and accumulated state.
                else:
                    # Use the committed definition and state as this conversation's baseline.
                    definition, state = topic["definition"], topic["state"]
                # Find older usable notes before deciding which source IDs to pin.
                note_ids = self.usable_note_sources(req.learner_id, topic_key)
                # Prefer the notes supplied specifically for this conversation.
                if req.notes.strip():
                    # Index the exact notes and pin that one document.
                    note_ids = [self.store.add_document(req.learner_id, topic_key, "study_notes", req.topic + " · supplied notes", req.notes)]
                # If there are no usable notes, check for an existing model-authored reference.
                elif not note_ids and not any(d['kind'] == 'model_reference' for d in self.store.documents(req.learner_id, topic_key)):
                    # Combine the topic overview and concept definitions into a searchable
                    # reference.
                    reference = definition["overview"] + "\n\n" + "\n\n".join(c["name"] + ": " + c["description"] for c in definition["concepts"])
                    # Save it with an explicit unverified model-reference label.
                    self.store.add_document(req.learner_id, topic_key, "model_reference", req.topic + " · model-authored reference (unverified)", reference)
                # Give the new conversation its own server-generated ID.
                cid = new_id("conversation")
                # List stored documents so a reference ID can be selected.
                documents = self.store.documents(req.learner_id, topic_key)
                # Label the pinned source type separately from its IDs.
                kind = 'study_notes' if note_ids else 'model_reference'
                # New conversations never inherit grade-only legacy documents.
                # Pin eligible notes or the saved model-reference document IDs.
                source_ids = note_ids or [d['document_id'] for d in documents if d['kind'] == 'model_reference']
                # Start the conversation record with its identity and owner/topic scope.
                payload = {"conversation_id": cid, "learner_id": req.learner_id, "topic_key": topic_key, "title": req.topic,
                           # Begin in the needs-question state with an initial version and
                           # timestamps.
                           "status": "needs_question", "version": 0, "created_at": now(), "updated_at": now(),
                           # Record the current topic snapshot as the starting point for later
                           # reporting.
                           "current_question_id": None, "answered_count": 0, "baseline_state": state,
                           # Record whether the reference came from supplied notes, topic memory or
                           # model text.
                           "last_error": None, "source_mode": "study_notes" if req.notes.strip() else "topic_memory" if topic else "model_reference",
                           # Store both the reference kind and exact IDs so other conversations
                           # cannot silently change it.
                           'reference_kind': kind, 'reference_source_ids': source_ids}
                # Save the record with the retry ID and request fingerprint.
                self.store.create_conversation(payload, req.request_id, request_hash)
            # Record the resulting IDs for the persistence stage.
            persisted['output'] = {'conversation_id': cid, 'topic_key': topic_key, 'committed': True}
        # Attach the newly known conversation ID to the execution trace.
        CURRENT.get().conversation_id = cid
        # Generate the first question after the creation transaction ends.
        return self.ensure_question(cid, req.learner_id)

    # Retrieve context and expose the search inputs and settings in a trace stage.
    def retrieve(self, conversation, state, concept_id):
        # Run the actual retrieval under the named observability stage.
        return call('retrieval', lambda: self._retrieve(conversation, state, concept_id),
                    # Record the target snapshot, BM25 ranking and within-scope fallback.
                    {'concept_id': concept_id, 'snapshot_id': state['snapshot_id']}, {'limit': 5, 'ranking': 'bm25', 'fallback': 'scoped_recent'})

    # Build the factual material and historical evidence lists used downstream.
    def _retrieve(self, conversation, state, concept_id):
        # Load the topic definition needed to name the target concept.
        topic = self.store.topic(conversation["learner_id"], conversation["topic_key"])
        # Look up the target's description in the saved map.
        concept = next(c for c in topic["definition"]["concepts"] if c["concept_id"] == concept_id)
        # Search using the concept name, its description and open skills.
        query = concept["name"] + " " + concept["description"] + " " + " ".join(state["concepts"][concept_id]["missing_criteria"])
        # Scope both searches to this learner and normalized topic.
        scope = (conversation["learner_id"], conversation["topic_key"], query)
        # Filter before ranking: learner evidence can never become factual material.
        # Read the conversation's pinned source document IDs.
        source_ids = conversation.get('reference_source_ids')
        # Legacy conversations have no pinned selection. Preserve their saved
        # question provenance rather than adopt notes added in another conversation.
        # Older sessions may lack a pinned source list.
        if source_ids is None:
            # Look for their already saved questions.
            questions = self.store.questions(conversation['conversation_id'])
            # Use the latest question as the available record of what material was used.
            if questions:
                # Extract only previously used reference passages from the stored generation
                # context.
                used = questions[-1]['generation_trace']['context']['retrieval']['materials']
                # Keep each cited document ID once, in its original order.
                source_ids = list(dict.fromkeys(m['source_id'] for m in used))
        # Use the pinned reference kind if present. Otherwise infer the source type for unpinned
        # legacy records.
        kind = conversation.get('reference_kind') or (None if source_ids is not None else 'study_notes' if any(
            # Prefer stored study notes when that legacy topic has them.
            d['kind'] == 'study_notes' for d in self.store.documents(*scope[:2])) else 'model_reference')
        # Fetch up to five matching factual passages.
        materials = self.store.retrieve(*scope, limit=5, material_kind=kind,
                                        # Restrict those results to the conversation's document
                                        # selection.
                                        source_ids=source_ids)
        # Return the query, search scope and retrieval method along with the hits.
        return {"query": query, "scope": {"learner_id": scope[0], "topic_key": scope[1]}, "method": "SQLite FTS5 · Porter/Unicode tokens · BM25",
                # Retrieve older learner evidence in its own search, with a separate limit of five.
                "materials": materials, "evidence": self.store.retrieve(*scope, evidence=True, limit=5)}

    # Change the reference for a conversation that is waiting for a question.
    def update_sources(self, cid, req):
        # Require a provider because the service will attempt generation after updating sources.
        self.require_provider()
        # Load the current conversation and material list before any model call.
        with self.db.connection():
            # Validate the learner scope.
            conversation = self.require_conversation(cid, req.learner_id)
            # Allow source changes only when a question is pending generation.
            if conversation['status'] != 'needs_question':
                # Protect the grading material of an already active question.
                raise HTTPException(409, 'Study material can be changed while a conversation is waiting for a question.')
            # Read the topic record for the current learner and conversation.
            topic = self.store.topic(req.learner_id, conversation['topic_key'])
            # Load the available reference document summaries.
            documents = self.store.documents(req.learner_id, conversation['topic_key'])
        # Choose between supplied notes and a model-authored reference.
        if req.mode == 'study_notes':
            # Apply the same minimum note check used during conversation creation.
            self.validate_material(req.notes)
            # Keep the original note text for storage and chunking.
            text = req.notes
        # Handle the explicit model-reference selection.
        else:
            # Reuse the saved reference if present. Otherwise author one without changing
            # the saved concept map, evidence, or learner state.
            # Use no new text if the existing model reference can be reused.
            text = None
            # Ask for model-authored reference text only if there is none to reuse.
            if not any(d['kind'] == 'model_reference' for d in documents):
                # Handle provider errors before trying to change saved source settings.
                try:
                    # Validate the returned topic map so it can supply usable reference text.
                    definition = TopicDefinition.model_validate(self.provider.build_topic(conversation['title'], '')).model_dump()
                    # Build the reference from the generated overview and concept descriptions.
                    text = definition['overview'] + '\n\n' + '\n\n'.join(c['name'] + ': ' + c['description'] for c in definition['concepts'])
                # Catch unavailable or invalid model output.
                except Exception:
                    # Give the learner a retry path while preserving saved evidence.
                    raise HTTPException(502, 'The reference could not be generated. Your saved evidence is unchanged; retry or supply study notes.') from None
        # Open a short transaction for the source selection and conversation update.
        with self.db.transaction():
            # Reload the conversation before writing to avoid replacing newer settings.
            current = self.require_conversation(cid, req.learner_id)
            # Compare the version and still-waiting status captured before the model call.
            if current['version'] != conversation['version'] or current['status'] != 'needs_question':
                # Ask the UI to reload if something else has changed this conversation.
                raise HTTPException(409, 'The conversation changed. Reload the saved conversation before changing its sources.')
            # Store and index the new source if text was provided or generated.
            if text is not None:
                # Save the document under the current learner, topic and requested source mode.
                sources = [self.store.add_document(req.learner_id, current['topic_key'], req.mode,
                           # Label model-authored references as unverified when constructing the
                           # document title.
                           current['title'] + (' · supplied notes' if req.mode == 'study_notes' else ' · model-authored reference (unverified)'), text)]
            # Otherwise use references already saved for this topic.
            else:
                # Select the saved model-reference document IDs.
                sources = [d['document_id'] for d in documents if d['kind'] == 'model_reference']
            # Pin the new kind and IDs on this conversation.
            current.update(reference_kind=req.mode, reference_source_ids=sources, source_mode=req.mode,
                           # Clear the generation error and advance the version and update time.
                           last_error=None, generation_failure_code=None, version=current['version'] + 1, updated_at=now())
            # Persist the updated source selection.
            self.store.save_conversation(current)
        # Attempt to generate using the new sources after committing them.
        return self.ensure_question(cid, req.learner_id)

    # Track generation as a separate execution, including retries after a saved answer.
    @traced('generation')
    # Return an existing active question or generate one from a committed state snapshot.
    def ensure_question(self, cid, learner_id):
        # Confirm inference is available.
        self.require_provider()
        # Collect planning input on a database connection before calling the model.
        with self.db.connection():
            # Read the learner's current session.
            conversation = self.require_conversation(cid, learner_id)
            # An existing question or a completed conversation needs no new question.
            if conversation["status"] == "completed" or conversation.get("current_question_id"):
                # Return the saved workspace view immediately.
                return self.session(cid, learner_id)
            # Read the topic map and current learner memory.
            topic = self.store.topic(learner_id, conversation["topic_key"])
            # Capture the snapshot used for this generation attempt.
            state = topic["state"]
            # Read this conversation's prior questions for planning and format selection.
            local_questions = self.store.questions(cid)
            # Read recent questions from all sessions on the topic for novelty checking.
            previous = self.store.topic_questions(learner_id, conversation["topic_key"])
            # Load accepted answers from this conversation for the deterministic planner.
            turns = self.store.answers(cid)
            # Choose concept, question kind and intent in Python.
            plan = call('planning', lambda: plan_next(state, topic['definition'], local_questions, turns),
                        # Record which snapshot and recent turn IDs informed that planning
                        # decision.
                        {'state_snapshot_id': state['snapshot_id'], 'prior_turn_ids': [t['turn_id'] for t in turns[-10:]]})
            # Fetch supporting material and separate diagnostic evidence for the target concept.
            retrieval = self.retrieve(conversation, state, plan["concept_id"])
            # Build the generation context around the saved topic, state and plan.
            context = {"topic": topic["definition"], "state": state, "plan": plan, "retrieval": retrieval,
                       # Allow only IDs belonging to retrieved reference material as citations.
                       "allowed_source_citations": [m['chunk_id'] for m in retrieval['materials']],
                       # Give the model a bounded set of earlier prompts to help it vary the next
                       # task.
                       "previous_questions": [{"prompt": q["prompt"], "kind": q["kind"], "concept_id": q["concept_id"]} for q in previous[-30:]],
                       # Label the generation prompt contract in the context.
                       "contract_version": QUESTION_PROMPT_VERSION}
        # Start a timer before the provider and retry loop.
        started = time.perf_counter()
        # Prepare a place for the accepted question or its structured failure.
        question, failure, code, attempts = None, None, None, 0
        # Give the service at most two generation attempts.
        for attempts in range(1, 3):
            # Clear the previous attempt's candidate and detected duplicate.
            candidate, similar = None, None
            # Handle generation and validation failures at the attempt boundary.
            try:
                # Trace the check that the search returned enough material.
                with stage('validation', {'material_count': len(retrieval['materials'])}, {'checks': ['minimum_source_material']}, attempts) as source_check:
                    # Count words across the retrieved reference passages.
                    if sum(len(normalize(m['content']).split()) for m in retrieval['materials']) < 8:
                        # Stop this attempt if there is too little material to ground a question.
                        raise ValueError('insufficient_source_material')
                    # Mark that the material precondition passed.
                    source_check['output'] = {'material_available': True}
                # Trace formulation with the plan, retrieval hits and saved snapshot as inputs.
                with stage('generation', {'plan': plan, 'retrieval': retrieval, 'state_snapshot_id': state['snapshot_id']},
                           # Record the model and prompt version for this attempt.
                           {'model': self.provider.model_name, 'prompt_version': QUESTION_PROMPT_VERSION}, attempts) as generated:
                    # Ask the model to propose a question from the prepared context.
                    raw = self.provider.generate_question(context)
                    # Validate fields and lengths against the question contract.
                    candidate = GeneratedQuestion.model_validate(raw).model_dump()
                    # Trace the public question and usage without exposing its private rubric.
                    generated['output'] = {**public_question(candidate), 'usage': raw.get('_usage')}
                # Check the generated result against the server's plan and retrieval context.
                with stage('validation', {'candidate_question': public_question(candidate)}, {'checks': ['target', 'citations', 'rubric_ids', 'required_criteria', 'novelty']}, attempts) as validated:
                    # Build the accepted set of factual chunk IDs for the validator.
                    allowed = {m["chunk_id"] for m in retrieval["materials"]}
                    # Verify the model followed both the assigned concept and kind.
                    if candidate["concept_id"] != plan["concept_id"] or candidate["kind"] != plan["kind"]:
                        # Reject a question that changed the objective chosen by the planner.
                        raise ValueError("target_mismatch")
                    # Compare proposed citations against the retrieved reference set.
                    if not set(candidate["source_citations"]).issubset(allowed):
                        # Reject references outside that set.
                        raise ValueError("unsupported_citations")
                    # Collect all criterion IDs from the proposed rubric.
                    ids = [c["criterion_id"] for c in candidate["criteria"]]
                    # Look for duplicate IDs that would make grading ambiguous.
                    if len(set(ids)) != len(ids):
                        # Reject a rubric that reused an ID for multiple entries.
                        raise ValueError("duplicate_rubric_ids")
                    # Map IDs to exact rubric wording for the recheck comparison.
                    criteria = {c["criterion_id"]: c["description"] for c in candidate["criteria"]}
                    # Verify that every skill the plan required was included verbatim.
                    if any(criteria.get(c["criterion_id"]) != c["description"] for c in plan["required_criteria"]):
                        # Reject questions that lose or change a required recheck.
                        raise ValueError("missing_recheck_criteria")
                    # Normalize wording for a more stable prompt similarity comparison.
                    normalized = normalize(candidate["prompt"]).casefold()
                    # Find a previous prompt more than 94 percent similar by SequenceMatcher.
                    similar = next((q for q in previous if SequenceMatcher(None, normalized, normalize(q["prompt"]).casefold()).ratio() > .94), None)
                    # Fail this attempt if a near duplicate matched.
                    if similar:
                        # Return the repeat reason to the controlled retry handling.
                        raise ValueError("repeated_question")
                    # Keep the candidate only after all deterministic checks passed.
                    question = candidate
                    # Retain provider usage with the saved question.
                    question["usage"] = raw.get("_usage")
                    # Mark the question's validation stage as successful.
                    validated['output'] = {'validated': True}
                # Stop generation attempts once a valid question is ready.
                break
            # Capture a failed result and prepare the next attempt when it can help.
            except Exception as error:
                # Distinguish an invalid returned question from a provider failure.
                failure = "generation_validation_failed" if isinstance(error, ValueError) else "generation_unavailable"
                # Preserve known validation reasons while using safe generic codes for other
                # errors.
                code = str(error) if isinstance(error, ValueError) and str(error) in VALIDATION_CODES else failure
                # Give the next attempt the failure code and the only citable reference IDs.
                context['validation_feedback'] = {'code': code, 'allowed_source_citations': context['allowed_source_citations'],
                    # Restate the plan's assigned concept and format.
                    'assigned_concept_id': plan['concept_id'], 'assigned_kind': plan['kind'],
                    # Tell the retry to preserve required rubric skills and create a genuinely
                    # different task.
                    'instruction': 'Cite ONLY these exact material chunk IDs; never cite learner evidence. Follow the exact concept and kind and retain required criterion IDs/descriptions verbatim. Do not reuse the rejected task or merely reword it. Supply a substantially different concrete case, example data and reasoning task in the question itself; asking the learner to invent a new example does not make a repeated prompt novel.'}
                # Add the rejected prompt when the provider returned one.
                if candidate:
                    # Give the model the exact prompt it must replace.
                    context['validation_feedback']['rejected_question_prompt'] = candidate['prompt']
                # Add extra guidance when the reason was similarity to a saved question.
                if similar:
                    # Show the earlier wording that matched.
                    context['validation_feedback']['similar_question_prompt'] = similar['prompt']
                    # Select a different task structure suited to the assigned question kind.
                    context['validation_feedback']['format_constraint'] = {
                        # Ask for a fresh comparison case when the kind is contrast.
                        'contrast': 'Use two short worked solutions or two learners\' claims about a fresh concrete case; ask the learner to compare and justify them. Do not reuse the previous direct definition-comparison format.',
                        # Ask for a new worked mistake and a repair when the kind is debug.
                        'debug': 'Present a different flawed worked example with concrete data and ask for the first invalid step and its repair.',
                        # Ask for concrete scenario data and reasoning when the kind is apply.
                        'apply': 'Use a new practical scenario with explicit input data and ask the learner to apply and justify the source rule.',
                        # Use a representation or counterexample when the kind is explain.
                        'explain': 'Use a concrete representation or counterexample and ask for an explanation of why the source rule applies, instead of restating a definition.',
                        # Provide a new scaffolded missing step when the kind is guided practice.
                        'guided_practice': 'Give a new partially worked case and ask the learner to complete and justify one missing step.',
                        # Change the setting and representation when testing transfer.
                        'transfer': 'Change the concrete setting and representation while using only the source principles, and ask the learner to justify how the rule transfers.'
                    # Use the instruction for the kind chosen by the planner.
                    }[plan['kind']]
                # Log the known code, attempt number and error type for diagnosis.
                logger.warning(json.dumps({"event": failure, "code": code, "conversation_id": cid, "attempt": attempts, "error_type": type(error).__name__}))
                # A model retry cannot fix missing source material.
                if code == 'insufficient_source_material':
                    # Preserve that specific condition for the workspace recovery screen.
                    failure = code
                    # Leave the retry loop so the learner can update the reference.
                    break
        # Open the transaction that commits the accepted question or tracks generation failure.
        with stage('persistence', {'kind': 'question', 'snapshot_id': state['snapshot_id']}) as persisted, self.db.transaction():
            # Reload the conversation after the model work.
            current = self.require_conversation(cid, learner_id)
            # Check the current topic snapshot under the same write transaction.
            current_topic = self.store.topic(learner_id, current["topic_key"])
            # Detect completion, an already-issued question or a concurrent conversation update.
            if current["version"] != conversation["version"] or current.get("current_question_id") or current["status"] == "completed":
                # Keep the newer saved result when a competing operation already changed the
                # session.
                return self.session(cid, learner_id)
            # Make sure the generated task still reflects the current topic memory.
            if current_topic["state"]["snapshot_id"] != state["snapshot_id"]:
                # Discard the candidate when its planning snapshot is stale.
                question, failure = None, "state_changed_retry"
            # Store an error only when there is no usable question to activate.
            current["last_error"] = failure if question is None else None
            # Retain the more detailed generation failure code for recovery.
            current['generation_failure_code'] = (code or failure) if question is None else None
            # Save a question only when generation and the concurrency checks passed.
            if question:
                # Assign the question an ID and attach its creation and decision context.
                question.update(question_id=new_id("question"), created_at=now(), generation_trace={
                    # Link the exact state, request trace and fingerprint of the generation
                    # context.
                    "trace_id": CURRENT.get().trace_id, "state_snapshot_id": state["snapshot_id"], "context_sha256": digest(context),
                    # Preserve the plan, retrieved context, source citations and model.
                    "context": context, "plan": plan, "source_citations": question["source_citations"], "model": self.provider.model_name,
                    # Record retry count, duration and prompt contract version.
                    "attempts": attempts, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "prompt_version": QUESTION_PROMPT_VERSION})
                # Save the private rubric and provenance with the accepted question.
                self.store.save_question(cid, question)
                # Activate the question for the learner to answer.
                current["current_question_id"], current["status"] = question["question_id"], "active"
            # With no usable question, keep the conversation in a recoverable state.
            else:
                # Leave the workspace waiting so generation can be retried on its own.
                current["status"] = "needs_question"
            # Advance the version so later writes can detect this update.
            current["version"] += 1
            # Stamp the latest change for listing and recovery.
            current["updated_at"] = now()
            # Save the updated question pointer or failure state.
            self.store.save_conversation(current)
            # Record the result of the persistence stage for later inspection.
            persisted['output'] = {'question_id': current['current_question_id'], 'committed': True, 'generation_failure': failure if question is None else None}
        # Reflect a failed generation in the execution summary.
        if question is None:
            # Set the trace status independently from the saved conversation status.
            CURRENT.get().status = 'failed'
            # Store the stable failure reason on the trace.
            CURRENT.get().metadata['failure_code'] = failure
        # A successful generation can link to the newly issued question.
        else:
            # Record that new question ID on the trace.
            CURRENT.get().question_id = question['question_id']
        # Return the latest committed workspace view.
        return self.session(cid, learner_id)

    # Tie the observation to the saved task's assigned concept.
    def ground_observation(self, observation, question, definition):
        # Verify the evidence still names the concept that was asked.
        if observation['concept_id'] != question['concept_id']:
            # Reject any attempt to move the answer to another concept.
            raise ValueError('wrong_concept')
        # Read the authoritative concept entry from the saved topic map.
        concept = next(c for c in definition['concepts'] if c['concept_id'] == observation['concept_id'])
        # Return the accepted concept along with the one eligible candidate.
        return {'accepted_concept': concept, 'candidate_concept_ids': [question['concept_id']],
                # Record that the link came from an assigned-concept check and attach its
                # supporting quote.
                'method': 'assigned_concept_constraint', 'source_quote': observation['source_quote'],
                # Make clear that this match has no similarity score and the topic map is
                # unverified.
                'similarity_score': None, 'taxonomy_provenance': 'model_authored_topic_map_unverified'}

    # Validate the evidence contract before any assessment state can change.
    def validate_observation(self, observation, context):
        # Load the saved question and its relevant state context.
        question, state = context["question"], context["state"]
        # Require the exact concept assigned to the question.
        if observation["concept_id"] != question["concept_id"]:
            # Reject a concept mismatch.
            raise ValueError("wrong_concept")
        # Read the raw quotation supplied by the provider adapter.
        quote = observation["source_quote"]
        # Require verbatim membership and up to three tokens based on how many the full response
        # contains.
        if quote not in context["response_text"] or len(analyze(quote)["token_spans"]) < min(3, analyze(context["response_text"])["token_count"]):
            # Reject fabricated or insufficiently informative quotations under this check.
            raise ValueError("unverifiable_quote")
        # Gather all criterion IDs from the actual saved rubric.
        all_ids = {c["criterion_id"] for c in question["criteria"]}
        # Turn both model judgments into sets for checking coverage and overlap.
        met, missing = set(observation["met_criteria"]), set(observation["missing_criteria"])
        # Require every rubric item once, in exactly one of the two judgment lists.
        if met & missing or met | missing != all_ids or len(met) != len(observation["met_criteria"]) or len(missing) != len(observation["missing_criteria"]):
            # Reject missing, extra, overlapping or duplicate rubric IDs.
            raise ValueError("invalid_rubric_partition")
        # An ambiguous answer cannot simultaneously claim met skills or contradictions.
        if observation["category"] == "ambiguous" and (met or observation["contradicts_hypothesis_ids"]):
            # Reject positive conclusions from an abstention.
            raise ValueError("unsupported_ambiguity_claim")
        # A misconception requires a named candidate belief and a skill that was missed.
        if observation["category"] == "misconception" and (not observation["misconception"] or not missing):
            # Reject a misconception label that lacks those supporting fields.
            raise ValueError("unsupported_misconception")
        # Full understanding requires that all current criteria were met.
        if observation["category"] == "understanding" and missing:
            # Reject any understanding label paired with a missing criterion.
            raise ValueError("incomplete_understanding")
        # Find the known hypotheses for this question's concept.
        hypotheses = state["concepts"][question["concept_id"]]["hypotheses"]
        # Read the specific belief IDs the model claims the answer contradicted.
        contradictions = observation["contradicts_hypothesis_ids"]
        # Forbid repeated IDs and IDs absent from this concept's saved beliefs.
        if len(set(contradictions)) != len(contradictions) or not set(contradictions).issubset(hypotheses):
            # Reject unknown or duplicate contradiction references.
            raise ValueError("unknown_contradiction")
        # Require a supported answer category and at least one met skill for contradiction.
        if contradictions and (observation["category"] not in ("understanding", "partial") or not met):
            # Reject a contradiction that lacks qualifying evidence.
            raise ValueError("unsupported_contradiction")
        # Check every citation against the retrieved factual passages for this answer.
        if not set(observation["source_citations"]).issubset({m["chunk_id"] for m in context["retrieval"]["materials"]}):
            # Reject citation IDs from outside the allowed material.
            raise ValueError("unsupported_source")

    # Journal the answer as an execution separate from the next question.
    @traced('answer')
    # Receive, interpret and commit one explanation before continuing the session.
    def answer(self, cid, req):
        # Ensure a configured model is available to interpret the answer.
        self.require_provider()
        # Start the timing window for extraction provenance.
        started = time.perf_counter()
        # Preserve hashes for pre-existing requests when the optional finish flag is false.
        # Fingerprint the submitted payload while preserving compatibility with older default
        # fields.
        request_hash = digest(req.model_dump(exclude={"request_id"}, exclude_defaults=True))
        # Read current conversation and prior memory before inference.
        with self.db.connection():
            # Ensure the submitted conversation belongs to the learner.
            conversation = self.require_conversation(cid, req.learner_id)
            # Look for an accepted answer already saved with this request ID.
            cached = self.store.answer(cid, req.request_id)
            # Return an existing identical answer without recalculating memory.
            if cached:
                # Compare the saved fingerprint against this submission.
                if cached[1] != request_hash:
                    # Reject a reused request ID with a different answer or intent.
                    raise HTTPException(409, "This request ID already saved a different answer.")
                # Mark the retry as a cache hit and link its accepted turn.
                CURRENT.get().metadata.update(cache_hit=True, turn_id=cached[0]['turn_id'],
                    # Point to the trace from the first accepted submission.
                    source_trace_id=cached[0]['extraction_trace'].get('trace_id'))
                # Return the committed session with that saved answer.
                return self.session(cid, req.learner_id)
            # Verify that this submission is for the only active question.
            if conversation["status"] != "active" or conversation.get("current_question_id") != req.question_id:
                # Refuse to attach an answer to a replaced or completed question.
                raise HTTPException(409, "This question is no longer active. Reload the conversation; your unsaved explanation is still in the editor.")
            # Load the stored prompt and private grading rubric.
            question = self.store.question(req.question_id)
            # Read the current topic revision.
            topic = self.store.topic(req.learner_id, conversation["topic_key"])
            # Keep that exact snapshot for the calculation and eventual write check.
            prior = topic["state"]
            # Build the evidence context from the question, state and original text.
            context = {"question": question, "state": prior, "response_text": req.response_text,
                       # Add matching references and the versioned evidence prompt contract.
                       "retrieval": self.retrieve(conversation, prior, question["concept_id"]), "contract_version": EVIDENCE_PROMPT_VERSION}
        # Record the validated response and learner-visible question in the input stage.
        call('input', lambda: {'response_text': req.response_text, 'question': public_question(question),
                             # Tag the input as learner text for provenance.
                             'session_id': cid, 'source_kind': 'learner_response', 'validated': True},
             # Link this stage to the question and caller request ID.
             {'question_id': req.question_id, 'request_id': req.request_id})
        # Persist a receipt with the original text before asking the model to read it.
        self.store.receive_submission(cid, req, CURRENT.get().trace_id)
        # Compute text features and original Unicode spans for inspection and scoring.
        nlp = call('preprocessing', lambda: analyze(req.response_text), {'original_text': req.response_text},
                   # Record the preprocessing version and explain what the offsets count.
                   {'version': PREPROCESSING_VERSION, 'offsets': 'original Unicode code points', 'negation_removed': False})
        # Start with no accepted observation and an empty list of retry reasons.
        observation, validation_attempts = None, []
        # Give the assessment service two attempts to extract valid evidence.
        for attempt in range(2):
            # Catch contract or provider errors before they can reach state integration.
            try:
                # Journal the model extraction from the current answer and saved snapshot.
                with stage('extraction', {'response_text': req.response_text, 'question_id': req.question_id, 'state_snapshot_id': prior['snapshot_id']},
                           # Record the model and prompt contract while treating confidence as
                           # uncalibrated.
                           {'model': self.provider.model_name, 'prompt_version': EVIDENCE_PROMPT_VERSION, 'confidence_calibrated': False}, attempt + 1) as extracted:
                    # Call the provider to select a quote and judge the current rubric.
                    raw = self.provider.evaluate(context)
                    # Limit the extraction trace to known evidence fields and model usage.
                    extracted['output'] = {k: v for k, v in raw.items() if k in ExtractedObservation.model_fields or k == '_usage'}
                # Start deterministic validation of the returned observation.
                with stage('validation', {'candidate': extracted['output']}, {'schema': 'ExtractedObservation', 'checks': ['schema', 'quote', 'rubric_partition', 'citations', 'contradiction_ids']}, attempt + 1) as validated:
                    # Parse the returned fields into the application evidence schema.
                    observation = ExtractedObservation.model_validate(raw).model_dump()
                    # Check exact quote, rubric integrity, concept, citations and belief IDs.
                    self.validate_observation(observation, context)
                    # Journal the successful checks and whether the answer led to an abstention.
                    validated['output'] = {'schema_valid': True, 'quote_verified': True, 'confidence_calibrated': False, 'abstained': observation['category'] == 'ambiguous'}
                # Record how accepted evidence links back to the assigned concept.
                call('grounding', lambda: self.ground_observation(observation, question, topic['definition']),
                     # Include both model-reported and expected concept IDs in the trace.
                     {'concept_id': observation['concept_id'], 'question_concept_id': question['concept_id']}, attempt=attempt + 1)
                # Stop retrying when the observation passes all checks.
                break
            # If an attempt fails, prepare a bounded retry with correction instructions.
            except Exception as error:
                # Clear any candidate that failed validation or grounding.
                observation = None
                # Keep a recognized reason code or record the exception type.
                code = str(error) if isinstance(error, ValueError) and str(error) in VALIDATION_CODES else type(error).__name__
                # Append the reason to the evidence provenance for the accepted retry, if any.
                validation_attempts.append(code)
                # Tell the next attempt what validation condition failed.
                context["validation_feedback"] = {
                    # Restate the assigned concept.
                    'code': code, 'assigned_concept_id': question['concept_id'],
                    # Supply only the current saved rubric IDs.
                    'allowed_criterion_ids': [c['criterion_id'] for c in question['criteria']],
                    # Supply the retrieved reference IDs that can be cited.
                    'allowed_source_citations': [m['chunk_id'] for m in context['retrieval']['materials']],
                    # Explain how to correct coverage and categories while keeping an exact quote.
                    'instruction': 'Correct the failed check. Partition only the CURRENT question rubric, not historical gaps. Every current criterion must appear exactly once. Use understanding only with no missing criteria; otherwise choose the supported category. Quote the current response exactly and cite only allowed reference sources.'}
        # If both attempts failed, exit before computing or saving accepted state.
        if observation is None:
            # Log identifiers for the rejected answer without logging the full raw response.
            logger.warning(json.dumps({"event": "assessment_evidence_rejected", "conversation_id": cid, "request_id": req.request_id}))
            # Return a recoverable error while leaving the submission receipt available.
            raise HTTPException(502, "The assessment model could not return evidence that passed validation. Your explanation is kept in the editor and no learner state was changed. Please retry this submission.")
        # Give the accepted answer turn a server-owned identity.
        turn_id = new_id('turn')
        # Give its evidence a separate ID and link it to the question and time.
        evidence = {**observation, "evidence_id": new_id("evidence"), "question_id": req.question_id, "timestamp": now(),
                    # Resolve the accepted quote to the first matching start and end offsets in the
                    # original response.
                    "source_start": req.response_text.index(observation["source_quote"]), "source_end": req.response_text.index(observation["source_quote"]) + len(observation["source_quote"]),
                    # Record the extractor and describe its confidence as uncalibrated.
                    "extractor_model": self.provider.model_name, "confidence_calibrated": False,
                    # Link the evidence to the execution, turn and session and record that
                    # validation passed.
                    "trace_id": CURRENT.get().trace_id, "source_interaction_id": turn_id, 'session_id': cid, 'validation_status': 'passed'}
        # Add an explicit evidence stage showing the chosen quoted source.
        call('evidence', lambda: evidence, {'source_kind': 'learner_response', 'source_quote': observation['source_quote']})
        # Start the deterministic state update trace.
        with stage('state_update', {'state_before': prior, 'evidence_id': evidence['evidence_id']}, {'policy': 'criterion-weighted-state-v1'}) as computed:
            # Apply the accepted evidence to a copy of the prior state.
            updated, change = integrate(prior, evidence, question, nlp)
            # Record both snapshots as a computation that is still awaiting persistence.
            computed['output'] = {'state_before': prior, 'state_after': updated, 'changes': change, 'persisted': False}
        # Assemble the accepted turn around the original task and response.
        turn = {"turn_id": turn_id, "timestamp": now(), "question": question, "response_text": req.response_text,
                # Store NLP features, evidence, before and after states and the exact calculation.
                "nlp": nlp, "evidence": evidence, "state_before": prior, "state_after": updated, "state_changes": change,
                # Link the model, snapshot and fingerprint of the extraction context.
                "extraction_trace": {"trace_id": CURRENT.get().trace_id, "context_sha256": digest(context), "state_snapshot_id": prior["snapshot_id"], "model": self.provider.model_name,
                    # Preserve retrieved passages, the retry reasons and prompt version.
                    "retrieval": context["retrieval"], "validation_attempts": validation_attempts, "prompt_version": EVIDENCE_PROMPT_VERSION,
                    # Record successful quote validation, elapsed time and provider usage.
                    "quote_validated": True, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "usage": raw.get("_usage")}}
        # Put accepted-candidate IDs on the execution for lookup.
        CURRENT.get().metadata.update(turn_id=turn['turn_id'], evidence_id=evidence['evidence_id'], snapshot_before_id=prior['snapshot_id'], snapshot_after_id=updated['snapshot_id'])
        # Start the short write transaction after all inference and computation.
        with stage('persistence', {'turn_id': turn['turn_id'], 'evidence_id': evidence['evidence_id'], 'expected_snapshot_id': prior['snapshot_id']}) as persisted, self.db.transaction():
            # Reload the conversation state under the write lock.
            current = self.require_conversation(cid, req.learner_id)
            # Check whether another attempt already committed the same request.
            cached = self.store.answer(cid, req.request_id)
            # If there is a saved answer, verify that it is the same logical submission.
            if cached:
                # Compare the new and saved request fingerprints.
                if cached[1] != request_hash:
                    # Reject the conflict if this ID now refers to different answer details.
                    raise HTTPException(409, "This request ID already saved a different answer.")
                # Use the already committed turn as the authoritative result.
                saved = cached[0]
                # Update the trace to point at the result that won the retry race.
                CURRENT.get().metadata.update(cache_hit=True, turn_id=saved['turn_id'],
                    # Reuse the evidence ID from that committed answer.
                    evidence_id=saved['evidence']['evidence_id'],
                    # Record its actual before-state snapshot ID.
                    snapshot_before_id=saved['state_before']['snapshot_id'],
                    # Record its actual after-state snapshot ID.
                    snapshot_after_id=saved['state_after']['snapshot_id'],
                    # Link to the earlier extraction trace.
                    source_trace_id=saved['extraction_trace'].get('trace_id'),
                    # Mark the losing attempt's candidate calculation as discarded.
                    discarded_computation=True)
                # Journal that this execution reused a saved turn instead of writing its own state.
                persisted['output'] = {'committed': False, 'reused_saved_turn_id': saved['turn_id']}
                # Complete this attempt's receipt by linking it to the existing accepted turn.
                self.store.complete_submission(CURRENT.get().trace_id, saved['turn_id'])
                # Return the authoritative session data without applying a second update.
                return self.session(cid, req.learner_id)
            # Reload the current topic projection before publishing new state.
            latest = self.store.topic(req.learner_id, current["topic_key"])
            # Compare the conversation version, active question and topic snapshot with the
            # originals.
            if current["version"] != conversation["version"] or current.get("current_question_id") != req.question_id or latest["state"]["snapshot_id"] != prior["snapshot_id"]:
                # Ask the learner to reload if concurrent work made the calculation stale.
                raise HTTPException(409, "Saved learner state changed while checking this answer. Reload, then retry your explanation.")
            # Insert the validated turn with its retry ID and fingerprint.
            self.store.save_answer(cid, req.request_id, request_hash, turn)
            # Mark the original-text receipt completed in the same transaction.
            self.store.complete_submission(CURRENT.get().trace_id, turn['turn_id'])
            # Update the current topic state and add its immutable history snapshot.
            self.store.put_topic(req.learner_id, current["topic_key"], topic["definition"], updated, latest["revision"] + 1)
            # Add this answer to a separate learner-evidence search index scope.
            self.store.add_chunk(req.learner_id, current["topic_key"], "learner_evidence", evidence["evidence_id"],
                                 # Make the concept, quote and claim searchable for later
                                 # diagnostic planning.
                                 question["concept_id"] + " " + topic["definition"]["title"] + " " + evidence["source_quote"] + " " + evidence["claim"],
                                 # Keep IDs and the quote needed to trace the result back to the
                                 # accepted turn.
                                 {"evidence_id": evidence["evidence_id"], "concept_id": evidence["concept_id"], "category": evidence["category"], "source_quote": evidence["source_quote"], "conversation_id": cid, "turn_id": turn["turn_id"]})
            # Clear the answered question and prepare the conversation to continue.
            current.update(current_question_id=None, status="needs_question", version=current["version"] + 1,
                           # Advance its answer count, version and update time.
                           answered_count=current["answered_count"] + 1, updated_at=now(), last_error=None)
            # Honor a request to finish with this accepted answer.
            if req.finish_after_answer:
                # Freeze the updated state as the ending snapshot without creating another
                # question.
                current.update(status="completed", ending_state=updated, finished_at=now())
            # Save the conversation status in the same transaction as the evidence and state.
            self.store.save_conversation(current)
            # Attach the committed record IDs and revision to the persistence stage.
            persisted['output'] = {'turn_id': turn['turn_id'], 'evidence_id': evidence['evidence_id'], 'snapshot_id': updated['snapshot_id'], 'revision': latest['revision'] + 1, 'committed': True}
        # Log the accepted save after the database transaction has ended.
        logger.info(json.dumps({"event": "assessment_answer_saved", "conversation_id": cid, "turn_id": turn["turn_id"], "snapshot_id": updated["snapshot_id"], "request_id": req.request_id}))
        # Return the completed workspace or generate using the newly committed memory.
        return self.session(cid, req.learner_id) if req.finish_after_answer else self.ensure_question(cid, req.learner_id)

    # Finish using already accepted evidence without assessing any unsent draft.
    def finish(self, cid, learner_id):
        # Update completion under a write transaction.
        with self.db.transaction():
            # Load the conversation in the caller's learner scope.
            conversation = self.require_conversation(cid, learner_id)
            # A repeated finish should leave the completed record alone.
            if conversation["status"] != "completed":
                # Load accepted turns to identify the last snapshot from this conversation.
                turns = self.store.answers(cid)
                # Use its last accepted state or the starting baseline if no answer was accepted.
                ending = turns[-1]["state_after"] if turns else conversation["baseline_state"]
                # Set the completed status, freeze its ending state and bump the version.
                conversation.update(status="completed", ending_state=ending, finished_at=now(), updated_at=now(), version=conversation["version"] + 1)
                # Store the completion metadata together.
                self.store.save_conversation(conversation)
        # Return the completed session view.
        return self.session(cid, learner_id)

    # Reopen a conversation and check if its earlier question still fits current memory.
    def resume(self, cid, learner_id):
        # Decide whether to reuse the question under a transaction.
        with self.db.transaction():
            # Load the learner's conversation.
            conversation = self.require_conversation(cid, learner_id)
            # Read the saved question if the conversation still has one.
            question = self.store.question(conversation.get("current_question_id"))
            # Load the topic state, which other conversations may have updated.
            topic = self.store.topic(learner_id, conversation["topic_key"])
            # Detect a saved question generated against an older topic snapshot.
            if question and question["generation_trace"]["state_snapshot_id"] != topic["state"]["snapshot_id"]:
                # Clear the stale question pointer so a fresh task can be issued.
                conversation["current_question_id"] = None
            # Set the status based on whether any current question can still be used.
            conversation.update(status="active" if conversation.get("current_question_id") else "needs_question", version=conversation["version"] + 1, updated_at=now())
            # Save the status and version change.
            self.store.save_conversation(conversation)
        # Return the still-valid question or generate a new one after the commit.
        return self.ensure_question(cid, learner_id)
