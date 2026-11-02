"""Database-first assessment loop. Model calls never hold a SQLite write lock."""
import hashlib
import json
import logging
import time
from difflib import SequenceMatcher
from fastapi import HTTPException
from .models import ExtractedObservation, GeneratedQuestion, TopicDefinition, new_id
from .nlp import PREPROCESSING_VERSION, analyze, normalize, text_hash
from .state import initialize, integrate, plan_next
from .storage import AssessmentStore, now
from .execution import CURRENT, VALIDATION_CODES, call, stage, traced
from .provider import EVIDENCE_PROMPT_VERSION, QUESTION_PROMPT_VERSION

logger = logging.getLogger("learner.pipeline")


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def public_question(question):
    if question is None:
        return None
    return {k: v for k, v in question.items() if k not in ("criteria", "reference_answer", "_usage")}


class AssessmentService:
    def __init__(self, db, provider):
        self.db, self.provider = db, provider
        self.store = AssessmentStore(db)

    def require_provider(self):
        if not self.provider.configured:
            raise HTTPException(503, "A question-generation model is not configured. Connect the Azure model in .env; saved conversations and reports remain available.")

    @staticmethod
    def validate_material(notes):
        # A minimum-size guard, not a claim that the material is factual or relevant.
        if len(normalize(notes).split()) < 8:
            raise HTTPException(422, 'Add at least 8 words of study material about the topic. A class or grade label alone cannot ground a question. Leave notes empty to use a model-authored reference.')

    def usable_note_sources(self, learner_id, topic_key):
        with self.db.connection() as conn:
            rows = conn.execute("SELECT document_id, raw_text FROM assessment_documents WHERE learner_id=? AND topic_key=? AND kind='study_notes' ORDER BY rowid", (learner_id, topic_key)).fetchall()
            return [r['document_id'] for r in rows if len(normalize(r['raw_text']).split()) >= 8]

    def require_conversation(self, cid, learner_id):
        conversation = self.store.conversation(cid, learner_id)
        if not conversation:
            raise HTTPException(404, "Conversation not found for this learner.")
        return conversation

    def session(self, cid, learner_id):
        with self.db.connection() as conn:
            if not conn.in_transaction:
                conn.execute("BEGIN")
            conversation = self.require_conversation(cid, learner_id)
            topic = self.store.topic(learner_id, conversation["topic_key"])
            state = conversation.get("ending_state") if conversation["status"] == "completed" else topic["state"]
            return {"conversation": {k: v for k, v in conversation.items() if k not in ("baseline_state", "ending_state")},
                    "definition": topic["definition"], "state": state,
                    "turns": self.store.answers(cid),
                    "active_question": public_question(self.store.question(conversation.get("current_question_id"))) if conversation["status"] != "completed" else None,
                    "sources": self.store.documents(learner_id, conversation["topic_key"])}

    @traced('create')
    def create(self, req):
        self.require_provider()
        if req.notes.strip():
            self.validate_material(req.notes)
        request_hash = digest(req.model_dump(exclude={"request_id"}))
        topic_key = text_hash(req.topic)[:40]
        with self.db.connection():
            cached = self.store.creation(req.learner_id, req.request_id)
            if cached:
                if cached[1] != request_hash:
                    raise HTTPException(409, "This start request was already used with different details.")
            topic = self.store.topic(req.learner_id, topic_key)
        # Generation may write; release the owning read connection before retrying it.
        if cached:
            CURRENT.get().metadata['cache_hit'] = True
            return self.ensure_question(cached[0]['conversation_id'], req.learner_id)
        definition = topic["definition"] if topic else None
        if definition is None:
            try:
                definition = call('topic', lambda: TopicDefinition.model_validate(self.provider.build_topic(req.topic, req.notes)).model_dump(),
                                  {'topic': req.topic, 'notes_character_count': len(req.notes)}, {'model': self.provider.model_name, 'contract': 'TopicDefinition'})
            except Exception as error:
                logger.warning(json.dumps({"event": "topic_generation_failed", "error_type": type(error).__name__}))
                raise HTTPException(502, "The model could not build a valid topic map. Your notes have not been saved; retry or add clearer source material.") from None
        with stage('persistence', {'kind': 'topic_and_conversation'}) as persisted, self.db.transaction():
            cached = self.store.creation(req.learner_id, req.request_id)
            if cached:
                if cached[1] != request_hash:
                    raise HTTPException(409, "This start request was already used with different details.")
                cid = cached[0]["conversation_id"]
            else:
                topic = self.store.topic(req.learner_id, topic_key)
                if topic is None:
                    state = initialize(definition)
                    self.store.put_topic(req.learner_id, topic_key, definition, state)
                else:
                    definition, state = topic["definition"], topic["state"]
                note_ids = self.usable_note_sources(req.learner_id, topic_key)
                if req.notes.strip():
                    note_ids = [self.store.add_document(req.learner_id, topic_key, "study_notes", req.topic + " · supplied notes", req.notes)]
                elif not note_ids and not any(d['kind'] == 'model_reference' for d in self.store.documents(req.learner_id, topic_key)):
                    reference = definition["overview"] + "\n\n" + "\n\n".join(c["name"] + ": " + c["description"] for c in definition["concepts"])
                    self.store.add_document(req.learner_id, topic_key, "model_reference", req.topic + " · model-authored reference (unverified)", reference)
                cid = new_id("conversation")
                documents = self.store.documents(req.learner_id, topic_key)
                kind = 'study_notes' if note_ids else 'model_reference'
                # New conversations never inherit grade-only legacy documents.
                source_ids = note_ids or [d['document_id'] for d in documents if d['kind'] == 'model_reference']
                payload = {"conversation_id": cid, "learner_id": req.learner_id, "topic_key": topic_key, "title": req.topic,
                           "status": "needs_question", "version": 0, "created_at": now(), "updated_at": now(),
                           "current_question_id": None, "answered_count": 0, "baseline_state": state,
                           "last_error": None, "source_mode": "study_notes" if req.notes.strip() else "topic_memory" if topic else "model_reference",
                           'reference_kind': kind, 'reference_source_ids': source_ids}
                self.store.create_conversation(payload, req.request_id, request_hash)
            persisted['output'] = {'conversation_id': cid, 'topic_key': topic_key, 'committed': True}
        CURRENT.get().conversation_id = cid
        return self.ensure_question(cid, req.learner_id)

    def retrieve(self, conversation, state, concept_id):
        return call('retrieval', lambda: self._retrieve(conversation, state, concept_id),
                    {'concept_id': concept_id, 'snapshot_id': state['snapshot_id']}, {'limit': 5, 'ranking': 'bm25', 'fallback': 'scoped_recent'})

    def _retrieve(self, conversation, state, concept_id):
        topic = self.store.topic(conversation["learner_id"], conversation["topic_key"])
        concept = next(c for c in topic["definition"]["concepts"] if c["concept_id"] == concept_id)
        query = concept["name"] + " " + concept["description"] + " " + " ".join(state["concepts"][concept_id]["missing_criteria"])
        scope = (conversation["learner_id"], conversation["topic_key"], query)
        # Filter before ranking: learner evidence can never become factual material.
        source_ids = conversation.get('reference_source_ids')
        # Legacy conversations have no pinned selection. Preserve their saved
        # question provenance rather than adopt notes added in another conversation.
        if source_ids is None:
            questions = self.store.questions(conversation['conversation_id'])
            if questions:
                used = questions[-1]['generation_trace']['context']['retrieval']['materials']
                source_ids = list(dict.fromkeys(m['source_id'] for m in used))
        kind = conversation.get('reference_kind') or (None if source_ids is not None else 'study_notes' if any(
            d['kind'] == 'study_notes' for d in self.store.documents(*scope[:2])) else 'model_reference')
        materials = self.store.retrieve(*scope, limit=5, material_kind=kind,
                                        source_ids=source_ids)
        return {"query": query, "scope": {"learner_id": scope[0], "topic_key": scope[1]}, "method": "SQLite FTS5 · Porter/Unicode tokens · BM25",
                "materials": materials, "evidence": self.store.retrieve(*scope, evidence=True, limit=5)}

    def update_sources(self, cid, req):
        self.require_provider()
        with self.db.connection():
            conversation = self.require_conversation(cid, req.learner_id)
            if conversation['status'] != 'needs_question':
                raise HTTPException(409, 'Study material can be changed while a conversation is waiting for a question.')
            topic = self.store.topic(req.learner_id, conversation['topic_key'])
            documents = self.store.documents(req.learner_id, conversation['topic_key'])
        if req.mode == 'study_notes':
            self.validate_material(req.notes)
            text = req.notes
        else:
            # Reuse the saved reference if present. Otherwise author one without changing
            # the saved concept map, evidence, or learner state.
            text = None
            if not any(d['kind'] == 'model_reference' for d in documents):
                try:
                    definition = TopicDefinition.model_validate(self.provider.build_topic(conversation['title'], '')).model_dump()
                    text = definition['overview'] + '\n\n' + '\n\n'.join(c['name'] + ': ' + c['description'] for c in definition['concepts'])
                except Exception:
                    raise HTTPException(502, 'The reference could not be generated. Your saved evidence is unchanged; retry or supply study notes.') from None
        with self.db.transaction():
            current = self.require_conversation(cid, req.learner_id)
            if current['version'] != conversation['version'] or current['status'] != 'needs_question':
                raise HTTPException(409, 'The conversation changed. Reload the saved conversation before changing its sources.')
            if text is not None:
                sources = [self.store.add_document(req.learner_id, current['topic_key'], req.mode,
                           current['title'] + (' · supplied notes' if req.mode == 'study_notes' else ' · model-authored reference (unverified)'), text)]
            else:
                sources = [d['document_id'] for d in documents if d['kind'] == 'model_reference']
            current.update(reference_kind=req.mode, reference_source_ids=sources, source_mode=req.mode,
                           last_error=None, generation_failure_code=None, version=current['version'] + 1, updated_at=now())
            self.store.save_conversation(current)
        return self.ensure_question(cid, req.learner_id)

    @traced('generation')
    def ensure_question(self, cid, learner_id):
        self.require_provider()
        with self.db.connection():
            conversation = self.require_conversation(cid, learner_id)
            if conversation["status"] == "completed" or conversation.get("current_question_id"):
                return self.session(cid, learner_id)
            topic = self.store.topic(learner_id, conversation["topic_key"])
            state = topic["state"]
            local_questions = self.store.questions(cid)
            previous = self.store.topic_questions(learner_id, conversation["topic_key"])
            turns = self.store.answers(cid)
            plan = call('planning', lambda: plan_next(state, topic['definition'], local_questions, turns),
                        {'state_snapshot_id': state['snapshot_id'], 'prior_turn_ids': [t['turn_id'] for t in turns[-10:]]})
            retrieval = self.retrieve(conversation, state, plan["concept_id"])
            context = {"topic": topic["definition"], "state": state, "plan": plan, "retrieval": retrieval,
                       "allowed_source_citations": [m['chunk_id'] for m in retrieval['materials']],
                       "previous_questions": [{"prompt": q["prompt"], "kind": q["kind"], "concept_id": q["concept_id"]} for q in previous[-30:]],
                       "contract_version": QUESTION_PROMPT_VERSION}
        started = time.perf_counter()
        question, failure, code, attempts = None, None, None, 0
        for attempts in range(1, 3):
            candidate, similar = None, None
            try:
                with stage('validation', {'material_count': len(retrieval['materials'])}, {'checks': ['minimum_source_material']}, attempts) as source_check:
                    if sum(len(normalize(m['content']).split()) for m in retrieval['materials']) < 8:
                        raise ValueError('insufficient_source_material')
                    source_check['output'] = {'material_available': True}
                with stage('generation', {'plan': plan, 'retrieval': retrieval, 'state_snapshot_id': state['snapshot_id']},
                           {'model': self.provider.model_name, 'prompt_version': QUESTION_PROMPT_VERSION}, attempts) as generated:
                    raw = self.provider.generate_question(context)
                    candidate = GeneratedQuestion.model_validate(raw).model_dump()
                    generated['output'] = {**public_question(candidate), 'usage': raw.get('_usage')}
                with stage('validation', {'candidate_question': public_question(candidate)}, {'checks': ['target', 'citations', 'rubric_ids', 'required_criteria', 'novelty']}, attempts) as validated:
                    allowed = {m["chunk_id"] for m in retrieval["materials"]}
                    if candidate["concept_id"] != plan["concept_id"] or candidate["kind"] != plan["kind"]:
                        raise ValueError("target_mismatch")
                    if not set(candidate["source_citations"]).issubset(allowed):
                        raise ValueError("unsupported_citations")
                    ids = [c["criterion_id"] for c in candidate["criteria"]]
                    if len(set(ids)) != len(ids):
                        raise ValueError("duplicate_rubric_ids")
                    criteria = {c["criterion_id"]: c["description"] for c in candidate["criteria"]}
                    if any(criteria.get(c["criterion_id"]) != c["description"] for c in plan["required_criteria"]):
                        raise ValueError("missing_recheck_criteria")
                    normalized = normalize(candidate["prompt"]).casefold()
                    similar = next((q for q in previous if SequenceMatcher(None, normalized, normalize(q["prompt"]).casefold()).ratio() > .94), None)
                    if similar:
                        raise ValueError("repeated_question")
                    question = candidate
                    question["usage"] = raw.get("_usage")
                    validated['output'] = {'validated': True}
                break
            except Exception as error:
                failure = "generation_validation_failed" if isinstance(error, ValueError) else "generation_unavailable"
                code = str(error) if isinstance(error, ValueError) and str(error) in VALIDATION_CODES else failure
                context['validation_feedback'] = {'code': code, 'allowed_source_citations': context['allowed_source_citations'],
                    'assigned_concept_id': plan['concept_id'], 'assigned_kind': plan['kind'],
                    'instruction': 'Cite ONLY these exact material chunk IDs; never cite learner evidence. Follow the exact concept and kind and retain required criterion IDs/descriptions verbatim. Do not reuse the rejected task or merely reword it. Supply a substantially different concrete case, example data and reasoning task in the question itself; asking the learner to invent a new example does not make a repeated prompt novel.'}
                if candidate:
                    context['validation_feedback']['rejected_question_prompt'] = candidate['prompt']
                if similar:
                    context['validation_feedback']['similar_question_prompt'] = similar['prompt']
                    context['validation_feedback']['format_constraint'] = {
                        'contrast': 'Use two short worked solutions or two learners\' claims about a fresh concrete case; ask the learner to compare and justify them. Do not reuse the previous direct definition-comparison format.',
                        'debug': 'Present a different flawed worked example with concrete data and ask for the first invalid step and its repair.',
                        'apply': 'Use a new practical scenario with explicit input data and ask the learner to apply and justify the source rule.',
                        'explain': 'Use a concrete representation or counterexample and ask for an explanation of why the source rule applies, instead of restating a definition.',
                        'guided_practice': 'Give a new partially worked case and ask the learner to complete and justify one missing step.',
                        'transfer': 'Change the concrete setting and representation while using only the source principles, and ask the learner to justify how the rule transfers.'
                    }[plan['kind']]
                logger.warning(json.dumps({"event": failure, "code": code, "conversation_id": cid, "attempt": attempts, "error_type": type(error).__name__}))
                if code == 'insufficient_source_material':
                    failure = code
                    break
        with stage('persistence', {'kind': 'question', 'snapshot_id': state['snapshot_id']}) as persisted, self.db.transaction():
            current = self.require_conversation(cid, learner_id)
            current_topic = self.store.topic(learner_id, current["topic_key"])
            if current["version"] != conversation["version"] or current.get("current_question_id") or current["status"] == "completed":
                return self.session(cid, learner_id)
            if current_topic["state"]["snapshot_id"] != state["snapshot_id"]:
                question, failure = None, "state_changed_retry"
            current["last_error"] = failure if question is None else None
            current['generation_failure_code'] = (code or failure) if question is None else None
            if question:
                question.update(question_id=new_id("question"), created_at=now(), generation_trace={
                    "trace_id": CURRENT.get().trace_id, "state_snapshot_id": state["snapshot_id"], "context_sha256": digest(context),
                    "context": context, "plan": plan, "source_citations": question["source_citations"], "model": self.provider.model_name,
                    "attempts": attempts, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "prompt_version": QUESTION_PROMPT_VERSION})
                self.store.save_question(cid, question)
                current["current_question_id"], current["status"] = question["question_id"], "active"
            else:
                current["status"] = "needs_question"
            current["version"] += 1
            current["updated_at"] = now()
            self.store.save_conversation(current)
            persisted['output'] = {'question_id': current['current_question_id'], 'committed': True, 'generation_failure': failure if question is None else None}
        if question is None:
            CURRENT.get().status = 'failed'
            CURRENT.get().metadata['failure_code'] = failure
        else:
            CURRENT.get().question_id = question['question_id']
        return self.session(cid, learner_id)

    def ground_observation(self, observation, question, definition):
        if observation['concept_id'] != question['concept_id']:
            raise ValueError('wrong_concept')
        concept = next(c for c in definition['concepts'] if c['concept_id'] == observation['concept_id'])
        return {'accepted_concept': concept, 'candidate_concept_ids': [question['concept_id']],
                'method': 'assigned_concept_constraint', 'source_quote': observation['source_quote'],
                'similarity_score': None, 'taxonomy_provenance': 'model_authored_topic_map_unverified'}

    def validate_observation(self, observation, context):
        question, state = context["question"], context["state"]
        if observation["concept_id"] != question["concept_id"]:
            raise ValueError("wrong_concept")
        quote = observation["source_quote"]
        if quote not in context["response_text"] or len(analyze(quote)["token_spans"]) < min(3, analyze(context["response_text"])["token_count"]):
            raise ValueError("unverifiable_quote")
        all_ids = {c["criterion_id"] for c in question["criteria"]}
        met, missing = set(observation["met_criteria"]), set(observation["missing_criteria"])
        if met & missing or met | missing != all_ids or len(met) != len(observation["met_criteria"]) or len(missing) != len(observation["missing_criteria"]):
            raise ValueError("invalid_rubric_partition")
        if observation["category"] == "ambiguous" and (met or observation["contradicts_hypothesis_ids"]):
            raise ValueError("unsupported_ambiguity_claim")
        if observation["category"] == "misconception" and (not observation["misconception"] or not missing):
            raise ValueError("unsupported_misconception")
        if observation["category"] == "understanding" and missing:
            raise ValueError("incomplete_understanding")
        hypotheses = state["concepts"][question["concept_id"]]["hypotheses"]
        contradictions = observation["contradicts_hypothesis_ids"]
        if len(set(contradictions)) != len(contradictions) or not set(contradictions).issubset(hypotheses):
            raise ValueError("unknown_contradiction")
        if contradictions and (observation["category"] not in ("understanding", "partial") or not met):
            raise ValueError("unsupported_contradiction")
        if not set(observation["source_citations"]).issubset({m["chunk_id"] for m in context["retrieval"]["materials"]}):
            raise ValueError("unsupported_source")

    @traced('answer')
    def answer(self, cid, req):
        self.require_provider()
        started = time.perf_counter()
        # Preserve hashes for pre-existing requests when the optional finish flag is false.
        request_hash = digest(req.model_dump(exclude={"request_id"}, exclude_defaults=True))
        with self.db.connection():
            conversation = self.require_conversation(cid, req.learner_id)
            cached = self.store.answer(cid, req.request_id)
            if cached:
                if cached[1] != request_hash:
                    raise HTTPException(409, "This request ID already saved a different answer.")
                CURRENT.get().metadata.update(cache_hit=True, turn_id=cached[0]['turn_id'],
                    source_trace_id=cached[0]['extraction_trace'].get('trace_id'))
                return self.session(cid, req.learner_id)
            if conversation["status"] != "active" or conversation.get("current_question_id") != req.question_id:
                raise HTTPException(409, "This question is no longer active. Reload the conversation; your unsaved explanation is still in the editor.")
            question = self.store.question(req.question_id)
            topic = self.store.topic(req.learner_id, conversation["topic_key"])
            prior = topic["state"]
            context = {"question": question, "state": prior, "response_text": req.response_text,
                       "retrieval": self.retrieve(conversation, prior, question["concept_id"]), "contract_version": EVIDENCE_PROMPT_VERSION}
        call('input', lambda: {'response_text': req.response_text, 'question': public_question(question),
                             'session_id': cid, 'source_kind': 'learner_response', 'validated': True},
             {'question_id': req.question_id, 'request_id': req.request_id})
        nlp = call('preprocessing', lambda: analyze(req.response_text), {'original_text': req.response_text},
                   {'version': PREPROCESSING_VERSION, 'offsets': 'original Unicode code points', 'negation_removed': False})
        observation, validation_attempts = None, []
        for attempt in range(2):
            try:
                with stage('extraction', {'response_text': req.response_text, 'question_id': req.question_id, 'state_snapshot_id': prior['snapshot_id']},
                           {'model': self.provider.model_name, 'prompt_version': EVIDENCE_PROMPT_VERSION, 'confidence_calibrated': False}, attempt + 1) as extracted:
                    raw = self.provider.evaluate(context)
                    extracted['output'] = {k: v for k, v in raw.items() if k in ExtractedObservation.model_fields or k == '_usage'}
                with stage('validation', {'candidate': extracted['output']}, {'schema': 'ExtractedObservation', 'checks': ['schema', 'quote', 'rubric_partition', 'citations', 'contradiction_ids']}, attempt + 1) as validated:
                    observation = ExtractedObservation.model_validate(raw).model_dump()
                    self.validate_observation(observation, context)
                    validated['output'] = {'schema_valid': True, 'quote_verified': True, 'confidence_calibrated': False, 'abstained': observation['category'] == 'ambiguous'}
                call('grounding', lambda: self.ground_observation(observation, question, topic['definition']),
                     {'concept_id': observation['concept_id'], 'question_concept_id': question['concept_id']}, attempt=attempt + 1)
                break
            except Exception as error:
                observation = None
                code = str(error) if isinstance(error, ValueError) and str(error) in VALIDATION_CODES else type(error).__name__
                validation_attempts.append(code)
                context["validation_feedback"] = {
                    'code': code, 'assigned_concept_id': question['concept_id'],
                    'allowed_criterion_ids': [c['criterion_id'] for c in question['criteria']],
                    'allowed_source_citations': [m['chunk_id'] for m in context['retrieval']['materials']],
                    'instruction': 'Correct the failed check. Partition only the CURRENT question rubric, not historical gaps. Every current criterion must appear exactly once. Use understanding only with no missing criteria; otherwise choose the supported category. Quote the current response exactly and cite only allowed reference sources.'}
        if observation is None:
            logger.warning(json.dumps({"event": "assessment_evidence_rejected", "conversation_id": cid, "request_id": req.request_id}))
            raise HTTPException(502, "The assessment model could not return evidence that passed validation. Your explanation is kept in the editor and no learner state was changed. Please retry this submission.")
        turn_id = new_id('turn')
        evidence = {**observation, "evidence_id": new_id("evidence"), "question_id": req.question_id, "timestamp": now(),
                    "source_start": req.response_text.index(observation["source_quote"]), "source_end": req.response_text.index(observation["source_quote"]) + len(observation["source_quote"]),
                    "extractor_model": self.provider.model_name, "confidence_calibrated": False,
                    "trace_id": CURRENT.get().trace_id, "source_interaction_id": turn_id, 'session_id': cid, 'validation_status': 'passed'}
        call('evidence', lambda: evidence, {'source_kind': 'learner_response', 'source_quote': observation['source_quote']})
        with stage('state_update', {'state_before': prior, 'evidence_id': evidence['evidence_id']}, {'policy': 'criterion-weighted-state-v1'}) as computed:
            updated, change = integrate(prior, evidence, question, nlp)
            computed['output'] = {'state_before': prior, 'state_after': updated, 'changes': change, 'persisted': False}
        turn = {"turn_id": turn_id, "timestamp": now(), "question": question, "response_text": req.response_text,
                "nlp": nlp, "evidence": evidence, "state_before": prior, "state_after": updated, "state_changes": change,
                "extraction_trace": {"trace_id": CURRENT.get().trace_id, "context_sha256": digest(context), "state_snapshot_id": prior["snapshot_id"], "model": self.provider.model_name,
                    "retrieval": context["retrieval"], "validation_attempts": validation_attempts, "prompt_version": EVIDENCE_PROMPT_VERSION,
                    "quote_validated": True, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "usage": raw.get("_usage")}}
        CURRENT.get().metadata.update(turn_id=turn['turn_id'], evidence_id=evidence['evidence_id'], snapshot_before_id=prior['snapshot_id'], snapshot_after_id=updated['snapshot_id'])
        with stage('persistence', {'turn_id': turn['turn_id'], 'evidence_id': evidence['evidence_id'], 'expected_snapshot_id': prior['snapshot_id']}) as persisted, self.db.transaction():
            current = self.require_conversation(cid, req.learner_id)
            cached = self.store.answer(cid, req.request_id)
            if cached:
                if cached[1] != request_hash:
                    raise HTTPException(409, "This request ID already saved a different answer.")
                saved = cached[0]
                CURRENT.get().metadata.update(cache_hit=True, turn_id=saved['turn_id'],
                    evidence_id=saved['evidence']['evidence_id'],
                    snapshot_before_id=saved['state_before']['snapshot_id'],
                    snapshot_after_id=saved['state_after']['snapshot_id'],
                    source_trace_id=saved['extraction_trace'].get('trace_id'),
                    discarded_computation=True)
                persisted['output'] = {'committed': False, 'reused_saved_turn_id': saved['turn_id']}
                return self.session(cid, req.learner_id)
            latest = self.store.topic(req.learner_id, current["topic_key"])
            if current["version"] != conversation["version"] or current.get("current_question_id") != req.question_id or latest["state"]["snapshot_id"] != prior["snapshot_id"]:
                raise HTTPException(409, "Saved learner state changed while checking this answer. Reload, then retry your explanation.")
            self.store.save_answer(cid, req.request_id, request_hash, turn)
            self.store.put_topic(req.learner_id, current["topic_key"], topic["definition"], updated, latest["revision"] + 1)
            self.store.add_chunk(req.learner_id, current["topic_key"], "learner_evidence", evidence["evidence_id"],
                                 question["concept_id"] + " " + topic["definition"]["title"] + " " + evidence["source_quote"] + " " + evidence["claim"],
                                 {"evidence_id": evidence["evidence_id"], "concept_id": evidence["concept_id"], "category": evidence["category"], "source_quote": evidence["source_quote"], "conversation_id": cid, "turn_id": turn["turn_id"]})
            current.update(current_question_id=None, status="needs_question", version=current["version"] + 1,
                           answered_count=current["answered_count"] + 1, updated_at=now(), last_error=None)
            if req.finish_after_answer:
                current.update(status="completed", ending_state=updated, finished_at=now())
            self.store.save_conversation(current)
            persisted['output'] = {'turn_id': turn['turn_id'], 'evidence_id': evidence['evidence_id'], 'snapshot_id': updated['snapshot_id'], 'revision': latest['revision'] + 1, 'committed': True}
        logger.info(json.dumps({"event": "assessment_answer_saved", "conversation_id": cid, "turn_id": turn["turn_id"], "snapshot_id": updated["snapshot_id"], "request_id": req.request_id}))
        return self.session(cid, req.learner_id) if req.finish_after_answer else self.ensure_question(cid, req.learner_id)

    def finish(self, cid, learner_id):
        with self.db.transaction():
            conversation = self.require_conversation(cid, learner_id)
            if conversation["status"] != "completed":
                turns = self.store.answers(cid)
                ending = turns[-1]["state_after"] if turns else conversation["baseline_state"]
                conversation.update(status="completed", ending_state=ending, finished_at=now(), updated_at=now(), version=conversation["version"] + 1)
                self.store.save_conversation(conversation)
        return self.session(cid, learner_id)

    def resume(self, cid, learner_id):
        with self.db.transaction():
            conversation = self.require_conversation(cid, learner_id)
            question = self.store.question(conversation.get("current_question_id"))
            topic = self.store.topic(learner_id, conversation["topic_key"])
            if question and question["generation_trace"]["state_snapshot_id"] != topic["state"]["snapshot_id"]:
                conversation["current_question_id"] = None
            conversation.update(status="active" if conversation.get("current_question_id") else "needs_question", version=conversation["version"] + 1, updated_at=now())
            self.store.save_conversation(conversation)
        return self.ensure_question(cid, learner_id)
