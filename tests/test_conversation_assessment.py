"""Real-flow regressions. Model responses below are test fixtures, never UI content."""
from copy import deepcopy
import pytest
from fastapi.testclient import TestClient
from apps.api.app import main


class FixtureProvider:
    configured = True
    model_name = "test-provider"

    def __init__(self):
        self.question_contexts = []
        self.fail_questions = False
        self.fabricate_quote = False

    def build_topic(self, topic, notes):
        return {"title": topic, "overview": "Tokens are spans of text; context changes meaning.", "concepts": [
            {"concept_id": "tokens", "name": "Tokenization", "description": "Split text into tokens while preserving boundaries.", "prerequisites": []},
            {"concept_id": "context", "name": "Context", "description": "Interpret tokens in their surrounding context.", "prerequisites": ["tokens"]},
        ]}

    def generate_question(self, context):
        self.question_contexts.append(deepcopy(context))
        if self.fail_questions:
            raise TimeoutError("private provider body")
        number = len(context["previous_questions"]) + 1
        tasks = ["Explain how token boundaries affect the interpretation of a sentence.", "A program merges all adjacent words into one unit. Analyze its mistake using the source distinction.", "Compare processing a new sentence with and without preserved boundaries. Justify the resulting difference.", "Design a fresh example that distinguishes a word from a token and explain why it works."]
        return {"concept_id": context["plan"]["concept_id"], "kind": context["plan"]["kind"],
                "prompt": tasks[(number - 1) % len(tasks)] + f" Assessment case {number}.",
                "expected_focus": "Give a reason and connect it to the concept.",
                "learning_support": "Review the source distinction before trying again." if context["plan"]["intent"] == "repair" else "",
                "criteria": context["plan"]["required_criteria"] or [{"criterion_id": "reason", "description": "Explains why token boundaries matter."}, {"criterion_id": "example", "description": "Connects a concrete example to token boundaries."}],
                "reference_answer": "Explain boundaries and contextual interpretation with an example.",
                "source_citations": [context["retrieval"]["materials"][0]["chunk_id"]]}

    def evaluate(self, context):
        text = context["response_text"]
        ambiguous = "guess" in text.lower()
        wrong = "wrong rule" in text.lower()
        hypotheses = list(context["state"]["concepts"][context["question"]["concept_id"]]["hypotheses"])
        return {"concept_id": context["question"]["concept_id"], "source_quote": "invented words" if self.fabricate_quote else text,
                "claim": "A candidate claim about token boundaries.", "category": "ambiguous" if ambiguous else "misconception" if wrong else "understanding",
                "confidence": .9, "met_criteria": [] if ambiguous or wrong else [c["criterion_id"] for c in context["question"]["criteria"]],
                "missing_criteria": [c["criterion_id"] for c in context["question"]["criteria"]] if ambiguous or wrong else [],
                "misconception": "Tokens always equal words" if wrong else None,
                "contradicts_hypothesis_ids": [] if wrong or ambiguous else hypotheses,
                "feedback": "Your example provides evidence we can check.", "coaching": "Use the source distinction, then try a fresh example.",
                "source_citations": [context["retrieval"]["materials"][0]["chunk_id"]]}


@pytest.fixture
def flow(monkeypatch):
    provider = FixtureProvider()
    monkeypatch.setattr(main, "assessment_provider", provider, raising=False)
    return TestClient(main.app), provider


def start(client, **overrides):
    return client.post("/api/v2/conversations", json={"learner_id": "real-user", "topic": "Natural language processing", "request_id": "start-1", **overrides})


def answer(client, session, text="Token boundaries preserve separate pieces of text and help us interpret a concrete example.", **overrides):
    return client.post(f"/api/v2/conversations/{session['conversation']['conversation_id']}/answers", json={
        "learner_id": "real-user", "question_id": session["active_question"]["question_id"], "response_text": text,
        "request_id": "answer-" + session["active_question"]["question_id"], **overrides})


def test_empty_first_start_and_resume_are_real_records(flow):
    client, provider = flow
    assert client.get("/api/v2/conversations", params={"learner_id": "real-user"}).json() == []
    response = start(client)
    assert response.status_code == 200, response.text
    session = response.json()
    assert session["turns"] == []
    assert session["active_question"]["generation_trace"]["state_snapshot_id"] == session["state"]["snapshot_id"]
    assert "criteria" not in session["active_question"]
    assert session["sources"][0]["kind"] == "model_reference"
    resumed = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}", params={"learner_id": "real-user"}).json()
    assert resumed == session
    assert len(provider.question_contexts) == 1
    assert start(client).json()["conversation"]["conversation_id"] == session["conversation"]["conversation_id"]


def test_question_generation_reads_saved_state_and_retrieved_evidence(flow):
    client, provider = flow
    session = start(client).json()
    result = answer(client, session, "I use a wrong rule because I think each token must always equal a full word.")
    assert result.status_code == 200, result.text
    updated = result.json()
    turn = updated["turns"][0]
    context = provider.question_contexts[-1]
    assert context["state"]["snapshot_id"] == turn["state_after"]["snapshot_id"]
    assert context["plan"]["intent"] == "repair"
    assert context["retrieval"]["evidence"][0]["metadata"]["evidence_id"] == turn["evidence"]["evidence_id"]
    assert context["plan"]["evidence_ids"]
    assert updated["active_question"]["question_id"] != session["active_question"]["question_id"]
    assert turn["nlp"]["token_spans"]
    assert turn["state_changes"]["weight"] > 0


def test_ambiguity_does_not_mint_mastery_and_report_is_traceable(flow):
    client, _ = flow
    session = start(client).json()
    updated = answer(client, session, "I guess, but I cannot explain why yet.").json()
    before = session["state"]["concepts"]
    after = updated["state"]["concepts"]
    assert [(c["alpha"], c["beta"]) for c in before.values()] == [(c["alpha"], c["beta"]) for c in after.values()]
    report = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}/report", params={"learner_id": "real-user"}).json()
    assert report["timeline"][0]["response_text"] == updated["turns"][0]["response_text"]
    assert report["timeline"][0]["state_after"]["snapshot_id"] == updated["state"]["snapshot_id"]
    assert report["unassessed_concepts"]
    assert report["evidence_of_progress"] is False


def test_validation_rejects_fabrication_without_changing_memory(flow):
    client, provider = flow
    session = start(client).json()
    provider.fabricate_quote = True
    result = answer(client, session)
    assert result.status_code == 502
    resumed = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}", params={"learner_id": "real-user"}).json()
    assert resumed["turns"] == []
    assert resumed["state"] == session["state"]


def test_generation_failure_preserves_answer_and_retries_only_question(flow):
    client, provider = flow
    session = start(client).json()
    provider.fail_questions = True
    result = answer(client, session)
    assert result.status_code == 200, result.text
    saved = result.json()
    assert len(saved["turns"]) == 1
    assert saved["active_question"] is None
    assert saved["conversation"]["status"] == "needs_question"
    provider.fail_questions = False
    resumed = client.post(f"/api/v2/conversations/{session['conversation']['conversation_id']}/next", json={"learner_id": "real-user"}).json()
    assert len(resumed["turns"]) == 1
    assert resumed["active_question"]
    assert resumed["state"]["snapshot_id"] == saved["state"]["snapshot_id"]


def test_topic_memory_reused_and_other_learners_isolated(flow):
    client, _ = flow
    session = start(client).json()
    updated = answer(client, session).json()
    second = start(client, request_id="start-2").json()
    assert second["turns"] == []
    assert second["state"]["snapshot_id"] == updated["state"]["snapshot_id"]
    other = start(client, learner_id="someone-else").json()
    assert all(c["informative_count"] == 0 for c in other["state"]["concepts"].values())
    assert client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}", params={"learner_id": "someone-else"}).status_code == 404


def test_notes_are_chunked_and_cited_without_cross_topic_retrieval(flow):
    client, provider = flow
    notes = "Tokenization divides text into tokens. A token need not be a whole word. Context affects how a token is interpreted in a sentence."
    session = start(client, notes=notes).json()
    assert session["sources"][0]["kind"] == "study_notes"
    material = provider.question_contexts[0]["retrieval"]["materials"]
    assert material and all(m["kind"] == "study_notes" for m in material)
    assert all(m["content"] in notes for m in material)


def test_idempotency_and_stale_questions(flow):
    client, _ = flow
    session = start(client).json()
    first = answer(client, session).json()
    duplicate = answer(client, session).json()
    assert len(duplicate["turns"]) == 1
    assert duplicate["state"] == first["state"]
    assert answer(client, session, request_id="different-attempt").status_code == 409
    assert answer(client, session, text="Changed response").status_code == 409


def test_contradiction_and_report_completion_do_not_count_coaching_as_learning(flow):
    client, _ = flow
    session = start(client).json()
    wrong = answer(client, session, "I use a wrong rule because I think a token must be exactly one word.").json()
    repaired = answer(client, wrong).json()
    cid = wrong["turns"][0]["evidence"]["concept_id"]
    hypotheses = repaired["state"]["concepts"][cid]["hypotheses"]
    assert hypotheses and all(h["contradicting_evidence_ids"] for h in hypotheses.values())
    assert repaired["state"]["concepts"][cid]["estimate"] > wrong["state"]["concepts"][cid]["estimate"]
    response = client.post(f"/api/v2/conversations/{session['conversation']['conversation_id']}/finish", json={"learner_id": "real-user"})
    assert response.json()["conversation"]["status"] == "completed"
    assert response.json()["active_question"] is None
    report = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}/report", params={"learner_id": "real-user"}).json()
    assert report["evidence_of_progress"] is True


def test_raw_text_survives_preprocessing_and_reload(flow):
    client, _ = flow
    notes = "  Tokens preserve boundaries.\nContext affects how the same word is interpreted.  "
    session = start(client, notes=notes).json()
    original = "  A token preserves boundaries.\nI use context to explain the meaning of a concrete example.  "
    updated = answer(client, session, original).json()
    assert updated["turns"][0]["response_text"] == original
    assert updated["turns"][0]["nlp"]["raw_text"] == original
    assert all(original[s["start"]:s["end"]] == s["text"] for s in updated["turns"][0]["nlp"]["sentences"])
    quote = updated["turns"][0]["evidence"]
    assert original[quote["source_start"]:quote["source_end"]] == quote["source_quote"]
    with main.db_manager.connection() as conn:
        assert conn.execute("SELECT raw_text FROM assessment_documents").fetchone()[0] == notes


def test_repeated_explanations_have_lower_weight(flow):
    client, _ = flow
    session = start(client).json()
    # Keep the same concept under repair; identical repeated wording is not independent evidence.
    text = "I use a wrong rule because I think a token must always equal a complete word."
    first = answer(client, session, text).json()
    second = answer(client, first, text).json()
    assert second["turns"][-1]["state_changes"]["duplicate_response"]
    assert second["turns"][-1]["state_changes"]["weight"] < first["turns"][-1]["state_changes"]["weight"]


def test_answer_bundle_rolls_back_on_storage_failure(flow, monkeypatch):
    from packages.learner_state_core.assessment.storage import AssessmentStore
    client, _ = flow
    session = start(client).json()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(AssessmentStore, "add_chunk", fail)
    failed = answer(client, session)
    assert failed.status_code == 500
    assert failed.json()['request_id'] == failed.headers['x-request-id']
    assert 'simulated storage failure' not in failed.text
    saved = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}", params={"learner_id": "real-user"}).json()
    assert saved["turns"] == []
    assert saved["state"] == session["state"]
    assert saved["active_question"] == session["active_question"]


def test_concurrent_answers_cannot_overwrite_topic_memory(flow):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    client, provider = flow
    session = start(client).json()
    barrier = Barrier(2)
    evaluate = provider.evaluate

    def delayed(context):
        result = evaluate(context)
        barrier.wait(timeout=10)
        return result

    provider.evaluate = delayed
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: answer(client, session, request_id=f"parallel-{i}").status_code, range(2)))
    assert sorted(results) == [200, 409]
    saved = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}", params={"learner_id": "real-user"}).json()
    assert len(saved["turns"]) == 1


def test_model_unavailable_does_not_insert_demo_fallback(flow):
    client, provider = flow
    provider.configured = False
    assert start(client).status_code == 503
    assert client.get("/api/v2/conversations", params={"learner_id": "real-user"}).json() == []


def test_unrelated_positive_evidence_does_not_erase_an_older_skill_gap():
    from packages.learner_state_core.assessment.state import initialize, integrate, plan_next
    from packages.learner_state_core.assessment.nlp import analyze
    definition = FixtureProvider().build_topic("NLP", "")
    state = initialize(definition)
    earlier = {"question_id": "q1", "criteria": [{"criterion_id": "reason", "description": "Explain how boundaries preserve distinct units."}]}
    evidence = {"evidence_id": "e1", "concept_id": "tokens", "confidence": .9, "category": "partial", "met_criteria": [], "missing_criteria": ["reason"], "misconception": None, "contradicts_hypothesis_ids": []}
    state, _ = integrate(state, evidence, earlier, analyze("I have not explained how boundaries preserve distinct units yet."))
    unrelated = {"question_id": "q2", "criteria": [{"criterion_id": "reason", "description": "Recognize punctuation as an independent token."}]}
    positive = {**evidence, "evidence_id": "e2", "category": "understanding", "met_criteria": ["reason"], "missing_criteria": []}
    state, _ = integrate(state, positive, unrelated, analyze("A punctuation mark can be represented as its own token."))
    assert earlier["criteria"][0]["description"] in state["concepts"]["tokens"]["missing_criteria"]
    plan = plan_next(state, definition, [], [])
    assert plan["intent"] == "repair"
    assert plan["required_criteria"][0]["description"] == earlier["criteria"][0]["description"]


def test_case_sensitive_source_changes_are_not_silently_deduplicated(flow):
    client, _ = flow
    a = start(client, notes="x = 1\nX = 2\nVariable names are case sensitive.").json()
    b = start(client, request_id="case-update", notes="x = 1\nx = 2\nVariable names are case sensitive.").json()
    assert len(a["sources"]) == 1
    assert len(b["sources"]) == 2


def test_report_never_exposes_the_unanswered_reference_answer(flow):
    client, _ = flow
    session = start(client).json()
    updated = answer(client, session).json()
    report = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}/report", params={"learner_id": "real-user"}).json()
    next_question = report["timeline"][0]["next_question"]
    assert next_question["question_id"] == updated["active_question"]["question_id"]
    assert "reference_answer" not in next_question
    assert "criteria" not in next_question


def test_case_sensitive_criteria_are_distinct_skills():
    from packages.learner_state_core.assessment.state import initialize, integrate
    from packages.learner_state_core.assessment.nlp import analyze
    state = initialize(FixtureProvider().build_topic("Code", ""))
    def observation(eid, met):
        return {"evidence_id": eid, "concept_id": "tokens", "confidence": .9, "category": "understanding" if met else "partial", "met_criteria": ["value"] if met else [], "missing_criteria": [] if met else ["value"], "misconception": None, "contradicts_hypothesis_ids": []}
    upper = {"question_id": "upper", "criteria": [{"criterion_id": "value", "description": "Explain the value of variable X in this program."}]}
    lower = {"question_id": "lower", "criteria": [{"criterion_id": "value", "description": "Explain the value of variable x in this program."}]}
    state, _ = integrate(state, observation("upper-gap", False), upper, analyze("I cannot yet explain the value of variable X in this program."))
    state, change = integrate(state, observation("lower-answer", True), lower, analyze("The lowercase variable x has the value assigned by this separate statement."))
    assert upper["criteria"][0]["description"] in state["concepts"]["tokens"]["missing_criteria"]
    assert change["resolved_criteria"] == []


def test_unresolved_belief_is_explicitly_checked_after_success_on_criteria(flow):
    client, provider = flow
    session = start(client).json()
    wrong = answer(client, session, "I use a wrong rule because I think every token must always be a full word.").json()
    evaluate = provider.evaluate

    def no_explicit_contradiction(context):
        result = evaluate(context)
        result["contradicts_hypothesis_ids"] = []
        return result

    provider.evaluate = no_explicit_contradiction
    repaired = answer(client, wrong).json()
    plan = repaired["active_question"]["generation_trace"]["plan"]
    assert plan["intent"] == "verify_hypothesis"
    assert plan["concept_id"] == wrong["turns"][0]["evidence"]["concept_id"]
    report = client.get(f"/api/v2/conversations/{session['conversation']['conversation_id']}/report", params={"learner_id": "real-user"}).json()
    assert report["remaining_gaps"][0]["supporting_quotes"][0]["evidence_id"] == wrong["turns"][0]["evidence"]["evidence_id"]
    assert all(q["category"] != "understanding" for q in report["remaining_gaps"][0]["supporting_quotes"])


def test_resume_uses_updated_topic_memory_after_another_conversation(flow):
    client, _ = flow
    first = start(client).json()
    cid = first["conversation"]["conversation_id"]
    client.post(f"/api/v2/conversations/{cid}/finish", json={"learner_id": "real-user"})
    second = start(client, request_id="another-start").json()
    changed = answer(client, second).json()
    resumed = client.post(f"/api/v2/conversations/{cid}/resume", json={"learner_id": "real-user"}).json()
    assert resumed["active_question"]["generation_trace"]["state_snapshot_id"] == changed["state"]["snapshot_id"]
    assert resumed["active_question"]["question_id"] != first["active_question"]["question_id"]


def test_new_conversation_generation_sees_earlier_topic_question_prompts(flow):
    client, provider = flow
    first = start(client).json()
    start(client, request_id="new-conversation")
    assert first["active_question"]["prompt"] in [q["prompt"] for q in provider.question_contexts[-1]["previous_questions"]]


def test_report_lists_only_sources_retrieved_for_its_own_conversation(flow):
    client, _ = flow
    first = start(client, notes="Token boundaries preserve distinct units in a sentence.").json()
    second = start(client, request_id="new-notes", notes="Context changes the meaning of a token in a different sentence.").json()
    report = client.get(f"/api/v2/conversations/{first['conversation']['conversation_id']}/report", params={"learner_id": "real-user"}).json()
    assert len(second["sources"]) == 2
    assert [s["document_id"] for s in report["source_documents"]] == [first["sources"][0]["document_id"]]
