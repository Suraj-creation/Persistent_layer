"""Regression coverage of the actual HTTP -> retrieval -> evidence -> durable state loop."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from packages.learner_state_core.db import LocalDatabaseManager
from packages.learner_state_core.evidence.extractor import RuleBasedEvidenceExtractor
from packages.learner_state_core.evidence.models import Interaction

CID = "fraction_addition_unlike_denominators"


@pytest.fixture
def flow(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNER_STATE_DB_PATH", str(tmp_path / "bootstrap.db"))
    from apps.api.app import main
    db = LocalDatabaseManager(str(tmp_path / "flow.db"))
    monkeypatch.setattr(main, "db_manager", db)
    monkeypatch.setenv("EVIDENCE_EXTRACTOR_MODE", "rules")
    return TestClient(main.app), db, main


def payload(learner="learner_A", text="I added the top and bottom numbers."):
    return dict(learner_id=learner, item_id="ITEM_FRAC_01", item_text="Solve: 1/2 + 1/3", item_concepts=[CID], response_text=text, selected_option="2/5", correctness=0)


def submit(client, key, data=None):
    response = client.post("/api/v1/interactions/process", json=data or payload(), headers={"Idempotency-Key": key})
    assert response.status_code == 200, response.text
    return response.json()


def test_exact_round_trip_and_restart_including_provenance_and_state(flow, monkeypatch):
    client, db, main = flow
    response = submit(client, "first")
    assert response["retrieval"]["hits"]
    assert response["evidence"]["retrieval_citation_ids"]
    restarted = LocalDatabaseManager(db.db_path)
    monkeypatch.setattr(main, "db_manager", restarted)
    assert client.get("/api/v1/learners/learner_A/state").json() == response["new_state"]
    assert client.get("/api/v1/learners/learner_A/evidence").json() == [response["evidence"]]
    assert client.get("/api/v1/learners/learner_A/events").json() == response["events"]
    replay = client.post("/api/v1/learners/learner_A/replay").json()
    assert replay["matches_persisted_state"]
    assert replay["final_snapshot"] == response["new_state"]
    assert client.get("/api/v1/learners/learner_A/retrieval-traces").json() == [response["retrieval"]]
    assert submit(client, "first") == response


def test_concurrent_retries_apply_once_and_distinct_events_are_not_lost(flow):
    client, db, _ = flow
    def send(key):
        return client.post("/api/v1/interactions/process", json=payload(), headers={"Idempotency-Key": key})
    # Warm the shared curriculum index before exercising concurrent writers.
    client.get("/api/v1/retrieval/status")
    with ThreadPoolExecutor(max_workers=6) as pool:
        retries = list(pool.map(send, ["retry"] * 6))
    assert all(response.status_code == 200 for response in retries)
    assert all(response.json() == retries[0].json() for response in retries)
    assert len(db.get_learner_evidence_ledger("learner_A")) == 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        distinct = list(pool.map(send, [f"event-{i}" for i in range(4)]))
    assert all(response.status_code == 200 for response in distinct)
    assert len(db.get_learner_evidence_ledger("learner_A")) == 5
    assert db.get_learner_snapshot("learner_A").concept_states[CID].interaction_count == 5
    assert client.post("/api/v1/learners/learner_A/replay").json()["matches_persisted_state"]


def test_failed_state_event_write_rolls_back_every_part_of_submission(flow, monkeypatch):
    client, db, _ = flow
    def fail(_):
        raise RuntimeError("injected write failure")
    monkeypatch.setattr(db, "save_audit_event", fail)
    response = client.post("/api/v1/interactions/process", json=payload(), headers={"Idempotency-Key": "will-fail"})
    assert response.status_code == 500
    assert "injected write failure" not in response.text
    assert response.headers["X-Request-ID"] == response.json()["request_id"]
    assert db.get_learner_interactions("learner_A") == []
    assert db.get_learner_evidence_ledger("learner_A") == []
    assert db.get_learner_audit_events("learner_A") == []
    assert db.get_request("learner_A", "will-fail") is None
    assert db.get_retrieval_traces("learner_A") == []


def test_key_conflict_rejected_and_same_key_scoped_to_learner(flow):
    client, db, _ = flow
    submit(client, "key")
    changed = payload(text="A completely different explanation.")
    assert client.post("/api/v1/interactions/process", json=changed, headers={"Idempotency-Key": "key"}).status_code == 409
    submit(client, "key", payload("learner_B"))
    assert len(db.get_learner_interactions("learner_A")) == len(db.get_learner_interactions("learner_B")) == 1
    assert client.get("/api/v1/learners/learner_C/evidence").json() == []
    assert client.get("/api/v1/learners/learner_C/conversation").json() == []


def test_diagnostic_response_is_scored_against_issued_item_and_durable(flow):
    client, db, _ = flow
    original = submit(client, "parent")
    probe = original["recommended_diagnostic_probe"]
    correct = next(option for option in probe["options"] if option["is_correct"])
    answer = {"learner_id": "learner_A", "parent_interaction_id": original["interaction"]["interaction_id"], "item_id": probe["item_id"], "option_key": correct["option_key"]}
    headers = {"Idempotency-Key": "probe-answer"}
    response = client.post("/api/v1/diagnostics/answer", json=answer, headers=headers)
    assert response.status_code == 200
    result = response.json()
    assert result["interaction"]["item_text"] == probe["question_text"]
    assert result["interaction"]["correctness"] == 1
    assert result["evidence"]["candidate_interpretation"]["category"] == "ambiguous"
    assert len(db.get_learner_interactions("learner_A")) == 2
    assert client.post("/api/v1/diagnostics/answer", json=answer, headers=headers).json() == result
    assert client.post("/api/v1/diagnostics/answer", json=answer).status_code == 409
    answer["learner_id"] = "learner_B"
    assert client.post("/api/v1/diagnostics/answer", json=answer).status_code == 409
    assert client.post("/api/v1/learners/learner_A/replay").json()["matches_persisted_state"]


def test_chat_restores_persisted_history_and_citations(flow):
    client, db, _ = flow
    request = dict(learner_id="chat_A", message="I added the top and bottom numbers.", selected_option="2/5")
    response = client.post("/api/v1/chat", json=request, headers={"Idempotency-Key": "chat-1"})
    assert response.status_code == 200
    result = response.json()
    assert "same unit denominator" in result["reply"]
    assert result["retrieval"]["hits"]
    history = client.get("/api/v1/learners/chat_A/conversation").json()
    assert history[0]["response"] == result
    assert history[0]["user"] == request["message"]
    assert client.post("/api/v1/chat", json=request, headers={"Idempotency-Key": "chat-1"}).json() == result
    assert len(db.get_turns("chat_A")) == 1
    assert client.get("/api/v1/learners/another/conversation").json() == []


def test_gap_and_review_metadata_survive_round_trip(flow):
    client, db, _ = flow
    result = submit(client, "gap", payload(text="What is a common denominator?"))
    assert result["new_state"]["active_prerequisite_gaps"] == ["fraction_equivalence"]
    assert db.get_learner_snapshot("learner_A").model_dump(mode="json") == result["new_state"]
    evidence = db.get_learner_evidence_ledger("learner_A")[0]
    assert evidence.schema_version == "3.0"
    assert evidence.prompt_version == "rules-v3"


@pytest.mark.parametrize("change", [dict(correctness=3), dict(latency_seconds=-1), dict(item_concepts=["unknown"]), dict(item_concepts=[CID, CID])])
def test_invalid_input_cannot_create_partial_state(flow, change):
    client, db, _ = flow
    data = {**payload(), **change}
    assert client.post("/api/v1/interactions/process", json=data).status_code == 422
    assert db.get_learner_interactions("learner_A") == []


@pytest.mark.parametrize("problem,answer,expected", [("1/2 + 1/3", "5/6", 1), ("1/4 + 1/2", "3/4", 1), ("1/4 + 1/2", "5/6", 0), ("1/0 + 1/2", "5/6", None), ("unrecognized problem", "D", None), ("1/2 * 1/3", "5/6", None), ("1/2 + 1/3 + 1/4", "5/6", None), ("1/2 + 1/3 * 2", "5/6", None)])
def test_fraction_scoring_has_no_magic_fallback(problem, answer, expected):
    from apps.api.app.main import check_fraction_correctness
    assert check_fraction_correctness(problem, answer) == expected


def test_clear_removes_all_personal_data_but_retains_public_curriculum(flow):
    client, db, _ = flow
    submit(client, "first")
    db.clear_learner_data("learner_A")
    assert db.get_learner_evidence_ledger("learner_A") == []
    assert db.get_learner_interactions("learner_A") == []
    assert db.get_retrieval_traces("learner_A") == []
    assert db.get_request("learner_A", "first") is None
    assert client.get("/api/v1/retrieval/status").json()["stores"][0]["documents"] == 10


def test_legacy_snapshot_is_preserved_as_explicit_replay_baseline(flow):
    client, db, main = flow
    old_interaction = Interaction(**payload(), interaction_id="legacy")
    old_evidence = RuleBasedEvidenceExtractor().extract(old_interaction)
    old_state = main.state_engine.initialize_learner_state("learner_A", list(main.FRACTION_TAXONOMY))
    old_state, _ = main.state_engine.update_state(old_state, old_evidence)
    old_state.policy_version = "beta-bayesian-v2.0"
    db.save_interaction(old_interaction)
    db.save_evidence(old_evidence)
    for state in old_state.concept_states.values():
        db.save_concept_state("learner_A", state)
    # Older databases have concept rows but no complete snapshot JSON.
    submit(client, "new-event")
    replay = client.post("/api/v1/learners/learner_A/replay").json()
    assert replay["replay_baseline"] == "preserved_legacy_snapshot"
    assert replay["baseline_evidence_count"] == 1
    assert replay["state_transitions_replayed"] == 1
    assert replay["matches_persisted_state"]
    assert len(db.get_learner_evidence_ledger("learner_A")) == 2


def test_snapshot_replacement_does_not_restore_removed_candidates(flow):
    client, db, _ = flow
    submit(client, "first")
    state = db.get_learner_snapshot("learner_A")
    state.concept_states[CID].candidate_misconceptions.clear()
    db.save_snapshot(state)
    assert db.get_learner_snapshot("learner_A").concept_states[CID].candidate_misconceptions == {}
    with db._get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM misconception_beliefs WHERE learner_id=?", ("learner_A",)).fetchone()[0] == 0


@pytest.mark.parametrize("reply,raises,expected_fragment", [("Try equal units [1].", False, "Try equal units [1]."), ("Unsupported answer [999].", False, "same unit denominator"), ("", True, "same unit denominator")])
def test_optional_tutor_provider_uses_citations_and_falls_back_without_partial_writes(flow, monkeypatch, reply, raises, expected_fragment):
    from types import SimpleNamespace
    client, db, main = flow
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        if raises:
            raise RuntimeError("provider unavailable")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    monkeypatch.setattr(main, "azure_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=generate))))
    monkeypatch.setenv("TUTOR_RESPONSE_MODE", "azure")
    request = {"learner_id": "provider_learner", "message": "I added the top and bottom numbers.", "selected_option": "2/5"}
    headers = {"Idempotency-Key": "provider-1"}
    response = client.post("/api/v1/chat", json=request, headers=headers)
    assert response.status_code == 200
    assert expected_fragment in response.json()["reply"]
    assert client.post("/api/v1/chat", json=request, headers=headers).json() == response.json()
    assert len(calls) == 1
    assert "CURRICULUM SOURCES" in calls[0]["messages"][0]["content"]
    assert len(db.get_learner_evidence_ledger("provider_learner")) == 1


def test_audit_report_is_repeatable_and_reports_heuristic_method(flow):
    client, _, _ = flow
    submit(client, "first")
    first = client.get('/api/v1/learners/learner_A/audit_report')
    assert first.status_code == 200
    assert first.json() == client.get('/api/v1/learners/learner_A/audit_report').json()
    assert 'hand-set heuristic weights; uncalibrated' in first.json()['report_markdown']
    assert first.json()['total_audit_events'] == 1


def test_multiple_workers_can_initialize_the_same_new_database(tmp_path):
    path = str(tmp_path / 'shared-startup.db')
    def initialize(index):
        manager = LocalDatabaseManager(path)
        manager.ensure_learner(f'worker-{index}')
        return manager
    with ThreadPoolExecutor(max_workers=6) as pool:
        managers = list(pool.map(initialize, range(6)))
    assert len(managers[0].list_learners()) == 6
