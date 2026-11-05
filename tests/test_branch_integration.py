"""Contracts specific to combining the improve and training branches."""
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from apps.api.app import main
from packages.learner_state_core.db import LocalDatabaseManager
from packages.learner_state_core.grounding.taxonomy import FRACTION_TAXONOMY
from packages.learner_state_core.state.engine import BayesianLearnerStateEngine
from test_conversation_assessment import flow, start, answer


@pytest.mark.parametrize("layout", ["improve", "training"])
def test_existing_branch_snapshots_migrate_without_losing_current_state_or_history(tmp_path, layout):
    path = str(tmp_path / f"{layout}.db")
    db = LocalDatabaseManager(path)
    initial = BayesianLearnerStateEngine().initialize_learner_state("existing", list(FRACTION_TAXONOMY))
    latest = initial.model_copy(update={"snapshot_id": "saved-latest", "as_of_interaction_id": "saved-interaction"}, deep=True)
    db.ensure_learner("existing")
    # Reproduce the exact conflicting schemas, not a new-final schema fixture.
    with db.connection() as conn:
        conn.execute("DROP TABLE learner_snapshots")
        conn.execute("DROP TABLE learner_snapshot_history")
        if layout == "improve":
            conn.execute("CREATE TABLE learner_snapshots(snapshot_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL REFERENCES learners(learner_id), payload_json TEXT NOT NULL)")
            for snapshot in (initial, latest):
                conn.execute("INSERT INTO learner_snapshots VALUES(?,?,?)", (snapshot.snapshot_id, "existing", snapshot.model_dump_json()))
        else:
            conn.execute("CREATE TABLE learner_snapshots(learner_id TEXT PRIMARY KEY REFERENCES learners(learner_id), payload_json TEXT NOT NULL)")
            conn.execute("INSERT INTO learner_snapshots VALUES(?,?)", ("existing", latest.model_dump_json()))
    # Concurrent startup and a repeated startup must both be safe and idempotent.
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: LocalDatabaseManager(path), range(3)))
    reopened = LocalDatabaseManager(path)
    assert reopened.get_learner_snapshot("existing") == latest
    history = reopened.get_snapshot_history("existing")
    assert [s["snapshot_id"] for s in history] == ([initial.snapshot_id, latest.snapshot_id] if layout == "improve" else [latest.snapshot_id])
    with reopened.connection() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_snapshot_history_and_current_projection_roll_back_together(tmp_path):
    db = LocalDatabaseManager(str(tmp_path / "rollback.db"))
    state = BayesianLearnerStateEngine().initialize_learner_state("learner", list(FRACTION_TAXONOMY))
    db.save_snapshot(state)
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.save_snapshot(state.model_copy(update={"snapshot_id": "never-committed"}))
            raise RuntimeError("force rollback")
    reopened = LocalDatabaseManager(db.db_path)
    assert reopened.get_learner_snapshot("learner") == state
    assert reopened.get_snapshot_history("learner") == [state.model_dump(mode="json")]


def test_reopening_does_not_resurrect_history_after_a_standalone_concept_update(tmp_path):
    db = LocalDatabaseManager(str(tmp_path / "standalone.db"))
    state = BayesianLearnerStateEngine().initialize_learner_state("learner", list(FRACTION_TAXONOMY))
    db.save_snapshot(state)
    concept = state.concept_states["fraction_basics"].model_copy(update={"alpha": 3.0, "mastery_estimate": .75})
    db.save_concept_state("learner", concept)
    reopened = LocalDatabaseManager(db.db_path)
    assert reopened.get_learner_snapshot("learner").concept_states["fraction_basics"].alpha == 3.0
    assert reopened.get_snapshot_history("learner") == [state.model_dump(mode="json")]


def test_fraction_conversation_protocols_keep_distinct_receipts_and_turn_shapes():
    client = TestClient(main.app)
    research = client.post("/api/v1/chat", headers={"Idempotency-Key": "same-key"}, json={"learner_id": "research", "message": "I added the top and bottom numbers."})
    workspace = client.post("/api/workspace/v1/chat", json={"learner_id": "legacy", "request_id": "same-key", "message": "I added the top and bottom numbers."})
    assert research.status_code == workspace.status_code == 200
    assert research.json()["retrieval"]["hits"]
    assert workspace.json()["trace"]["quote_validated"]
    assert main.db_manager.get_turns("research")[0]["response"]["reply"] == research.json()["reply"]
    assert main.db_manager.get_learning_turns("legacy")[0]["trace"] == workspace.json()["trace"]
    assert main.db_manager.get_learning_turns("research") == []
    assert main.db_manager.get_turns("legacy") == []


def test_research_graph_and_assessment_retrieval_coexist_without_changing_topic_memory(flow):
    client, _ = flow
    session = start(client).json()
    saved = answer(client, session, finish_after_answer=True).json()
    cid = saved["conversation"]["conversation_id"]
    query = client.post("/api/v1/retrieval/query", json={"query": "Why do fractions need a common denominator?"})
    assert query.status_code == 200
    assert query.json()["hits"]
    restored = client.get(f"/api/v2/conversations/{cid}", params={"learner_id": "real-user"}).json()
    assert restored["state"] == saved["state"]
    assert restored["turns"] == saved["turns"]
    traces = client.get("/api/v2/observability/traces", params={"learner_id": "real-user"}).json()
    assert traces["items"]
    assert all(trace["status"] == "completed" for trace in traces["items"])
    with main.db_manager.connection() as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
