from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from packages.learner_state_core.db import LocalDatabaseManager


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Keep API tests isolated from the developer's persistent learner database.
    db_path = tmp_path / "learner_state.db"
    monkeypatch.setenv("LEARNER_STATE_DB_PATH", str(db_path))
    from apps.api.app import main as api_main

    monkeypatch.setattr(api_main, "db_manager", LocalDatabaseManager(str(db_path)))
    return TestClient(api_main.app)


def test_api_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["concepts_loaded"] > 0


def test_api_taxonomy(client):
    res = client.get("/api/v1/taxonomy")
    assert res.status_code == 200
    data = res.json()
    assert "fraction_addition_unlike_denominators" in data


def test_training_report_exposes_aggregate_results_and_verified_artifact_metadata(client, monkeypatch, tmp_path):
    # The repository artifacts resolve from the API module location, not the
    # process working directory used to launch Uvicorn or its tests.
    monkeypatch.chdir(tmp_path)
    res = client.get("/api/v1/training-report")

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "available"
    assert data["runtime_boundary"]["bkt_used_for_live_learner_state"] is False
    assert data["runtime_boundary"]["map_classifier_trained_or_used"] is False

    bkt = data["assistments_bkt"]
    assert bkt["status"] == "evaluated"
    assert bkt["data_audit"]["rows_after_support_filter"] == 49658
    assert bkt["summary_across_splits"]
    bkt_summary = next(row for row in bkt["summary_across_splits"] if row["name"] == "bkt")
    assert bkt_summary["metrics"]["auc"]["mean_across_splits"] == pytest.approx(0.6638595166)
    assert len(bkt["seeds"]) == 2
    assert len(bkt["skill_metrics"]) == 8
    assert len(bkt["skill_parameters"]) == 8
    assert all(artifact["checksum_verified"] for split in bkt["seeds"] for artifact in split["artifacts"])
    assert "Percents" in bkt["skill_metrics"]
    assert bkt["skill_metrics"]["Percents"]["rows"] == 21

    assert data["eedi_graph"]["status"] == "audited"
    assert data["eedi_graph"]["row_counts"]["misconception-map.csv"] == 19294
    assert data["map_classifier"]["status"] == "prepared_not_trained"
    assert data["map_classifier"]["model_available"] is False
    assert "learner_predictions" not in data
    assert "raw_rows" not in data


def test_training_report_returns_partial_status_when_research_artifacts_are_missing(tmp_path):
    from apps.api.app.training_report import build_training_research_summary

    data = build_training_research_summary(tmp_path)

    assert data["status"] == "partial"
    assert data["assistments_bkt"]["status"] == "unavailable"
    assert data["eedi_graph"]["status"] == "unavailable"
    assert len(data["warnings"]) == 3


def test_training_report_handles_malformed_artifact_metadata(tmp_path):
    import json

    from apps.api.app.training_report import build_training_research_summary

    model_dir = tmp_path / "models/assistments_2009_2010_bkt"
    model_dir.mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    (model_dir / "metrics.json").write_text(json.dumps({"audit": "invalid"}), encoding="utf-8")
    (model_dir / "robustness.json").write_text(json.dumps({"seed_results": "invalid", "summary": []}), encoding="utf-8")
    (tmp_path / "experiments/eedi_graph_audit.json").write_text(
        json.dumps({"row_counts": [], "unique_counts": {}}), encoding="utf-8"
    )

    data = build_training_research_summary(tmp_path)

    assert data["status"] == "partial"
    assert data["assistments_bkt"]["status"] == "unavailable"
    assert data["eedi_graph"]["status"] == "unavailable"
    assert len(data["warnings"]) == 3


def test_api_process_student_a_and_diagnostic_probe(client):
    """
    Test processing an interaction for Student A via HTTP API:
    Should extract misconception and immediately return a recommended diagnostic probe with EIG.
    """
    payload = {
        "learner_id": "api_student_A",
        "item_id": "ITEM_FRAC_01",
        "item_text": "What is 1/2 + 1/3?",
        "item_concepts": ["fraction_addition_unlike_denominators"],
        "response_text": "I added the top and bottom numbers.",
        "selected_option": "2/5",
        "correctness": 0,
        "latency_seconds": 32.5,
    }

    res = client.post("/api/v1/interactions/process", json=payload)
    assert res.status_code == 200
    data = res.json()

    # Verify evidence
    evidence = data["evidence"]
    assert evidence["candidate_interpretation"]["category"] == "misconception"
    assert evidence["candidate_interpretation"]["hypothesis_id"] == "denominator_addition_rule"

    # Verify new state
    new_state = data["new_state"]
    cstate = new_state["concept_states"]["fraction_addition_unlike_denominators"]
    assert "denominator_addition_rule" in cstate["candidate_misconceptions"]

    # Verify recommended diagnostic probe
    probe = data["recommended_diagnostic_probe"]
    assert probe is not None
    assert probe["concept_id"] == "fraction_addition_unlike_denominators"
    assert "Expected Information Gain" in data["diagnostic_rationale"]

    # Verify state replay via API
    replay_res = client.post("/api/v1/learners/api_student_A/replay")
    assert replay_res.status_code == 200
    replay_data = replay_res.json()
    assert replay_data["replay_status"] == "REPLAY_COMPLETED"
    assert replay_data["total_interactions_replayed"] == 1
