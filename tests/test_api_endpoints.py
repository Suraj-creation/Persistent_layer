from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from apps.api.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


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
    assert replay_data["replay_status"] == "SUCCESS_IDENTICAL_PROJECTION"
    assert replay_data["total_interactions_replayed"] == 1
