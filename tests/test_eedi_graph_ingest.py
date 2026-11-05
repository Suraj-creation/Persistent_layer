import pandas as pd
import pytest

from experiments import ingest_eedi_misconceptions_graph as ingest


def graph_frames():
    edges = pd.DataFrame(
        {
            "MisconceptionId": [100, 100],
            "MisconceptionName": ["Example error", "Example error"],
            "PercentageMisconceptionExistsInThisConstruct": [70.0, 30.0],
            "ConstructId": [1, 2],
            "ConstructName": ["Construct one", "Construct two"],
            "SubTopicSubjectId": [10, 10],
            "SubTopicName": ["Subtopic", "Subtopic"],
            "TopicSubjectId": [20, 20],
            "TopicName": ["Topic", "Topic"],
            "SubjectSubjectId": [30, 30],
            "SubjectName": ["Number", "Number"],
        }
    )
    constructs = pd.DataFrame(
        {
            "SubjectName": ["Number", "Number"],
            "SubjectSubjectId": [30, 30],
            "TopicName": ["Topic", "Topic"],
            "TopicSubjectId": [20, 20],
            "SubTopicName": ["Subtopic", "Subtopic"],
            "SubTopicSubjectId": [10, 10],
            "ConstructName": ["Construct one", "Construct two"],
            "ConstructId": [1, 2],
        }
    )
    return {"misconception-map.csv": edges, "construct-map.csv": constructs}


def use_small_expected_tables(monkeypatch):
    monkeypatch.setattr(
        ingest,
        "FILES",
        {
            "misconception-map.csv": {
                **ingest.FILES["misconception-map.csv"],
                "rows": 2,
            },
            "construct-map.csv": {
                **ingest.FILES["construct-map.csv"],
                "rows": 2,
            },
        },
    )


def test_eedi_graph_audit_checks_keys_links_and_weight_totals(monkeypatch):
    use_small_expected_tables(monkeypatch)

    audit = ingest.audit_graph(graph_frames())

    assert audit["row_counts"] == {
        "misconception-map.csv": 2,
        "construct-map.csv": 2,
    }
    assert audit["unique_counts"]["misconceptions"] == 1
    assert audit["unique_counts"]["construct_catalogue"] == 2
    assert audit["weight_validation"]["all_groups_sum_to_100_within_tolerance"]


def test_eedi_graph_audit_rejects_unlinked_construct_id(monkeypatch):
    use_small_expected_tables(monkeypatch)
    frames = graph_frames()
    frames["misconception-map.csv"].loc[0, "ConstructId"] = 99

    with pytest.raises(ValueError, match="absent from construct-map"):
        ingest.audit_graph(frames)


def test_eedi_graph_audit_rejects_weight_groups_that_do_not_sum_to_100(monkeypatch):
    use_small_expected_tables(monkeypatch)
    frames = graph_frames()
    frames["misconception-map.csv"].loc[1, "PercentageMisconceptionExistsInThisConstruct"] = 25.0

    with pytest.raises(ValueError, match="do not sum to 100"):
        ingest.audit_graph(frames)
