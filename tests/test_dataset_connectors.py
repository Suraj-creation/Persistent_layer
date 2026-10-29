from __future__ import annotations

import pytest
from packages.learner_state_core.data.eedi_loader import EediDataLoader
from packages.learner_state_core.data.assistments_loader import FoundationalAssistLoader
from packages.learner_state_core.data.splitter import LeakageSafeSplitter


def test_eedi_loader_and_audit():
    loader = EediDataLoader()
    df = loader.load()
    assert len(df) > 0
    assert "explanation" in df.columns
    assert "user_id" in df.columns

    audit = loader.audit()
    assert audit.total_records == len(df)
    assert audit.total_students > 0
    assert audit.text_availability_rate > 0.50
    assert audit.misconception_label_rate > 0.20
    assert "SUITABLE" in audit.suitability_verdict

    canonical_records = loader.to_canonical_interactions()
    assert len(canonical_records) == len(df)
    interaction, gt_misc = canonical_records[0]
    assert interaction.learner_id.startswith("student_")
    assert interaction.response_text is not None


def test_assistments_loader_and_audit():
    loader = FoundationalAssistLoader()
    df = loader.load()
    assert len(df) > 0
    assert "answer_text" in df.columns

    audit = loader.audit()
    assert audit.total_records == len(df)
    assert audit.total_students > 0
    assert "FoundationalASSIST" in audit.dataset_name

    interactions = loader.to_canonical_interactions()
    assert len(interactions) == len(df)


def test_leakage_safe_splitter():
    loader = EediDataLoader()
    df = loader.load()

    # Test single student split
    train_df, test_df = LeakageSafeSplitter.student_split(df, student_col="user_id", test_ratio=0.25)
    train_users = set(train_df["user_id"])
    test_users = set(test_df["user_id"])

    # Disjoint sets: NO OVERLAP!
    assert len(train_users.intersection(test_users)) == 0
    assert len(train_users) + len(test_users) == df["user_id"].nunique()

    # Test k-fold split
    folds = LeakageSafeSplitter.student_k_fold(df, student_col="user_id", k=3)
    assert len(folds) == 3
    for tr, te in folds:
        assert len(set(tr["user_id"]).intersection(set(te["user_id"]))) == 0
