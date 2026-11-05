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


def test_synthetic_loaders_and_canonical_records_are_repeatable():
    from pandas.testing import assert_frame_equal
    for loader_type in (EediDataLoader, FoundationalAssistLoader):
        first, second = loader_type(), loader_type()
        assert_frame_equal(first.load(), second.load())
        a, b = first.to_canonical_interactions(), second.to_canonical_interactions()
        assert a == b


def test_splitter_does_not_change_global_random_state():
    import numpy as np
    df = EediDataLoader().load()
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    LeakageSafeSplitter.student_split(df)
    assert np.random.random() == expected


@pytest.mark.parametrize("ratio", [0, 1, -1, 2])
def test_splitter_rejects_invalid_ratio(ratio):
    with pytest.raises(ValueError):
        LeakageSafeSplitter.student_split(EediDataLoader().load(), test_ratio=ratio)
