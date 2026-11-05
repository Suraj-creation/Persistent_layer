import pandas as pd
import pytest

from experiments.train_assistments_bkt import (
    prepare_interactions,
    filter_supported_skills,
    select_skill_subset,
    split_by_student,
)


def test_prepare_interactions_keeps_only_supported_ordered_targets():
    raw = pd.DataFrame(
        [
            {"order_id": 20, "user_id": 1, "problem_id": 2, "original": 1, "correct": 1, "first_action": 0, "skill_name": "Fractions"},
            {"order_id": 10, "user_id": 1, "problem_id": 1, "original": 1, "correct": 0, "first_action": 0, "skill_name": "Fractions"},
            {"order_id": 30, "user_id": 1, "problem_id": 3, "original": 0, "correct": 1, "first_action": 0, "skill_name": "Fractions"},
            {"order_id": 40, "user_id": 1, "problem_id": 4, "original": 1, "correct": 0, "first_action": 1, "skill_name": "Fractions"},
            {"order_id": 50, "user_id": 2, "problem_id": 5, "original": 1, "correct": 1, "first_action": 0, "skill_name": None},
        ]
    )

    prepared, audit = prepare_interactions(raw)

    assert prepared[["user_id", "order_id", "correct"]].to_dict("records") == [
        {"user_id": "1", "order_id": 10, "correct": 0},
        {"user_id": "1", "order_id": 20, "correct": 1},
    ]
    assert audit["raw_rows"] == 5
    assert audit["model_rows"] == 2


def test_prepare_interactions_rejects_duplicate_events():
    raw = pd.DataFrame(
        [
            {"order_id": 1, "user_id": 1, "problem_id": 1, "original": 1, "correct": 1, "first_action": 0, "skill_name": "Fractions"},
            {"order_id": 1, "user_id": 1, "problem_id": 1, "original": 1, "correct": 1, "first_action": 0, "skill_name": "Fractions"},
        ]
    )

    with pytest.raises(ValueError, match="repeated user/problem/order"):
        prepare_interactions(raw)


def test_student_split_keeps_learners_disjoint_and_deterministic():
    frame = pd.DataFrame(
        {"user_id": [str(i) for i in range(10) for _ in range(2)], "event": range(20)}
    )

    train_a, test_a = split_by_student(frame, test_ratio=0.2, seed=42)
    train_b, test_b = split_by_student(frame, test_ratio=0.2, seed=42)

    assert set(train_a.user_id).isdisjoint(set(test_a.user_id))
    assert sorted(test_a.user_id.unique()) == sorted(test_b.user_id.unique())
    assert len(train_a) + len(test_a) == len(frame)


def test_skill_subset_focuses_on_fraction_domain_and_rejects_bad_regex():
    frame = pd.DataFrame(
        {"skill_name": ["Addition and Subtraction Fractions", "Equation Solving"]}
    )

    selected = select_skill_subset(frame, "fraction")

    assert selected.skill_name.tolist() == ["Addition and Subtraction Fractions"]
    with pytest.raises(ValueError, match="Invalid --skill-regex"):
        select_skill_subset(frame, "[")


def test_filter_supported_skills_removes_underrepresented_labels():
    frame = pd.DataFrame(
        {
            "user_id": ["1", "2", "1"],
            "skill_name": ["Fractions", "Fractions", "Rare Skill"],
            "correct": [1, 0, 1],
        }
    )

    selected = filter_supported_skills(frame, min_rows=2, min_students=2)

    assert selected.skill_name.tolist() == ["Fractions", "Fractions"]
