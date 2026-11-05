import pandas as pd
import pytest

from experiments.evaluate_assistments_bkt_robustness import (
    add_skill_opportunity,
    predict_skill_prevalence,
)


def test_skill_opportunity_counts_follow_learner_skill_order():
    frame = pd.DataFrame(
        {
            "user_id": ["a", "a", "a", "b"],
            "skill_name": ["Fractions", "Percents", "Fractions", "Fractions"],
            "order_id": [4, 2, 8, 3],
            "correct": [1, 0, 0, 1],
        }
    )

    result = add_skill_opportunity(frame)

    assert result[["user_id", "skill_name", "skill_opportunity"]].to_dict("records") == [
        {"user_id": "a", "skill_name": "Percents", "skill_opportunity": 1},
        {"user_id": "a", "skill_name": "Fractions", "skill_opportunity": 1},
        {"user_id": "a", "skill_name": "Fractions", "skill_opportunity": 2},
        {"user_id": "b", "skill_name": "Fractions", "skill_opportunity": 1},
    ]


def test_skill_prevalence_is_smoothed_and_unknown_skill_falls_back_to_global():
    train = pd.DataFrame(
        {
            "skill_name": ["Fractions", "Fractions", "Percents"],
            "correct": [1, 0, 1],
        }
    )
    test = pd.DataFrame({"skill_name": ["Fractions", "Unseen"]})

    result = predict_skill_prevalence(train, test, prior_strength=1.0)

    assert result[0] == pytest.approx(5 / 9)
    assert result[1] == pytest.approx(2 / 3)
