import numpy as np
import pandas as pd
import pytest

from experiments.train_map_misconception_classifier import (
    _model_input,
    load_training_rows,
    make_model,
    map_at_3,
)


def test_model_input_preserves_math_symbols():
    text = _model_input("Compare 1/2 and 0.5", "1/2 > 0.5", "I used 1/2 > 0.5")

    assert "1/2" in text
    assert ">" in text
    assert "0.5" in text


def test_load_training_rows_drops_conflicts_and_exact_duplicates(tmp_path):
    source = pd.DataFrame(
        [
            {"QuestionId": 1, "QuestionText": "Q1", "MC_Answer": "A", "StudentExplanation": "same input", "Category": "False_Misconception", "Misconception": "A"},
            {"QuestionId": 1, "QuestionText": "Q1", "MC_Answer": "A", "StudentExplanation": "same input", "Category": "False_Misconception", "Misconception": "B"},
            {"QuestionId": 2, "QuestionText": "Q2", "MC_Answer": "B", "StudentExplanation": "unique input one", "Category": "False_Misconception", "Misconception": "C"},
            {"QuestionId": 2, "QuestionText": "Q2", "MC_Answer": "B", "StudentExplanation": "unique input one", "Category": "False_Misconception", "Misconception": "C"},
            {"QuestionId": 3, "QuestionText": "Q3", "MC_Answer": "C", "StudentExplanation": "unique input two", "Category": "True_Correct", "Misconception": "NA"},
        ]
    )
    csv_path = tmp_path / "train.csv"
    source.to_csv(csv_path, index=False)

    rows, audit = load_training_rows(csv_path)

    assert len(rows) == 2
    assert audit["dropped_conflicting_input_rows"] == 2
    assert audit["dropped_exact_duplicate_rows"] == 1
    assert rows["target"].nunique() == 2


def test_map_at_3_uses_reciprocal_rank():
    labels = np.asarray(["A", "B", "C"])
    probabilities = np.asarray(
        [
            [0.2, 0.5, 0.3],  # B ranks first
            [0.6, 0.3, 0.1],  # B ranks second
        ]
    )

    assert map_at_3(np.asarray(["B", "B"]), probabilities, labels) == pytest.approx(0.75)


def test_tfidf_logistic_model_fits_smoke_corpus():
    model = make_model(max_features=200, max_iter=100, seed=42)
    texts = [
        "Question: compare 1/2 and 1/3 Student explanation: add the denominators",
        "Question: compare 1/4 and 1/5 Student explanation: add the denominators",
        "Question: compare 2/3 and 1/6 Student explanation: use a common denominator",
        "Question: compare 3/4 and 1/8 Student explanation: use a common denominator",
    ]
    labels = ["False_Misconception:Denominator", "False_Misconception:Denominator", "True_Correct:NA", "True_Correct:NA"]

    model.fit(texts, labels)
    probabilities = model.predict_proba(texts)

    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
