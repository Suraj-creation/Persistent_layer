#!/usr/bin/env python3
"""Train a reproducible TF-IDF + logistic-regression baseline on MAP.

The script expects an authorized local copy of the MAP competition's
``train.csv``. It deliberately does not download data or accept Kaggle rules.
The MAP data-access rules require a participant to accept the terms before
access and limit use to non-commercial purposes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


REQUIRED_COLUMNS = {
    "QuestionId",
    "QuestionText",
    "MC_Answer",
    "StudentExplanation",
    "Category",
    "Misconception",
}


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _label_component(value: Any) -> str:
    cleaned = _clean_text(value)
    if cleaned.lower() in {"", "na", "n/a", "nan", "none", "null"}:
        return "NA"
    return cleaned


def _model_input(question: Any, answer: Any, explanation: Any) -> str:
    return (
        f"Question: {_clean_text(question)}\n"
        f"Selected answer: {_clean_text(answer)}\n"
        f"Student explanation: {_clean_text(explanation)}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_rows(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"MAP training file not found: {path}. Place an authorized train.csv there after accepting the dataset rules."
        )

    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "This file does not match the MAP training schema. Missing columns: "
            + ", ".join(missing)
        )

    rows = pd.DataFrame(
        {
            "question_id": df["QuestionId"].map(_clean_text),
            "model_input": [
                _model_input(q, a, e)
                for q, a, e in zip(
                    df["QuestionText"], df["MC_Answer"], df["StudentExplanation"]
                )
            ],
            "category": df["Category"].map(_label_component),
            "misconception": df["Misconception"].map(_label_component),
        }
    )
    rows["target"] = rows["category"] + ":" + rows["misconception"]
    rows = rows[rows["question_id"].ne("") & rows["model_input"].str.strip().ne("")]

    labels_per_input = rows.groupby("model_input", sort=False)["target"].nunique()
    conflicting_inputs = set(labels_per_input[labels_per_input > 1].index)
    conflict_rows = int(rows["model_input"].isin(conflicting_inputs).sum())
    if conflicting_inputs:
        rows = rows[~rows["model_input"].isin(conflicting_inputs)].copy()

    before_dedup = len(rows)
    rows = rows.drop_duplicates(subset=["model_input", "target"], keep="first").reset_index(drop=True)
    exact_duplicate_rows = before_dedup - len(rows)
    rows["group_id"] = rows["model_input"].map(
        lambda text: hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
    )

    if rows["target"].nunique() < 2:
        raise ValueError("MAP training data must contain at least two distinct Category:Misconception labels.")

    audit = {
        "source_rows": int(len(df)),
        "usable_rows": int(len(rows)),
        "distinct_questions": int(rows["question_id"].nunique()),
        "distinct_joint_labels": int(rows["target"].nunique()),
        "dropped_conflicting_input_rows": conflict_rows,
        "dropped_exact_duplicate_rows": exact_duplicate_rows,
    }
    return rows, audit


def make_model(max_features: int, max_iter: int, seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=max_features // 2,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ],
        transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.7},
    )
    classifier = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def map_at_3(y_true: np.ndarray, probabilities: np.ndarray, labels: np.ndarray) -> float:
    k = min(3, len(labels))
    top_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
    scores = []
    for truth, row in zip(y_true, top_indices):
        ranked = labels[row]
        try:
            rank = list(ranked).index(truth) + 1
        except ValueError:
            scores.append(0.0)
        else:
            scores.append(1.0 / rank)
    return float(np.mean(scores)) if scores else 0.0


def top_label_ece(y_true: np.ndarray, probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    confidences = probabilities.max(axis=1)
    predictions = labels[probabilities.argmax(axis=1)]
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        in_bin = (confidences >= edges[idx]) & (
            confidences < edges[idx + 1] if idx < bins - 1 else confidences <= edges[idx + 1]
        )
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(correct[in_bin].mean()) - float(confidences[in_bin].mean()))
    return float(ece)


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    model_labels: np.ndarray,
    all_labels: np.ndarray,
    seen_in_training: np.ndarray | None = None,
) -> dict[str, Any]:
    predictions = model_labels[probabilities.argmax(axis=1)]
    label_positions = {label: idx for idx, label in enumerate(all_labels)}
    one_hot = np.zeros((len(y_true), len(all_labels)), dtype=np.float64)
    for row_idx, truth in enumerate(y_true):
        one_hot[row_idx, label_positions[truth]] = 1.0

    full_probabilities = np.zeros_like(one_hot)
    for model_idx, label in enumerate(model_labels):
        full_probabilities[:, label_positions[label]] = probabilities[:, model_idx]

    per_label_precision, per_label_recall, per_label_f1, per_label_support = precision_recall_fscore_support(
        y_true, predictions, labels=all_labels, zero_division=0
    )

    if seen_in_training is None:
        model_label_set = set(model_labels)
        seen_in_training = np.asarray([label in model_label_set for label in y_true])

    validation_labels = set(y_true)
    seen_validation_labels = set(y_true[seen_in_training])

    return {
        "validation_rows": int(len(y_true)),
        "validation_rows_with_label_seen_in_training": int(seen_in_training.sum()),
        "validation_row_label_coverage": round(float(seen_in_training.mean()) if len(y_true) else 0.0, 6),
        "validation_distinct_labels_seen_in_training": len(seen_validation_labels),
        "validation_distinct_label_coverage": round(
            len(seen_validation_labels) / len(validation_labels) if validation_labels else 0.0,
            6,
        ),
        "map_at_3": round(map_at_3(y_true, probabilities, model_labels), 6),
        "top_1_accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "macro_f1_over_all_observed_labels": round(
            float(f1_score(y_true, predictions, labels=all_labels, average="macro", zero_division=0)), 6
        ),
        "per_label_metrics": {
            label: {
                "precision": round(float(per_label_precision[idx]), 6),
                "recall": round(float(per_label_recall[idx]), 6),
                "f1": round(float(per_label_f1[idx]), 6),
                "support": int(per_label_support[idx]),
            }
            for idx, label in enumerate(all_labels)
        },
        "multiclass_brier_score_over_all_observed_labels": round(
            float(np.mean(np.sum((one_hot - full_probabilities) ** 2, axis=1))) if len(y_true) else 0.0,
            6,
        ),
        "top_label_ece_10_bins": round(top_label_ece(y_true, probabilities, model_labels), 6),
        "note": "Confidence metrics are uncalibrated diagnostics; fit calibration on held-out data before using probabilities in learner state.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/raw/map/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/map_misconception_classifier"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=80000)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--confirm-map-data-access-authorization",
        action="store_true",
        help="Confirm you have accepted the MAP competition rules and are using the data for non-commercial research/education.",
    )
    args = parser.parse_args()
    if not args.confirm_map_data_access_authorization:
        parser.error(
            "MAP data use is gated by its competition rules. Sign in and accept the rules yourself before using the dataset, then rerun with --confirm-map-data-access-authorization."
        )
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    rows, audit = load_training_rows(args.train_csv)
    if rows["group_id"].nunique() < args.folds:
        raise ValueError("There are fewer unique model inputs than requested validation folds.")

    all_labels = np.asarray(sorted(rows["target"].unique()))
    label_positions = {label: idx for idx, label in enumerate(all_labels)}
    oof_probabilities = np.zeros((len(rows), len(all_labels)), dtype=np.float64)
    oof_label_seen_in_training = np.zeros(len(rows), dtype=bool)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(rows["model_input"], rows["target"], groups=rows["group_id"])
    ):
        train_rows = rows.iloc[train_indices]
        validation_rows = rows.iloc[validation_indices]
        if train_rows["target"].nunique() < 2:
            raise ValueError(f"Fold {fold} has fewer than two training labels; reduce --folds or revise the split.")

        validation_model = make_model(args.max_features, args.max_iter, args.seed)
        validation_model.fit(train_rows["model_input"], train_rows["target"])
        fold_probabilities = validation_model.predict_proba(validation_rows["model_input"])
        fold_labels = validation_model.named_steps["classifier"].classes_
        for fold_label_index, label in enumerate(fold_labels):
            oof_probabilities[validation_indices, label_positions[label]] = fold_probabilities[:, fold_label_index]
        oof_label_seen_in_training[validation_indices] = validation_rows["target"].isin(fold_labels).to_numpy()

    metrics = evaluate(
        rows["target"].to_numpy(),
        oof_probabilities,
        all_labels,
        all_labels,
        oof_label_seen_in_training,
    )

    final_model = make_model(args.max_features, args.max_iter, args.seed)
    final_model.fit(rows["model_input"], rows["target"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, args.output_dir / "model.joblib")
    final_labels = final_model.named_steps["classifier"].classes_.tolist()
    (args.output_dir / "labels.json").write_text(
        json.dumps(final_labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = {
        "task": "MAP Category:Misconception classification",
        "model": "word+character TF-IDF with class-balanced logistic regression",
        "dataset_file": str(args.train_csv),
        "dataset_sha256": _sha256(args.train_csv),
        "source_license": "CC BY-NC 4.0; MAP competition rules apply",
        "usage_scope": "non-commercial research and education only",
        "validation": {
            "split_method": "StratifiedGroupKFold out-of-fold predictions grouped by exact normalized model input",
            "folds": args.folds,
            "seed": args.seed,
            "metrics": metrics,
        },
        "data_audit": audit,
        "training_rows_after_cleaning": int(len(rows)),
        "distinct_questions": int(rows["question_id"].nunique()),
        "joint_label_count": int(rows["target"].nunique()),
        "classes_in_final_artifact": len(final_labels),
        "hyperparameters": {
            "word_ngram_range": [1, 2],
            "character_ngram_range": [3, 5],
            "min_document_frequency": 2,
            "word_max_features": args.max_features,
            "character_max_features": args.max_features // 2,
            "logistic_regression_C": 4.0,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
        },
        "versions": {"python": __import__("sys").version.split()[0], "scikit_learn": sklearn.__version__},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "This classifier proposes a response-level label. It does not infer a longitudinal learner state or prove a student's cognition.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
