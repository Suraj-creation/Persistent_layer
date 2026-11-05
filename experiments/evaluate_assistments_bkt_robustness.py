"""Repeat the ASSISTments BKT holdout and compare low-complexity predictors.

This is a correctness-only knowledge-tracing experiment. It trains a separate
non-forgetting BKT model for each student split and compares the predictions
with constant prevalence, smoothed skill prevalence, and a skill-plus-opportunity
logistic baseline. The script never writes learner-level predictions or rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:  # Support both `python -m experiments...` and direct script execution.
    from experiments.train_assistments_bkt import (
        DEFAULT_DATA,
        REQUIRED_COLUMNS,
        SOURCE_PAGE,
        binary_metrics,
        filter_supported_skills,
        prepare_interactions,
        select_skill_subset,
        split_by_student,
    )
except ModuleNotFoundError as exc:
    if exc.name != "experiments":
        raise
    from train_assistments_bkt import (
        DEFAULT_DATA,
        REQUIRED_COLUMNS,
        SOURCE_PAGE,
        binary_metrics,
        filter_supported_skills,
        prepare_interactions,
        select_skill_subset,
        split_by_student,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "models/assistments_2009_2010_bkt/robustness.json"
DEFAULT_SEEDS = (42, 7)


def add_skill_opportunity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each learner's one-based opportunity count within a released skill."""
    result = frame.sort_values(["user_id", "order_id"], kind="mergesort").copy()
    result["skill_opportunity"] = (
        result.groupby(["user_id", "skill_name"], sort=False).cumcount() + 1
    )
    return result.reset_index(drop=True)


def predict_skill_prevalence(
    train: pd.DataFrame, test: pd.DataFrame, prior_strength: float = 20.0
) -> np.ndarray:
    """Estimate a skill rate with a global-mean prior to stabilize small skills."""
    global_rate = float(train["correct"].mean())
    stats = train.groupby("skill_name")["correct"].agg(["sum", "count"])
    rates = (stats["sum"] + prior_strength * global_rate) / (
        stats["count"] + prior_strength
    )
    return test["skill_name"].map(rates).fillna(global_rate).to_numpy(dtype=float)


def fit_opportunity_model(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Fit a compact, leakage-safe skill and log-opportunity correctness model."""
    train_features = train[["skill_name", "skill_opportunity"]].copy()
    test_features = test[["skill_name", "skill_opportunity"]].copy()
    train_features["log_skill_opportunity"] = np.log1p(
        train_features.pop("skill_opportunity").astype(float)
    )
    test_features["log_skill_opportunity"] = np.log1p(
        test_features.pop("skill_opportunity").astype(float)
    )

    transform = ColumnTransformer(
        [
            ("skill", OneHotEncoder(handle_unknown="ignore"), ["skill_name"]),
            ("opportunity", StandardScaler(), ["log_skill_opportunity"]),
        ]
    )
    predictor = make_pipeline(
        transform,
        LogisticRegression(max_iter=1000, random_state=0),
    )
    predictor.fit(train_features, train["correct"].to_numpy(dtype=int))
    return predictor.predict_proba(test_features)[:, 1]


def _summarize_seed_scores(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(seed_results[0]["models"])
    metrics = ("auc", "accuracy_at_0_5", "brier_score", "ece_10_bins_uncalibrated")
    summary: dict[str, Any] = {}
    for model_name in names:
        summary[model_name] = {}
        for metric in metrics:
            values = [row["models"][model_name][metric] for row in seed_results]
            finite = np.asarray([v for v in values if v is not None], dtype=float)
            summary[model_name][metric] = {
                "mean_across_splits": float(finite.mean()) if finite.size else None,
                "sd_across_splits": float(finite.std(ddof=1)) if finite.size > 1 else None,
            }
    return summary


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    if not args.seeds:
        raise ValueError("At least one split seed is required")
    if args.skill_prior_strength < 0:
        raise ValueError("--skill-prior-strength must be nonnegative")

    raw = pd.read_csv(
        data_path,
        encoding="latin1",
        low_memory=False,
        usecols=lambda column: column in REQUIRED_COLUMNS,
    )
    interactions, _ = prepare_interactions(raw)
    interactions = select_skill_subset(interactions, args.skill_regex)
    interactions = filter_supported_skills(
        interactions, args.min_skill_rows, args.min_skill_students
    )
    interactions = add_skill_opportunity(interactions)
    if interactions.empty:
        raise ValueError("No rows remain after the configured data filters")

    from pyBKT.models import Model

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        train_df, test_df = split_by_student(interactions, args.test_ratio, seed)
        train_skills = set(train_df["skill_name"].unique())
        evaluable_test = test_df.loc[test_df["skill_name"].isin(train_skills)].copy()
        if evaluable_test.empty:
            raise ValueError(f"Seed {seed} produced no test events with trained skills")

        model = Model(seed=seed, num_fits=args.num_fits, parallel=False)
        reused_existing_model = False
        artifact_dir = PROJECT_ROOT / "models/assistments_2009_2010_bkt"
        metrics_path = artifact_dir / "metrics.json"
        expected_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if seed == 42 and args.reuse_seed_42_artifact and metrics_path.is_file():
            recorded = json.loads(metrics_path.read_text())
            if recorded.get("provenance", {}).get("data_file_sha256") == expected_hash:
                model.load(str(artifact_dir / "model.pkl"))
                reused_existing_model = True
        if not reused_existing_model:
            model.fit(
                data=train_df[["user_id", "skill_name", "correct"]],
                skills=".*",
            )
        model_path = artifact_dir / ("model.pkl" if seed == 42 else f"model_seed_{seed}.pkl")
        params_path = artifact_dir / (
            "parameters.csv" if seed == 42 else f"parameters_seed_{seed}.csv"
        )
        if not reused_existing_model:
            model.save(str(model_path))
            model.params().reset_index().to_csv(params_path, index=False)
        prediction_frame = model.predict(
            data=evaluable_test[["user_id", "skill_name", "correct"]]
        )
        if len(prediction_frame) != len(evaluable_test):
            raise RuntimeError(
                f"pyBKT returned {len(prediction_frame)} predictions for "
                f"{len(evaluable_test)} held-out rows on seed {seed}"
            )
        y_true = prediction_frame["correct"].to_numpy(dtype=int)
        y_bkt = np.clip(
            prediction_frame["correct_predictions"].to_numpy(dtype=float), 0.0, 1.0
        )
        if not np.isfinite(y_bkt).all():
            raise RuntimeError(f"pyBKT returned non-finite scores for seed {seed}")

        train_prevalence = float(train_df["correct"].mean())
        predictions = {
            "bkt": y_bkt,
            "constant_train_prevalence": np.full(len(y_true), train_prevalence),
            "smoothed_skill_prevalence": predict_skill_prevalence(
                train_df, evaluable_test, args.skill_prior_strength
            ),
            "skill_plus_opportunity_logistic": fit_opportunity_model(
                train_df, evaluable_test
            ),
        }
        seed_results.append(
            {
                "seed": seed,
                "bkt_model_source": (
                    "existing seed-42 artifact with matching dataset checksum"
                    if reused_existing_model
                    else "fit on this seed's training learners"
                ),
                "bkt_artifacts": {
                    "model": str(model_path.relative_to(PROJECT_ROOT)),
                    "parameters": str(params_path.relative_to(PROJECT_ROOT)),
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "parameter_sha256": hashlib.sha256(params_path.read_bytes()).hexdigest(),
                },
                "split_counts": {
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "test_rows_evaluated": int(len(y_true)),
                    "unseen_skill_rows_excluded": int(len(test_df) - len(y_true)),
                    "train_students": int(train_df["user_id"].nunique()),
                    "test_students": int(test_df["user_id"].nunique()),
                },
                "models": {
                    name: binary_metrics(y_true, probs)
                    for name, probs in predictions.items()
                },
            }
        )
        print(
            f"seed={seed} rows={len(y_true):,} "
            f"BKT AUC={seed_results[-1]['models']['bkt']['auc']:.4f} "
            f"opportunity AUC="
            f"{seed_results[-1]['models']['skill_plus_opportunity_logistic']['auc']:.4f}",
            flush=True,
        )
        # Persist completed splits so an interrupted serial BKT run can resume
        # without losing finished evaluations.
        checkpoint = {
            "dataset": "ASSISTments 2009-2010 corrected skill-builder release",
            "source_page": SOURCE_PAGE,
            "status": "in_progress",
            "completed_seeds": [row["seed"] for row in seed_results],
            "seed_results": seed_results,
            "provenance": {
                "dataset_file_sha256": expected_hash,
                "dataset_file_name": data_path.name,
            },
        }
        output_path.write_text(json.dumps(checkpoint, indent=2) + "\n")

    results = {
        "dataset": "ASSISTments 2009-2010 corrected skill-builder release",
        "task": "predict first-attempt correctness from a held-out student's observed history",
        "source_page": SOURCE_PAGE,
        "split_design": "repeated student-level holdouts; all model parameters fit on training learners only",
        "seeds": list(args.seeds),
        "test_ratio": args.test_ratio,
        "skill_regex": args.skill_regex or "(all skills)",
        "skill_support": {
            "min_rows": args.min_skill_rows,
            "min_students": args.min_skill_students,
        },
        "skill_prevalence_prior_strength": args.skill_prior_strength,
        "bkt": {
            "num_fits": args.num_fits,
            "forgetting_enabled": False,
            "parallel": False,
        },
        "summary": _summarize_seed_scores(seed_results),
        "paired_bkt_minus_baseline_mean": {
            baseline: {
                metric: float(
                    np.mean(
                        [
                            row["models"]["bkt"][metric]
                            - row["models"][baseline][metric]
                            for row in seed_results
                        ]
                    )
                )
                for metric in ("auc", "brier_score")
            }
            for baseline in (
                "constant_train_prevalence",
                "smoothed_skill_prevalence",
                "skill_plus_opportunity_logistic",
            )
        },
        "seed_results": seed_results,
        "interpretation_limits": [
            "Across-seed standard deviations describe split sensitivity; they are not confidence intervals.",
            "The opportunity model uses only skill identity and within-learner skill opportunity count.",
            "All baselines are correctness predictors, not misconception or reasoning classifiers.",
            "This historical source does not establish transfer to the app's learners or learning impact.",
            "No learner-level predictions or raw dataset rows are written to the report.",
        ],
        "provenance": {
            "dataset_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "dataset_file_name": data_path.name,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "versions": {
                "python": platform.python_version(),
                "pyBKT": importlib.metadata.version("pyBKT"),
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scikit-learn": importlib.metadata.version("scikit-learn"),
            },
        },
    }
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved aggregate robustness report to {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--skill-regex", default="fraction|percent")
    parser.add_argument("--min-skill-rows", type=int, default=100)
    parser.add_argument("--min-skill-students", type=int, default=20)
    parser.add_argument("--num-fits", type=int, default=1)
    parser.add_argument("--skill-prior-strength", type=float, default=20.0)
    parser.add_argument(
        "--reuse-seed-42-artifact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the tracked seed-42 BKT artifact when its dataset checksum matches.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
