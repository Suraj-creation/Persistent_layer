"""Train and evaluate a correctness-only BKT baseline on ASSISTments 2009-2010.

The script uses the corrected skill-builder release. It keeps only original
problems where a student attempted an answer before asking for help, so the
target is first-attempt correctness rather than a hint request encoded as an
incorrect response. Students are split before fitting. On held-out students,
their earlier responses are available to predict their later responses.

The resulting model is a knowledge-tracing baseline for the released skill
names. It is not a misconception-from-explanation model and is not directly
compatible with the prototype's fraction taxonomy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data/raw/assistments/assistments_2009_2010_corrected.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "models/assistments_2009_2010_bkt"
SOURCE_PAGE = (
    "https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/"
    "skill-builder-data-2009-2010"
)
SOURCE_FILE = "https://drive.google.com/file/d/1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE/view"

REQUIRED_COLUMNS = {
    "order_id",
    "user_id",
    "problem_id",
    "original",
    "correct",
    "first_action",
    "skill_name",
}


# Prepare chronological first-attempt records for offline correctness prediction.
def prepare_interactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select clean first-attempt, original-problem events and sort chronologically."""
    # Check that the corrected ASSISTments file contains the required fields.
    missing = REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Count the original input for the data audit.
    audit = {"raw_rows": int(len(raw))}
    # Make a working copy of only the fields used in training.
    frame = raw[list(REQUIRED_COLUMNS)].copy()
    # Parse the original-problem flag numerically.
    frame["original"] = pd.to_numeric(frame["original"], errors="coerce")
    # Parse the first-action indicator.
    frame["first_action"] = pd.to_numeric(frame["first_action"], errors="coerce")
    # Parse correctness into its numeric representation.
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce")
    # Convert the event order to a numeric value before sorting.
    frame["order_id"] = pd.to_numeric(frame["order_id"], errors="coerce")

    # Keep the original problems in the logged dataset.
    frame = frame.loc[frame["original"].eq(1)].copy()
    # Count the events that passed that filter.
    audit["original_problem_rows"] = int(len(frame))
    # Keep the rows whose first recorded action was an attempt.
    frame = frame.loc[frame["first_action"].eq(0)].copy()
    # Record how many first-attempt rows passed.
    audit["attempt_first_rows"] = int(len(frame))
    # Normalize skill labels and replace missing labels with empty text for filtering.
    frame["skill_name"] = frame["skill_name"].fillna("").astype(str).str.strip()
    # Require learner, problem and order IDs, a skill and a binary correctness label.
    valid = (
        frame["user_id"].notna()
        & frame["problem_id"].notna()
        & frame["order_id"].notna()
        & frame["skill_name"].ne("")
        & frame["correct"].isin([0, 1])
    )
    # Keep only rows meeting those data quality requirements.
    frame = frame.loc[valid].copy()
    # Count the clean model rows.
    audit["model_rows"] = int(len(frame))
    # Record how many original rows were removed by task and quality filters.
    audit["dropped_after_task_filters"] = audit["raw_rows"] - audit["model_rows"]

    # Look for repeated events using learner, problem and request order IDs.
    duplicate_keys = frame.duplicated(["user_id", "problem_id", "order_id"], keep=False)
    # Store how many rows were involved in repeated event keys.
    audit["duplicate_event_rows"] = int(duplicate_keys.sum())
    # Stop on unexpected duplicates so training does not count the same event twice.
    if duplicate_keys.any():
        raise ValueError(
            "Corrected ASSISTments release contains repeated user/problem/order events; "
            "inspect duplicates before training."
        )

    # Use strings as learner identifiers for the model.
    frame["user_id"] = frame["user_id"].astype(str)
    # Standardize the skill name column to text.
    frame["skill_name"] = frame["skill_name"].astype(str)
    # Keep observed correctness as a binary integer.
    frame["correct"] = frame["correct"].astype(int)
    # Sort within each learner by event order with a stable sort.
    frame = frame.sort_values(["user_id", "order_id"], kind="mergesort").reset_index(drop=True)
    # Return the compact model table and the audit.
    return frame[["user_id", "skill_name", "correct", "order_id", "problem_id"]], audit


# Hold out whole students so their rows cannot appear in both training and evaluation.
def split_by_student(
    frame: pd.DataFrame, test_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Validate the requested test share.
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    # List unique learner IDs before drawing the split.
    students = frame["user_id"].drop_duplicates().to_numpy(copy=True)
    # Require enough learners for both training and test.
    if len(students) < 2:
        raise ValueError("At least two students are required for a held-out split")
    # Shuffle student IDs using the reproducible seed.
    np.random.default_rng(seed).shuffle(students)
    # Keep at least one learner in each partition.
    n_test = min(max(1, math.ceil(len(students) * test_ratio)), len(students) - 1)
    # Assign the first part of the shuffled ID list to test.
    test_students = set(students[:n_test])
    # Keep all the held-out learners' rows together.
    test = frame.loc[frame["user_id"].isin(test_students)].copy().reset_index(drop=True)
    # Keep the remaining learners' rows together in training.
    train = frame.loc[~frame["user_id"].isin(test_students)].copy().reset_index(drop=True)
    # Return both chronologically ordered partitions.
    return train, test


# Apply an optional experiment-specific skill filter.
def select_skill_subset(frame: pd.DataFrame, skill_regex: str) -> pd.DataFrame:
    if not skill_regex:
        return frame.copy().reset_index(drop=True)
    try:
        selected = frame["skill_name"].str.contains(skill_regex, case=False, regex=True)
    except re.error as exc:
        raise ValueError(f"Invalid --skill-regex: {exc}") from exc
    return frame.loc[selected].copy().reset_index(drop=True)


# Exclude rare skills below the chosen row or student support.
def filter_supported_skills(
    frame: pd.DataFrame, min_rows: int, min_students: int
) -> pd.DataFrame:
    # Count rows and unique learners for each skill.
    support = frame.groupby("skill_name").agg(
        rows=("correct", "size"), students=("user_id", "nunique")
    )
    # Keep only skills meeting both thresholds.
    keep = support.index[
        (support["rows"] >= min_rows) & (support["students"] >= min_students)
    ]
    return frame.loc[frame["skill_name"].isin(keep)].copy().reset_index(drop=True)


# Measure calibration using bins of predicted probabilities.
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= edges[i]) & (
            y_prob < edges[i + 1] if i < n_bins - 1 else y_prob <= edges[i + 1]
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


# Compare the correctness forecasts with actual labels.
def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | int | None]:
    y_hat = (y_prob >= 0.5).astype(int)
    auc: float | None = None
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_prob))
    return {
        "rows": int(len(y_true)),
        "auc": auc,
        "accuracy_at_0_5": float(accuracy_score(y_true, y_hat)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece_10_bins_uncalibrated": compute_ece(y_true, y_prob),
    }


# Run the offline ASSISTments fit and held-out comparison.
def train(args: argparse.Namespace) -> dict[str, Any]:
    # Resolve the local corrected dataset path.
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    # Read the data with the source encoding and restrict it to training columns.
    raw = pd.read_csv(
        data_path,
        encoding="latin1",
        low_memory=False,
        usecols=lambda column: column in REQUIRED_COLUMNS,
    )
    # Clean and audit the original dataset.
    interactions, audit = prepare_interactions(raw)
    # Limit the skill scope if the experiment requested a particular subject subset.
    interactions = select_skill_subset(interactions, args.skill_regex)
    audit["skill_regex"] = args.skill_regex or "(all skills)"
    audit["rows_after_regex_filter"] = int(len(interactions))
    audit["skills_after_regex_filter"] = int(interactions["skill_name"].nunique())
    # Apply the minimum support thresholds for stable fitting.
    filtered_by_support = filter_supported_skills(
        interactions, args.min_skill_rows, args.min_skill_students
    )
    audit["min_skill_rows"] = args.min_skill_rows
    audit["min_skill_students"] = args.min_skill_students
    audit["rows_after_support_filter"] = int(len(filtered_by_support))
    audit["skills_after_support_filter"] = int(
        filtered_by_support["skill_name"].nunique()
    )
    audit["rows_dropped_for_low_skill_support"] = int(
        len(interactions) - len(filtered_by_support)
    )
    # Use the filtered rows for the student split.
    interactions = filtered_by_support
    if interactions.empty:
        raise ValueError("No rows remain after applying --skill-regex")
    print(
        f"Prepared {len(interactions):,} events across "
        f"{interactions['user_id'].nunique():,} students and "
        f"{interactions['skill_name'].nunique():,} skills."
    )
    # Split by whole students without shuffling the order of their responses.
    train_df, test_df = split_by_student(interactions, args.test_ratio, args.seed)

    # Remember which skills the model will have seen during fitting.
    train_skills = set(train_df["skill_name"].unique())
    # Count the skills present in the held-out set.
    test_skills = set(test_df["skill_name"].unique())
    # Evaluate only test skills the model could learn from training.
    evaluable_test = test_df.loc[test_df["skill_name"].isin(train_skills)].copy()
    # Report how many held-out events were outside that common skill scope.
    unseen_skill_rows = int(len(test_df) - len(evaluable_test))
    if evaluable_test.empty:
        raise ValueError("No held-out test rows have a skill represented in training")

    # Import the classical BKT implementation for this offline run.
    from pyBKT.models import Model

    # pyBKT's parallel implementation creates a fresh process pool per EM
    # iteration. Serial fitting avoids process explosions on many-core hosts.
    # Initialize the seeded model in serial mode.
    model = Model(seed=args.seed, num_fits=args.num_fits, parallel=False)
    print(
        f"Fitting serial BKT on {len(train_df):,} events from "
        f"{train_df['user_id'].nunique():,} students...",
        flush=True,
    )
    # Estimate BKT parameters from binary correctness sequences only.
    model.fit(
        # Give fitting only learner IDs, skill labels and correctness observations.
        data=train_df[["user_id", "skill_name", "correct"]],
        # Fit all skills in the filtered training table.
        skills=".*",
    )
    # Predict later correctness within the held-out learner sequences.
    prediction_frame = model.predict(
        # Give pyBKT the observed sequence in each fold's evaluation input.
        data=evaluable_test[["user_id", "skill_name", "correct"]]
    )
    prediction_frame = prediction_frame.copy()
    # Read the actual held-out correctness labels.
    y_true = prediction_frame["correct"].to_numpy(dtype=int)
    # Read each probability forecast from the library output.
    y_prob = prediction_frame["correct_predictions"].to_numpy(dtype=float)
    # Discard rows with non-finite forecasts.
    valid = np.isfinite(y_prob)
    # Use the same rows for labels and bounded probabilities.
    y_true, y_prob = y_true[valid], np.clip(y_prob[valid], 0.0, 1.0)
    # Keep the prediction table aligned with the metric arrays.
    prediction_frame = prediction_frame.loc[valid].copy()
    # Report outcome metrics per skill as well as overall.
    per_skill_metrics = {
        str(skill): binary_metrics(
            rows["correct"].to_numpy(dtype=int),
            np.clip(rows["correct_predictions"].to_numpy(dtype=float), 0.0, 1.0),
        )
        for skill, rows in prediction_frame.groupby("skill_name", sort=True)
    }

    # Compute the base correctness rate from the training set alone.
    train_prevalence = float(train_df["correct"].mean())
    # Use that fixed rate as a simple prediction baseline.
    majority_prob = np.full(len(y_true), train_prevalence, dtype=float)
    # Assemble the audit, held-out metrics, settings and comparison scope.
    metrics = {
        "dataset": "ASSISTments 2009-2010 corrected skill-builder release",
        "target": "correct on first attempt, limited to original problems with first_action == 0",
        "split": "student-level holdout; responses within each held-out student's sequence remain chronological",
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "num_fits": args.num_fits,
        "forgetting_enabled": False,
        "fit_mode": "serial; parallel=False to avoid pyBKT's per-iteration process-pool overhead",
        "skill_regex": args.skill_regex or "(all skills)",
        "included_skills": sorted(train_skills),
        "audit": audit,
        "split_counts": {
            "train_rows": int(len(train_df)),
            "test_rows_before_unseen_skill_filter": int(len(test_df)),
            "test_rows_evaluated": int(len(y_true)),
            "test_rows_skipped_unseen_skill": unseen_skill_rows,
            "train_students": int(train_df["user_id"].nunique()),
            "test_students": int(test_df["user_id"].nunique()),
            "train_skills": int(len(train_skills)),
            "test_skills": int(len(test_skills)),
            "test_skill_coverage": round(
                len(test_skills & train_skills) / max(len(test_skills), 1), 6
            ),
        },
        "heldout_metrics": binary_metrics(y_true, y_prob),
        "heldout_metrics_by_skill": per_skill_metrics,
        "constant_train_prevalence_baseline": binary_metrics(y_true, majority_prob),
        "calibration_note": "Brier and ECE are reported without post-hoc calibration; ECE is bin-dependent.",
        "scope_limitations": [
            "This is correctness-only knowledge tracing, not misconception classification.",
            "The 2009-2010 skill labels are not mapped to this repository's fraction taxonomy.",
            "Held-out users come from the same historical source and do not establish transfer to the app's learners.",
            "Only original problems and attempt-first rows were retained; scaffolding and hint-first events were excluded.",
        ],
    }

    # Resolve the output folder for this offline experiment.
    output_dir = Path(args.output_dir).expanduser().resolve()
    # Create the folder if it does not yet exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the fitted model as an experiment artifact.
    model.save(str(output_dir / "model.pkl"))
    # Read the fitted skill parameters in table form.
    params = model.params().reset_index()
    # Save a readable parameter table beside the model.
    params.to_csv(output_dir / "parameters.csv", index=False)
    parameter_values = params["value"].to_numpy(dtype=float)
    estimated_params = params.loc[params["param"].ne("forgets"), "value"].to_numpy(dtype=float)
    # Summarize finite and boundary parameter values for interpretation.
    metrics["parameter_diagnostics"] = {
        "finite_parameter_count": int(np.isfinite(parameter_values).sum()),
        "total_parameter_count": int(len(parameter_values)),
        "estimated_parameter_boundary_count_excluding_fixed_forgetting": int(
            ((estimated_params <= 1e-8) | (estimated_params >= 1.0 - 1e-8)).sum()
        ),
        "fixed_forgetting_parameters_at_zero": int(
            ((params["param"] == "forgets") & (params["value"].abs() <= 1e-8)).sum()
        ),
    }
    # Record data identity and library versions for reproducible comparison.
    metrics["provenance"] = {
        "source_page": SOURCE_PAGE,
        "corrected_source_file": SOURCE_FILE,
        "data_file_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "data_file_name": data_path.name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            package: version(package)
            for package in ("pyBKT", "numpy", "pandas", "scikit-learn")
        },
    }
    # Write the complete experiment report for the training workspace.
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved model and report under {output_dir}")
    # Return the same metrics that were saved.
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Local corrected ASSISTments CSV")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--skill-regex",
        default="fraction|percent",
        help="Regex selecting skill names; default focuses on the fraction/percent prototype domain. Empty string selects all.",
    )
    parser.add_argument("--min-skill-rows", type=int, default=100)
    parser.add_argument("--min-skill-students", type=int, default=20)
    parser.add_argument("--num-fits", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    results = train(parse_args())
    print(json.dumps(results, indent=2))
