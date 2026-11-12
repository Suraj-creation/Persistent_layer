"""Build a safe, aggregate-only view of the checked-in research artifacts.

The web API deliberately reports metadata and fitted parameter tables without
loading serialized model objects or exposing learner-level rows/predictions.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BKT_ARTIFACT_DIR = Path("models/assistments_2009_2010_bkt")


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _read_json(path: Path, warnings: list[str], label: str) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        warnings.append(f"{label} metadata unavailable: {exc}")
        return None


def _safe_repo_path(root: Path, relative_path: str) -> Path | None:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _artifact_metadata(
    root: Path,
    relative_path: str,
    expected_sha256: str | None,
    artifact_type: str,
) -> dict[str, Any]:
    path = _safe_repo_path(root, relative_path)
    actual_sha256 = _sha256(path) if path is not None and path.is_file() else None
    return {
        "name": Path(relative_path).name,
        "path": relative_path,
        "type": artifact_type,
        "available": actual_sha256 is not None,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "checksum_verified": bool(expected_sha256 and actual_sha256 == expected_sha256),
    }


def _load_skill_parameters(root: Path, seed_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_skill: dict[str, dict[str, Any]] = {}
    for seed_result in seed_results:
        seed = str(seed_result.get("seed"))
        artifact_paths = _mapping(seed_result.get("bkt_artifacts"))
        relative_path = artifact_paths.get("parameters")
        if not isinstance(relative_path, str):
            continue
        path = _safe_repo_path(root, relative_path)
        if path is None or not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as source:
                for row in csv.DictReader(source):
                    skill = row.get("skill")
                    parameter = row.get("param")
                    if not skill or not parameter:
                        continue
                    try:
                        value = float(row["value"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    by_skill.setdefault(skill, {}).setdefault(seed, {})[parameter] = value
        except OSError:
            continue
    return [
        {"skill": skill, "by_seed": by_skill[skill]}
        for skill in sorted(by_skill)
    ]


def build_training_research_summary(repository_root: Path | None = None) -> dict[str, Any]:
    """Return aggregate training, audit, and artifact metadata for Researcher Studio."""
    root = (repository_root or REPOSITORY_ROOT).resolve()
    warnings: list[str] = []
    model_dir = root / BKT_ARTIFACT_DIR
    metrics = _read_json(model_dir / "metrics.json", warnings, "BKT run")
    robustness = _read_json(model_dir / "robustness.json", warnings, "BKT robustness")
    graph = _read_json(root / "experiments/eedi_graph_audit.json", warnings, "Eedi graph audit")

    metrics_data = _mapping(metrics)
    robustness_data = _mapping(robustness)
    graph_data = _mapping(graph)
    raw_seed_results = robustness_data.get("seed_results", [])
    seed_results = [result for result in raw_seed_results if isinstance(result, dict)] if isinstance(raw_seed_results, list) else []
    valid_metrics = bool(metrics) and isinstance(metrics_data.get("heldout_metrics"), dict) and isinstance(metrics_data.get("audit"), dict)
    valid_robustness = bool(robustness) and isinstance(raw_seed_results, list) and isinstance(robustness_data.get("summary"), dict)
    valid_graph = bool(graph) and isinstance(graph_data.get("row_counts"), dict) and isinstance(graph_data.get("unique_counts"), dict)
    if metrics and not valid_metrics:
        warnings.append("BKT run metadata is missing required evaluation fields.")
    if robustness and not valid_robustness:
        warnings.append("BKT robustness metadata is missing required split or summary fields.")
    if graph and not valid_graph:
        warnings.append("Eedi graph audit metadata is missing required aggregate counts.")
    seeds: list[dict[str, Any]] = []
    for result in seed_results:
        artifact_refs = _mapping(result.get("bkt_artifacts"))
        seed_artifacts = []
        for key, artifact_type in (("model", "serialized_model"), ("parameters", "parameter_csv")):
            relative_path = artifact_refs.get(key)
            if isinstance(relative_path, str):
                checksum_key = "parameter_sha256" if key == "parameters" else "model_sha256"
                seed_artifacts.append(
                    _artifact_metadata(root, relative_path, artifact_refs.get(checksum_key), artifact_type)
                )
        split_counts = _mapping(result.get("split_counts"))
        seeds.append({
            "seed": result.get("seed"),
            "train_rows": split_counts.get("train_rows"),
            "test_rows": split_counts.get("test_rows"),
            "train_students": split_counts.get("train_students"),
            "test_students": split_counts.get("test_students"),
            "models": _mapping(result.get("models")),
            "artifacts": seed_artifacts,
        })

    model_summary = []
    summary_models = _mapping(robustness_data.get("summary"))
    for model_name, metric_values in summary_models.items():
        if isinstance(metric_values, dict):
            model_summary.append({"name": model_name, "metrics": metric_values})

    data_audit = _mapping(metrics_data.get("audit"))
    primary_split = _mapping(metrics_data.get("split_counts"))
    heldout_metrics = _mapping(metrics_data.get("heldout_metrics"))
    model_artifacts = []
    if metrics:
        model_artifacts.extend([
            _artifact_metadata(
                root,
                str(BKT_ARTIFACT_DIR / "model.pkl"),
                next((_mapping(a.get("bkt_artifacts")).get("model_sha256") for a in seed_results if a.get("seed") == 42), None),
                "serialized_model",
            ),
            _artifact_metadata(
                root,
                str(BKT_ARTIFACT_DIR / "metrics.json"),
                None,
                "evaluation_metrics",
            ),
            _artifact_metadata(
                root,
                str(BKT_ARTIFACT_DIR / "robustness.json"),
                None,
                "robustness_metrics",
            ),
        ])

    return {
        "status": "available" if valid_metrics and valid_robustness and valid_graph else "partial",
        "purpose": (
            "Research artifacts for a fraction-focused prototype that records learner language as "
            "auditable, concept-linked evidence and explores cautious learner-state updates."
        ),
        "runtime_boundary": {
            "bkt_used_for_live_learner_state": False,
            "map_classifier_trained_or_used": False,
            "eedi_graph_used_for_runtime_retrieval": False,
            "live_state_method": "Rule-based or optional Azure-backed evidence extraction followed by heuristic state updates.",
            "notice": "These research artifacts are not connected to live prediction or learner-state updates.",
        },
        "assistments_bkt": {
            "status": "evaluated" if valid_robustness else "unavailable",
            "dataset": robustness_data.get("dataset") or metrics_data.get("dataset"),
            "source_url": robustness_data.get("source_page"),
            "task": "Predict later first-attempt correctness from an observed skill sequence for held-out learners.",
            "model": "pyBKT non-forgetting Bayesian Knowledge Tracing; serial fit; one initialization per split.",
            "target": "correct on first attempt for original problems where first_action == 0",
            "skill_filter": robustness_data.get("skill_regex"),
            "split_design": robustness_data.get("split_design"),
            "test_ratio": robustness_data.get("test_ratio"),
            "data_audit": data_audit,
            "primary_split": primary_split,
            "primary_split_metrics": heldout_metrics,
            "summary_across_splits": model_summary,
            "paired_bkt_minus_baselines": _mapping(robustness_data.get("paired_bkt_minus_baseline_mean")),
            "seeds": seeds,
            "skill_metrics": _mapping(metrics_data.get("heldout_metrics_by_skill")),
            "skill_parameters": _load_skill_parameters(root, seed_results),
            "artifacts": model_artifacts,
            "data_sha256": _mapping(robustness_data.get("provenance")).get("dataset_file_sha256")
            or _mapping(metrics_data.get("provenance")).get("data_file_sha256"),
            "trained_at_utc": _mapping(metrics_data.get("provenance")).get("trained_at_utc"),
            "evaluated_at_utc": _mapping(robustness_data.get("provenance")).get("evaluated_at_utc"),
            "versions": _mapping(robustness_data.get("provenance")).get("versions", {}),
            "limitations": robustness_data.get("interpretation_limits", [])
            if isinstance(robustness_data.get("interpretation_limits", []), list) else [],
            "calibration_note": metrics_data.get("calibration_note"),
        },
        "eedi_graph": {
            "status": "audited" if valid_graph else "unavailable",
            "dataset": graph_data.get("dataset"),
            "revision": graph_data.get("revision"),
            "license": graph_data.get("license"),
            "attribution": graph_data.get("attribution"),
            "source_url": graph_data.get("source_url"),
            "row_counts": _mapping(graph_data.get("row_counts")),
            "unique_counts": _mapping(graph_data.get("unique_counts")),
            "weight_validation": _mapping(graph_data.get("weight_validation")),
            "file_sha256": _mapping(graph_data.get("file_sha256")),
            "data_role": graph_data.get("data_role"),
            "runtime_integrated": False,
        },
        "map_classifier": {
            "status": "prepared_not_trained",
            "model_available": False,
            "reason": "The authorized MAP train.csv is not present; competition terms must be accepted before access, and use is restricted to non-commercial purposes.",
            "method": "Word and character TF-IDF features with class-balanced logistic regression.",
            "planned_validation": "Five-fold grouped out-of-fold evaluation; identical normalized inputs stay in one fold.",
            "planned_metrics": ["MAP@3", "top-1 accuracy", "macro-F1", "per-label precision/recall/F1", "label coverage", "multiclass Brier", "top-label ECE"],
            "runtime_integrated": False,
        },
        "other_dataset_status": [
            {
                "dataset": "FoundationalASSIST",
                "status": "researched_not_downloaded_or_trained",
                "role": "Potential separate longitudinal correctness task after access review.",
                "limitation": "No verified misconception labels; short or numeric answers are not reasoning explanations.",
            },
            {
                "dataset": "Synthetic fraction fixture",
                "status": "used_in_existing_demo_only",
                "role": "Exercises the existing benchmark and presentation paths.",
                "limitation": "Generated wording and labels are coupled; scores are not empirical training results.",
            },
        ],
        "related_reports": [
            "TRAINING_EXPERIMENT_REPORT.md",
            "DATA_AND_TRAINING_PLAN.md",
            "experiments/assistments_bkt_report.md",
            "experiments/benchmark_report.md",
        ],
        "warnings": warnings,
    }
