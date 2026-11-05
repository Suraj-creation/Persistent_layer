"""Download and audit the public, version-pinned Eedi misconception graph.

This graph contains curriculum constructs and misconception-to-construct edges;
it contains no student answers and is not a response-classification dataset.
The raw files stay under the Git-ignored data/raw directory. The checked-in
audit contains only aggregate counts, source revision, and file checksums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEDI_REPO = "Eedi/Eedi-Misconceptions-Graph"
EEDI_REVISION = "2ae0839debf73c0ce7dd6578d8ea8b99e7be8631"
EEDI_LICENSE = "CC BY 4.0"
EEDI_ATTRIBUTION = (
    '"Eedi Misconceptions Graph v1.0" by Eedi, licensed under CC BY 4.0.'
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/raw/eedi_misconceptions_graph"
DEFAULT_AUDIT = PROJECT_ROOT / "experiments/eedi_graph_audit.json"
FILES = {
    "misconception-map.csv": {
        "path": "data/misconception-map.csv",
        "rows": 19294,
        "sha256": "3937e12b96ba2659ab16f060edfffeb6f35a1cce5fb60caedf6955a1b6fed0c2",
        "required_columns": {
            "MisconceptionId",
            "MisconceptionName",
            "PercentageMisconceptionExistsInThisConstruct",
            "ConstructId",
            "ConstructName",
            "SubTopicSubjectId",
            "SubTopicName",
            "TopicSubjectId",
            "TopicName",
            "SubjectSubjectId",
            "SubjectName",
        },
    },
    "construct-map.csv": {
        "path": "data/construct-map.csv",
        "rows": 2258,
        "sha256": "45a0fb21687f9b13cc59bb9aa011a39347637e304a8c239ee6d447b5bd77051b",
        "required_columns": {
            "SubjectName",
            "SubjectSubjectId",
            "TopicName",
            "TopicSubjectId",
            "SubTopicName",
            "SubTopicSubjectId",
            "ConstructName",
            "ConstructId",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pinned_file(relative_path: str, destination: Path) -> None:
    url = (
        f"https://huggingface.co/datasets/{EEDI_REPO}/resolve/"
        f"{EEDI_REVISION}/{relative_path}"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PersistentLayer-EediGraphIngest/1.0"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            while chunk := response.read(1024 * 1024):
                temp_file.write(chunk)
    os.replace(temp_path, destination)


def audit_graph(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate the v1.0 schema, keys, links, and edge-weight totals."""
    missing_files = set(FILES) - set(frames)
    if missing_files:
        raise ValueError(f"Missing graph tables: {sorted(missing_files)}")

    for filename, spec in FILES.items():
        frame = frames[filename]
        missing_columns = spec["required_columns"] - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{filename} is missing required columns: {sorted(missing_columns)}"
            )
        if len(frame) != spec["rows"]:
            raise ValueError(
                f"{filename} has {len(frame)} rows; pinned v1.0 expects {spec['rows']}"
            )

    edges = frames["misconception-map.csv"]
    constructs = frames["construct-map.csv"]
    for filename, frame, columns in (
        (
            "misconception-map.csv",
            edges,
            ["MisconceptionId", "MisconceptionName", "ConstructId", "ConstructName"],
        ),
        (
            "construct-map.csv",
            constructs,
            ["ConstructId", "ConstructName", "SubjectName", "TopicName", "SubTopicName"],
        ),
    ):
        if frame[columns].isna().any().any():
            raise ValueError(f"{filename} contains null identity or hierarchy values")

    if constructs["ConstructId"].duplicated().any():
        raise ValueError("construct-map.csv contains duplicate ConstructId values")
    if edges["MisconceptionId"].isna().any():
        raise ValueError("misconception-map.csv contains null MisconceptionId values")
    if not set(edges["ConstructId"]).issubset(set(constructs["ConstructId"])):
        raise ValueError("misconception-map.csv contains ConstructId values absent from construct-map.csv")

    names = constructs.set_index("ConstructId")["ConstructName"]
    edge_names = edges["ConstructId"].map(names)
    if not edge_names.reset_index(drop=True).equals(
        edges["ConstructName"].reset_index(drop=True)
    ):
        raise ValueError("ConstructName values do not match the construct catalogue")

    weights = pd.to_numeric(
        edges["PercentageMisconceptionExistsInThisConstruct"], errors="coerce"
    )
    if weights.isna().any() or ((weights < 0) | (weights > 100)).any():
        raise ValueError("Misconception-to-construct weights must be numeric percentages in [0, 100]")
    totals = weights.groupby(edges["MisconceptionId"]).sum()
    bad_totals = totals[(totals - 100.0).abs() > 0.1]
    if not bad_totals.empty:
        raise ValueError(
            f"{len(bad_totals)} misconception edge groups do not sum to 100 within 0.1"
        )

    return {
        "row_counts": {filename: int(len(frame)) for filename, frame in frames.items()},
        "unique_counts": {
            "misconceptions": int(edges["MisconceptionId"].nunique()),
            "constructs_in_edges": int(edges["ConstructId"].nunique()),
            "construct_catalogue": int(constructs["ConstructId"].nunique()),
            "subjects": int(constructs["SubjectName"].nunique()),
            "topics": int(constructs["TopicSubjectId"].nunique()),
            "subtopics": int(constructs["SubTopicSubjectId"].nunique()),
        },
        "weight_validation": {
            "misconception_groups_checked": int(len(totals)),
            "all_groups_sum_to_100_within_tolerance": True,
            "tolerance_percentage_points": 0.1,
        },
    }


def ingest(data_dir: Path, audit_path: Path, download: bool) -> dict[str, Any]:
    data_dir = data_dir.expanduser().resolve()
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for filename, spec in FILES.items():
        path = data_dir / filename
        if download and not path.exists():
            download_pinned_file(spec["path"], path)
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. Pass --download to fetch pinned Eedi revision {EEDI_REVISION}."
            )
        frames[filename] = pd.read_csv(path, low_memory=False)
        hashes[filename] = sha256_file(path)
        if hashes[filename] != spec["sha256"]:
            raise ValueError(
                f"{filename} checksum does not match pinned revision {EEDI_REVISION}"
            )

    audit = audit_graph(frames)
    audit.update(
        {
            "dataset": "Eedi Misconceptions Graph v1.0",
            "repo": EEDI_REPO,
            "revision": EEDI_REVISION,
            "license": EEDI_LICENSE,
            "attribution": EEDI_ATTRIBUTION,
            "source_url": f"https://huggingface.co/datasets/{EEDI_REPO}/tree/{EEDI_REVISION}",
            "file_sha256": hashes,
            "data_role": "curriculum grounding only; no learner responses or student-level labels",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    audit_path = audit_path.expanduser().resolve()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"Saved aggregate audit to {audit_path}")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest(args.data_dir, args.audit, download=args.download)
