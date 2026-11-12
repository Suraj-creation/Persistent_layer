"""Index curated curriculum and optionally the checksum-verified Eedi graph."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.learner_state_core.retrieval import KnowledgeStore
from packages.learner_state_core.retrieval.corpus import curriculum_documents, eedi_documents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=ROOT / "data/learner_state.db")
    parser.add_argument("--include-eedi", action="store_true")
    parser.add_argument("--audit-output", type=Path, default=ROOT / "experiments/retrieval_index_audit.json")
    args = parser.parse_args()
    store = KnowledgeStore(str(args.database))
    reports = [store.sync(curriculum_documents())]
    if args.include_eedi:
        reports.append(store.sync(eedi_documents(ROOT / "data/raw/eedi_misconceptions_graph", ROOT / "experiments/eedi_graph_audit.json"), "eedi_research"))
    audit = {"stores": reports, "learner_data_indexed": False, "eedi_role": "External curriculum research only; no app concept crosswalk or learner evidence", "eedi_attribution": '"Eedi Misconceptions Graph v1.0" by Eedi, licensed under CC BY 4.0.', "eedi_revision": "2ae0839debf73c0ce7dd6578d8ea8b99e7be8631"}
    args.audit_output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
