"""Evaluate retrieval on a small, authored regression set, not learner outcomes."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packages.learner_state_core.retrieval import KnowledgeStore
from packages.learner_state_core.retrieval.corpus import curriculum_documents


def evaluate(store: KnowledgeStore, cases: list[dict]):
    results = {}
    for mode in ("naive", "local", "global", "hybrid", "mix"):
        rows = []
        for case in cases:
            start = time.perf_counter()
            retrieved = store.query(case["query"], mode=mode, top_k=3)
            elapsed_ms = 1000 * (time.perf_counter() - start)
            repeat = store.query(case["query"], mode=mode, top_k=3)
            ranked = [hit.document_id for hit in retrieved.hits]
            relevant = set(case["relevant"])
            first_rank = next((i for i, document in enumerate(ranked, 1) if document in relevant), None)
            rows.append({"query": case["query"], "relevant": sorted(relevant), "retrieved_document_ids": ranked, "recall_at_3": len(set(ranked) & relevant) / len(relevant) if relevant else None, "reciprocal_rank_at_3": 1 / first_rank if first_rank else (0 if relevant else None), "top_1_relevant": ranked[0] in relevant if ranked and relevant else (False if relevant else None), "expected_abstention": not relevant, "abstained": retrieved.abstained, "repeat_identical": retrieved.model_dump() == repeat.model_dump(), "citations_resolve": all(hit.citation_id in retrieved.context and hit.source for hit in retrieved.hits), "latency_ms": round(elapsed_ms, 3)})
        positive = [row for row in rows if not row["expected_abstention"]]
        negative = [row for row in rows if row["expected_abstention"]]
        results[mode] = {"summary": {"recall_at_3": statistics.mean(row["recall_at_3"] for row in positive), "mrr_at_3": statistics.mean(row["reciprocal_rank_at_3"] for row in positive), "top_1_accuracy": statistics.mean(row["top_1_relevant"] for row in positive), "out_of_scope_abstention_rate": statistics.mean(row["abstained"] for row in negative), "repeat_determinism_rate": statistics.mean(row["repeat_identical"] for row in rows), "citation_resolution_rate": statistics.mean(row["citations_resolve"] for row in rows), "median_query_latency_ms": statistics.median(row["latency_ms"] for row in rows)}, "queries": rows}
    return {"evaluation_scope": "12 repository-authored regression queries: 10 in-scope and 2 out-of-scope; no student labels, independent relevance review, or learning-impact claim.", "corpus": store.status(), "modes": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/retrieval_evaluation.json")
    args = parser.parse_args()
    cases_path = ROOT / "experiments/retrieval_cases.json"
    with tempfile.TemporaryDirectory() as directory:
        store = KnowledgeStore(str(Path(directory) / "evaluation.db"))
        store.sync(curriculum_documents())
        report = evaluate(store, json.loads(cases_path.read_text()))
    report["cases_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({mode: result["summary"] for mode, result in report["modes"].items()}, indent=2))
