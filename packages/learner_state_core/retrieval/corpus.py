"""Trusted curriculum sources and explicitly separate Eedi research graph import."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..grounding.taxonomy import FRACTION_TAXONOMY

CURRICULUM_REVISION = "fractions-authored-v2"
TEACHING_NOTES = {
    "fraction_basics": "The denominator counts equal parts in one whole and sets the unit size. The numerator counts how many of those equal parts are taken. Ask the learner what the denominator means before inferring a persistent gap.",
    "fraction_equivalence": "Multiply or divide the numerator and denominator by the same nonzero factor to preserve the fraction's value. For example 1/2 = 2/4 = 3/6. Adding the same number does not preserve value: 1/2 is not 2/3.",
    "fraction_addition_like_denominators": "When adding fractions with the same denominator, add the numerators and keep the denominator: 1/4 + 2/4 = 3/4. The size of each equal part stays the same.",
    "fraction_addition_unlike_denominators": "To add fractions with different denominators, find equivalent fractions with a common denominator and add their numerators. For 1/2 + 1/3 use denominator 6: 3/6 + 2/6 = 5/6. For 1/4 + 1/2 use denominator 4: 1/4 + 2/4 = 3/4. Ask what equal unit size both fractions can share. A correct answer alone does not prove the learner's reasoning.",
}


def curriculum_documents() -> list[dict]:
    documents = []
    for cid, concept in sorted(FRACTION_TAXONOMY.items()):
        documents.append({
            "document_id": f"curriculum:{cid}", "title": concept.concept_name,
            "text": concept.description + " " + TEACHING_NOTES[cid],
            "source": f"repository:packages/learner_state_core/grounding/taxonomy.py#{cid}",
            "revision": CURRICULUM_REVISION, "license": "Repository-authored curriculum",
            "entities": [{"id": cid, "name": concept.concept_name, "kind": "concept"}],
            "edges": [{"source": cid, "target": prerequisite, "relation": "requires", "weight": 1.0} for prerequisite in concept.prerequisite_concept_ids],
        })
        for misconception in concept.known_misconceptions:
            documents.append({
                "document_id": f"curriculum:{cid}:{misconception.misconception_id}",
                "title": misconception.name,
                "text": f"Concept: {concept.concept_name}. {misconception.description} Typical pattern: {misconception.typical_distractor_pattern} Teaching guidance: {misconception.remediation_hint} This is a possible response interpretation, not a diagnosis of an individual learner.",
                "source": f"repository:packages/learner_state_core/grounding/taxonomy.py#{misconception.misconception_id}",
                "revision": CURRICULUM_REVISION, "license": "Repository-authored curriculum",
                "entities": [{"id": misconception.misconception_id, "name": misconception.name, "kind": "candidate_pattern"}],
                "edges": [{"source": misconception.misconception_id, "target": cid, "relation": "associated_with", "weight": 1.0}],
            })
    return documents


def eedi_documents(data_dir: Path, audit_path: Path) -> list[dict]:
    """Only admit exactly audited files. Eedi IDs never become app concept IDs."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for filename, expected in audit["file_sha256"].items():
        if hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Eedi checksum mismatch: {filename}")
    common = {"revision": audit["revision"], "license": audit["license"], "source": audit["source_url"], "attribution": audit["attribution"]}
    docs = []
    with (data_dir / "construct-map.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            eid = "eedi:construct:" + row["ConstructId"]
            topic = "eedi:topic:" + row["TopicSubjectId"]
            docs.append({**common, "document_id": eid, "title": row["ConstructName"], "text": f"Subject: {row['SubjectName']}. Topic: {row['TopicName']}. Subtopic: {row['SubTopicName']}. Construct: {row['ConstructName']}. Curriculum relation only; no learner response or reviewed app taxonomy mapping.", "entities": [{"id": eid, "name": row["ConstructName"], "kind": "external_construct"}, {"id": topic, "name": row["TopicName"], "kind": "external_topic"}], "edges": [{"source": eid, "target": topic, "relation": "within_topic"}]})
    grouped = {}
    with (data_dir / "misconception-map.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            eid = "eedi:misconception:" + row["MisconceptionId"]
            doc = grouped.setdefault(eid, {**common, "document_id": eid, "title": row["MisconceptionName"], "text": row["MisconceptionName"] + ". External curriculum pattern only; no individual learner evidence.", "entities": [{"id": eid, "name": row["MisconceptionName"], "kind": "external_misconception"}], "edges": []})
            doc["edges"].append({"source": eid, "target": "eedi:construct:" + row["ConstructId"], "relation": "associated_with", "weight": float(row["PercentageMisconceptionExistsInThisConstruct"]) / 100})
    return docs + list(grouped.values())
