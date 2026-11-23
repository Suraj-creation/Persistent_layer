"""Trusted curriculum sources and explicitly separate Eedi research graph import."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from ..grounding.taxonomy import FRACTION_TAXONOMY

# Version the authored fraction material so citations can identify its revision.
CURRICULUM_REVISION = "fractions-authored-v2"
# Keep the explanations and examples used to build the sandbox's reference documents.
TEACHING_NOTES = {
    "fraction_basics": "The denominator counts equal parts in one whole and sets the unit size. The numerator counts how many of those equal parts are taken. Ask the learner what the denominator means before inferring a persistent gap.",
    "fraction_equivalence": "Multiply or divide the numerator and denominator by the same nonzero factor to preserve the fraction's value. For example 1/2 = 2/4 = 3/6. Adding the same number does not preserve value: 1/2 is not 2/3.",
    "fraction_addition_like_denominators": "When adding fractions with the same denominator, add the numerators and keep the denominator: 1/4 + 2/4 = 3/4. The size of each equal part stays the same.",
    "fraction_addition_unlike_denominators": "To add fractions with different denominators, find equivalent fractions with a common denominator and add their numerators. For 1/2 + 1/3 use denominator 6: 3/6 + 2/6 = 5/6. For 1/4 + 1/2 use denominator 4: 1/4 + 2/4 = 3/4. Ask what equal unit size both fractions can share. A correct answer alone does not prove the learner's reasoning.",
}


# Turn the curated fraction curriculum into documents for the research index.
def curriculum_documents() -> list[dict]:
    # Collect both skill explanations and candidate misconception resources.
    documents = []
    # Walk the curated skills in a stable order.
    for cid, concept in sorted(FRACTION_TAXONOMY.items()):
        # Create one reference document for this curriculum skill.
        documents.append({
            # Use a curriculum-prefixed source ID and the concept's display name.
            "document_id": f"curriculum:{cid}", "title": concept.concept_name,
            # Combine the curriculum definition with the authored teaching note.
            "text": concept.description + " " + TEACHING_NOTES[cid],
            # Link the source to the corresponding repository taxonomy entry.
            "source": f"repository:packages/learner_state_core/grounding/taxonomy.py#{cid}",
            # Preserve the revision and authorship information.
            "revision": CURRICULUM_REVISION, "license": "Repository-authored curriculum",
            # Add the concept entity used by graph search.
            "entities": [{"id": cid, "name": concept.concept_name, "kind": "concept"}],
            # Connect this skill to each of its curated prerequisites.
            "edges": [{"source": cid, "target": prerequisite, "relation": "requires", "weight": 1.0} for prerequisite in concept.prerequisite_concept_ids],
        })
        # Index each known candidate misconception for this concept.
        for misconception in concept.known_misconceptions:
            # Create a related document that describes the possible reasoning pattern.
            documents.append({
                # Give the document its own ID within the curriculum collection.
                "document_id": f"curriculum:{cid}:{misconception.misconception_id}",
                # Use the misconception's name as the title.
                "title": misconception.name,
                # Include the pattern, a typical response and correction guidance.
                "text": f"Concept: {concept.concept_name}. {misconception.description} Typical pattern: {misconception.typical_distractor_pattern} Teaching guidance: {misconception.remediation_hint} This is a possible response interpretation, not a diagnosis of an individual learner.",
                # Point to the original taxonomy entry for this pattern.
                "source": f"repository:packages/learner_state_core/grounding/taxonomy.py#{misconception.misconception_id}",
                # Record that the material is repository-authored.
                "revision": CURRICULUM_REVISION, "license": "Repository-authored curriculum",
                # Add a corresponding graph entity for the candidate pattern.
                "entities": [{"id": misconception.misconception_id, "name": misconception.name, "kind": "candidate_pattern"}],
                # Connect the pattern to the affected skill.
                "edges": [{"source": misconception.misconception_id, "target": cid, "relation": "associated_with", "weight": 1.0}],
            })
    # Return the complete authored collection for synchronization.
    return documents


# Build external Eedi graph documents only from the audited data files.
def eedi_documents(data_dir: Path, audit_path: Path) -> list[dict]:
    """Only admit exactly audited files. Eedi IDs never become app concept IDs."""
    # Read revision, source and fingerprint information from the audit.
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    # Recheck every audited input file before turning it into search material.
    for filename, expected in audit["file_sha256"].items():
        # Compare the actual SHA-256 digest with the one the audit recorded.
        if hashlib.sha256((data_dir / filename).read_bytes()).hexdigest() != expected:
            # Refuse an index input that changed since the audit.
            raise ValueError(f"Eedi checksum mismatch: {filename}")
    # Carry the source URL, license, revision and attribution onto all imported documents.
    common = {"revision": audit["revision"], "license": audit["license"], "source": audit["source_url"], "attribution": audit["attribution"]}
    # Collect the external curriculum construct documents.
    docs = []
    # Open the construct table with support for a UTF-8 byte-order marker.
    with (data_dir / "construct-map.csv").open(encoding="utf-8-sig", newline="") as source:
        # Read each construct row by its column names.
        for row in csv.DictReader(source):
            # Give its construct a distinct Eedi-prefixed graph ID.
            eid = "eedi:construct:" + row["ConstructId"]
            # Give its topic a distinct external ID as well.
            topic = "eedi:topic:" + row["TopicSubjectId"]
            # Create a construct document and graph connection to its topic.
            docs.append({**common, "document_id": eid, "title": row["ConstructName"], "text": f"Subject: {row['SubjectName']}. Topic: {row['TopicName']}. Subtopic: {row['SubTopicName']}. Construct: {row['ConstructName']}. Curriculum relation only; no learner response or reviewed app taxonomy mapping.", "entities": [{"id": eid, "name": row["ConstructName"], "kind": "external_construct"}, {"id": topic, "name": row["TopicName"], "kind": "external_topic"}], "edges": [{"source": eid, "target": topic, "relation": "within_topic"}]})
    # Group rows for one misconception under a single external document.
    grouped = {}
    # Open the audited misconception-to-construct mapping table.
    with (data_dir / "misconception-map.csv").open(encoding="utf-8-sig", newline="") as source:
        # Process each relationship row.
        for row in csv.DictReader(source):
            # Prefix the external misconception ID so it cannot be confused with a live concept.
            eid = "eedi:misconception:" + row["MisconceptionId"]
            # Create a reusable source document for the pattern the first time it appears.
            doc = grouped.setdefault(eid, {**common, "document_id": eid, "title": row["MisconceptionName"], "text": row["MisconceptionName"] + ". External curriculum pattern only; no individual learner evidence.", "entities": [{"id": eid, "name": row["MisconceptionName"], "kind": "external_misconception"}], "edges": []})
            # Link that pattern to this construct and scale the documented percentage to zero
            # through one.
            doc["edges"].append({"source": eid, "target": "eedi:construct:" + row["ConstructId"], "relation": "associated_with", "weight": float(row["PercentageMisconceptionExistsInThisConstruct"]) / 100})
    # Return both source groups for the separate Eedi research index.
    return docs + list(grouped.values())
