"""Reproducible evaluation of saved responses against supplied annotations.

No model inference, invented labels, or future-prediction claims occur here.
"""
import hashlib
import json
from typing import Literal
from pydantic import BaseModel, Field
from .execution import ExecutionStore
from .models import new_id
from .storage import now

# List the categories accepted in the annotation dataset.
CATEGORIES = ['understanding', 'partial', 'misconception', 'prerequisite_gap', 'slip', 'ambiguous']


# Define an externally supplied judgment for one accepted turn.
class GoldAnnotation(BaseModel):
    # Link the supplied annotation to the saved learner answer.
    turn_id: str
    # Record the expert's category judgment.
    category: Literal['understanding', 'partial', 'misconception', 'prerequisite_gap', 'slip', 'ambiguous']
    # Record the annotated concept.
    concept_id: str
    # Allow a separate yes or no judgment about the presence of a misconception.
    misconception_present: bool | None = None
    # Allow a separate judgment of whether the recorded claim is faithful to the response.
    claim_faithful: bool | None = None


# Define the required labels and provenance for a scoring run.
class AnnotationDataset(BaseModel):
    # Name the dataset or response collection.
    dataset: str = Field(min_length=3, max_length=200)
    # Preserve who or what supplied the annotation judgments.
    annotation_source: str = Field(min_length=3, max_length=500)
    # Require a bounded nonempty set of labels.
    annotations: list[GoldAnnotation] = Field(min_length=1, max_length=2000)


# Install persistent storage for evaluation output.
def install_schema(conn):
    # Save entire evaluation records separately from learner state.
    conn.execute('CREATE TABLE IF NOT EXISTS assessment_evaluations(run_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, payload_json TEXT NOT NULL)')
    # Index the learner for scoped evaluation lists.
    conn.execute('CREATE INDEX IF NOT EXISTS assessment_evaluations_owner ON assessment_evaluations(learner_id)')


# Create a reproducible digest of structured evaluation input.
def digest(value):
    # Sort keys and retain the original Unicode text in the hash input.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


# Compare saved current-response interpretations with supplied annotations.
def evaluate_annotations(db, learner_id, payload):
    # Parse and validate the annotation dataset.
    data = AnnotationDataset.model_validate(payload).model_dump()
    # Collect the saved turn IDs to be evaluated.
    ids = [a['turn_id'] for a in data['annotations']]
    # Ensure that one turn cannot be counted twice for one run.
    if len(ids) != len(set(ids)):
        # Reject duplicated turn labels.
        raise ValueError('duplicate_annotation')
    # Prepare examples and a store for retrieving accepted answers.
    rows, store = [], ExecutionStore(db)
    # Walk the supplied expert labels.
    for annotation in data['annotations']:
        # Load the accepted turn only within the corresponding learner scope.
        record = store.record(learner_id, annotation['turn_id'])
        # Require it to exist before measuring agreement.
        if record is None:
            # Reject a missing or out-of-scope record.
            raise ValueError('record_not_found')
        # Store the annotation and a hash of the exact saved turn used.
        rows.append({'annotation': annotation, 'record_sha256': digest(record),
                     # Compare the label with the original assessment's saved category and concept.
                     'prediction': {'category': record['evidence']['category'], 'concept_id': record['evidence']['concept_id']},
                     # Preserve the model and trace that produced the stored claim.
                     'extractor_model': record['evidence']['extractor_model'], 'trace_id': record['extraction_trace'].get('trace_id')})
    # Macro-F1 includes classes present in either the gold or predicted labels.
    # Compute category metrics over classes present in the known or predicted labels.
    classes = sorted({r['annotation']['category'] for r in rows} | {r['prediction']['category'] for r in rows})
    # Collect results for each observed class.
    per_class = []
    # Measure classification performance for one class at a time.
    for label in classes:
        # Count saved predictions that correctly match this label.
        tp = sum(r['annotation']['category'] == label and r['prediction']['category'] == label for r in rows)
        # Count predictions of this label where the annotation disagrees.
        fp = sum(r['annotation']['category'] != label and r['prediction']['category'] == label for r in rows)
        # Count true examples of this label missed by the saved prediction.
        fn = sum(r['annotation']['category'] == label and r['prediction']['category'] != label for r in rows)
        # Derive precision and recall, using zero when there is no relevant denominator.
        precision, recall = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
        # Also calculate the class F1 score from the same counts.
        per_class.append({'category': label, 'precision': precision, 'recall': recall, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
    # Limit binary misconception scoring to annotations with a supplied yes or no label.
    binary = [r for r in rows if r['annotation']['misconception_present'] is not None]
    # Count the actual positive labels and use that support for metric availability.
    positive = sum(r['annotation']['misconception_present'] for r in binary)
    # Count correctly identified misconceptions.
    tp = sum(r['annotation']['misconception_present'] and r['prediction']['category'] == 'misconception' for r in binary)
    # Count predictions of misconceptions on examples labeled absent.
    fp = sum(not r['annotation']['misconception_present'] and r['prediction']['category'] == 'misconception' for r in binary)
    # Count annotated misconceptions that extraction missed.
    fn = sum(r['annotation']['misconception_present'] and r['prediction']['category'] != 'misconception' for r in binary)
    # Collect only the expert judgments that explicitly addressed claim faithfulness.
    faithful = [r['annotation']['claim_faithful'] for r in rows if r['annotation']['claim_faithful'] is not None]
    # Calculate category agreement across the saved responses.
    metrics = {'category_accuracy': sum(r['annotation']['category'] == r['prediction']['category'] for r in rows) / len(rows),
               # Calculate agreement on the saved concept links.
               'concept_accuracy': sum(r['annotation']['concept_id'] == r['prediction']['concept_id'] for r in rows) / len(rows),
               # Give each observed category equal weight in the macro precision.
               'category_macro_precision': sum(c['precision'] for c in per_class) / len(classes),
               # Give each category equal weight in the macro recall.
               'category_macro_recall': sum(c['recall'] for c in per_class) / len(classes),
               # Give each category equal weight in the macro F1.
               'category_macro_f1': sum(c['f1'] for c in per_class) / len(classes),
               # Calculate misconception precision when positive support exists.
               'misconception_precision': tp / (tp + fp) if positive and tp + fp else 0.0 if positive else None,
               # Calculate misconception recall from the supplied positives.
               'misconception_recall': tp / (tp + fn) if positive else None,
               # Calculate binary misconception F1 with the measured confusion counts.
               'misconception_f1': 2 * tp / (2 * tp + fp + fn) if positive else None,
               # Calculate claim faithfulness from the subset with supplied judgments.
               'expert_claim_faithfulness': sum(faithful) / len(faithful) if faithful else None}
    # Give this run its own identity, timestamp and annotation provenance.
    result = {'run_id': new_id('evaluation'), 'created_at': now(), 'dataset': data['dataset'], 'annotation_source': data['annotation_source'],
              # Define the evaluated task as diagnosis of the current explanation.
              'task': 'current_response_diagnosis', 'method': 'saved_v2_evidence_vs_supplied_annotations_v1', 'sample_size': len(rows),
              # Store hashes of the labels and evaluated records and identify the extractor models.
              'annotations_sha256': digest(data), 'record_set_sha256': digest(rows), 'models': sorted({r['extractor_model'] for r in rows}),
              # Record the eligible sample size for every metric alongside its value.
              'metrics': metrics, 'metric_support': {key: len(faithful) if key == 'expert_claim_faithfulness' else len(binary) if key.startswith('misconception_') else len(rows) for key in metrics},
              # Include class metrics and the evaluated pairs so the run is inspectable.
              'per_class': per_class, 'baselines': [], 'records': rows,
              # Carry the scope and limitations with the saved results.
              'limitations': ['Annotation provenance is supplied by the evaluator and is not independently verified.',
                              'Concept association is constrained to the saved question concept, not open-domain concept discovery.',
                              'Misconception precision/recall/F1 refer to binary presence, not misconception-identity classification.',
                              'No positive gold misconception labels means misconception F1 is not evaluated.',
                              'This evaluates current-response diagnosis, not future-answer prediction, calibration, learning gains or H1–H6 baseline comparisons.']}
    # Open a transaction to retain the evaluation for later scorecard use.
    with db.transaction() as conn:
        # Insert the complete measurement record under the learner ID.
        conn.execute('INSERT INTO assessment_evaluations VALUES(?,?,?)', (result['run_id'], learner_id, json.dumps(result)))
    # Return the exact run payload that was persisted.
    return result


# List recent evaluation runs for a local learner.
def runs(db, learner_id, limit=100):
    # Open a connection for the scoped run list.
    with db.connection() as conn:
        # Read the latest matching run records up to the requested limit.
        rows = conn.execute('SELECT payload_json FROM assessment_evaluations WHERE learner_id=? ORDER BY rowid DESC LIMIT ?', (learner_id, limit)).fetchall()
    # Decode their saved results for the API.
    return [json.loads(r[0]) for r in rows]


# Look up one evaluation by learner and run ID.
def get_run(db, learner_id, run_id):
    # Open the connection for the exact lookup.
    with db.connection() as conn:
        # Scope the query by both identity fields.
        row = conn.execute('SELECT payload_json FROM assessment_evaluations WHERE learner_id=? AND run_id=?', (learner_id, run_id)).fetchone()
    # Return the decoded result if the learner has that run.
    return json.loads(row[0]) if row else None
