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

CATEGORIES = ['understanding', 'partial', 'misconception', 'prerequisite_gap', 'slip', 'ambiguous']


class GoldAnnotation(BaseModel):
    turn_id: str
    category: Literal['understanding', 'partial', 'misconception', 'prerequisite_gap', 'slip', 'ambiguous']
    concept_id: str
    misconception_present: bool | None = None
    claim_faithful: bool | None = None


class AnnotationDataset(BaseModel):
    dataset: str = Field(min_length=3, max_length=200)
    annotation_source: str = Field(min_length=3, max_length=500)
    annotations: list[GoldAnnotation] = Field(min_length=1, max_length=2000)


def install_schema(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS assessment_evaluations(run_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, payload_json TEXT NOT NULL)')
    conn.execute('CREATE INDEX IF NOT EXISTS assessment_evaluations_owner ON assessment_evaluations(learner_id)')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate_annotations(db, learner_id, payload):
    data = AnnotationDataset.model_validate(payload).model_dump()
    ids = [a['turn_id'] for a in data['annotations']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_annotation')
    rows, store = [], ExecutionStore(db)
    for annotation in data['annotations']:
        record = store.record(learner_id, annotation['turn_id'])
        if record is None:
            raise ValueError('record_not_found')
        rows.append({'annotation': annotation, 'record_sha256': digest(record),
                     'prediction': {'category': record['evidence']['category'], 'concept_id': record['evidence']['concept_id']},
                     'extractor_model': record['evidence']['extractor_model'], 'trace_id': record['extraction_trace'].get('trace_id')})
    # Macro-F1 includes classes present in either the gold or predicted labels.
    classes = sorted({r['annotation']['category'] for r in rows} | {r['prediction']['category'] for r in rows})
    per_class = []
    for label in classes:
        tp = sum(r['annotation']['category'] == label and r['prediction']['category'] == label for r in rows)
        fp = sum(r['annotation']['category'] != label and r['prediction']['category'] == label for r in rows)
        fn = sum(r['annotation']['category'] == label and r['prediction']['category'] != label for r in rows)
        precision, recall = tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0
        per_class.append({'category': label, 'precision': precision, 'recall': recall, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0})
    binary = [r for r in rows if r['annotation']['misconception_present'] is not None]
    positive = sum(r['annotation']['misconception_present'] for r in binary)
    tp = sum(r['annotation']['misconception_present'] and r['prediction']['category'] == 'misconception' for r in binary)
    fp = sum(not r['annotation']['misconception_present'] and r['prediction']['category'] == 'misconception' for r in binary)
    fn = sum(r['annotation']['misconception_present'] and r['prediction']['category'] != 'misconception' for r in binary)
    faithful = [r['annotation']['claim_faithful'] for r in rows if r['annotation']['claim_faithful'] is not None]
    metrics = {'category_accuracy': sum(r['annotation']['category'] == r['prediction']['category'] for r in rows) / len(rows),
               'concept_accuracy': sum(r['annotation']['concept_id'] == r['prediction']['concept_id'] for r in rows) / len(rows),
               'category_macro_precision': sum(c['precision'] for c in per_class) / len(classes),
               'category_macro_recall': sum(c['recall'] for c in per_class) / len(classes),
               'category_macro_f1': sum(c['f1'] for c in per_class) / len(classes),
               'misconception_precision': tp / (tp + fp) if positive and tp + fp else 0.0 if positive else None,
               'misconception_recall': tp / (tp + fn) if positive else None,
               'misconception_f1': 2 * tp / (2 * tp + fp + fn) if positive else None,
               'expert_claim_faithfulness': sum(faithful) / len(faithful) if faithful else None}
    result = {'run_id': new_id('evaluation'), 'created_at': now(), 'dataset': data['dataset'], 'annotation_source': data['annotation_source'],
              'task': 'current_response_diagnosis', 'method': 'saved_v2_evidence_vs_supplied_annotations_v1', 'sample_size': len(rows),
              'annotations_sha256': digest(data), 'record_set_sha256': digest(rows), 'models': sorted({r['extractor_model'] for r in rows}),
              'metrics': metrics, 'metric_support': {key: len(faithful) if key == 'expert_claim_faithfulness' else len(binary) if key.startswith('misconception_') else len(rows) for key in metrics},
              'per_class': per_class, 'baselines': [], 'records': rows,
              'limitations': ['Annotation provenance is supplied by the evaluator and is not independently verified.',
                              'Concept association is constrained to the saved question concept, not open-domain concept discovery.',
                              'Misconception precision/recall/F1 refer to binary presence, not misconception-identity classification.',
                              'No positive gold misconception labels means misconception F1 is not evaluated.',
                              'This evaluates current-response diagnosis, not future-answer prediction, calibration, learning gains or H1–H6 baseline comparisons.']}
    with db.transaction() as conn:
        conn.execute('INSERT INTO assessment_evaluations VALUES(?,?,?)', (result['run_id'], learner_id, json.dumps(result)))
    return result


def runs(db, learner_id, limit=100):
    with db.connection() as conn:
        rows = conn.execute('SELECT payload_json FROM assessment_evaluations WHERE learner_id=? ORDER BY rowid DESC LIMIT ?', (learner_id, limit)).fetchall()
    return [json.loads(r[0]) for r in rows]


def get_run(db, learner_id, run_id):
    with db.connection() as conn:
        row = conn.execute('SELECT payload_json FROM assessment_evaluations WHERE learner_id=? AND run_id=?', (learner_id, run_id)).fetchone()
    return json.loads(row[0]) if row else None
