"""Operational measurements and explicit research gaps; no generated research scores."""
import hashlib
import json
from pathlib import Path
from .storage import now
from .evaluation import runs, get_run


def research_status(db, learner_id, run_id=None):
    with db.connection() as conn:
        rows = conn.execute('SELECT a.payload_json FROM assessment_answers a JOIN assessment_conversations c ON c.conversation_id=a.conversation_id WHERE c.learner_id=?', (learner_id,)).fetchall()
        snapshot_ids = {r[0] for r in conn.execute('SELECT snapshot_id FROM assessment_state_history WHERE learner_id=?', (learner_id,))}
    turns = [json.loads(r[0]) for r in rows]
    quote_matches = sum(t['response_text'][t['evidence']['source_start']:t['evidence']['source_end']] == t['evidence']['source_quote'] for t in turns)
    state_links = sum(t['state_after']['snapshot_id'] in snapshot_ids and t['evidence']['evidence_id'] in t['state_after']['concepts'][t['evidence']['concept_id']]['evidence_ids'] for t in turns)
    metrics = [('Prediction', 'Next-answer AUC', 'Requires held-out future responses and comparable baselines.'),
               ('Prediction', 'Next-answer log loss', 'Requires held-out future responses.'),
               ('Diagnosis', 'Misconception precision / recall / F1', 'Insufficient expert-labelled v2 responses.'),
               ('Diagnosis', 'Concept identification / expert agreement', 'Requires independent expert concept labels.'),
               ('Evidence', 'Semantic faithfulness', 'Exact quote matching does not establish semantic faithfulness.'),
               ('State', 'Brier score / ECE', 'Heuristic scores have not been calibrated against longitudinal labels.'),
               ('Adaptation', 'Information gain / questions to resolution', 'Not evaluated; current planner uses heuristics.'),
               ('Learning outcomes', 'Retention / transfer gain', 'Requires a suitable learner study and control group.')]
    scorecard = [dict(dimension=d, metric=m, baseline=None, text_baseline=None, proposed=None, improvement=None, status='Not evaluated', evidence=e) for d, m, e in metrics]
    for label, numerator in [('Verbatim quote integrity', quote_matches), ('Persisted state/evidence linkage', state_links)]:
        scorecard.append(dict(dimension='Operations', metric=label, baseline=None, text_baseline=None,
                              proposed=numerator / len(turns) if turns else None, improvement=None,
                              status='Measured telemetry' if turns else 'No submitted evidence',
                              evidence=f'{numerator}/{len(turns)} saved responses; deterministic integrity check, not research validation.'))
    descriptions = [
        ('H1', 'Language adds evidence beyond right/wrong marks.', 'Evidence extraction', 'Correctness-only KT', 'Held-out responses with correctness and expert diagnosis labels', 'AUC/log loss for future prediction; diagnosis F1 separately'),
        ('H2', 'Structured evidence beats raw text alone.', 'Structured evidence and deterministic integration', 'TF-IDF/raw text classifier', 'Identical labelled splits and matched task horizons', 'Diagnosis precision/recall/F1 and future-answer AUC'),
        ('H3', 'Concept grounding reduces misreadings.', 'Assigned-concept constrained grounding', 'Extraction with grounding removed', 'Expert concept labels and ambiguity annotations', 'Concept accuracy; extraction error analysis'),
        ('H4', 'Keeping history beats one-answer-at-a-time approaches.', 'Persistent state and source retrieval', 'Stateless extraction', 'Time-ordered multi-interaction learner trajectories', 'Future-answer log loss; contradiction and update consistency'),
        ('H5', 'Showing uncertainty reduces overconfident diagnoses.', 'Abstention, confidence display, ambiguous-state holds', 'Same pipeline without uncertainty/abstention', 'Expert labels and confidence judgments', 'Brier score/ECE; overdiagnosis and abstention coverage'),
        ('H6', 'Smart question choice needs fewer questions.', 'Heuristic adaptive planner; information gain not implemented', 'Random, fixed-order and prerequisite selection', 'Validated preceding stages and controlled question-selection study', 'Questions to resolve uncertainty; measured information gain'),
    ]
    hypotheses = [dict(id=i, hypothesis=h, capability=c, baseline=b, required_data=data, evaluation=m,
                       status='Implemented capability; research test pending', tested=False, findings=None) for i, h, c, b, data, m in descriptions]
    path = Path(__file__).resolve().parents[3] / 'experiments' / 'benchmark_results.json'
    legacy = {'comparable': False, 'status': 'Not available'}
    if path.exists():
        try:
            content = path.read_bytes()
            legacy.update(status='Historical artifact; not v2 validation', artifact='experiments/benchmark_results.json',
                          sha256=hashlib.sha256(content).hexdigest(), results=json.loads(content),
                          limitations=['No saved run timestamp, configuration hash or exact dataset revision.',
                                       'Text baseline sees the current explanation; persistent-state tier predicts before seeing the current response.',
                                       'These task horizons are not comparable. Do not infer learning gains or use the stored deltas as valid improvements.',
                                       'Legacy fixture/synthetic dataset results do not validate the current model-assisted assessment.'])
        except (ValueError, OSError):
            legacy['status'] = 'Historical artifact unreadable'
    evaluations = runs(db, learner_id)
    selected = get_run(db, learner_id, run_id) if run_id else None
    if run_id and selected is None:
        raise ValueError('evaluation_not_found')
    if selected:
        fields = [('Diagnosis', 'Concept identification accuracy', 'concept_accuracy'),
                  ('Diagnosis', 'Category macro precision', 'category_macro_precision'),
                  ('Diagnosis', 'Category macro recall', 'category_macro_recall'),
                  ('Diagnosis', 'Category macro F1', 'category_macro_f1'),
                  ('Diagnosis', 'Binary misconception precision', 'misconception_precision'),
                  ('Diagnosis', 'Binary misconception recall', 'misconception_recall'),
                  ('Diagnosis', 'Binary misconception F1', 'misconception_f1'),
                  ('Evidence', 'Expert-annotated claim faithfulness', 'expert_claim_faithfulness')]
        scorecard = [r for r in scorecard if r['dimension'] not in ('Diagnosis', 'Evidence')]
        for dimension, label, field in fields:
            value = selected['metrics'][field]
            scorecard.append(dict(dimension=dimension, metric=label, baseline=None, text_baseline=None, proposed=value, improvement=None,
                                  status='Measured on supplied annotations' if value is not None else 'Insufficient labelled data',
                                  evidence=f"{selected['run_id']} · {selected.get('metric_support', {}).get(field, selected['sample_size'])} eligible labels · {selected['annotation_source']}"))
    return {'generated_at': now(), 'dataset': selected['dataset'] if selected else 'Current local learner records', 'model': ' / '.join(selected['models']) if selected else 'v2 criterion-weighted-state-v1',
            'sample_size': selected['sample_size'] if selected else len(turns), 'operational_sample_size': len(turns), 'scorecard': scorecard, 'hypotheses': hypotheses, 'legacy_benchmark': legacy,
            'evaluation_runs': [{k: v for k, v in r.items() if k != 'records'} for r in evaluations],
            'selected_evaluation': selected,
            'notice': 'Evaluation uses supplied annotations; inspect provenance and limitations. Baseline comparisons and H1–H6 remain untested.' if selected else 'Select a saved evaluation run to inspect supplied-label results. Features and integrity tests do not prove H1–H6.' if evaluations else 'No reproducible labelled v2 research evaluation has been recorded. Features and integrity tests do not prove H1–H6.'}
