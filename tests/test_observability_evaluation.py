import pytest
from apps.api.app import main
from test_conversation_assessment import flow, start, answer


def test_annotation_evaluation_is_reproducible_scoped_and_keeps_tasks_separate(flow):
    from packages.learner_state_core.assessment.evaluation import evaluate_annotations
    client, _ = flow
    session = start(client).json()
    saved = answer(client, session).json()['turns'][0]
    annotation = {'turn_id': saved['turn_id'], 'category': 'understanding', 'concept_id': 'tokens', 'misconception_present': False, 'claim_faithful': True}
    result = evaluate_annotations(main.db_manager, 'real-user', {'dataset': 'controlled fixture annotation', 'annotation_source': 'unit test, not expert research', 'annotations': [annotation]})
    assert result['metrics']['category_accuracy'] == 1
    assert result['metrics']['concept_accuracy'] == 1
    assert result['metrics']['expert_claim_faithfulness'] == 1
    assert result['metric_support']['expert_claim_faithfulness'] == 1
    assert result['metrics']['misconception_f1'] is None  # no positive gold labels
    assert result['task'] == 'current_response_diagnosis'
    assert result['baselines'] == []
    assert result['annotations_sha256']
    repeated = evaluate_annotations(main.db_manager, 'real-user', {'dataset': 'controlled fixture annotation', 'annotation_source': 'unit test, not expert research', 'annotations': [annotation]})
    assert repeated['annotations_sha256'] == result['annotations_sha256']
    with pytest.raises(ValueError, match='record_not_found'):
        evaluate_annotations(main.db_manager, 'different-user', {'dataset': 'controlled fixture annotation', 'annotation_source': 'unit test', 'annotations': [annotation]})
    response = client.get('/api/v2/observability/research', params={'learner_id':'real-user', 'run_id': result['run_id']})
    assert response.status_code == 200
    research = response.json()
    assert research['evaluation_runs']
    assert any(row['proposed'] == 1 for row in research['scorecard'] if row['metric'] == 'Concept identification accuracy')
    assert all(row['proposed'] is None for row in research['scorecard'] if row['dimension'] == 'Prediction')
    assert all(h['tested'] is False for h in research['hypotheses'])


def test_duplicate_annotation_cannot_inflate_evaluation_sample(flow):
    from packages.learner_state_core.assessment.evaluation import evaluate_annotations
    client, _ = flow
    saved = answer(client, start(client).json()).json()['turns'][0]
    annotation = {'turn_id':saved['turn_id'], 'category':'understanding', 'concept_id':'tokens'}
    with pytest.raises(ValueError, match='duplicate_annotation'):
        evaluate_annotations(main.db_manager, 'real-user', {'dataset':'duplicate input', 'annotation_source':'unit test', 'annotations':[annotation, annotation]})
