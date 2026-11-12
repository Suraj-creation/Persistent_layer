"""Regression for historical gap IDs leaking into the active question rubric."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from packages.learner_state_core.assessment.provider import AssessmentProvider
from test_conversation_assessment import flow, start, answer


def evaluation_context():
    return {
        'question': {'question_id':'question_current', 'concept_id':'addition', 'kind':'contrast',
            'prompt':'Distinguish the sum from the addition operation.',
            'expected_focus':'Name the result and distinguish it from the operation.',
            'criteria':[{'criterion_id':'current_sum', 'description':'Names the result of addition.'}],
            'reference_answer':'The sum is the result of the addition operation.',
            'generation_trace':{'context':{'plan':{'missing_criteria':['old_associativity']}}},
            'usage':{'prompt_tokens':1000}},
        'state':{'snapshot_id':'before', 'concepts':{'addition':{'hypotheses':{}, 'missing_criteria':['Explain associativity.']}, 'other':{'hypotheses':{}}}},
        'response_text':'Addition is the operation; its result is the sum.',
        'retrieval':{'materials':[{'chunk_id':'source_current','content':'The result is called the sum.'}], 'evidence':[]},
        'contract_version':'rubric-evidence-v2',
    }


def valid_observation():
    return dict(concept_id='addition', source_quote='its result is the sum.', claim='The result of addition is called the sum.',
        category='understanding', confidence=.9, met_criteria=['current_sum'], missing_criteria=[],
        misconception=None, contradicts_hypothesis_ids=[], feedback='You distinguished the result from the operation.',
        coaching='', source_citations=['source_current'])


def test_provider_contract_rejects_old_gap_ids_and_removes_nested_question_history(monkeypatch):
    provider = AssessmentProvider(object(), 'test')
    captured = []
    monkeypatch.setattr(provider, '_call', lambda schema, instructions, payload: captured.append((schema,payload)))
    context = evaluation_context()
    original = deepcopy(context)
    provider.evaluate(context)
    schema,payload = captured[0]
    assert 'generation_trace' not in payload['question']
    assert 'usage' not in payload['question']
    assert set(payload['state']['concepts']) == {'addition'}
    assert 'missing_criteria' not in payload['state']['concepts']['addition']
    assert 'evidence' not in payload['retrieval']
    assert payload['allowed_criterion_ids'] == ['current_sum']
    assert payload['allowed_source_citations'] == ['source_current']
    assert schema.model_validate(valid_observation()).met_criteria == ['current_sum']
    invalid = {**valid_observation(), 'missing_criteria':['old_associativity']}
    with pytest.raises(ValidationError):
        schema.model_validate(invalid)
    with pytest.raises(ValidationError):
        schema.model_validate({**valid_observation(), 'concept_id':'other'})
    with pytest.raises(ValidationError):
        schema.model_validate({**valid_observation(), 'source_citations':['learner_evidence_id']})
    assert context == original


def test_rubric_failure_retry_receives_exact_code_and_active_ids(flow):
    client, provider = flow
    session = start(client).json()
    evaluate = provider.evaluate
    feedback = []
    def first_invalid(context):
        result = evaluate(context)
        if 'validation_feedback' not in context:
            result['missing_criteria'] = ['old_gap_not_in_this_question']
        else:
            feedback.append(context['validation_feedback'])
        return result
    provider.evaluate = first_invalid
    response = answer(client, session)
    assert response.status_code == 200, response.text
    assert isinstance(feedback[0], dict)
    assert feedback[0]['code'] == 'invalid_rubric_partition'
    assert feedback[0]['allowed_criterion_ids'] == ['reason','example']
    result = response.json()
    assert len(result['turns']) == 1
    assert result['active_question'] is not None
    assert result['active_question']['question_id'] != session['active_question']['question_id']
    assert result['active_question']['generation_trace']['state_snapshot_id'] == result['state']['snapshot_id']


def test_repeated_invalid_evidence_preserves_state_and_attributes_failure_to_model(flow):
    client, provider = flow
    session = start(client).json()
    evaluate = provider.evaluate
    def invalid(context):
        return {**evaluate(context), 'missing_criteria':['unrelated_old_gap']}
    provider.evaluate = invalid
    response = answer(client, session)
    assert response.status_code == 502
    assert 'assessment model' in response.json()['detail'].lower()
    saved = client.get('/api/v2/conversations/' + session['conversation']['conversation_id'], params={'learner_id':'real-user'}).json()
    assert saved['state'] == session['state']
    assert saved['turns'] == []
