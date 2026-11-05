"""Long-conversation generation must preserve the deterministic planner target."""
import pytest
from pydantic import ValidationError
from packages.learner_state_core.assessment.provider import AssessmentProvider
from test_conversation_assessment import flow, start, answer


def test_question_schema_cannot_override_planned_concept_or_kind(monkeypatch):
    provider = AssessmentProvider(object(), 'test')
    captured = []
    monkeypatch.setattr(provider, '_call', lambda schema, instructions, payload: captured.append(schema))
    provider.generate_question({'allowed_source_citations':['material_current'], 'plan':{'concept_id':'addition', 'kind':'contrast'}})
    schema = captured[0]
    valid = {'concept_id':'addition', 'kind':'contrast', 'prompt':'Distinguish the addition operation from the sum using a fresh example.',
             'expected_focus':'Explain the distinction using the source material.', 'learning_support':'',
             'criteria':[{'criterion_id':'distinction','description':'Distinguishes the operation from its result.'}],
             'reference_answer':'Addition is the operation and the sum is the result.', 'source_citations':['material_current']}
    assert schema.model_validate(valid).concept_id == 'addition'
    with pytest.raises(ValidationError):
        schema.model_validate({**valid, 'concept_id':'old_concept_from_history'})
    with pytest.raises(ValidationError):
        schema.model_validate({**valid, 'kind':'guided_practice'})


def test_repeated_question_retry_identifies_rejected_and_conflicting_text(flow):
    client, provider = flow
    session = start(client).json()
    generate = provider.generate_question
    feedback = []
    def repeat_then_correct(context):
        question = generate(context)
        if 'validation_feedback' not in context:
            question['prompt'] = session['active_question']['prompt']
        else:
            feedback.append(context['validation_feedback'])
        return question
    provider.generate_question = repeat_then_correct
    response = answer(client, session)
    assert response.status_code == 200, response.text
    assert response.json()['active_question'] is not None
    assert feedback[0]['code'] == 'repeated_question'
    assert feedback[0]['rejected_question_prompt'] == session['active_question']['prompt']
    assert feedback[0]['similar_question_prompt'] == session['active_question']['prompt']
    assert feedback[0]['format_constraint']
