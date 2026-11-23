"""The model selects evidence; it never has to retype Unicode source text."""
import pytest
from pydantic import ValidationError

from packages.learner_state_core.assessment.provider import AssessmentProvider
from test_evidence_contract_recovery import evaluation_context, valid_observation


@pytest.mark.parametrize('text', [
    'Proper: numerator < denominator. Improper: numerator ≥ denominator. Mixed: whole + fraction.',
    '  中文解释：分子 ≥ 分母。\r\nHindi: अंश ≥ हर। Emoji 🧠. LaTeX: $\\frac{7}{4}$.  ',
    ('A fraction compares equal parts to the whole.\n' * 176)[:7950] + ' ≥ 1',
    'No',
    '???',
    ' ' * 7900 + 'I cannot explain yet.',
])
def test_model_selects_original_passage_without_copying_symbols(monkeypatch, text):
    provider = AssessmentProvider(object(), 'test')
    context = evaluation_context()
    context['response_text'] = text
    original = dict(valid_observation())
    selected_text = []

    def select(schema, instructions, payload):
        passages = payload['response_passages']
        assert passages
        for passage in passages:
            assert text[passage['start']:passage['end']] == passage['text']
        properties = schema.model_json_schema()['properties']
        assert 'source_quote' not in properties
        # Copying the >= symbol produced a corrupt quote in the real failure.
        # The wire contract must instead accept only IDs of original passages.
        wire = {k: v for k, v in original.items() if k != 'source_quote'}
        wire['source_passage_id'] = passages[-1]['passage_id']
        wire['reasoning_status'] = 'substantive'
        selected_text.append(passages[-1]['text'])
        with pytest.raises(ValidationError):
            schema.model_validate({**wire, 'source_passage_id': 'invented'})
        return schema.model_validate(wire).model_dump()

    monkeypatch.setattr(provider, '_call', select)
    result = provider.evaluate(context)
    assert result['source_quote'] == selected_text[0]
    assert 'source_passage_id' not in result
    assert result['met_criteria'] == ['current_sum']
    assert context['response_text'] == text


def test_every_original_character_is_available_for_evidence_selection(monkeypatch):
    text = 'First explanation.\n' + 'Long reasoning ≥ with equal parts. ' * 100 + '\nNo.'
    provider = AssessmentProvider(object(), 'test')
    context = evaluation_context()
    context['response_text'] = text

    def select(schema, instructions, payload):
        covered = set()
        for passage in payload['response_passages']:
            covered.update(range(passage['start'], passage['end']))
        assert covered == set(range(len(text)))
        wire = {k:v for k,v in valid_observation().items() if k != 'source_quote'}
        return {**wire, 'reasoning_status':'substantive', 'source_passage_id': payload['response_passages'][0]['passage_id']}

    monkeypatch.setattr(provider, '_call', select)
    assert provider.evaluate(context)['source_quote'].startswith('First explanation.')


def test_help_only_judgment_cannot_be_scored_as_prerequisite_gap(monkeypatch):
    provider = AssessmentProvider(object(), 'test')
    context = evaluation_context()
    context['response_text'] = 'I am not sure; I cannot explain the reasoning yet. Can you help me?'

    def select(schema, instructions, payload):
        wire = {k:v for k,v in valid_observation().items() if k != 'source_quote'}
        wire.update(source_passage_id=payload['response_passages'][0]['passage_id'],
                    reasoning_status='help_or_uncertainty_only', category='prerequisite_gap',
                    met_criteria=[], missing_criteria=['current_sum'])
        with pytest.raises(ValidationError, match='ambiguous'):
            schema.model_validate(wire)
        return schema.model_validate({**wire, 'category':'ambiguous'}).model_dump()

    monkeypatch.setattr(provider, '_call', select)
    assert provider.evaluate(context)['category'] == 'ambiguous'
