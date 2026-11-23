"""Integration contracts: real API/storage, fixture inference explicitly not research data."""
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from test_conversation_assessment import flow, start, answer
from apps.api.app import main


def traces(client, **filters):
    response = client.get('/api/v2/observability/traces', params={'learner_id': 'real-user', **filters})
    assert response.status_code == 200, response.text
    return response.json()


def detail(client, trace_id, learner='real-user'):
    return client.get('/api/v2/observability/traces/' + trace_id, params={'learner_id': learner})


def test_replayed_answer_links_saved_evidence_without_claiming_new_processing(flow):
    client, _ = flow
    session = start(client).json()
    saved = answer(client, session).json()['turns'][0]
    assert answer(client, session).status_code == 200
    replay = detail(client, traces(client, operation='answer')['items'][0]['trace_id']).json()
    assert replay['metadata']['cache_hit'] is True
    assert replay['metadata']['turn_id'] == saved['turn_id']
    assert replay['metadata']['source_trace_id'] == saved['extraction_trace']['trace_id']
    assert replay['stages'] == []
    assert client.get('/api/v2/observability/records/' + replay['metadata']['turn_id'], params={'learner_id':'real-user'}).json() == saved


def test_preprocessing_recognizes_typographic_negation_without_changing_raw_offsets():
    from packages.learner_state_core.assessment.nlp import analyze
    text = '😀 I don’t know. I can’t explain why, maybe it won’t work.'
    result = analyze(text)
    assert result['raw_text'] == text
    assert result['negation_cues'] == ["don't", "can't", "won't"]
    assert result['uncertainty_cues'] == ["don't know", 'maybe']
    assert all(text[t['start']:t['end']] == t['token'] for t in result['token_spans'])


def test_concurrent_identical_replay_links_winning_record_not_unsaved_candidate(flow):
    from threading import Barrier
    client, provider = flow
    session = start(client).json()
    barrier, evaluate = Barrier(2), provider.evaluate
    def parallel(context):
        result = evaluate(context)
        barrier.wait(timeout=10)
        return result
    provider.evaluate = parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: answer(client, session, finish_after_answer=True), range(2)))
    assert [r.status_code for r in responses] == [200, 200]
    saved = responses[0].json()['turns'][0]
    assert len(responses[1].json()['turns']) == 1
    executions = [detail(client, item['trace_id']).json() for item in traces(client, operation='answer')['items']]
    replay = next(item for item in executions if item['trace_id'] != saved['extraction_trace']['trace_id'])
    assert replay['metadata']['turn_id'] == saved['turn_id']
    assert replay['metadata']['source_trace_id'] == saved['extraction_trace']['trace_id']
    assert replay['metadata']['cache_hit'] is True
    assert next(s for s in replay['stages'] if s['key'] == 'persistence')['output']['committed'] is False


def test_committed_trace_links_actual_text_evidence_state_and_history(flow):
    client, _ = flow
    session = start(client).json()
    result = answer(client, session).json()
    turn = result['turns'][0]
    rows = traces(client, operation='answer')['items']
    assert len(rows) == 1
    trace = detail(client, rows[0]['trace_id']).json()
    assert trace['status'] == 'completed'
    stages = {s['key']: s for s in trace['stages']}
    assert stages['preprocessing']['output']['raw_text'] == turn['response_text']
    assert stages['preprocessing']['parameters']['version'] == turn['nlp']['version']
    assert stages['evidence']['output']['evidence_id'] == turn['evidence']['evidence_id']
    assert stages['state_update']['output']['state_after'] == turn['state_after']
    assert stages['persistence']['output']['snapshot_id'] == turn['state_after']['snapshot_id']
    assert turn['extraction_trace']['trace_id'] == trace['trace_id']
    assert turn['evidence']['source_interaction_id'] == turn['turn_id']
    assert all(s['duration_ms'] >= 0 for s in trace['stages'])
    assert [s['sequence'] for s in trace['stages']] == list(range(1, len(trace['stages']) + 1))
    assert detail(client, trace['trace_id']).json() == trace
    assert detail(client, trace['trace_id'], 'someone-else').status_code == 404
    assert traces(client, conversation_id='nonexistent')['items'] == []
    assert traces(client, limit=1)['has_more']


def test_retry_records_rejected_attempts_without_leaking_error_body(flow):
    client, provider = flow
    session = start(client).json()
    evaluate = provider.evaluate
    calls = []
    def once_invalid(context):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('AZURE_API_KEY=must-never-appear')
        return evaluate(context)
    provider.evaluate = once_invalid
    assert answer(client, session).status_code == 200
    trace = detail(client, traces(client, operation='answer')['items'][0]['trace_id']).json()
    attempts = [s for s in trace['stages'] if s['key'] == 'extraction']
    assert [s['status'] for s in attempts] == ['failed', 'completed']
    assert [s['attempt'] for s in attempts] == [1, 2]
    assert 'must-never-appear' not in json.dumps(trace)


def test_invalid_model_output_is_failed_and_never_commits_state(flow):
    client, provider = flow
    session = start(client).json()
    provider.fabricate_quote = True
    assert answer(client, session).status_code == 502
    trace = detail(client, traces(client, operation='answer')['items'][0]['trace_id']).json()
    assert trace['status'] == 'failed'
    assert len([s for s in trace['stages'] if s['key'] == 'validation' and s['status'] == 'failed']) == 2
    assert not any(s['key'] == 'persistence' for s in trace['stages'])


def test_persistence_rollback_leaves_failed_trace_but_no_learner_update(flow, monkeypatch):
    from packages.learner_state_core.assessment.storage import AssessmentStore
    client, _ = flow
    session = start(client).json()
    def fail(*args, **kwargs):
        raise RuntimeError('private database information')
    monkeypatch.setattr(AssessmentStore, 'add_chunk', fail)
    failed = answer(client, session)
    assert failed.status_code == 500
    assert failed.json()['request_id'] == failed.headers['x-request-id']
    assert 'private database information' not in failed.text
    trace = detail(client, traces(client, operation='answer')['items'][0]['trace_id']).json()
    assert trace['status'] == 'failed'
    assert next(s for s in trace['stages'] if s['key'] == 'persistence')['status'] == 'failed'
    saved = client.get('/api/v2/conversations/' + session['conversation']['conversation_id'], params={'learner_id': 'real-user'}).json()
    assert saved['state'] == session['state'] and saved['turns'] == []


def test_live_running_stage_is_visible_before_model_returns(flow):
    client, provider = flow
    session = start(client).json()
    entered, release = Event(), Event()
    evaluate = provider.evaluate
    def slow(context):
        entered.set()
        assert release.wait(10)
        return evaluate(context)
    provider.evaluate = slow
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(answer, client, session)
        try:
            assert entered.wait(10)
            live = traces(client, operation='answer')['items'][0]
            assert live['status'] == 'running'
            trace = detail(client, live['trace_id']).json()
            assert trace['stages'][-1]['key'] == 'extraction'
            assert trace['stages'][-1]['status'] == 'running'
            assert trace['stages'][-1]['completed_at'] is None
        finally:
            release.set()
        assert pending.result().status_code == 200


@pytest.mark.parametrize('text', [
    'Both numerator and denominator must be multiplied by the same number to preserve the fraction.',
    'I use a wrong rule because changing only the numerator preserves the value of the fraction.',
    'I used the common denominator correctly but calculated seven times two as thirteen by mistake.',
    'I guess this might be right but cannot explain the reason.',
    '2/3', '½ × 2 ≠ 1/4; I do not change only the numerator.',
    'I multipled numrator and denominatr by same numbr so the ratio stays the same.',
])
def test_language_scenarios_retain_verbatim_spans_and_actual_outcomes(flow, text):
    client, provider = flow
    # This test checks transport/provenance, not the semantic accuracy of a fixture model.
    session = start(client).json()
    assert answer(client, session, text).status_code == 200
    trace = detail(client, traces(client, operation='answer')['items'][0]['trace_id']).json()
    nlp = next(s['output'] for s in trace['stages'] if s['key'] == 'preprocessing')
    assert nlp['raw_text'] == text
    assert all(text[t['start']:t['end']] == t['token'] for t in nlp['token_spans'])


def test_research_matrix_does_not_invent_research_metrics(flow):
    client, _ = flow
    response = client.get('/api/v2/observability/research', params={'learner_id': 'real-user'})
    assert response.status_code == 200
    data = response.json()
    assert len(data['hypotheses']) == 6
    assert all(h['tested'] is False for h in data['hypotheses'])
    assert all(row['proposed'] is None for row in data['scorecard'] if row['dimension'] != 'Operations')
    assert data['legacy_benchmark']['comparable'] is False
    assert client.get('/api/v2/observability/traces', params={'learner_id': 'real-user', 'limit': 501}).status_code == 422


def test_math_tokens_preserve_original_symbol_spans():
    from packages.learner_state_core.assessment.nlp import analyze
    text = '½ × 2 ≠ 1/4; not -2.5'
    result = analyze(text)
    assert {'½', '×', '≠', '/', '-', '2.5'}.issubset({s['token'] for s in result['token_spans']})
    assert result['negation_cues'] == ['not']


def test_sentence_boundaries_retain_newline_text_and_decimal_numbers():
    from packages.learner_state_core.assessment.nlp import analyze
    result = analyze('First line\nThe value is 2.5. I do not round it.')
    sentences = [s['text'].strip() for s in result['sentences']]
    assert sentences == ['First line', 'The value is 2.5.', 'I do not round it.']


def test_history_records_remain_inspectable_without_fabricated_stage_events(flow):
    client, _ = flow
    session = start(client).json()
    saved = answer(client, session).json()['turns'][0]
    listed = client.get('/api/v2/observability/records', params={'learner_id': 'real-user'}).json()
    assert listed['items'][0]['turn_id'] == saved['turn_id']
    path = '/api/v2/observability/records/' + saved['turn_id']
    assert client.get(path, params={'learner_id': 'real-user'}).json() == saved
    assert client.get(path, params={'learner_id': 'different-user'}).status_code == 404


def test_trace_storage_outage_does_not_abort_answer(flow, monkeypatch):
    from packages.learner_state_core.assessment.execution import ExecutionStore
    client, _ = flow
    session = start(client).json()
    monkeypatch.setattr(ExecutionStore, 'write', lambda *args: False)
    result = answer(client, session)
    assert result.status_code == 200
    assert len(result.json()['turns']) == 1


def test_retention_removes_expired_trace_content_without_deleting_evidence(flow):
    from packages.learner_state_core.assessment.execution import ExecutionStore
    client, _ = flow
    session = start(client).json()
    saved = answer(client, session).json()['turns'][0]
    with main.db_manager.connection() as conn:
        conn.execute("UPDATE nlp_executions SET started_at='2000-01-01T00:00:00+00:00'")
    ExecutionStore(main.db_manager).prune()
    assert traces(client)['items'] == []
    with main.db_manager.connection() as conn:
        assert conn.execute('SELECT count(*) FROM nlp_stages').fetchone()[0] == 0
    assert client.get('/api/v2/observability/records/' + saved['turn_id'], params={'learner_id': 'real-user'}).status_code == 200


def test_concurrent_same_question_records_distinct_attempts_and_one_commit(flow):
    from threading import Barrier
    client, provider = flow
    session = start(client).json()
    barrier, evaluate = Barrier(2), provider.evaluate
    def parallel(context):
        raw = evaluate(context)
        barrier.wait(timeout=10)
        return raw
    provider.evaluate = parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda i: answer(client, session, request_id=f'parallel-observe-{i}'), range(2)))
    assert sorted(r.status_code for r in responses) == [200, 409]
    items = traces(client, operation='answer')['items']
    assert len(items) == 2 and len({i['trace_id'] for i in items}) == 2
    assert sorted(i['status'] for i in items) == ['completed', 'failed']


def test_retry_creation_commits_generation_before_trace_completion(flow):
    client, provider = flow
    provider.fail_questions = True
    first = start(client).json()
    assert first['active_question'] is None
    provider.fail_questions = False
    recovered = start(client).json()
    assert recovered['active_question']
    generation = traces(client, operation='generation')['items'][0]
    assert generation['status'] == 'completed'
    trace = detail(client, generation['trace_id']).json()
    assert all(s['status'] != 'running' for s in trace['stages'])


def test_malformed_structured_output_is_rejected_and_each_attempt_is_visible(flow):
    client, provider = flow
    session = start(client).json()
    provider.evaluate = lambda context: {'concept_id': context['question']['concept_id']}
    assert answer(client, session).status_code == 502
    trace = detail(client, traces(client, operation='answer')['items'][0]['trace_id']).json()
    failed = [s for s in trace['stages'] if s['key'] == 'validation']
    assert len(failed) == 2
    assert all(s['status'] == 'failed' and s['error']['type'] == 'ValidationError' for s in failed)
    assert not any(s['key'] == 'persistence' for s in trace['stages'])


def test_separate_learners_execute_concurrently_without_mixing_traces(flow):
    from threading import Barrier
    client, provider = flow
    first = start(client).json()
    second = start(client, learner_id='separate-user').json()
    barrier, evaluate = Barrier(2), provider.evaluate
    def parallel(context):
        raw = evaluate(context)
        barrier.wait(timeout=10)
        return raw
    provider.evaluate = parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(answer, client, first)
        two = pool.submit(answer, client, second, learner_id='separate-user')
        assert one.result().status_code == two.result().status_code == 200
    mine = traces(client, operation='answer')['items']
    others = client.get('/api/v2/observability/traces', params={'learner_id':'separate-user', 'operation':'answer'}).json()['items']
    assert len(mine) == len(others) == 1
    assert mine[0]['trace_id'] != others[0]['trace_id']
    assert detail(client, others[0]['trace_id']).status_code == 404
