"""Regressions for the real short-notes/citation failure and state invariants."""
from test_conversation_assessment import flow, start, answer
from apps.api.app import main


def legacy_blocked(client, provider):
    first = start(client).json()
    saved = answer(client, first).json()
    # Reproduce data saved by older versions, without going through new input validation.
    store = main.assessments().store
    with main.db_manager.transaction():
        short_id = store.add_document('real-user', first['conversation']['topic_key'], 'study_notes', 'Grade only', 'Class 6th')
    provider.fail_questions = True
    blocked = start(client, request_id='legacy-blocked').json()
    provider.fail_questions = False
    with main.db_manager.transaction():
        conversation = store.conversation(blocked['conversation']['conversation_id'], 'real-user')
        conversation.update(reference_kind='study_notes', reference_source_ids=[short_id], current_question_id=None, status='needs_question')
        store.save_conversation(conversation)
    blocked = client.post(f"/api/v2/conversations/{conversation['conversation_id']}/next", json={'learner_id': 'real-user'}).json()
    return blocked, saved


def test_grade_only_notes_are_rejected_before_creating_records(flow):
    client, provider = flow
    result = start(client, notes='Class 6th')
    assert result.status_code == 422
    assert 'study material' in result.json()['detail']
    assert provider.question_contexts == []
    assert client.get('/api/v2/conversations', params={'learner_id': 'real-user'}).json() == []


def test_existing_short_notes_have_actionable_failure_and_no_model_retry(flow):
    client, provider = flow
    blocked, saved = legacy_blocked(client, provider)
    assert blocked['conversation']['last_error'] == 'insufficient_source_material'
    assert blocked['state'] == saved['state']
    calls = len(provider.question_contexts)
    cid = blocked['conversation']['conversation_id']
    retried = client.post(f'/api/v2/conversations/{cid}/next', json={'learner_id': 'real-user'}).json()
    assert retried['active_question'] is None
    assert len(provider.question_contexts) == calls
    assert retried['state'] == saved['state']


def test_source_recovery_uses_reference_without_mutating_topic_memory(flow):
    client, provider = flow
    blocked, saved = legacy_blocked(client, provider)
    cid = blocked['conversation']['conversation_id']
    recovered = client.post(f'/api/v2/conversations/{cid}/sources', json={'learner_id': 'real-user', 'mode': 'model_reference'})
    assert recovered.status_code == 200, recovered.text
    session = recovered.json()
    assert session['active_question'] is not None
    assert session['state'] == saved['state']
    assert session['turns'] == []
    assert all(m['kind'] == 'model_reference' for m in provider.question_contexts[-1]['retrieval']['materials'])
    assert client.post(f'/api/v2/conversations/{cid}/sources', json={'learner_id': 'another-user', 'mode': 'model_reference'}).status_code == 404
    assert client.post(f'/api/v2/conversations/{cid}/sources', json={'learner_id': 'real-user', 'mode': 'model_reference'}).status_code == 409


def test_source_recovery_accepts_notes_and_rejects_short_replacements(flow):
    client, provider = flow
    blocked, saved = legacy_blocked(client, provider)
    cid = blocked['conversation']['conversation_id']
    path = f'/api/v2/conversations/{cid}/sources'
    assert client.post(path, json={'learner_id': 'real-user', 'mode': 'study_notes', 'notes': 'Class 6th'}).status_code == 422
    notes = 'Token boundaries preserve separate units of text and allow interpretation in context.'
    result = client.post(path, json={'learner_id': 'real-user', 'mode': 'study_notes', 'notes': notes}).json()
    assert result['active_question']
    assert result['state'] == saved['state']
    assert all(m['content'] in notes for m in provider.question_contexts[-1]['retrieval']['materials'])


def test_failed_creation_is_not_reported_as_completed(flow):
    client, provider = flow
    provider.fail_questions = True
    session = start(client).json()
    traces = client.get('/api/v2/observability/traces', params={'learner_id': 'real-user', 'operation': 'create'}).json()['items']
    assert traces[0]['status'] == 'failed'
    assert traces[0]['conversation_id'] == session['conversation']['conversation_id']


def test_generation_and_observability_reads_never_change_saved_learning_state(flow):
    client, provider = flow
    original = start(client).json()
    provider.fail_questions = True
    saved = answer(client, original).json()
    cid = saved['conversation']['conversation_id']
    for _ in range(3):
        retry = client.post(f'/api/v2/conversations/{cid}/next', json={'learner_id': 'real-user'}).json()
        assert retry['state'] == saved['state']
        assert retry['turns'] == saved['turns']
        for path in ['traces', 'records', 'research']:
            response = client.get(f'/api/v2/observability/{path}', params={'learner_id': 'real-user'})
            assert response.status_code == 200, response.text
    provider.fail_questions = False
    recovered = client.post(f'/api/v2/conversations/{cid}/next', json={'learner_id': 'real-user'}).json()
    assert recovered['active_question']
    assert recovered['state'] == saved['state']
    assert recovered['conversation']['answered_count'] == 1


def test_invalid_citations_produce_specific_safe_feedback_then_recover(flow):
    client, provider = flow
    generate = provider.generate_question
    def invalid(context):
        result = generate(context)
        result['source_citations'] = ['learner_evidence_is_not_reference_material']
        return result
    provider.generate_question = invalid
    blocked = start(client).json()
    assert blocked['conversation']['generation_failure_code'] == 'unsupported_citations'
    assert provider.question_contexts[-1]['allowed_source_citations']
    assert provider.question_contexts[-1]['validation_feedback']['code'] == 'unsupported_citations'
    provider.generate_question = generate
    recovered = client.post(f"/api/v2/conversations/{blocked['conversation']['conversation_id']}/next", json={'learner_id': 'real-user'}).json()
    assert recovered['active_question']
    assert recovered['state'] == blocked['state']


def test_provider_schema_restricts_citations_to_actual_material(monkeypatch):
    from packages.learner_state_core.assessment.provider import AssessmentProvider
    provider = AssessmentProvider(object(), 'test')
    captured = []
    monkeypatch.setattr(provider, '_call', lambda schema, instructions, payload: captured.append(schema))
    provider.generate_question({'allowed_source_citations': ['material_1'], 'plan': {'concept_id': 'tokens', 'kind': 'explain'}})
    schema = captured[0]
    properties = schema.model_json_schema()['properties']
    assert properties['source_citations']['items'].get('const') == 'material_1'


def test_repairing_one_conversation_cannot_replace_another_question_sources(flow):
    client, provider = flow
    first = start(client).json()
    provider.fail_questions = True
    second = start(client, request_id='second-conversation').json()
    provider.fail_questions = False
    second_id = second['conversation']['conversation_id']
    repaired = client.post(f'/api/v2/conversations/{second_id}/sources', json={
        'learner_id': 'real-user', 'mode': 'study_notes',
        'notes': 'Token boundaries preserve separate units of text and support contextual interpretation.'})
    assert repaired.status_code == 200
    evaluate = provider.evaluate
    contexts = []
    def capture(context):
        contexts.append(context)
        return evaluate(context)
    provider.evaluate = capture
    response = answer(client, first)
    assert response.status_code == 200, response.text
    assert all(m['kind'] == 'model_reference' for m in contexts[0]['retrieval']['materials'])
    assert {m['source_id'] for m in contexts[0]['retrieval']['materials']} == {
        m['source_id'] for m in first['active_question']['generation_trace']['context']['retrieval']['materials']}


def test_new_conversations_do_not_reuse_legacy_grade_only_notes(flow):
    client, provider = flow
    blocked, saved = legacy_blocked(client, provider)
    new = start(client, request_id='after-recovery-without-notes').json()
    assert new['active_question'] is not None, new['conversation']
    assert new['state'] == saved['state']
    assert all(m['kind'] == 'model_reference' for m in provider.question_contexts[-1]['retrieval']['materials'])
