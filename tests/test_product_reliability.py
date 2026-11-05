"""Product contracts verified against real temporary SQLite, with fixture inference."""
import json
import sqlite3

import pytest
from test_conversation_assessment import flow, start, answer
from test_nlp_observability import traces, detail
from apps.api.app import main
from packages.learner_state_core.assessment.execution import ExecutionStore


def test_failed_submission_is_durable_and_recovers_without_invalid_evidence(flow):
    client, provider = flow
    session = start(client).json()
    cid = session['conversation']['conversation_id']
    original = 'My original explanation must survive a failed model call and a browser restart.'
    provider.fabricate_quote = True
    assert answer(client, session, original).status_code == 502
    resumed = client.get(f'/api/v2/conversations/{cid}', params={'learner_id': 'real-user'}).json()
    assert resumed['pending_submission']['response_text'] == original
    assert resumed['pending_submission']['status'] == 'failed'
    assert resumed['state'] == session['state']
    assert resumed['turns'] == []
    with main.db_manager.connection() as conn:
        row = conn.execute('SELECT * FROM assessment_submissions').fetchone()
        assert row['response_text'] == original
        assert row['question_id'] == session['active_question']['question_id']
        assert row['turn_id'] is None
        assert conn.execute('SELECT count(*) FROM assessment_answers').fetchone()[0] == 0
    provider.fabricate_quote = False
    saved = answer(client, session, original, finish_after_answer=True).json()
    assert len(saved['turns']) == 1
    assert saved['pending_submission'] is None
    with main.db_manager.connection() as conn:
        rows = conn.execute('SELECT status,turn_id FROM assessment_submissions ORDER BY rowid').fetchall()
        assert [r['status'] for r in rows] == ['failed', 'completed']
        assert rows[-1]['turn_id'] == saved['turns'][0]['turn_id']
        assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
    assert client.get(f'/api/v2/conversations/{cid}', params={'learner_id': 'another-user'}).status_code == 404


def test_filtered_totals_and_duration_exclude_untimed_and_running_rows(flow):
    client, _ = flow
    session = start(client).json()
    answer(client, session, finish_after_answer=True)
    with main.db_manager.connection() as conn:
        conn.execute("UPDATE nlp_executions SET duration_ms=100 WHERE operation='answer'")
        conn.execute("UPDATE nlp_executions SET duration_ms=NULL WHERE operation='create'")
        conn.execute("UPDATE nlp_executions SET status='running',duration_ms=9000 WHERE operation='generation'")
    result = traces(client, operation='answer', limit=1)
    assert result['summary']['total'] == 1
    assert result['summary']['average_ms'] == 100
    assert traces(client)['summary']['average_ms'] == 100
    assert traces(client, status='failed')['summary']['total'] == 0
    assert traces(client, status='failed')['summary']['success_rate'] is None


def test_execution_lookup_stage_date_and_error_filters_use_same_population(flow):
    client, provider = flow
    session = start(client).json()
    provider.fabricate_quote = True
    answer(client, session)
    failed = traces(client, operation='answer')['items'][0]
    result = traces(client, search=failed['trace_id'], stage='validation', error_only=True)
    assert len(result['items']) == result['summary']['total'] == 1
    assert result['items'][0]['trace_id'] == failed['trace_id']
    assert result['summary']['retry_count'] == 1
    assert traces(client, search='does-not-exist')['summary']['total'] == 0
    assert traces(client, started_after='2099-01-01T00:00:00Z')['items'] == []
    assert client.get('/api/v2/observability/traces', params={'learner_id':'real-user','started_after':'invalid'}).status_code == 422


def test_partial_telemetry_marks_successful_business_operation_incomplete(flow, monkeypatch):
    client, _ = flow
    session = start(client).json()
    original_write = ExecutionStore.write
    calls = 0
    def miss_one(self, action):
        nonlocal calls
        calls += 1
        if calls == 5:
            return False
        return original_write(self, action)
    monkeypatch.setattr(ExecutionStore, 'write', miss_one)
    result = answer(client, session, finish_after_answer=True)
    assert result.status_code == 200
    execution = detail(client, result.json()['turns'][0]['extraction_trace']['trace_id']).json()
    assert execution['status'] == 'completed'
    assert execution['telemetry_status'] == 'incomplete'
    assert execution['incomplete_warning']
    assert result.json()['turns'][0]['evidence']['validation_status'] == 'passed'


def test_database_failure_returns_safe_json_and_preserves_submission(flow, monkeypatch):
    from packages.learner_state_core.assessment.storage import AssessmentStore
    client, _ = flow
    session = start(client).json()
    def fail(*args):
        raise sqlite3.OperationalError('private filesystem path and secret')
    monkeypatch.setattr(AssessmentStore, 'save_answer', fail)
    result = answer(client, session)
    assert result.status_code == 503
    assert 'private' not in result.text
    assert result.headers.get('x-request-id')
    with main.db_manager.connection() as conn:
        assert conn.execute('SELECT count(*) FROM assessment_answers').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM assessment_state_history').fetchone()[0] == 1
        assert conn.execute('SELECT response_text FROM assessment_submissions').fetchone()[0]


@pytest.mark.parametrize('case,expected', [
    ('unfinished_stage', 'incomplete'), ('missing_stage', 'incomplete'),
    ('legacy', 'unknown'), ('complete', 'complete'), ('running', 'incomplete'),
])
def test_summary_and_detail_agree_on_telemetry_completeness(flow, case, expected):
    client, _ = flow
    session = start(client).json()
    saved = answer(client, session, finish_after_answer=True).json()
    trace_id = saved['turns'][0]['extraction_trace']['trace_id']
    with main.db_manager.connection() as conn:
        if case in ('unfinished_stage', 'legacy'):
            conn.execute("UPDATE nlp_executions SET metadata_json='{}' WHERE trace_id=?", (trace_id,))
        if case == 'unfinished_stage':
            conn.execute("UPDATE nlp_stages SET payload_json=json_set(payload_json,'$.status','running') WHERE trace_id=? AND sequence=1", (trace_id,))
        elif case == 'missing_stage':
            conn.execute('DELETE FROM nlp_stages WHERE trace_id=? AND sequence=1', (trace_id,))
        elif case == 'running':
            conn.execute("UPDATE nlp_executions SET status='running' WHERE trace_id=?", (trace_id,))
    assert detail(client, trace_id).json()['telemetry_status'] == expected
    summary = traces(client, search=trace_id)['summary']
    assert summary['total'] == 1
    assert summary['incomplete_telemetry'] == int(expected == 'incomplete')
