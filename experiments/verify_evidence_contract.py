"""Replay a failed submission on a disposable DB copy using the configured model.

python experiments/verify_evidence_contract.py --trace-id nlp_...
The original database is opened read-only. No learner text is exported.
"""
import argparse
import json
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-id', required=True)
    args = parser.parse_args()
    from apps.api.app import main as api
    from packages.learner_state_core.db import LocalDatabaseManager
    from fastapi.testclient import TestClient
    assert api.assessment_provider.configured, 'Configured provider required'
    with tempfile.TemporaryDirectory(prefix='evidence-contract-replay-') as temporary:
        target = Path(temporary) / 'replay.db'
        # db_manager exposes the resolved path; never replace the original file.
        source_path = api.db_manager.db_path
        with closing(sqlite3.connect(Path(source_path).resolve().as_uri() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(target)) as destination:
                source.backup(destination)
        api.db_manager = LocalDatabaseManager(str(target))
        with api.db_manager.connection() as conn:
            trace = conn.execute('SELECT * FROM nlp_executions WHERE trace_id=?', (args.trace_id,)).fetchone()
            assert trace and trace['operation'] == 'answer'
            stages = [json.loads(r[0]) for r in conn.execute('SELECT payload_json FROM nlp_stages WHERE trace_id=? ORDER BY sequence', (args.trace_id,))]
        original = next(s['output']['response_text'] for s in stages if s['key'] == 'input')
        client = TestClient(api.app)
        path = '/api/v2/conversations/' + trace['conversation_id']
        before = client.get(path, params={'learner_id':trace['learner_id']}).json()
        assert before['active_question']['question_id'] == trace['question_id'], 'Failed question is no longer active; use its saved snapshot for a historical replay.'
        result = client.post(path + '/answers', json={'learner_id':trace['learner_id'], 'question_id':trace['question_id'], 'response_text':original, 'request_id':'verification-replay-' + args.trace_id})
        assert result.status_code == 200, result.text
        saved = result.json()
        assert len(saved['turns']) == len(before['turns']) + 1
        turn = saved['turns'][-1]
        assert turn['response_text'] == original
        rubric = {c['criterion_id'] for c in turn['question']['criteria']}
        assert set(turn['evidence']['met_criteria']) | set(turn['evidence']['missing_criteria']) == rubric
        assert turn['extraction_trace']['quote_validated'] is True
        assert saved['active_question'] is not None, saved['conversation']
        assert saved['active_question']['question_id'] != trace['question_id']
        assert saved['active_question']['generation_trace']['state_snapshot_id'] == saved['state']['snapshot_id']
        print(json.dumps({'mode':'live_configured_model', 'original_database_unchanged':True,
            'saved_exact_response':True, 'active_rubric_only':True, 'quote_validated':True,
            'one_answer_committed':True, 'distinct_next_question':True,
            'next_question_uses_committed_state':True, 'category':turn['evidence']['category'],
            'validation_retries':len(turn['extraction_trace']['validation_attempts']),
            'prompt_version':turn['extraction_trace']['prompt_version']}))


if __name__ == '__main__':
    main()
