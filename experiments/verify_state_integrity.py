"""Save SQL-backed duplicate, abstention and contradiction evidence without AI calls."""
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def main():
    with tempfile.TemporaryDirectory(prefix='learner-state-integrity-') as temp:
        os.environ['LEARNER_DB_PATH'] = str(Path(temp) / 'check.db')
        os.environ['LEARNER_EXTRACTOR'] = 'rules'
        from apps.api.app import main as api
        from fastapi.testclient import TestClient
        from test_conversation_assessment import FixtureProvider, start, answer

        api.assessment_provider = FixtureProvider()
        client = TestClient(api.app)
        scenarios = {}

        def accepted(response):
            assert response.status_code == 200, response.text
            return response.json()

        def reference(session):
            turn = session['turns'][-1]
            return {'conversation_id':session['conversation']['conversation_id'],
                    'turn_id':turn['turn_id'], 'evidence_id':turn['evidence']['evidence_id'],
                    'trace_id':turn['extraction_trace']['trace_id'],
                    'snapshot_before':turn['state_before']['snapshot_id'],
                    'snapshot_after':turn['state_after']['snapshot_id'],
                    'state_changes':turn['state_changes']}

        initial = accepted(start(client, topic='Duplicate verification', request_id='duplicate'))
        text = 'I use a wrong rule because I think a token must always equal a complete word.'
        first = accepted(answer(client, initial, text))
        replay = accepted(answer(client, initial, text))
        assert first['state'] == replay['state'] and len(replay['turns']) == 1
        repeated = accepted(answer(client, first, text))
        assert repeated['turns'][-1]['state_changes']['duplicate_response']
        assert repeated['turns'][-1]['state_changes']['weight'] < first['turns'][-1]['state_changes']['weight']
        scenarios['duplicate'] = {'idempotent_replay_added_answers':0, 'first':reference(first), 'repeated_on_next_question':reference(repeated)}

        initial = accepted(start(client, topic='Abstention verification', request_id='abstention'))
        held = accepted(answer(client, initial, 'I guess, but I cannot explain why yet.'))
        parameters = lambda state: {key:{name:concept[name] for name in ('alpha','beta')} for key,concept in state['concepts'].items()}
        assert parameters(initial['state']) == parameters(held['state'])
        scenarios['abstention'] = {'before':parameters(initial['state']), 'after':parameters(held['state']), **reference(held)}

        initial = accepted(start(client, topic='Contradiction verification', request_id='contradiction'))
        wrong = accepted(answer(client, initial, 'I use a wrong rule because I think a token must be exactly one word.'))
        repaired = accepted(answer(client, wrong))
        concept_id = wrong['turns'][-1]['evidence']['concept_id']
        hypotheses = repaired['state']['concepts'][concept_id]['hypotheses']
        assert hypotheses and all(h['contradicting_evidence_ids'] for h in hypotheses.values())
        scenarios['contradiction'] = {'before':reference(wrong), 'after':reference(repaired), 'hypotheses':hypotheses}

        with api.db_manager.connection() as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
            assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
            verified = []
            for row in conn.execute('SELECT * FROM assessment_answers'):
                turn = json.loads(row['payload_json'])
                assert turn['evidence']['source_interaction_id'] == row['turn_id']
                assert conn.execute('SELECT 1 FROM assessment_questions WHERE question_id=? AND conversation_id=?', (turn['question']['question_id'],row['conversation_id'])).fetchone()
                snapshot = conn.execute('SELECT payload_json FROM assessment_state_history WHERE snapshot_id=?', (turn['state_after']['snapshot_id'],)).fetchone()
                assert json.loads(snapshot[0]) == turn['state_after']
                assert conn.execute("SELECT 1 FROM assessment_submissions WHERE turn_id=? AND status='completed'", (row['turn_id'],)).fetchone()
                assert conn.execute("SELECT 1 FROM nlp_executions WHERE trace_id=? AND status='completed'", (turn['extraction_trace']['trace_id'],)).fetchone()
                verified.append(row['turn_id'])
            tables = ['assessment_conversations','assessment_questions','assessment_answers','assessment_state_history','assessment_submissions','assessment_documents','assessment_chunks','nlp_executions','nlp_stages']
            counts = {table:conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in tables}
        output = ROOT / 'docs/verification/product-transformation/state-integrity.json'
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'mode':'fixture', 'synthetic':True, 'integrity':'ok', 'foreign_key_violations':0, 'verified_turns':verified, 'counts':counts, 'scenarios':scenarios},indent=2),encoding='utf-8')
        print(json.dumps({'scenarios':list(scenarios), 'verified_answers':len(verified), 'integrity':'ok', 'output':str(output)}))


if __name__ == '__main__':
    main()
