"""Run the actual UI/API on an isolated database. No real learner data is touched.

python experiments/verify_nlp_observability.py [--live]
Fixture inference is the default. --live uses the configured provider and synthetic text.
The resulting integrity measurements are NOT research or learning-outcome scores.
"""
import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs' / 'verification' / 'nlp-observability')
    args = parser.parse_args()
    from apps.api.app import main as api
    from packages.learner_state_core.db import LocalDatabaseManager
    from fastapi.staticfiles import StaticFiles
    from playwright.sync_api import sync_playwright, expect
    import uvicorn
    args.output.mkdir(parents=True, exist_ok=True)
    if args.live and not api.assessment_provider.configured:
        raise RuntimeError('Live provider is not configured; no illustrative fallback is permitted.')
    if not args.live:
        from test_conversation_assessment import FixtureProvider
        api.assessment_provider = FixtureProvider()
    with tempfile.TemporaryDirectory(prefix='nlp-observability-check-') as temporary:
        api.db_manager = LocalDatabaseManager(str(Path(temporary) / 'verification.db'))
        api.app.mount('/', StaticFiles(directory=ROOT / 'apps' / 'web' / 'dist', html=True), name='verification-ui')
        with socket.socket() as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(api.app, host='127.0.0.1', port=port, log_level='warning'))
        worker = threading.Thread(target=server.run, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started:
                if time.monotonic() > deadline:
                    raise RuntimeError('Verification server did not start.')
                time.sleep(.05)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, channel='chromium')
                context = browser.new_context(viewport={'width': 1440, 'height': 1000})
                context.add_init_script("localStorage.setItem('learner-state:guide:v2','seen'); localStorage.setItem('learner-state:identity:v2','verification_synthetic_learner');")
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{port}')
                topic = 'Equivalent fractions' if args.live else 'Tokenization'
                notes = ('Equivalent fractions represent the same value. Multiply or divide both numerator and denominator by the same nonzero number. Changing only the numerator usually changes the value. A fraction denotes a ratio of numerator to denominator.' if args.live else 'Tokenization divides text into tokens while preserving boundaries. A token need not be a whole word. Context affects interpretation of a token in a sentence.')
                page.get_by_label('Topic to assess').fill(topic)
                page.get_by_label('Your study notes').fill(notes)
                page.get_by_role('button', name='Start my assessment').click()
                response_box = page.locator('.active-assessment textarea')
                expect(response_box).to_be_visible(timeout=120000)
                first_question = page.locator('#active-question-heading').inner_text()
                response_text = ('I multiply both numerator and denominator by the same nonzero number, so their ratio stays unchanged. For example, 1/2 = 2/4 because multiplying both parts by two preserves the value; multiplying only the numerator would change it.' if args.live else 'Token boundaries preserve distinct pieces of text. For example, separate word spans allow a program to interpret the surrounding context without merging every word together.')
                response_box.fill(response_text)
                page.get_by_role('button', name='Submit & next question', exact=True).click()
                # Navigation remains available while the actual inference is in flight.
                page.get_by_role('button', name='NLP Observability', exact=True).click()
                expect(page.get_by_role('heading', name='NLP Observability', exact=True)).to_be_visible()
                operation_select = page.locator('.nlp-filters label').filter(has_text='Execution').locator('select')
                operation_select.select_option('answer')
                expect(page.locator('.nlp-execution-list > button').first).to_be_visible(timeout=120000)
                # Wait for the real evidence stage, rather than a fixed animation timer.
                expect(page.locator('.nlp-stage').filter(has_text='Cognitive Evidence')).to_be_visible(timeout=120000)
                expect(page.locator('.nlp-execution-detail .nlp-execution-meta .nlp-status')).to_have_text('completed', timeout=120000)
                page.screenshot(path=str(args.output / 'desktop-timeline.png'), full_page=True)
                # Completion of extraction alone is insufficient: the learner must see
                # a distinct next question backed by the newly persisted snapshot.
                page.get_by_role('button', name='Learning workspace', exact=True).first.click()
                expect(response_box).to_be_visible(timeout=120000)
                assert page.locator('#active-question-heading').inner_text() != first_question
                cid = page.evaluate("localStorage.getItem('learner-state:conversation:v2')")
                session_url = f'http://127.0.0.1:{port}/api/v2/conversations/{cid}?learner_id=verification_synthetic_learner'
                saved = context.request.get(session_url).json()
                assert saved['active_question'] is not None, saved['conversation']
                assert saved['active_question']['generation_trace']['state_snapshot_id'] == saved['state']['snapshot_id']
                assert saved['conversation']['answered_count'] == 1
                page.screenshot(path=str(args.output / 'desktop-next-question.png'), full_page=True)
                page.reload()
                expect(response_box).to_be_visible()
                reloaded = context.request.get(session_url).json()
                assert reloaded['state'] == saved['state']
                assert reloaded['active_question']['question_id'] == saved['active_question']['question_id']
                page.get_by_role('button', name='NLP Observability', exact=True).click()
                page.locator('.nlp-filters label').filter(has_text='Execution').locator('select').select_option('answer')
                expect(page.locator('.nlp-execution-meta .nlp-status')).to_have_text('completed')
                page.get_by_role('button', name='Text transformations', exact=True).click()
                expect(page.get_by_role('heading', name='Original learner text')).to_be_visible()
                assert page.locator('.nlp-inspector blockquote').first.inner_text() == response_text
                page.screenshot(path=str(args.output / 'desktop-transformations.png'), full_page=True)
                page.get_by_role('button', name='Cognitive Evidence', exact=True).click()
                expect(page.get_by_role('heading', name='What the learner said')).to_be_visible()
                page.get_by_role('button', name='State changes', exact=True).click()
                expect(page.get_by_text('The transaction committed this state.', exact=True)).to_be_visible()
                page.screenshot(path=str(args.output / 'desktop-state.png'), full_page=True)
                page.get_by_role('button', name='Saved responses', exact=True).click()
                expect(page.locator('.nlp-record-list > button').first).to_be_visible()
                page.locator('.nlp-record-list > button').first.click()
                expect(page.get_by_role('heading', name='Saved response record')).to_be_visible()
                page.route('**/api/v2/observability/records?**', lambda route: route.fulfill(status=503, json={'detail':'Temporary records outage'}))
                page.get_by_role('button', name='Refresh observability', exact=True).click()
                expect(page.get_by_role('alert').filter(has_text='Temporary records outage')).to_be_visible()
                page.unroute('**/api/v2/observability/records?**')
                page.get_by_role('button', name='Refresh observability', exact=True).click()
                expect(page.locator('.nlp-record-list > button').first).to_be_visible()
                page.get_by_role('button', name='NLP techniques', exact=True).click()
                page.get_by_label('Search NLP techniques').fill('BM25')
                expect(page.locator('.nlp-techniques article')).to_have_count(1)
                page.get_by_role('button', name='Research scorecard', exact=True).click()
                expect(page.get_by_role('heading', name='Six hypotheses, six open research tests')).to_be_visible()
                pending_rows = page.locator('.nlp-scorecard tbody tr').filter(has_text='Not evaluated')
                expect(pending_rows).to_have_count(8)
                page.screenshot(path=str(args.output / 'desktop-research.png'), full_page=True)
                page.set_viewport_size({'width': 390, 'height': 844})
                page.get_by_role('button', name='Execution traces', exact=True).click()
                expect(page.locator('.nlp-execution-detail h2')).to_be_visible()
                page.screenshot(path=str(args.output / 'mobile-timeline.png'), full_page=True)
                overflow = page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
                assert not overflow, 'Mobile layout overflows the viewport'
                assert context.request.get(session_url).json()['state'] == saved['state'], 'Observability reads changed learner memory'
                page.reload()
                page.get_by_role('button', name='NLP Observability', exact=True).click()
                page.locator('.nlp-filters label').filter(has_text='Execution').locator('select').select_option('answer')
                expect(page.locator('.nlp-execution-meta .nlp-status')).to_have_text('completed')
                # The API boundary is exercised using a different local profile.
                traces = context.request.get(f'http://127.0.0.1:{port}/api/v2/observability/traces?learner_id=verification_synthetic_learner&operation=answer').json()
                trace_id = traces['items'][0]['trace_id']
                assert context.request.get(f'http://127.0.0.1:{port}/api/v2/observability/traces/{trace_id}?learner_id=another_profile').status == 404
                trace = context.request.get(f'http://127.0.0.1:{port}/api/v2/observability/traces/{trace_id}?learner_id=verification_synthetic_learner').json()
                records = context.request.get(f'http://127.0.0.1:{port}/api/v2/observability/records?learner_id=verification_synthetic_learner').json()
                turn = context.request.get(f"http://127.0.0.1:{port}/api/v2/observability/records/{records['items'][0]['turn_id']}?learner_id=verification_synthetic_learner").json()
                all_traces = context.request.get(f'http://127.0.0.1:{port}/api/v2/observability/traces?learner_id=verification_synthetic_learner').json()
                linked = [context.request.get(f"http://127.0.0.1:{port}/api/v2/observability/traces/{item['trace_id']}?learner_id=verification_synthetic_learner").json() for item in all_traces['items']]
                with api.db_manager.connection() as conn:
                    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
                    stored = json.loads(conn.execute('SELECT payload_json FROM assessment_answers WHERE turn_id=?', (turn['turn_id'],)).fetchone()[0])
                    assert stored == turn
                    assert conn.execute('SELECT question_id FROM assessment_questions WHERE question_id=?', (turn['question']['question_id'],)).fetchone()
                    assert conn.execute('SELECT snapshot_id FROM assessment_state_history WHERE snapshot_id=?', (turn['state_after']['snapshot_id'],)).fetchone()
                    receipt = dict(conn.execute('SELECT status,turn_id,question_id FROM assessment_submissions WHERE trace_id=?', (trace_id,)).fetchone())
                    assert receipt['status'] == 'completed' and receipt['turn_id'] == turn['turn_id']
                    database_counts = {table: conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in ['assessment_answers','assessment_questions','assessment_state_history','assessment_submissions','nlp_executions','nlp_stages']}
                report = {'provider_mode': 'live_configured_model' if args.live else 'fixture_inference',
                          'database_verification': {'integrity':'ok','foreign_key_violations':0,'counts':database_counts,'submission':receipt},
                          'synthetic_learner': True, 'is_research_evaluation': False,
                          'checks': ['UI topic creation', 'UI answer submission', 'distinct visible next question', 'next question uses committed state', 'question and memory survive reload', 'observability leaves learner state unchanged', 'navigation during inference', 'persisted trace', 'text/evidence/state inspectors', 'historical saved response', 'records outage and refresh recovery', 'technique search', 'truthful research matrix', 'desktop/mobile', 'reload', 'cross-profile isolation'],
                          'javascript_errors': errors, 'mobile_overflow': overflow,
                          'execution': trace, 'linked_executions': linked, 'saved_database_record': turn}
                assert errors == [], errors
                output = args.output / ('live-execution.json' if args.live else 'fixture-execution.json')
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({'mode': report['provider_mode'], 'checks_passed': len(report['checks']), 'javascript_errors': len(errors), 'output': str(output)}))
                browser.close()
        finally:
            server.should_exit = True
            worker.join(timeout=5)


if __name__ == '__main__':
    main()
