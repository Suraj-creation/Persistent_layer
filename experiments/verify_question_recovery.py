"""Browser regressions for blocked generation, source repair and persistent memory.

Runs against a disposable SQLite database; --live uses the configured cloud model
with synthetic learner text. Screenshots and a check manifest are verification
artifacts, never research evaluation data. Build apps/web before running.
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
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/verification/question-recovery')
    args = parser.parse_args()
    from apps.api.app import main as api
    from packages.learner_state_core.db import LocalDatabaseManager
    from playwright.sync_api import sync_playwright, expect
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    if not args.live:
        from test_conversation_assessment import FixtureProvider
        api.assessment_provider = FixtureProvider()
        original_generate = api.assessment_provider.generate_question
        tasks = [
            'Explain how token boundaries preserve distinct pieces of text in a sentence.',
            'A parser merges adjacent words. Diagnose which information is lost and why.',
            'Compare punctuation handling in two possible sentence segmentation strategies.',
            'Design a vocabulary that represents an unfamiliar term as several pieces.',
            'Analyze why splitting a contraction by spaces can retain ambiguity.',
            'Apply the source distinction to the representation of an unseen compound.',
            'Justify preserving the order of tokens when interpreting a sentence.',
            'Explain a case where one written word contains several model tokens.',
        ]
        def varied(context):
            result = original_generate(context)
            result['prompt'] = tasks[len(context['previous_questions']) % len(tasks)]
            return result
        api.assessment_provider.generate_question = varied
    assert api.assessment_provider.configured, 'Configured model required for --live'
    mode = 'live' if args.live else 'fixture'
    output = args.output / mode
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    with tempfile.TemporaryDirectory(prefix='question-recovery-') as temporary:
        api.db_manager = LocalDatabaseManager(str(Path(temporary) / 'isolated.db'))
        api.app.mount('/', StaticFiles(directory=ROOT / 'apps/web/dist', html=True))
        with socket.socket() as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        server = uvicorn.Server(uvicorn.Config(api.app, host='127.0.0.1', port=port, log_level='warning'))
        worker = threading.Thread(target=server.run, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 20
            while not server.started:
                assert time.monotonic() < deadline, 'Verification server failed to start'
                time.sleep(.05)
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True, channel='chromium')
                context = browser.new_context(viewport={'width': 1560, 'height': 1050})
                context.add_init_script("localStorage.setItem('learner-state:guide:v2','seen');localStorage.setItem('learner-state:identity:v2','synthetic_recovery');")
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(base)
                page.screenshot(path=str(output / 'setup-desktop.png'), full_page=True)
                topic = 'Photosynthesis' if args.live else 'Tokenization'
                notes = ('Plants capture light energy with chlorophyll in chloroplasts. Photosynthesis uses carbon dioxide and water to produce sugars and release oxygen. Light-dependent reactions capture energy; the Calvin cycle uses stored chemical energy to fix carbon into sugars.' if args.live else 'Tokenization divides text into tokens while preserving boundaries. A token need not be a whole word. Context affects interpretation of a token in a sentence.')
                page.get_by_label('Topic to assess').fill(topic)
                page.get_by_label('Your study notes').fill('Class 6th')
                page.get_by_role('button', name='Start my assessment', exact=True).click()
                expect(page.get_by_role('alert')).to_contain_text('8 words of study material')
                assert context.request.get(base + '/api/v2/conversations?learner_id=synthetic_recovery').json() == []
                checks.append('grade-only notes rejected without creating a conversation')
                page.get_by_label('Your study notes').fill(notes)
                page.get_by_role('button', name='Start my assessment', exact=True).click()
                response = page.get_by_label('Explain your thinking', exact=True)
                expect(response).to_be_visible(timeout=150000)
                cid = page.evaluate("localStorage.getItem('learner-state:conversation:v2')")
                url = base + f'/api/v2/conversations/{cid}?learner_id=synthetic_recovery'
                original = context.request.get(url).json()
                first_id = original['active_question']['question_id']
                response.fill('I cannot remember the reasoning yet and I am unsure how these parts fit together.')
                page.get_by_role('button', name='Submit & next question', exact=True).click()
                expect(response).to_be_enabled(timeout=150000)
                saved = context.request.get(url).json()
                assert saved['conversation']['answered_count'] == 1
                assert saved['active_question'] is not None, saved['conversation']
                assert saved['active_question']['question_id'] != first_id
                assert saved['active_question']['generation_trace']['state_snapshot_id'] == saved['state']['snapshot_id']
                checks.append('answer commits once and a distinct next question is displayed')
                page.screenshot(path=str(output / 'learning-desktop.png'), full_page=True)
                page.reload()
                expect(response).to_be_visible()
                assert context.request.get(url).json()['state'] == saved['state']
                checks.append('reload preserves question and topic memory')

                # Force a real server-side generation failure after evidence commits.
                generate = api.assessment_provider.generate_question
                def unavailable(_context):
                    raise TimeoutError('synthetic generation outage')
                api.assessment_provider.generate_question = unavailable
                response.fill('I still cannot explain the relationship and I need to revisit the study material.')
                page.get_by_role('button', name='Submit & next question', exact=True).click()
                expect(page.get_by_role('heading', name='Question generation is paused.')).to_be_visible(timeout=150000)
                blocked = context.request.get(url).json()
                assert blocked['conversation']['answered_count'] == 2
                page.get_by_role('button', name='Generate next question', exact=True).click()
                expect(page.get_by_role('button', name='Generate next question', exact=True)).to_be_enabled(timeout=150000)
                retried = context.request.get(url).json()
                assert retried['state'] == blocked['state']
                assert retried['turns'] == blocked['turns']
                checks.append('generation failure and retry preserve answers and estimates')
                api.assessment_provider.generate_question = generate
                page.get_by_role('button', name='Generate next question', exact=True).click()
                expect(response).to_be_visible(timeout=150000)
                recovered = context.request.get(url).json()
                assert recovered['state'] == blocked['state']
                checks.append('generation retry recovers without duplicate evidence')

                # Reproduce the old data defect, then exercise both source-repair controls.
                store = api.assessments().store
                with api.db_manager.transaction():
                    short_id = store.add_document('synthetic_recovery', recovered['conversation']['topic_key'], 'study_notes', 'Legacy grade-only note', 'Class 6th')
                    legacy = store.conversation(cid, 'synthetic_recovery')
                    legacy.update(current_question_id=None, status='needs_question', reference_kind='study_notes', reference_source_ids=[short_id])
                    store.save_conversation(legacy)
                context.request.post(base + f'/api/v2/conversations/{cid}/next', data={'learner_id': 'synthetic_recovery'})
                page.reload()
                expect(page.get_by_role('heading', name='Add material to continue.')).to_be_visible()
                page.get_by_text('Update study material', exact=True).click()
                page.screenshot(path=str(output / 'source-recovery-desktop.png'), full_page=True)
                page.get_by_label('Replacement study notes').fill(notes)
                page.get_by_role('button', name='Save notes & generate', exact=False).click()
                expect(response).to_be_visible(timeout=150000)
                repaired = context.request.get(url).json()
                assert repaired['state'] == blocked['state']
                assert all(m['content'] in notes for m in repaired['active_question']['generation_trace']['context']['retrieval']['materials'])
                checks.append('legacy short material can be replaced without changing evidence')
                with api.db_manager.transaction():
                    legacy = store.conversation(cid, 'synthetic_recovery')
                    legacy.update(current_question_id=None, status='needs_question', reference_source_ids=[short_id])
                    store.save_conversation(legacy)
                context.request.post(base + f'/api/v2/conversations/{cid}/next', data={'learner_id': 'synthetic_recovery'})
                page.reload()
                expect(page.get_by_role('heading', name='Add material to continue.')).to_be_visible()
                page.get_by_text('Update study material', exact=True).click()
                page.get_by_role('button', name='Use model reference instead', exact=True).click()
                expect(response).to_be_visible(timeout=150000)
                reference = context.request.get(url).json()
                assert reference['state'] == blocked['state']
                assert all(m['kind'] == 'model_reference' for m in reference['active_question']['generation_trace']['context']['retrieval']['materials'])
                checks.append('explicit model reference recovery uses only labeled model sources')

                page.get_by_role('button', name='NLP Observability', exact=True).click()
                expect(page.locator('.nlp-execution-list > button').first).to_be_visible()
                page.screenshot(path=str(output / 'observability-desktop.png'), full_page=True)
                assert context.request.get(url).json()['state'] == blocked['state']
                checks.append('observability navigation leaves learner memory unchanged')
                page.get_by_role('button', name='Learning workspace', exact=True).first.click()
                page.get_by_role('button', name='Finish & view report', exact=True).click()
                expect(page.locator('.conversation-report')).to_be_visible(timeout=150000)
                report = context.request.get(base + f'/api/v2/conversations/{cid}/report?learner_id=synthetic_recovery').json()
                assert report['answered_count'] == 2
                assert report['ending_state'] == blocked['state']
                context.request.post(base + f'/api/v2/conversations/{cid}/resume', data={'learner_id': 'synthetic_recovery'})
                page.reload()
                expect(response).to_be_visible()
                assert context.request.get(url).json()['state'] == blocked['state']
                checks.append('finish/report/resume retain the submitted evidence')

                page.set_viewport_size({'width': 390, 'height': 844})
                response.scroll_into_view_if_needed()
                page.screenshot(path=str(output / 'learning-mobile.png'), full_page=True)
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Mobile overflow'
                assert page.locator('#active-question-heading').evaluate('(e) => parseFloat(getComputedStyle(e).fontSize)') >= 20
                assert response.evaluate('(e) => parseFloat(getComputedStyle(e).fontSize)') >= 15
                checks.append('mobile question, answer field and actions are readable without overflow')
                assert not errors, errors
                checks.append('zero browser JavaScript errors')
                (output / 'checks.json').write_text(json.dumps({'mode': mode, 'synthetic_data': True, 'checks': checks, 'javascript_errors': errors, 'final_answer_count': 2, 'state_unchanged_during_retries_and_recovery': True}, indent=2), encoding='utf-8')
                print(json.dumps({'mode': mode, 'checks_passed': len(checks), 'output': str(output)}))
                browser.close()
        finally:
            server.should_exit = True
            worker.join(timeout=5)


if __name__ == '__main__':
    main()
