"""Browser regressions for viewport fit and truthful NLP inspectors.

Uses fixture inference and a disposable database; never edits real learner data.
Build apps/web first. Screenshots and results go to docs/verification/workspace-feedback.
"""
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def main():
    from apps.api.app import main as api
    from packages.learner_state_core.db import LocalDatabaseManager
    from test_conversation_assessment import FixtureProvider
    from fastapi.staticfiles import StaticFiles
    from playwright.sync_api import sync_playwright, expect
    import uvicorn

    output = ROOT / 'docs/verification/workspace-feedback'
    output.mkdir(parents=True, exist_ok=True)
    api.assessment_provider = FixtureProvider()
    with tempfile.TemporaryDirectory(prefix='workspace-feedback-') as temporary:
        api.db_manager = LocalDatabaseManager(str(Path(temporary) / 'test.db'))
        api.app.mount('/', StaticFiles(directory=ROOT / 'apps/web/dist', html=True), name='feedback-ui')
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
                assert time.monotonic() < deadline, 'Server did not start'
                time.sleep(.05)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(viewport={'width':1366,'height':768}, reduced_motion='reduce')
                context.add_init_script("localStorage.setItem('learner-state:guide:v2','seen'); localStorage.setItem('learner-state:identity:v2','feedback_check');")
                page = context.new_page()
                page.set_default_timeout(30000)
                expect.set_options(timeout=30000)
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(base)
                print('Verification UI ready', flush=True)
                expect(page.locator('.setup-form')).to_be_visible()
                measurements = []
                for width, height in [(1366,768), (1280,720), (1024,768), (390,844)]:
                    page.set_viewport_size({'width':width,'height':height})
                    page.evaluate('window.scrollTo(0,0)')
                    bounds = page.get_by_role('button', name='Start my assessment').bounding_box()
                    document = page.evaluate('({width:innerWidth, height:innerHeight, scrollWidth:document.documentElement.scrollWidth, scrollHeight:document.documentElement.scrollHeight})')
                    measurements.append({'viewport':[width,height], 'start_button':bounds, 'document':document})
                    assert bounds['y'] >= 0 and bounds['y'] + bounds['height'] <= height, measurements[-1]
                    assert document['scrollWidth'] <= width, document
                    assert document['scrollHeight'] <= height + 1, document
                    page.screenshot(path=str(output / f'setup-{width}.png'))
                print('Setup viewport checks passed', flush=True)
                page.set_viewport_size({'width':1366,'height':768})
                page.get_by_label('Topic to assess').fill('Tokenization')
                page.get_by_label('Your study notes').fill('Tokenization divides text into spans while preserving boundaries. Context changes how tokens are interpreted in a sentence.')
                page.get_by_role('button', name='Start my assessment').click()
                expect(page.locator('#assessment-response')).to_be_visible()
                for label in ['Submit & finish', 'Submit & next question', 'Finish & view report']:
                    bounds = page.get_by_role('button', name=label, exact=True).bounding_box()
                    assert bounds['y'] + bounds['height'] <= 768, (label, bounds)
                assert page.evaluate('document.documentElement.scrollHeight <= innerHeight + 1')
                page.screenshot(path=str(output / 'question-desktop.png'))
                # Long answers exceed bounded trace arrays, but the inspector must
                # recover every token from the persisted record without blank markers.
                text = 'Token boundaries preserve context. ' * 65
                page.get_by_label('Explain your thinking').fill(text)
                page.get_by_role('button', name='Submit & next question', exact=True).click()
                expect(page.locator('.conversation-history')).to_be_visible()
                page.get_by_role('button', name='NLP Observability', exact=True).click()
                expect(page.locator('.nlp-execution-detail h2')).to_have_text('Response processing')
                page.get_by_role('button', name='Text transformations', exact=True).click()
                expect(page.get_by_role('heading', name='Original learner text')).to_be_visible()
                expect(page.locator('.nlp-record-source')).to_be_visible()
                token_count = page.locator('.nlp-token-list > span').count()
                assert token_count > 160, token_count
                assert all(page.locator('.nlp-token-list strong').nth(i).inner_text() for i in range(token_count))
                page.screenshot(path=str(output / 'text-desktop.png'))
                page.get_by_role('button', name='Cognitive Evidence', exact=True).click()
                expect(page.get_by_role('heading', name='Criteria met', exact=True)).to_be_visible()
                expect(page.get_by_text('Explains why token boundaries matter.', exact=True)).to_be_visible()
                page.screenshot(path=str(output / 'evidence-desktop.png'))
                page.get_by_role('button', name='State changes', exact=True).click()
                expect(page.get_by_text('The transaction committed this state.', exact=True)).to_be_visible()
                expect(page.locator('.nlp-state-table tbody tr')).to_have_count(2)
                page.screenshot(path=str(output / 'state-desktop.png'))
                print('Complete text, evidence and state inspectors checked', flush=True)
                # Explicitly selecting generation must explain applicability and
                # navigate to the answer, rather than showing a dead-end notice.
                page.locator('.nlp-execution-list > button').filter(has_text='Question generation').first.click()
                expect(page.locator('.nlp-execution-detail h2')).to_have_text('Question generation')
                page.get_by_role('button', name='Cognitive Evidence', exact=True).click()
                expect(page.get_by_role('heading', name='Choose a learner response to inspect this step')).to_be_visible()
                page.get_by_role('button', name='Inspect latest learner response').click()
                expect(page.get_by_role('heading', name='What the learner said')).to_be_visible()
                cid = page.evaluate("localStorage.getItem('learner-state:conversation:v2')")
                session_url = f'{base}/api/v2/conversations/{cid}?learner_id=feedback_check'
                saved = context.request.get(session_url).json()
                # A rejected quote cannot become accepted Cognitive Evidence.
                api.assessment_provider.fabricate_quote = True
                rejected = context.request.post(f'{base}/api/v2/conversations/{cid}/answers', data={'learner_id':'feedback_check', 'question_id':saved['active_question']['question_id'], 'response_text':'I explain token boundaries using a new concrete sentence.', 'request_id':'rejected'})
                assert rejected.status == 502
                page.get_by_role('button', name='Refresh observability').click()
                expect(page.locator('.nlp-execution-list > button').first).to_contain_text('failed')
                page.locator('.nlp-execution-list > button').first.click()
                page.get_by_role('button', name='Cognitive Evidence', exact=True).click()
                expect(page.get_by_role('heading', name='No accepted output for this step')).to_be_visible()
                page.get_by_role('button', name='State changes', exact=True).click()
                expect(page.get_by_role('heading', name='No accepted output for this step')).to_be_visible()
                assert context.request.get(session_url).json()['state'] == saved['state']
                page.get_by_role('button', name='Learning workspace', exact=True).first.click()
                for width, height in [(1024,768), (390,844), (320,568)]:
                    page.set_viewport_size({'width':width, 'height':height})
                    page.get_by_role('button', name='Saved topic memory', exact=True).click()
                    expect(page.get_by_role('heading', name='Saved topic memory', exact=True)).to_be_visible()
                    page.get_by_role('button', name='Current question', exact=True).click()
                    expect(page.locator('#assessment-response')).to_be_visible()
                    page.get_by_role('button', name='Why this question?', exact=True).click()
                    expect(page.locator('#inspector-decision')).to_have_attribute('aria-selected', 'true')
                    page.get_by_role('button', name='Current question', exact=True).click()
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
                    page.screenshot(path=str(output / f'question-{width}.png'))
                page.set_viewport_size({'width':1024, 'height':768})
                page.get_by_role('button', name='Saved topic memory', exact=True).click()
                page.get_by_role('button', name='Show workspace guide', exact=True).click()
                expect(page.locator('.guide-tip')).to_be_visible()
                expect(page.locator('#assessment-response')).to_be_visible()
                for target in ['question', 'expectation', 'inspector']:
                    expect(page.locator(f'[data-guide="{target}"]')).to_be_visible()
                    expect(page.locator('.guide-highlight')).to_have_js_property('clientWidth', page.locator(f'[data-guide="{target}"]').evaluate('(element) => Math.round(element.getBoundingClientRect().width + 8)'), timeout=5000)
                    page.locator('.guide-actions').get_by_role('button', name='Next', exact=True).click()
                page.get_by_role('button', name='Close guide', exact=True).click()
                # An ambiguous response records an observation, holds numeric
                # estimates, and remains inspectable through an idempotent replay.
                api.assessment_provider.fabricate_quote = False
                hold_payload = {'learner_id':'feedback_check', 'question_id':saved['active_question']['question_id'], 'response_text':'I guess maybe I am not sure about this token boundary.', 'request_id':'held-response', 'finish_after_answer':True}
                held_response = context.request.post(f'{base}/api/v2/conversations/{cid}/answers', data=hold_payload)
                assert held_response.status == 200, held_response.text()
                held = held_response.json()
                assert all(c['estimate'] == saved['state']['concepts'][key]['estimate'] for key,c in held['state']['concepts'].items())
                replay = context.request.post(f'{base}/api/v2/conversations/{cid}/answers', data=hold_payload)
                assert replay.status == 200
                assert len(replay.json()['turns']) == len(held['turns'])
                page.get_by_role('button', name='NLP Observability', exact=True).click()
                expect(page.locator('.nlp-execution-detail h2')).to_have_text('Response processing')
                page.get_by_role('button', name='State changes', exact=True).click()
                expect(page.get_by_role('heading', name='Estimate held', exact=True)).to_be_visible()
                expect(page.get_by_text('The transaction committed this state.', exact=True)).to_be_visible()
                expect(page.get_by_text('No new NLP stages ran and no new evidence was added.', exact=False)).to_be_visible()
                assert page.evaluate('document.documentElement.scrollHeight <= innerHeight + 1')
                page.screenshot(path=str(output / 'held-replay-desktop.png'))
                assert not errors, errors
                result = {'provider':'fixture', 'synthetic_data':True, 'setup_measurements':measurements, 'complete_tokens_displayed':token_count, 'checks':['setup fits viewport', 'desktop question actions visible', 'memory navigation', 'default response selection', 'complete long-answer tokens', 'criterion descriptions', 'committed state comparison', 'generation-to-response navigation', 'rejected evidence absent', 'failed request preserves state', 'mobile no horizontal overflow', 'guide reveals target pane', 'ambiguous response holds estimates', 'replay loads original saved response'], 'javascript_errors':errors}
                (output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
                print(json.dumps(result))
                browser.close()
        finally:
            server.should_exit = True
            worker.join(timeout=5)


if __name__ == '__main__':
    main()
