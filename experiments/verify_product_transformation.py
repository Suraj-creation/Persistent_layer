"""Browser + SQL regression for the improve branch; fixture model, isolated SQLite."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
OUTPUT = ROOT / 'docs/verification/product-transformation/browser'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design-inspector', type=Path, help='Optional installed Taste inspect.mjs path')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='learner-product-check-') as temp:
        os.environ['LEARNER_DB_PATH'] = str(Path(temp) / 'check.db')
        os.environ['LEARNER_EXTRACTOR'] = 'rules'
        from apps.api.app import main as api
        from test_conversation_assessment import FixtureProvider
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import HTMLResponse
        from playwright.sync_api import sync_playwright, expect
        import uvicorn

        provider = FixtureProvider()
        api.assessment_provider = provider
        inspection_conversation = None
        if args.design_inspector:
            @api.app.get('/__visual_check', response_class=HTMLResponse)
            def visual_check(screen: str = 'setup'):
                # This route exists only in the isolated verification harness.
                identity = 'product_check' if screen == 'assessment' else 'visual_empty'
                initialization = f"localStorage.setItem('learner-state:identity:v2',{json.dumps(identity)});localStorage.setItem('learner-state:guide:v2','seen');"
                initialization += f"localStorage.setItem('learner-state:conversation:v2',{json.dumps(inspection_conversation)});" if screen == 'assessment' else "localStorage.removeItem('learner-state:conversation:v2');"
                return (ROOT / 'apps/web/dist/index.html').read_text(encoding='utf-8').replace('<head>', '<head><script>' + initialization + '</script>')
        api.app.mount('/', StaticFiles(directory=ROOT / 'apps/web/dist', html=True))
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        server = uvicorn.Server(uvicorn.Config(api.app, host='127.0.0.1', port=port, log_level='warning'))
        worker = threading.Thread(target=server.run, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 20
            while not server.started:
                assert time.monotonic() < deadline
                time.sleep(.05)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(viewport={'width':1440,'height':900}, reduced_motion='reduce')
                context.add_init_script("localStorage.setItem('learner-state:identity:v2','product_check'); localStorage.setItem('learner-state:guide:v2','seen');")
                page = context.new_page()
                page.set_default_timeout(20000)
                expect.set_options(timeout=20000)
                errors, requests, measurements = [], [], []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('request', lambda request: requests.append(request.url))
                page.goto(base)
                expect(page.get_by_label('Topic to assess')).to_be_visible()
                for width,height in [(1440,900),(1024,768),(768,1024),(390,844),(320,568)]:
                    page.set_viewport_size({'width':width,'height':height})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
                    page.screenshot(path=str(OUTPUT / f'setup-{width}.png'))
                    measurements.append({'width':width,'horizontal_overflow':False})
                assert not any('MathRenderer' in request for request in requests)
                page.set_viewport_size({'width':1440,'height':900})
                page.get_by_label('Topic to assess').fill('Tokenization')
                page.get_by_label('Your study notes').fill('Tokenization divides text into spans while preserving boundaries. Context changes how tokens are interpreted in a sentence.')
                page.get_by_role('button', name='Start my assessment').click()
                expect(page.get_by_label('Explain your thinking')).to_be_visible()
                page.screenshot(path=str(OUTPUT / 'assessment-desktop.png'))
                original = 'Token boundaries preserve distinct pieces of text, including words with meaning in context.'
                provider.fabricate_quote = True
                page.get_by_label('Explain your thinking').fill(original)
                page.get_by_role('button', name='Submit & next question', exact=True).click()
                expect(page.get_by_role('alert')).to_contain_text('could not return evidence')
                page.evaluate("Object.keys(localStorage).filter(key => key.startsWith('learner-state:draft')).forEach(key => localStorage.removeItem(key))")
                page.reload()
                expect(page.get_by_label('Explain your thinking')).to_have_value(original)
                expect(page.locator('.recovered-submission')).to_be_visible()
                page.screenshot(path=str(OUTPUT / 'recovered-submission.png'))
                provider.fabricate_quote = False
                page.get_by_role('button', name='Submit & next question', exact=True).click()
                expect(page.locator('.conversation-history')).to_be_visible()
                cid = page.evaluate("localStorage.getItem('learner-state:conversation:v2')")
                inspection_conversation = cid
                session_url = f'{base}/api/v2/conversations/{cid}?learner_id=product_check'
                session = context.request.get(session_url).json()
                trace_id = session['turns'][0]['extraction_trace']['trace_id']
                page.get_by_role('button', name='NLP Observability', exact=True).click()
                page.get_by_label('Execution lookup', exact=True).fill(trace_id)
                expect(page.locator('.nlp-execution-list > button')).to_have_count(1)
                expect(page.locator('.nlp-summary > span').first).to_have_text('1 executions')
                page.get_by_text('Stage, error & date filters', exact=False).first.click()
                page.get_by_label('Pipeline stage').select_option('validation')
                expect(page.locator('.nlp-execution-list > button')).to_have_count(1)
                page.get_by_label('Errors and recovered retries').check()
                expect(page.get_by_role('heading', name='No executions match these filters.')).to_be_visible()
                page.get_by_role('button', name='Clear filters', exact=True).click()
                page.get_by_label('From (UTC)', exact=True).fill('2099-01-01')
                expect(page.get_by_role('heading', name='No executions match these filters.')).to_be_visible()
                page.get_by_role('button', name='Clear filters', exact=True).click()
                page.locator('.nlp-advanced-filters > summary').click()
                expect(page.locator('.nlp-execution-list > button')).to_have_count(5)
                page.get_by_text('Performance & reliability', exact=True).click()
                expect(page.get_by_text('Provider retries', exact=True)).to_be_visible()
                page.screenshot(path=str(OUTPUT / 'execution-desktop.png'))
                for width,height in [(1024,768),(768,1024),(390,844),(320,568)]:
                    page.set_viewport_size({'width':width,'height':height})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width
                    page.screenshot(path=str(OUTPUT / f'execution-{width}.png'))
                page.set_viewport_size({'width':1440,'height':900})
                page.get_by_role('button', name='Conversations', exact=True).click()
                page.get_by_role('button', name='All conversations', exact=True).click()
                page.get_by_label('Search conversations').fill('missing-topic')
                expect(page.get_by_text('No conversations match.', exact=False)).to_be_visible()
                page.get_by_role('button', name='Clear filters', exact=True).click()
                expect(page.locator('.conversation-collection > button')).to_have_count(1)
                page.screenshot(path=str(OUTPUT / 'history-desktop.png'))
                page.locator('.conversation-collection > button').click()
                page.get_by_role('button', name='Reports', exact=True).click()
                expect(page.locator('.conversation-report h1')).to_have_text('Tokenization')
                page.screenshot(path=str(OUTPUT / 'report-desktop.png'))
                page.get_by_role('button', name='Learning workspace', exact=True).click()
                held = []
                page.route('**/report?**', lambda route: held.append(route))
                page.get_by_role('button', name='Reports', exact=True).click()
                expect(page.get_by_text('Building the report from saved evidence…', exact=True)).to_be_visible()
                page.get_by_role('button', name='All reports', exact=True).click()
                expect(page.get_by_label('Search conversations')).to_be_visible()
                for route in held:
                    route.abort()
                page.unroute('**/report?**')
                page.locator('.conversation-collection > button').click()
                expect(page.locator('.conversation-report h1')).to_have_text('Tokenization')
                page.get_by_role('button', name='Learning workspace', exact=True).click()
                page.get_by_role('button', name='Show workspace guide', exact=True).click()
                expect(page.locator('.guide-tip')).to_be_visible()
                page.keyboard.press('Escape')
                expect(page.locator('.guide-tip')).not_to_be_visible()
                expect(page.get_by_role('button', name='Show workspace guide', exact=True)).to_be_focused()
                # A missing split bundle must leave navigation and a recovery action.
                offline_context = browser.new_context()
                offline_context.add_init_script("localStorage.setItem('learner-state:guide:v2','seen');")
                offline_page = offline_context.new_page()
                offline_page.route('**/assets/NlpObservability-*.js', lambda route: route.abort())
                offline_page.goto(base)
                offline_page.get_by_role('button', name='NLP Observability', exact=True).click()
                expect(offline_page.get_by_role('alert')).to_contain_text('This view could not load')
                expect(offline_page.get_by_role('button', name='Learning workspace', exact=True)).to_be_visible()
                offline_page.unroute('**/assets/NlpObservability-*.js')
                offline_page.get_by_role('button', name='Reload workspace', exact=True).click()
                offline_page.get_by_role('button', name='NLP Observability', exact=True).click()
                expect(offline_page.get_by_label('Execution lookup', exact=True)).to_be_visible()
                offline_context.close()
                # Inspect real SQLite rows, not only the HTTP response.
                with api.db_manager.connection() as conn:
                    assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
                    assert conn.execute('PRAGMA foreign_key_check').fetchall() == []
                    answer_row = conn.execute('SELECT * FROM assessment_answers').fetchone()
                    saved = json.loads(answer_row['payload_json'])
                    assert saved['question']['question_id'] == conn.execute('SELECT question_id FROM assessment_questions ORDER BY rowid LIMIT 1').fetchone()[0]
                    assert saved['evidence']['source_interaction_id'] == answer_row['turn_id']
                    assert conn.execute('SELECT snapshot_id FROM assessment_state_history WHERE snapshot_id=?', (saved['state_after']['snapshot_id'],)).fetchone()
                    receipts = [dict(row) for row in conn.execute('SELECT trace_id,question_id,status,turn_id FROM assessment_submissions ORDER BY rowid')]
                    assert [r['status'] for r in receipts] == ['failed','completed']
                    assert conn.execute('SELECT count(*) FROM nlp_stages').fetchone()[0] > 0
                    counts = {table:conn.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in ['assessment_answers','assessment_questions','assessment_state_history','assessment_submissions','nlp_executions','nlp_stages']}
                assert not errors, errors
                result = {'mode':'fixture', 'synthetic':True, 'viewports':measurements, 'browser_errors':errors,
                          'trace_id':trace_id, 'turn_id':saved['turn_id'], 'evidence_id':saved['evidence']['evidence_id'],
                          'snapshot_id':saved['state_after']['snapshot_id'], 'database_counts':counts, 'receipts':receipts,
                          'checks':['failed submission recovered without local draft','filtered totals','execution lookup','stage filter','error filter','UTC date filter','performance metrics','history search','report reload','cancelled report loading recovery','keyboard guide dismissal and focus restoration','SQLite relationships','foreign keys','integrity','on-demand math loading','split bundle failure recovery']}
                (OUTPUT / 'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(json.dumps(result), flush=True)
                browser.close()
            if args.design_inspector:
                for screen in ('setup', 'assessment'):
                    inspected = subprocess.run(['node', str(args.design_inspector), '--app', f'{base}/__visual_check?screen={screen}', '--out', str(OUTPUT.parent / ('taste-' + screen))], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
                    (OUTPUT.parent / ('taste-' + screen + '.txt')).write_text(inspected.stdout + inspected.stderr, encoding='utf-8')
                    print(json.dumps({'design_inspector':screen, 'exit_code':inspected.returncode}), flush=True)
        finally:
            server.should_exit = True
            worker.join(timeout=5)


if __name__ == '__main__':
    main()
