"""Browser-driven long-session assessment verification on a disposable database.

Build apps/web, then run:
  python experiments/verify_extended_assessment.py --answers 205
  python experiments/verify_extended_assessment.py --live --answers 3

Every answer is submitted through browser controls. Live answers use deliberately
ambiguous/wrong text followed by saved reference answers; this is an integration
test, not a learning-outcome evaluation. The real learner database is not used.
"""
import argparse
import hashlib
import json
import logging
import math
import random
import socket
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def assert_turn(turn, before, after):
    assert turn['state_before'] == before
    assert turn['state_after'] == after
    text, evidence, nlp = turn['response_text'], turn['evidence'], turn['nlp']
    assert nlp['raw_text'] == text
    assert text[evidence['source_start']:evidence['source_end']] == evidence['source_quote']
    assert all(text[t['start']:t['end']] == t['token'] for t in nlp['token_spans'])
    assert all(text[s['start']:s['end']] == s['text'] for s in nlp['sentences'])
    criteria = {c['criterion_id'] for c in turn['question']['criteria']}
    met, missing = set(evidence['met_criteria']), set(evidence['missing_criteria'])
    assert not met & missing and met | missing == criteria
    cid = evidence['concept_id']
    assert cid == turn['question']['concept_id']
    old, new = before['concepts'][cid], after['concepts'][cid]
    assert new['observation_count'] == old['observation_count'] + 1
    assert new['evidence_ids'] == old['evidence_ids'] + [evidence['evidence_id']]
    duplicate = nlp['response_hash'] in old['response_hashes']
    meaningful = evidence['category'] != 'ambiguous' and evidence['confidence'] >= .55 and nlp['token_count'] >= 4
    weight = round(evidence['confidence'] * (.15 if duplicate else 1) * min(nlp['token_count'] / 16, 1), 6) if meaningful else 0
    coverage = len(met) / len(criteria)
    alpha = old['alpha'] + round(weight * coverage, 6)
    beta = old['beta'] + round(weight * (1 - coverage) * (.15 if evidence['category'] == 'slip' else 1), 6)
    assert abs(new['alpha'] - alpha) < 1e-9 and abs(new['beta'] - beta) < 1e-9
    assert new['estimate'] == round(alpha / (alpha + beta), 5)
    assert new['uncertainty'] == round(math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))), 5)
    assert turn['state_changes']['weight'] == weight
    assert turn['state_changes']['duplicate_response'] == duplicate
    for key in before['concepts']:
        if key != cid:
            assert before['concepts'][key] == after['concepts'][key]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--answers', type=int, help='Normal next-question submissions; two finish/resume submissions are added.')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/verification/extended-assessment')
    args = parser.parse_args()
    if args.answers is None:
        args.answers = 3 if args.live else 205
    if args.live and args.answers not in (2,3):
        parser.error('Live checks are capped at five questions total. Use --answers 2 or 3 (four or five total answers including finish/resume). Use fixture mode for long runs.')
    if not args.live and args.answers < 10:
        parser.error('Use at least ten answers for the fixture stress check.')
    from apps.api.app import main as api
    from packages.learner_state_core.db import LocalDatabaseManager
    from fastapi.staticfiles import StaticFiles
    from playwright.sync_api import sync_playwright, expect
    import uvicorn
    logging.getLogger('learner.pipeline').setLevel(logging.WARNING)

    if not args.live:
        from test_conversation_assessment import FixtureProvider
        class ExtendedProvider(FixtureProvider):
            def __init__(self):
                super().__init__()
                self.generated = 0

            def generate_question(self, context):
                if self.fail_questions:
                    raise TimeoutError('Injected generation outage for recovery verification')
                self.generated += 1
                result = super().generate_question(context)
                words = ('river forest window candle bicycle engine planet notebook lantern garden mountain camera '
                         'silver wooden orange purple copper paper smooth rough quiet bright narrow distant '
                         'panda tiger rabbit falcon otter dolphin robin turtle orchid willow maple cedar '
                         'bridge tunnel harbor station school library village meadow morning evening summer winter '
                         'walks waits turns reads writes draws counts groups splits joins keeps moves').split()
                sample = random.Random(self.generated).sample(words, 12)
                result['prompt'] = 'Analyze this text sample: "' + ' '.join(sample) + '". Explain how token boundaries and context affect its representation, using the source distinction.'
                return result

            def evaluate(self, context):
                result = super().evaluate(context)
                text = context['response_text']
                ids = [c['criterion_id'] for c in context['question']['criteria']]
                for prefix, category in [('Partial reasoning:', 'partial'), ('Calculation slip:', 'slip'), ('Prerequisite difficulty:', 'prerequisite_gap')]:
                    if text.startswith(prefix):
                        result.update(category=category, met_criteria=ids[:-1], missing_criteria=ids[-1:], contradicts_hypothesis_ids=[])
                if text.startswith('Low confidence:'):
                    result['confidence'] = .4
                return result
        api.assessment_provider = ExtendedProvider()
    assert api.assessment_provider.configured, 'Live model is not configured'
    live_question_calls = 0
    if args.live:
        generate_live = api.assessment_provider.generate_question
        def budgeted_generation(context):
            nonlocal live_question_calls
            if live_question_calls >= 5:
                raise RuntimeError('live_question_budget_exhausted')
            live_question_calls += 1
            return generate_live(context)
        api.assessment_provider.generate_question = budgeted_generation
    mode = 'live' if args.live else 'fixture'
    output = args.output / mode
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'mode':mode, 'synthetic_data':True, 'research_evaluation':False, 'requested_next_submissions':args.answers,
                'answers':[], 'reloads':[], 'checks':[], 'javascript_errors':[], 'model':api.assessment_provider.model_name}
    with tempfile.TemporaryDirectory(prefix='extended-assessment-') as temporary:
        api.db_manager = LocalDatabaseManager(str(Path(temporary) / 'test.db'))
        api.app.mount('/', StaticFiles(directory=ROOT / 'apps/web/dist', html=True), name='extended-ui')
        with socket.socket() as available:
            available.bind(('127.0.0.1', 0))
            port = available.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        server = uvicorn.Server(uvicorn.Config(api.app, host='127.0.0.1', port=port, log_level='warning'))
        worker = threading.Thread(target=server.run, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                assert time.monotonic() < deadline
                time.sleep(.05)
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(viewport={'width':1440,'height':1000}, reduced_motion='reduce', accept_downloads=True)
                learner = 'extended_' + mode
                context.add_init_script(f"localStorage.setItem('learner-state:guide:v2','seen');localStorage.setItem('learner-state:identity:v2','{learner}');")
                page = context.new_page()
                page.set_default_timeout(30000)
                expect.set_options(timeout=30000)
                page.on('pageerror', lambda error: manifest['javascript_errors'].append(str(error)))
                page.goto(base)
                page.get_by_label('Topic to assess').fill('Addition and its properties' if args.live else 'Tokenization')
                notes = ('Addition combines quantities into a total called the sum and is written with a plus sign. An addend is a number being added. Adding zero leaves a number unchanged. Addition of whole numbers is commutative: a+b=b+a. It is associative: (a+b)+c=a+(b+c). A missing addend can be found by subtracting the known addend from the sum. For example, 3+4=7 and 7-3=4.' if args.live else 'Tokenization divides text into spans while preserving boundaries. A token need not be a whole word. Context changes interpretation. Punctuation and numbers can be separate tokens; preserving order avoids merging unrelated units.')
                page.get_by_label('Your study notes').fill(notes)
                with page.expect_response(lambda r: r.request.method == 'POST' and r.url.endswith('/api/v2/conversations'), timeout=120000) as creation:
                    page.get_by_role('button', name='Start my assessment', exact=True).click()
                assert creation.value.status == 200
                session = creation.value.json()
                cid = session['conversation']['conversation_id']
                manifest['conversation_id'] = cid
                endpoint = base + f'/api/v2/conversations/{cid}'
                session_url = endpoint + f'?learner_id={learner}'
                expect(page.locator('#assessment-response')).to_be_visible(timeout=120000)
                known_questions, known_evidence, known_snapshots = set(), set(), {session['state']['snapshot_id']}
                known_prompts = set()
                immutable_turns = []

                def get_json(url):
                    # Playwright retains API response bodies until disposed;
                    # keeping hundreds of full-history reads exhausts the test
                    # driver. Retry only read-only transport resets, never an
                    # answer submission.
                    response = context.request.get(url, max_retries=2)
                    try:
                        assert response.ok, response.text()
                        return response.json()
                    finally:
                        response.dispose()

                def read_session():
                    return get_json(session_url)

                def reload_and_check(number, draft=''):
                    state, qid = session['state'], session['active_question']['question_id']
                    if draft:
                        page.get_by_label('Explain your thinking', exact=True).fill(draft)
                    page.reload()
                    expect(page.locator('#assessment-response')).to_be_visible(timeout=120000)
                    expect(page.locator('#assessment-response')).to_have_value(draft)
                    actual = read_session()
                    assert actual['state'] == state and actual['active_question']['question_id'] == qid
                    assert actual['conversation']['answered_count'] == number
                    manifest['reloads'].append({'after_answers':number, 'draft_preserved':bool(draft)})

                def inspect_saved(number):
                    state = session['state']
                    turn = session['turns'][-1]
                    page.get_by_role('button', name='NLP Observability', exact=True).click()
                    page.locator('.nlp-filters label').filter(has_text='Execution').locator('select').select_option('answer')
                    page.locator('.nlp-filters label').filter(has_text='Question ID').locator('input').fill(turn['question']['question_id'])
                    expect(page.locator('.nlp-execution-detail h2')).to_have_text('Response processing')
                    page.get_by_role('button', name='Text transformations', exact=True).click()
                    expect(page.locator('.nlp-record-source')).to_contain_text(turn['turn_id'])
                    assert page.locator('.nlp-inspector blockquote').first.inner_text() == turn['response_text']
                    page.get_by_role('button', name='Cognitive Evidence', exact=True).click()
                    assert page.locator('.nlp-inspector blockquote').first.inner_text() == turn['evidence']['source_quote']
                    page.get_by_role('button', name='State changes', exact=True).click()
                    expect(page.get_by_text('The transaction committed this state.', exact=True)).to_be_visible()
                    expect(page.locator('.nlp-state-table tbody tr')).to_have_count(len(state['concepts']))
                    with page.expect_download() as download:
                        page.get_by_role('button', name='Export complete record', exact=True).click()
                    record = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
                    assert record == turn
                    assert read_session()['state'] == state
                    page.screenshot(path=str(output / f'after-{number}-state.png'))
                    page.get_by_role('button', name='Learning workspace', exact=True).first.click()
                    if session['active_question']:
                        expect(page.locator('#assessment-response')).to_be_visible()
                    else:
                        expect(page.get_by_role('heading', name='This conversation is saved.', exact=True)).to_be_visible()
                    manifest['checks'].append(f'text/evidence/state UI and JSON export after {number} answers')

                def submit(text, finish=False):
                    old = session
                    qid = old['active_question']['question_id']
                    assert qid not in known_questions
                    page.get_by_label('Explain your thinking', exact=True).fill(text)
                    with page.expect_response(lambda r: r.request.method == 'POST' and r.url == endpoint + '/answers', timeout=180000) as pending:
                        page.get_by_role('button', name='Submit & finish' if finish else 'Submit & next question', exact=True).click()
                    response = pending.value
                    if response.status != 200:
                        page.screenshot(path=str(output / 'failure.png'))
                        manifest['failed_submission'] = {'number':len(old['turns'])+1, 'question':old['active_question'], 'response':text}
                        traces = get_json(base + f'/api/v2/observability/traces?learner_id={learner}&operation=answer&limit=1')
                        if traces['items']:
                            tid = traces['items'][0]['trace_id']
                            manifest['failed_execution'] = get_json(base + f'/api/v2/observability/traces/{tid}?learner_id={learner}')
                    assert response.status == 200, response.text()
                    # Chromium can evict large bodies from the test inspector
                    # cache even though the application reads them successfully.
                    # Long-run checks use a fresh persisted session read and an
                    # exact visible-question comparison, not that debug cache.
                    result = response.json() if args.live else read_session()
                    count = len(old['turns']) + 1
                    assert result['conversation']['answered_count'] == count and len(result['turns']) == count
                    assert [digest(t) for t in result['turns'][:-1]] == immutable_turns
                    turn = result['turns'][-1]
                    assert turn['response_text'] == text and turn['question']['question_id'] == qid
                    assert_turn(turn, old['state'], result['state'])
                    assert turn['evidence']['evidence_id'] not in known_evidence
                    assert result['state']['snapshot_id'] not in known_snapshots
                    known_questions.add(qid); known_evidence.add(turn['evidence']['evidence_id']); known_snapshots.add(result['state']['snapshot_id'])
                    known_prompts.add(turn['question']['prompt'])
                    immutable_turns.append(digest(turn))
                    with api.db_manager.connection() as conn:
                        persisted = json.loads(conn.execute('SELECT payload_json FROM assessment_answers WHERE turn_id=?', (turn['turn_id'],)).fetchone()[0])
                        snapshot = json.loads(conn.execute('SELECT payload_json FROM assessment_state_history WHERE snapshot_id=?', (result['state']['snapshot_id'],)).fetchone()[0])
                        assert conn.execute('SELECT count(*) FROM assessment_answers WHERE conversation_id=?', (cid,)).fetchone()[0] == count
                        assert persisted == turn and snapshot == result['state']
                        trace = [json.loads(r[0]) for r in conn.execute('SELECT payload_json FROM nlp_stages WHERE trace_id=? ORDER BY sequence', (turn['extraction_trace']['trace_id'],))]
                        assert next(s for s in trace if s['key']=='persistence')['output']['committed'] is True
                        assert next(s for s in trace if s['key']=='persistence')['output']['snapshot_id'] == snapshot['snapshot_id']
                        assert next(s for s in trace if s['key']=='evidence')['output']['evidence_id'] == turn['evidence']['evidence_id']
                    if finish:
                        assert result['conversation']['status'] == 'completed' and result['active_question'] is None
                        expect(page.locator('.conversation-report')).to_be_visible(timeout=120000)
                    else:
                        if result['active_question'] is None:
                            manifest['failed_generation'] = result['conversation']
                            manifest['last_saved_evidence'] = turn['evidence']
                            traces = get_json(base + f'/api/v2/observability/traces?learner_id={learner}&operation=generation&limit=1')
                            if traces['items']:
                                tid = traces['items'][0]['trace_id']
                                manifest['failed_execution'] = get_json(base + f'/api/v2/observability/traces/{tid}?learner_id={learner}')
                            page.screenshot(path=str(output / 'generation-failure.png'))
                        assert result['active_question'] is not None, result['conversation']
                        next_q = result['active_question']
                        assert next_q['question_id'] not in known_questions and next_q['prompt'] not in known_prompts
                        assert next_q['generation_trace']['state_snapshot_id'] == result['state']['snapshot_id']
                        expect(page.locator('#assessment-response')).to_have_value('', timeout=120000)
                        if not args.live:
                            expect(page.locator('#active-question-heading')).to_have_text(next_q['prompt'])
                        expect(page.get_by_role('button', name='Submit & next question', exact=True)).to_be_disabled()
                    manifest['answers'].append({'number':count, 'question_id':qid, 'turn_id':turn['turn_id'],
                        'evidence_id':turn['evidence']['evidence_id'], 'snapshot_id':result['state']['snapshot_id'],
                        'category':turn['evidence']['category'], 'weight':turn['state_changes']['weight'],
                        'duplicate':turn['state_changes']['duplicate_response'], 'validation_retries':len(turn['extraction_trace']['validation_attempts']),
                        'generation_attempts':result['active_question']['generation_trace']['attempts'] if result['active_question'] else None,
                        'generation_prompt_version':result['active_question']['generation_trace']['prompt_version'] if result['active_question'] else None,
                        'next_question_id':result['active_question']['question_id'] if result['active_question'] else None})
                    (output / 'progress.json').write_text(json.dumps({'mode':mode, 'status':'running', 'verified_answers':count}), encoding='utf-8')
                    return result

                def answer_text(number):
                    if args.live:
                        if number == 1:
                            return 'I am not sure how to explain this yet; maybe I need to review the meaning of addition.'
                        if number == 2:
                            return 'I think addition always multiplies the quantities: for example, adding 3 and 4 gives 12 because I multiply them.'
                        private = api.assessments().store.question(session['active_question']['question_id'])
                        return private['reference_answer'] + f' I used the stated relationship to reason through this case, number {number}. I reject the earlier multiplication rule: addition combines quantities, while multiplication is a different operation. For example, 3 + 4 is 7, whereas 3 times 4 is 12.'
                    samples = [
                        'I use a wrong rule because I believe every token must always be one entire word, so punctuation can never have its own token.',
                        'I use a wrong rule because I believe every token must always be one entire word, so punctuation can never have its own token.',
                        'I guess maybe tokens are words but I am not sure how the boundaries work.',
                        'Token boundaries preserve distinct pieces of text. A token need not equal a word; splitting punctuation separately retains useful context and preserves order.',
                        'Partial reasoning: A program can separate text into spans, but I have not connected the example to the boundary distinction.',
                        'Calculation slip: The conceptual boundary rule is to keep distinct pieces separate; I applied it but miscounted one punctuation token.',
                        'Low confidence: Tokenization identifies separate spans and preserves their order so surrounding context can affect interpretation.',
                        'Prerequisite difficulty: I have difficulty identifying separate written words, so I cannot yet justify splitting this sentence into spans.',
                        'Token boundaries and surrounding context matter. In "red, river", the comma is distinct punctuation; the spans preserve order and do not force every token to equal a word.',
                    ]
                    text = samples[(number - 1) % len(samples)]
                    return text if number in (1,2) else text + f' Verification example {number}.'

                print(f'{mode}: browser assessment started', flush=True)
                for number in range(1, args.answers + 1):
                    session = submit(answer_text(number))
                    if number in {1,3,10,30,100,160,200}:
                        reload_and_check(number, 'Unsent explanation survives reload.' if number in {3,30,160} else '')
                    if number in {3,10,30,100,args.answers}:
                        inspect_saved(number)
                    if number <= 3 or number % 10 == 0 or args.live:
                        print(f'{mode}: {number} answers saved; question {number + 1} visible; evidence/state verified', flush=True)
                    if number == 3:
                        state, qid = session['state'], session['active_question']['question_id']
                        page.get_by_role('button', name='Finish & view report', exact=True).click()
                        expect(page.locator('.conversation-report')).to_be_visible()
                        report = get_json(endpoint + f'/report?learner_id={learner}')
                        assert report['answered_count'] == 3 and report['ending_state'] == state
                        page.get_by_role('button', name='Continue this assessment', exact=True).click()
                        expect(page.locator('#assessment-response')).to_be_visible(timeout=120000)
                        session = read_session()
                        assert session['state'] == state and session['active_question']['question_id'] == qid
                        manifest['checks'].append('finish/report/resume after three answers has no automatic stop or state update')
                page.screenshot(path=str(output / 'long-session-question.png'))
                # Submit-and-finish, export, reload and continue after the long run.
                session = submit(answer_text(args.answers + 1), finish=True)
                with page.expect_download() as download:
                    page.get_by_role('button', name='Export JSON', exact=True).click()
                report = json.loads(Path(download.value.path()).read_text(encoding='utf-8'))
                assert report['answered_count'] == args.answers + 1 and len(report['timeline']) == args.answers + 1
                assert report['ending_state'] == session['state']
                page.screenshot(path=str(output / 'completed-report.png'))
                page.reload()
                expect(page.get_by_role('heading', name='This conversation is saved.', exact=True)).to_be_visible()
                page.get_by_role('button', name='Continue assessment', exact=True).click()
                expect(page.locator('#assessment-response')).to_be_visible(timeout=120000)
                session = read_session()
                assert session['state'] == report['ending_state']
                session = submit(answer_text(args.answers + 2), finish=args.live)
                inspect_saved(args.answers + 2)
                assert not manifest['javascript_errors'], manifest['javascript_errors']
                assert len(known_evidence) == args.answers + 2
                assert len(known_snapshots) == args.answers + 3
                with api.db_manager.connection() as conn:
                    manifest['database_counts'] = {
                        'answers':conn.execute('SELECT count(*) FROM assessment_answers WHERE conversation_id=?', (cid,)).fetchone()[0],
                        'questions':conn.execute('SELECT count(*) FROM assessment_questions WHERE conversation_id=?', (cid,)).fetchone()[0],
                        'snapshots':conn.execute('SELECT count(*) FROM assessment_state_history WHERE learner_id=?', (learner,)).fetchone()[0],
                        'retrievable_evidence':conn.execute("SELECT count(*) FROM assessment_chunks WHERE learner_id=? AND kind='learner_evidence'", (learner,)).fetchone()[0]}
                assert manifest['database_counts'] == {'answers':args.answers+2, 'questions':args.answers+2 if args.live else args.answers+3, 'snapshots':args.answers+3, 'retrievable_evidence':args.answers+2}
                categories = Counter(item['category'] for item in manifest['answers'])
                if not args.live:
                    assert set(categories) >= {'understanding','partial','misconception','prerequisite_gap','slip','ambiguous'}
                    assert any(item['duplicate'] for item in manifest['answers'])
                    assert any(item['weight']==0 for item in manifest['answers'])
                manifest.update(status='passed', answers_saved=len(session['turns']), categories=dict(categories),
                    next_question_visible=bool(session['active_question']), state_digest=digest(session['state']), unique_evidence=len(known_evidence), unique_snapshots=len(known_snapshots))
                manifest['checks'].extend(['every answer submitted through the browser', 'each quote and NLP span matches original response',
                    'each rubric partition uses only current criteria', 'independent state-formula checks',
                    'prior answer records remain immutable', 'one persisted answer and snapshot per submission',
                    'each next question uses its committed snapshot', 'draft and question survive reloads',
                    'submit-and-finish and full JSON report export', 'reload completed conversation and continue beyond long run',
                    'observability reads do not change memory', 'zero browser JavaScript errors'])
                browser.close()
        except Exception as error:
            manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if args.live:
                manifest['live_question_generation_calls'] = live_question_calls
            (output / 'progress.json').write_text(json.dumps({'mode':mode, 'status':manifest['status'], 'verified_answers':len(manifest['answers'])}), encoding='utf-8')
            (output / 'results.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            server.should_exit = True
            worker.join(timeout=5)
    print(json.dumps({k:manifest[k] for k in ['mode','status','answers_saved','categories','unique_evidence','unique_snapshots','javascript_errors']}), flush=True)


if __name__ == '__main__':
    main()
