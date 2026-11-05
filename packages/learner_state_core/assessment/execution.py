"""Independent, bounded execution journal. Telemetry failure never aborts learning."""
import contextvars
import functools
import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from .models import new_id
from .storage import now
from .execution_queries import TELEMETRY_STATUS_SQL, execution_filter, summarize

logger = logging.getLogger('learner.observability')
CURRENT = contextvars.ContextVar('assessment_execution', default=None)
REQUEST_ID = contextvars.ContextVar('http_request_id', default=None)
VALIDATION_CODES = {'wrong_concept', 'unverifiable_quote', 'invalid_rubric_partition', 'unsupported_ambiguity_claim',
                    'unsupported_misconception', 'incomplete_understanding', 'unknown_contradiction', 'unsupported_contradiction',
                    'unsupported_source', 'target_mismatch', 'unsupported_citations', 'duplicate_rubric_ids',
                    'missing_recheck_criteria', 'repeated_question', 'insufficient_source_material'}

# This registry describes implementations, never hypothetical textbook operations.
TECHNIQUES = {
    'input': ('Input validation', 'Pydantic request contract', 'assessment/models.py', 'Keep the original observation separate from generated material.', 'Reject invalid requests before interpretation; no text cleaning replaces the observation.'),
    'topic': ('Concept map construction', 'Model-assisted structured extraction', 'assessment/provider.py:build_topic', 'Define the assessment scope and prerequisite graph.', 'Generated relationships are unverified, not an expert curriculum. Graph identities and cycles are checked deterministically.'),
    'preprocessing': ('Text transformations', 'NFKC · regex tokens/sentences · lexical features', 'assessment/nlp.py:analyze', 'Expose original offsets, mathematical symbols, keywords, negation and uncertainty.', 'Lightweight domain-neutral spans preserve symbols and negation. POS/dependency models add no validated benefit here.'),
    'retrieval': ('Scoped source retrieval', 'SQLite FTS5 · Porter/unicode61 · BM25', 'assessment/storage.py:retrieve', 'Ground inference in saved source passages and historical evidence.', 'Lexical ranking is local and inspectable. Dense embeddings and TF-IDF are alternatives, not executed in this path.'),
    'planning': ('Diagnostic question planning', 'Deterministic criterion/prerequisite policy', 'assessment/state.py:plan_next', 'Choose a repair, hypothesis check or transfer objective from saved evidence.', 'Heuristic planning is not measured information gain or a validated learning intervention.'),
    'generation': ('Question formulation', 'Schema-constrained pretrained model', 'assessment/provider.py:generate_question', 'Formulate a fresh question using the saved plan and retrieved sources.', 'The model writes the task; deterministic policy owns its objective. Private rubrics are excluded from telemetry.'),
    'extraction': ('Cognitive interpretation', 'Schema-constrained pretrained model', 'assessment/provider.py:evaluate', 'Extract a quoted claim, rubric judgments and candidate difficulty.', 'Model interpretation handles varied language; confidence is uncalibrated and source validation does not prove semantic truth.'),
    'grounding': ('Concept grounding', 'Assigned-concept identity constraint', 'assessment/service.py:ground_observation', 'Associate the extracted claim with the exact saved question concept.', 'This constrains candidates; it is not open-world concept discovery or embedding similarity.'),
    'validation': ('Evidence validation', 'Pydantic · exact spans · rubric and reference checks', 'assessment/service.py:validate_observation', 'Reject fabricated quotations, invalid rubric partitions and unsupported contradictions.', 'Deterministic validation establishes provenance, not expert agreement or calibrated confidence.'),
    'evidence': ('Cognitive Evidence', 'Validated structured evidence assembly', 'assessment/service.py:answer', 'Keep source quote, model claim and candidate inference distinct with trace references.', 'An articulated misconception remains a revisable hypothesis, not a permanent learner label.'),
    'state_update': ('Learner-state computation', 'Criterion-weighted Beta update', 'assessment/state.py:integrate', 'Update evidence-supported estimates and weaken contradicted hypotheses.', 'Ambiguity holds numeric estimates; duplicates are damped. These weights require research calibration.'),
    'persistence': ('Atomic persistence', 'SQLite WAL transaction · optimistic version check', 'assessment/service.py', 'Commit an answer, evidence and immutable state snapshot together.', 'A computed state is not a saved state until the transaction commits.'),
}


def catalog():
    return [dict(key=k, name=v[0], technique=v[1], implementation='packages/learner_state_core/' + v[2],
                 purpose=v[3], justification=v[4]) for k, v in TECHNIQUES.items()]


def bounded(value, depth=0):
    """Minimize payloads and omit credentials/private question answers by key."""
    if depth > 12:
        return {'omitted': 'maximum depth'}
    if hasattr(value, 'model_dump'):
        value = value.model_dump(mode='json')
    if isinstance(value, dict):
        result = {}
        for k, v in list(value.items())[:160]:
            if any(word in k.lower() for word in ('api_key', 'authorization', 'password', 'secret', 'credential', 'endpoint')) or k in ('reference_answer', 'criteria'):
                continue
            result[k] = bounded(v, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        result = [bounded(v, depth + 1) for v in value[:160]]
        if len(value) > 160:
            result.append({'omitted_items': len(value) - 160})
        return result
    if isinstance(value, str):
        return value if len(value) <= 12000 else value[:12000] + '\n[truncated]'
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return type(value).__name__


def install_schema(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS nlp_executions (
        ordinal INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT UNIQUE NOT NULL,
        learner_id TEXT NOT NULL, conversation_id TEXT, question_id TEXT,
        operation TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
        completed_at TEXT, duration_ms REAL, metadata_json TEXT NOT NULL)''')
    conn.execute('''CREATE TABLE IF NOT EXISTS nlp_stages (
        stage_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES nlp_executions(trace_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL, payload_json TEXT NOT NULL, UNIQUE(trace_id, sequence))''')
    conn.execute('CREATE INDEX IF NOT EXISTS nlp_execution_scope ON nlp_executions(learner_id, conversation_id, ordinal)')
    conn.execute('CREATE INDEX IF NOT EXISTS nlp_execution_time ON nlp_executions(started_at)')


class ExecutionStore:
    def __init__(self, db):
        self.db = db

    def write(self, action):
        # Never borrow the learner transaction: failed writes must remain observable.
        try:
            with self.db.independent_connection() as conn:
                conn.execute('PRAGMA busy_timeout=150')
                action(conn)
            return True
        except Exception as error:
            logger.warning('execution_journal_unavailable error_type=%s', type(error).__name__)
            return False

    def prune(self):
        try:
            days = min(3650, max(1, int(os.getenv('NLP_TRACE_RETENTION_DAYS', '30'))))
        except ValueError:
            days = 30
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        # Includes abandoned executions; retention bounds raw trace content even after a crash.
        self.write(lambda c: c.execute('DELETE FROM nlp_executions WHERE started_at < ?', (cutoff,)))

    def list(self, learner_id, conversation_id=None, operation=None, status=None, question_id=None, limit=30, before=None, **filters):
        self.prune()
        where, values = execution_filter(learner_id, conversation_id, operation, status, question_id, **filters)
        if before is not None:
            where += ' AND e.ordinal < ?'
            values.append(before)
        with self.db.connection() as conn:
            rows = conn.execute('SELECT e.* FROM nlp_executions e WHERE ' + where + ' ORDER BY e.ordinal DESC LIMIT ?', (*values, limit + 1)).fetchall()
        items = [self.decode(r) for r in rows[:limit]]
        return {'items': items, 'has_more': len(rows) > limit, 'next_cursor': items[-1]['ordinal'] if len(rows) > limit else None}

    @staticmethod
    def decode(row):
        result = dict(row)
        result['metadata'] = json.loads(result.pop('metadata_json'))
        result['execution_id'] = result['trace_id']
        return result

    def get(self, learner_id, trace_id):
        self.prune()
        with self.db.connection() as conn:
            row = conn.execute(f'SELECT e.*, {TELEMETRY_STATUS_SQL} AS telemetry_status FROM nlp_executions e WHERE e.learner_id=? AND e.trace_id=?', (learner_id, trace_id)).fetchone()
            if row is None:
                return None
            result = self.decode(row)
            result['stages'] = [json.loads(r[0]) for r in conn.execute('SELECT payload_json FROM nlp_stages WHERE trace_id=? ORDER BY sequence', (trace_id,))]
        incomplete = result['telemetry_status'] == 'incomplete'
        result['incomplete_warning'] = ('No completion was recorded; execution may still be running or may have been interrupted.' if result['status'] == 'running'
            else 'Some execution events could not be recorded. Business status and saved evidence are separate from telemetry completeness.' if incomplete else None)
        return result

    def summary(self, learner_id, conversation_id=None, **filters):
        where, values = execution_filter(learner_id, conversation_id, **filters)
        with self.db.connection() as conn:
            return summarize(conn, where, values)

    def records(self, learner_id, conversation_id=None, limit=30, before=None):
        clauses, values = ['c.learner_id=?'], [learner_id]
        if conversation_id:
            clauses.append('c.conversation_id=?')
            values.append(conversation_id)
        if before is not None:
            clauses.append('a.rowid < ?')
            values.append(before)
        with self.db.connection() as conn:
            rows = conn.execute('SELECT a.rowid AS ordinal, a.turn_id, a.conversation_id, a.payload_json FROM assessment_answers a JOIN assessment_conversations c ON c.conversation_id=a.conversation_id WHERE ' + ' AND '.join(clauses) + ' ORDER BY a.rowid DESC LIMIT ?', (*values, limit + 1)).fetchall()
        items = []
        for row in rows[:limit]:
            turn = json.loads(row['payload_json'])
            items.append(dict(ordinal=row['ordinal'], turn_id=row['turn_id'], conversation_id=row['conversation_id'],
                              timestamp=turn['timestamp'], question_id=turn['question']['question_id'],
                              question=turn['question']['prompt'], trace_id=turn['extraction_trace'].get('trace_id'),
                              evidence_id=turn['evidence']['evidence_id'], snapshot_id=turn['state_after']['snapshot_id']))
        return {'items': items, 'has_more': len(rows) > limit, 'next_cursor': items[-1]['ordinal'] if len(rows) > limit else None}

    def record(self, learner_id, turn_id):
        with self.db.connection() as conn:
            row = conn.execute('SELECT a.payload_json FROM assessment_answers a JOIN assessment_conversations c ON c.conversation_id=a.conversation_id WHERE a.turn_id=? AND c.learner_id=?', (turn_id, learner_id)).fetchone()
        return json.loads(row[0]) if row else None


class Execution:
    def __init__(self, db, learner_id, operation, conversation_id=None, question_id=None):
        self.store = ExecutionStore(db)
        self.trace_id, self.learner_id = new_id('nlp'), learner_id
        self.operation, self.conversation_id, self.question_id = operation, conversation_id, question_id
        self.started, self.started_at = time.perf_counter(), now()
        self.sequence, self.last_stage, self.active_stage = 0, None, None
        parent = CURRENT.get()
        self.metadata = {'parent_trace_id': parent.trace_id if parent else None, 'confidence_calibrated': False,
                         'http_request_id': REQUEST_ID.get(), 'telemetry_write_failures': 0}
        self.status = 'running'
        self.store.prune()
        self.write(lambda c: c.execute('INSERT INTO nlp_executions(trace_id,learner_id,conversation_id,question_id,operation,status,started_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)',
            (self.trace_id, learner_id, conversation_id, question_id, operation, self.status, self.started_at, json.dumps(self.metadata))))

    def write(self, action):
        if not self.store.write(action):
            self.metadata['telemetry_write_failures'] += 1

    def finish(self, error=None):
        status = 'failed' if error or self.status == 'failed' else 'completed'
        if error:
            self.metadata['error'] = {'type': type(error).__name__, 'code': 'execution_failed', 'http_status': getattr(error, 'status_code', None)}
        duration = round((time.perf_counter() - self.started) * 1000, 3)
        self.metadata['expected_stage_count'] = self.sequence
        self.write(lambda c: c.execute('UPDATE nlp_executions SET status=?,conversation_id=?,question_id=?,completed_at=?,duration_ms=?,metadata_json=? WHERE trace_id=?',
            (status, self.conversation_id, self.question_id, now(), duration, json.dumps(bounded(self.metadata)), self.trace_id)))

    @contextmanager
    def stage(self, key, inputs=None, parameters=None, attempt=1):
        self.sequence += 1
        info = TECHNIQUES[key]
        payload = dict(stage_id=new_id('stage'), trace_id=self.trace_id, sequence=self.sequence, key=key, name=info[0],
                       parent_stage_id=self.active_stage, previous_stage_id=self.last_stage, technique=info[1],
                       implementation='packages/learner_state_core/' + info[2], input=bounded(inputs), parameters=bounded(parameters or {}),
                       status='running', attempt=attempt, started_at=now(), completed_at=None, duration_ms=None, output=None, error=None)
        def save():
            encoded = json.dumps(payload, ensure_ascii=False)
            if len(encoded.encode()) > 192000:
                payload['output'] = {'omitted': 'payload exceeds 192 KB; inspect the linked saved record', 'record_reference': self.metadata.get('turn_id')}
                payload['input'] = {'omitted': 'payload exceeds 192 KB; inspect the linked saved record'}
                encoded = json.dumps(payload, ensure_ascii=False)
            self.write(lambda c: c.execute('INSERT INTO nlp_stages VALUES(?,?,?,?) ON CONFLICT(stage_id) DO UPDATE SET payload_json=excluded.payload_json',
                                               (payload['stage_id'], self.trace_id, payload['sequence'], encoded)))
        save()
        started, parent = time.perf_counter(), self.active_stage
        self.active_stage = payload['stage_id']
        try:
            yield payload
            payload['status'] = 'completed'
        except Exception as error:
            payload['status'] = 'failed'
            # Provider and validation exception messages can contain private payloads.
            code = str(error) if isinstance(error, ValueError) and str(error) in VALIDATION_CODES else 'stage_failed'
            payload['error'] = {'type': type(error).__name__, 'code': code}
            raise
        finally:
            payload['output'] = bounded(payload['output'])
            payload['completed_at'] = now()
            payload['duration_ms'] = round((time.perf_counter() - started) * 1000, 3)
            self.active_stage, self.last_stage = parent, payload['stage_id']
            save()


@contextmanager
def stage(key, inputs=None, parameters=None, attempt=1):
    execution = CURRENT.get()
    if execution is None:
        yield {'output': None}
    else:
        with execution.stage(key, inputs, parameters, attempt) as payload:
            yield payload


def call(key, function, inputs=None, parameters=None, attempt=1):
    with stage(key, inputs, parameters, attempt) as span:
        result = function()
        span['output'] = result
        return result


def traced(operation):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(self, *args, **kwargs):
            if operation == 'create':
                req = args[0]
                learner, cid, qid = req.learner_id, None, None
            elif operation == 'answer':
                cid, req = args
                learner, qid = req.learner_id, req.question_id
            else:
                cid, learner = args
                qid = None
            execution = Execution(self.db, learner, operation, cid, qid)
            if operation in ('create', 'answer'):
                execution.metadata['request_id'] = req.request_id
            token, error = CURRENT.set(execution), None
            try:
                result = function(self, *args, **kwargs)
                if isinstance(result, dict) and 'conversation' in result:
                    execution.conversation_id = result['conversation']['conversation_id']
                    if operation == 'create' and result['conversation']['status'] == 'needs_question':
                        execution.status = 'failed'
                        execution.metadata['failure_code'] = result['conversation'].get('last_error')
                return result
            except Exception as caught:
                error = caught
                if operation == 'answer':
                    try:
                        self.store.fail_submission(execution.trace_id, type(caught).__name__)
                    except Exception as journal_error:
                        logger.warning('submission_status_unavailable error_type=%s', type(journal_error).__name__)
                raise
            finally:
                CURRENT.reset(token)
                execution.finish(error)
        return wrapped
    return decorate
