"""Independent, bounded execution journal. Telemetry failure never aborts learning."""
# Keep active execution state isolated between concurrent requests.
import contextvars
# Preserve the wrapped service methods' metadata.
import functools
# Serialize trace payloads for SQLite.
import json
# Report journal write failures without stopping the learner flow.
import logging
# Read the configured trace retention period.
import os
# Measure elapsed durations with a monotonic clock.
import time
# Record stage outcomes when their context-managed blocks exit.
from contextlib import contextmanager
# Calculate the UTC retention cutoff.
from datetime import datetime, timedelta, timezone

# Generate unique execution and stage identifiers.
from .models import new_id
# Use the assessment timestamp format in journal rows.
from .storage import now
# Share trace filtering and completeness formulas across result views.
from .execution_queries import TELEMETRY_STATUS_SQL, execution_filter, summarize

# Send telemetry-specific messages to the observability logger.
logger = logging.getLogger('learner.observability')
# Store the current trace for stages inside the same request.
CURRENT = contextvars.ContextVar('assessment_execution', default=None)
# Carry the HTTP request ID into the trace without adding it to every method argument.
REQUEST_ID = contextvars.ContextVar('http_request_id', default=None)
# Permit specific evidence validation codes in a recorded error.
VALIDATION_CODES = {'wrong_concept', 'unverifiable_quote', 'invalid_rubric_partition', 'unsupported_ambiguity_claim',
                    # Include category and hypothesis consistency checks among the safe codes.
                    'unsupported_misconception', 'incomplete_understanding', 'unknown_contradiction', 'unsupported_contradiction',
                    # Include source and generated-rubric validation codes.
                    'unsupported_source', 'target_mismatch', 'unsupported_citations', 'duplicate_rubric_ids',
                    # Include target rechecks, repeated questions and insufficient material.
                    'missing_recheck_criteria', 'repeated_question', 'insufficient_source_material'}

# This registry describes implementations, never hypothetical textbook operations.
# Describe each implemented stage and link the UI to its Python implementation.
TECHNIQUES = {
    # Document the boundary between raw learner input and generated content.
    'input': ('Input validation', 'Pydantic request contract', 'assessment/models.py', 'Keep the original observation separate from generated material.', 'Reject invalid requests before interpretation; no text cleaning replaces the observation.'),
    # Identify the model-assisted topic map and server-side graph checks.
    'topic': ('Concept map construction', 'Model-assisted structured extraction', 'assessment/provider.py:build_topic', 'Define the assessment scope and prerequisite graph.', 'Generated relationships are unverified, not an expert curriculum. Graph identities and cycles are checked deterministically.'),
    # Explain which text transformations are actually applied.
    'preprocessing': ('Text transformations', 'NFKC · regex tokens/sentences · lexical features', 'assessment/nlp.py:analyze', 'Expose original offsets, mathematical symbols, keywords, negation and uncertainty.', 'Lightweight domain-neutral spans preserve symbols and negation. POS/dependency models add no validated benefit here.'),
    # Identify the specific search index and ranking used in live assessment.
    'retrieval': ('Scoped source retrieval', 'SQLite FTS5 · Porter/unicode61 · BM25', 'assessment/storage.py:retrieve', 'Ground inference in saved source passages and historical evidence.', 'Lexical ranking is local and inspectable. Dense embeddings and TF-IDF are alternatives, not executed in this path.'),
    # Identify the Python policy that chooses the next diagnostic objective.
    'planning': ('Diagnostic question planning', 'Deterministic criterion/prerequisite policy', 'assessment/state.py:plan_next', 'Choose a repair, hypothesis check or transfer objective from saved evidence.', 'Heuristic planning is not measured information gain or a validated learning intervention.'),
    # Identify formulation as the model's job after planning.
    'generation': ('Question formulation', 'Schema-constrained pretrained model', 'assessment/provider.py:generate_question', 'Formulate a fresh question using the saved plan and retrieved sources.', 'The model writes the task; deterministic policy owns its objective. Private rubrics are excluded from telemetry.'),
    # Identify the provider's scoped evidence interpretation.
    'extraction': ('Cognitive interpretation', 'Schema-constrained pretrained model', 'assessment/provider.py:evaluate', 'Extract a quoted claim, rubric judgments and candidate difficulty.', 'Model interpretation handles varied language; confidence is uncalibrated and source validation does not prove semantic truth.'),
    # Identify the exact concept constraint rather than an approximate similarity search.
    'grounding': ('Concept grounding', 'Assigned-concept identity constraint', 'assessment/service.py:ground_observation', 'Associate the extracted claim with the exact saved question concept.', 'This constrains candidates; it is not open-world concept discovery or embedding similarity.'),
    # Summarize the evidence acceptance checks and what they establish.
    'validation': ('Evidence validation', 'Pydantic · exact spans · rubric and reference checks', 'assessment/service.py:validate_observation', 'Reject fabricated quotations, invalid rubric partitions and unsupported contradictions.', 'Deterministic validation establishes provenance, not expert agreement or calibrated confidence.'),
    # Link the evidence assembly step to the answer service.
    'evidence': ('Cognitive Evidence', 'Validated structured evidence assembly', 'assessment/service.py:answer', 'Keep source quote, model claim and candidate inference distinct with trace references.', 'An articulated misconception remains a revisable hypothesis, not a permanent learner label.'),
    # Link trace calculations to the deterministic state policy.
    'state_update': ('Learner-state computation', 'Criterion-weighted Beta update', 'assessment/state.py:integrate', 'Update evidence-supported estimates and weaken contradicted hypotheses.', 'Ambiguity holds numeric estimates; duplicates are damped. These weights require research calibration.'),
    # Identify when the validated update becomes durable.
    'persistence': ('Atomic persistence', 'SQLite WAL transaction · optimistic version check', 'assessment/service.py', 'Commit an answer, evidence and immutable state snapshot together.', 'A computed state is not a saved state until the transaction commits.'),
}


# Expose the catalog of real implementation stages to the workspace.
def catalog():
    # Return stage names, methods and clickable implementation paths for inspection.
    return [dict(key=k, name=v[0], technique=v[1], implementation='packages/learner_state_core/' + v[2],
                 # Include the stated purpose and limits of each registered technique.
                 purpose=v[3], justification=v[4]) for k, v in TECHNIQUES.items()]


# Redact private fields and limit the size of trace values recursively.
def bounded(value, depth=0):
    """Minimize payloads and omit credentials/private question answers by key."""
    # Stop recursive serialization at the configured nesting limit.
    if depth > 12:
        # Describe the omitted nested content.
        return {'omitted': 'maximum depth'}
    # Convert Pydantic objects to their serializable representation.
    if hasattr(value, 'model_dump'):
        # Dump models as JSON-compatible field values.
        value = value.model_dump(mode='json')
    # Walk mappings so sensitive fields can be excluded by key.
    if isinstance(value, dict):
        # Start a filtered copy of the original mapping.
        result = {}
        # Limit the number of fields inspected in one mapping.
        for k, v in list(value.items())[:160]:
            # Exclude credential-like fields and the question's private answer and rubric.
            if any(word in k.lower() for word in ('api_key', 'authorization', 'password', 'secret', 'credential', 'endpoint')) or k in ('reference_answer', 'criteria'):
                # Skip the excluded field entirely.
                continue
            # Apply the same rules to nested values.
            result[k] = bounded(v, depth + 1)
        # Return the filtered mapping.
        return result
    # Limit long collections so traces remain inspectable.
    if isinstance(value, (list, tuple)):
        # Keep up to 160 entries and filter each one.
        result = [bounded(v, depth + 1) for v in value[:160]]
        # Indicate whether further entries were omitted.
        if len(value) > 160:
            # Record how many entries are outside the trace limit.
            result.append({'omitted_items': len(value) - 160})
        # Return the bounded collection.
        return result
    # Apply a per-value size limit to strings.
    if isinstance(value, str):
        # Keep up to 12,000 characters and mark a truncated value.
        return value if len(value) <= 12000 else value[:12000] + '\n[truncated]'
    # Keep JSON-compatible scalar values directly.
    if value is None or isinstance(value, (int, float, bool)):
        # Return the original scalar.
        return value
    # Use the object's type name when the value has no supported representation.
    return type(value).__name__


# Create the tables for live execution summaries and individual stages.
def install_schema(conn):
    # Save one summary row per execution with an ordered cursor and current status.
    conn.execute('''CREATE TABLE IF NOT EXISTS nlp_executions (
        ordinal INTEGER PRIMARY KEY AUTOINCREMENT, trace_id TEXT UNIQUE NOT NULL,
        learner_id TEXT NOT NULL, conversation_id TEXT, question_id TEXT,
        operation TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
        completed_at TEXT, duration_ms REAL, metadata_json TEXT NOT NULL)''')
    # Save independently updatable stage rows that are removed with the parent execution.
    conn.execute('''CREATE TABLE IF NOT EXISTS nlp_stages (
        stage_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES nlp_executions(trace_id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL, payload_json TEXT NOT NULL, UNIQUE(trace_id, sequence))''')
    # Index owner and conversation for paged trace lists.
    conn.execute('CREATE INDEX IF NOT EXISTS nlp_execution_scope ON nlp_executions(learner_id, conversation_id, ordinal)')
    # Index the start time for date filtering and retention.
    conn.execute('CREATE INDEX IF NOT EXISTS nlp_execution_time ON nlp_executions(started_at)')


# Keep trace persistence and reads separate from accepted learner records.
class ExecutionStore:
    # Inject the shared database infrastructure.
    def __init__(self, db):
        # Keep access to an independent SQLite connection for journal writes.
        self.db = db

    # Attempt journal writes without letting a telemetry failure cancel learning.
    def write(self, action):
        # Never borrow the learner transaction: failed writes must remain observable.
        # Allow any failed attempt to return a safe failure indicator.
        try:
            # Open a connection that is independent of the accepted answer transaction.
            with self.db.independent_connection() as conn:
                # Bound how long a telemetry write waits for SQLite.
                conn.execute('PRAGMA busy_timeout=150')
                # Apply the provided journal insertion, update or cleanup.
                action(conn)
            # Return success only after the connection exits normally.
            return True
        # Catch a telemetry storage failure.
        except Exception as error:
            # Log only the safe exception type.
            logger.warning('execution_journal_unavailable error_type=%s', type(error).__name__)
            # Let the calling execution count the missed write.
            return False

    # Remove expired raw executions using the configured retention window.
    def prune(self):
        # Read retention settings with a safe fallback.
        try:
            # Bound the configured number of retention days and default to 30.
            days = min(3650, max(1, int(os.getenv('NLP_TRACE_RETENTION_DAYS', '30'))))
        # Handle an invalid configured value.
        except ValueError:
            # Use the default retention when the configuration cannot be parsed.
            days = 30
        # Derive the expiry cutoff from the current UTC time.
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        # Includes abandoned executions; retention bounds raw trace content even after a crash.
        # Remove old executions, which also removes their associated stage rows.
        self.write(lambda c: c.execute('DELETE FROM nlp_executions WHERE started_at < ?', (cutoff,)))

    # List trace summaries within the requested learner scope.
    def list(self, learner_id, conversation_id=None, operation=None, status=None, question_id=None, limit=30, before=None, **filters):
        # Apply retention before serving the trace list.
        self.prune()
        # Build the same search conditions used by the aggregate summary.
        where, values = execution_filter(learner_id, conversation_id, operation, status, question_id, **filters)
        # Apply the page cursor only to the list.
        if before is not None:
            # Limit this page to executions older than the cursor.
            where += ' AND e.ordinal < ?'
            # Bind the cursor value separately from the query.
            values.append(before)
        # Read the trace page from the shared database.
        with self.db.connection() as conn:
            # Read one extra row to tell whether another page exists.
            rows = conn.execute('SELECT e.* FROM nlp_executions e WHERE ' + where + ' ORDER BY e.ordinal DESC LIMIT ?', (*values, limit + 1)).fetchall()
        # Decode up to the requested limit, excluding the extra paging marker.
        items = [self.decode(r) for r in rows[:limit]]
        # Return the page and its next cursor.
        return {'items': items, 'has_more': len(rows) > limit, 'next_cursor': items[-1]['ordinal'] if len(rows) > limit else None}

    # The decoder has no dependency on store configuration.
    @staticmethod
    # Convert a saved trace row into the shape used by the inspection UI.
    def decode(row):
        # Copy the SQL row into an ordinary mapping.
        result = dict(row)
        # Decode the metadata JSON and expose it as structured data.
        result['metadata'] = json.loads(result.pop('metadata_json'))
        # Expose the stable trace ID under the name the UI uses for an execution.
        result['execution_id'] = result['trace_id']
        # Return the normalized execution summary.
        return result

    # Load an execution plus its recorded stages by trace and learner ID.
    def get(self, learner_id, trace_id):
        # Apply retention before trying to read execution detail.
        self.prune()
        # Use the same read connection for the summary and stages.
        with self.db.connection() as conn:
            # Compute telemetry completeness from stored status and stage counts.
            row = conn.execute(f'SELECT e.*, {TELEMETRY_STATUS_SQL} AS telemetry_status FROM nlp_executions e WHERE e.learner_id=? AND e.trace_id=?', (learner_id, trace_id)).fetchone()
            # Stop when there was no trace within the requested learner scope.
            if row is None:
                # Return no record so the API can report a scoped 404.
                return None
            # Decode the persistent trace record.
            result = self.decode(row)
            # Return stage records in the order they were started.
            result['stages'] = [json.loads(r[0]) for r in conn.execute('SELECT payload_json FROM nlp_stages WHERE trace_id=? ORDER BY sequence', (trace_id,))]
        # Distinguish incomplete telemetry from the business outcome of the operation.
        incomplete = result['telemetry_status'] == 'incomplete'
        # Explain that a running trace may still be collecting events.
        result['incomplete_warning'] = ('No completion was recorded; execution may still be running or may have been interrupted.' if result['status'] == 'running'
            # Use a different explanation for a completed operation with missing telemetry.
            else 'Some execution events could not be recorded. Business status and saved evidence are separate from telemetry completeness.' if incomplete else None)
        # Return both business status and telemetry status.
        return result

    # Summarize the entire filtered set of traces.
    def summary(self, learner_id, conversation_id=None, **filters):
        # Apply the same filters used in the trace list.
        where, values = execution_filter(learner_id, conversation_id, **filters)
        # Use a single connection for the aggregate queries.
        with self.db.connection() as conn:
            # Calculate aggregate metrics with the same predicates as the list.
            return summarize(conn, where, values)

    # List accepted assessment records from the long-lived answer table.
    def records(self, learner_id, conversation_id=None, limit=30, before=None):
        # Begin with mandatory learner scope.
        clauses, values = ['c.learner_id=?'], [learner_id]
        # Apply an optional conversation restriction.
        if conversation_id:
            # Add its predicate to the list query.
            clauses.append('c.conversation_id=?')
            # Bind the requested conversation ID.
            values.append(conversation_id)
        # Apply older-page filtering when a cursor is supplied.
        if before is not None:
            # Limit results to rows before that response ordinal.
            clauses.append('a.rowid < ?')
            # Bind the cursor to the saved-record query.
            values.append(before)
        # Read from the durable answer table.
        with self.db.connection() as conn:
            # Read accepted turns in learner scope even if their live traces have expired.
            rows = conn.execute('SELECT a.rowid AS ordinal, a.turn_id, a.conversation_id, a.payload_json FROM assessment_answers a JOIN assessment_conversations c ON c.conversation_id=a.conversation_id WHERE ' + ' AND '.join(clauses) + ' ORDER BY a.rowid DESC LIMIT ?', (*values, limit + 1)).fetchall()
        # Start the compact record list for this page.
        items = []
        # Walk the requested rows and leave the sentinel row for paging.
        for row in rows[:limit]:
            # Decode the accepted turn JSON.
            turn = json.loads(row['payload_json'])
            # Expose compact identifiers and display text for the saved-response list.
            items.append(dict(ordinal=row['ordinal'], turn_id=row['turn_id'], conversation_id=row['conversation_id'],
                              # Attach the answer time and saved question ID.
                              timestamp=turn['timestamp'], question_id=turn['question']['question_id'],
                              # Include the prompt and its trace ID when one was recorded.
                              question=turn['question']['prompt'], trace_id=turn['extraction_trace'].get('trace_id'),
                              # Include evidence and the committed state snapshot for lookup.
                              evidence_id=turn['evidence']['evidence_id'], snapshot_id=turn['state_after']['snapshot_id']))
        # Return the page and ordinal of its last included item when more remain.
        return {'items': items, 'has_more': len(rows) > limit, 'next_cursor': items[-1]['ordinal'] if len(rows) > limit else None}

    # Look up the full immutable accepted record for the detail view.
    def record(self, learner_id, turn_id):
        # Use a read connection for exact turn lookup.
        with self.db.connection() as conn:
            # Constrain the answer lookup by the learner who owns the conversation.
            row = conn.execute('SELECT a.payload_json FROM assessment_answers a JOIN assessment_conversations c ON c.conversation_id=a.conversation_id WHERE a.turn_id=? AND c.learner_id=?', (turn_id, learner_id)).fetchone()
        # Decode the accepted turn if it belongs to the requested learner.
        return json.loads(row[0]) if row else None


# Represent one execution of the create, answer or generation workflow.
class Execution:
    # Set up the initial trace with its operation context.
    def __init__(self, db, learner_id, operation, conversation_id=None, question_id=None):
        # Bind journal writes to the provided database.
        self.store = ExecutionStore(db)
        # Create a trace ID and associate it with the current learner.
        self.trace_id, self.learner_id = new_id('nlp'), learner_id
        # Retain the operation and any conversation and question IDs already known.
        self.operation, self.conversation_id, self.question_id = operation, conversation_id, question_id
        # Measure durations with a monotonic clock and store the wall-clock start separately.
        self.started, self.started_at = time.perf_counter(), now()
        # Initialize the stage sequence and parent links.
        self.sequence, self.last_stage, self.active_stage = 0, None, None
        # Check if this operation was called inside another traced operation.
        parent = CURRENT.get()
        # Record links to the parent trace and originating HTTP request.
        self.metadata = {'parent_trace_id': parent.trace_id if parent else None, 'confidence_calibrated': False,
                         # Track failed journal writes separately from inference or storage
                         # outcomes.
                         'http_request_id': REQUEST_ID.get(), 'telemetry_write_failures': 0}
        # Record the execution as running until the wrapped operation exits.
        self.status = 'running'
        # Apply retention when recording a new execution.
        self.store.prune()
        # Save the trace summary row before any stages begin.
        self.write(lambda c: c.execute('INSERT INTO nlp_executions(trace_id,learner_id,conversation_id,question_id,operation,status,started_at,metadata_json) VALUES(?,?,?,?,?,?,?,?)',
            # Bind the caller's identifiers and metadata as separate SQL values.
            (self.trace_id, learner_id, conversation_id, question_id, operation, self.status, self.started_at, json.dumps(self.metadata))))

    # Send a specific write through the best-effort journal.
    def write(self, action):
        # Detect whether the record could be written.
        if not self.store.write(action):
            # Count write failures so completeness can be assessed later.
            self.metadata['telemetry_write_failures'] += 1

    # Close the top-level execution summary with its outcome.
    def finish(self, error=None):
        # Mark errors and explicitly failed generation as failures.
        status = 'failed' if error or self.status == 'failed' else 'completed'
        # Record a safe error summary when the wrapped operation raised.
        if error:
            # Keep the exception type and HTTP status without including private payloads.
            self.metadata['error'] = {'type': type(error).__name__, 'code': 'execution_failed', 'http_status': getattr(error, 'status_code', None)}
        # Measure how long the entire operation took in milliseconds.
        duration = round((time.perf_counter() - self.started) * 1000, 3)
        # Save how many stages should exist even if a journal write was missed.
        self.metadata['expected_stage_count'] = self.sequence
        # Update the persisted execution record with status and final IDs.
        self.write(lambda c: c.execute('UPDATE nlp_executions SET status=?,conversation_id=?,question_id=?,completed_at=?,duration_ms=?,metadata_json=? WHERE trace_id=?',
            # Store the completion timestamp, elapsed duration and bounded metadata.
            (status, self.conversation_id, self.question_id, now(), duration, json.dumps(bounded(self.metadata)), self.trace_id)))

    # Make stage start and finish follow the lifetime of a Python context block.
    @contextmanager
    # Record a named operation, its attempt number, inputs and result.
    def stage(self, key, inputs=None, parameters=None, attempt=1):
        # Give each stage a sequence number within the execution.
        self.sequence += 1
        # Load the descriptive label and code reference from the technique catalog.
        info = TECHNIQUES[key]
        # Build the stage record with unique IDs and its position in the trace.
        payload = dict(stage_id=new_id('stage'), trace_id=self.trace_id, sequence=self.sequence, key=key, name=info[0],
                       # Link the parent and the previously completed stage.
                       parent_stage_id=self.active_stage, previous_stage_id=self.last_stage, technique=info[1],
                       # Bound the captured inputs and parameters and identify the implementation.
                       implementation='packages/learner_state_core/' + info[2], input=bounded(inputs), parameters=bounded(parameters or {}),
                       # Initially expose the stage as running without an outcome.
                       status='running', attempt=attempt, started_at=now(), completed_at=None, duration_ms=None, output=None, error=None)
        # Define how to persist the current version of this stage.
        def save():
            # Preserve Unicode when serializing the trace payload.
            encoded = json.dumps(payload, ensure_ascii=False)
            # Check the total serialized size before storing it.
            if len(encoded.encode()) > 192000:
                # Replace oversized output with a link to the accepted record when available.
                payload['output'] = {'omitted': 'payload exceeds 192 KB; inspect the linked saved record', 'record_reference': self.metadata.get('turn_id')}
                # Replace oversized input with a short inspection message.
                payload['input'] = {'omitted': 'payload exceeds 192 KB; inspect the linked saved record'}
                # Serialize the smaller payload.
                encoded = json.dumps(payload, ensure_ascii=False)
            # Insert the new stage or update it when its outcome is known.
            self.write(lambda c: c.execute('INSERT INTO nlp_stages VALUES(?,?,?,?) ON CONFLICT(stage_id) DO UPDATE SET payload_json=excluded.payload_json',
                                               # Bind the identifiers, sequence and encoded payload.
                                               (payload['stage_id'], self.trace_id, payload['sequence'], encoded)))
        # Persist the start so live polling can display a running stage.
        save()
        # Capture the timing start and previous parent pointer.
        started, parent = time.perf_counter(), self.active_stage
        # Make this stage the parent of any inner stages.
        self.active_stage = payload['stage_id']
        # Journal the outcome whether the enclosed work returns or raises.
        try:
            # Let the caller fill in the output as it runs the stage.
            yield payload
            # Mark normal completion if the caller exits without an exception.
            payload['status'] = 'completed'
        # Capture failures from inside the stage.
        except Exception as error:
            # Mark this individual stage as failed.
            payload['status'] = 'failed'
            # Provider and validation exception messages can contain private payloads.
            # Retain only recognized validation reasons or a generic safe code.
            code = str(error) if isinstance(error, ValueError) and str(error) in VALIDATION_CODES else 'stage_failed'
            # Include the exception type for troubleshooting.
            payload['error'] = {'type': type(error).__name__, 'code': code}
            # Let the business workflow handle the original error.
            raise
        # Finish the record for successful and failed stages.
        finally:
            # Redact and limit the captured output.
            payload['output'] = bounded(payload['output'])
            # Record the time when the stage ended.
            payload['completed_at'] = now()
            # Calculate the stage's elapsed time in milliseconds.
            payload['duration_ms'] = round((time.perf_counter() - started) * 1000, 3)
            # Restore the parent pointer and remember this completed stage.
            self.active_stage, self.last_stage = parent, payload['stage_id']
            # Save the completed stage record.
            save()


# Offer the same context-manager interface with or without tracing enabled.
@contextmanager
# Select the current execution automatically for a service stage.
def stage(key, inputs=None, parameters=None, attempt=1):
    # Look up the execution local to this request or nested operation.
    execution = CURRENT.get()
    # Tests or helper calls can run without an active trace.
    if execution is None:
        # Yield an ordinary output container in that case.
        yield {'output': None}
    # Route through the real recorder when an execution exists.
    else:
        # Use its named stage with the current attempt number.
        with execution.stage(key, inputs, parameters, attempt) as payload:
            # Give that recorded payload back to the caller.
            yield payload


# Wrap a single function call when its returned value is the trace output.
def call(key, function, inputs=None, parameters=None, attempt=1):
    # Start the named trace stage.
    with stage(key, inputs, parameters, attempt) as span:
        # Run the real processing function.
        result = function()
        # Capture what it returned.
        span['output'] = result
        # Return the same value to the service.
        return result


# Decorate the service's create, generation and answer entry points with tracing.
def traced(operation):
    # Receive the actual method being wrapped.
    def decorate(function):
        # Preserve the method's metadata.
        @functools.wraps(function)
        # Create one tracked execution around each service call.
        def wrapped(self, *args, **kwargs):
            # Creating a conversation starts before a conversation ID is assigned.
            if operation == 'create':
                # Read the request envelope from the method arguments.
                req = args[0]
                # Start the trace with the learner ID and empty conversation and question IDs.
                learner, cid, qid = req.learner_id, None, None
            # An answer already refers to a particular question.
            elif operation == 'answer':
                # Read the conversation ID and request.
                cid, req = args
                # Record its explicit learner and question IDs.
                learner, qid = req.learner_id, req.question_id
            # A generation call already has the conversation and learner arguments.
            else:
                # Take those IDs from the wrapped method.
                cid, learner = args
                # Leave the question ID empty until generation succeeds.
                qid = None
            # Start the execution journal for these identifiers.
            execution = Execution(self.db, learner, operation, cid, qid)
            # Recognize the traced operations with explicit retry request IDs.
            if operation in ('create', 'answer'):
                # Store the request ID for correlation.
                execution.metadata['request_id'] = req.request_id
            # Set this execution as the request-local stage target and keep a reset token.
            token, error = CURRENT.set(execution), None
            # Run the underlying operation while tracking its outcome.
            try:
                # Execute the actual service logic with its original arguments.
                result = function(self, *args, **kwargs)
                # A successful session result can supply the stored conversation ID.
                if isinstance(result, dict) and 'conversation' in result:
                    # Attach the resolved conversation to the execution summary.
                    execution.conversation_id = result['conversation']['conversation_id']
                    # A create response can save a conversation while its question generation
                    # failed.
                    if operation == 'create' and result['conversation']['status'] == 'needs_question':
                        # Mark that operation failed for observability.
                        execution.status = 'failed'
                        # Carry the persisted generation failure reason into the trace.
                        execution.metadata['failure_code'] = result['conversation'].get('last_error')
                # Pass the service result back to the API.
                return result
            # On failure, preserve the exception for the execution outcome.
            except Exception as caught:
                # Keep the error object until finalizing the trace.
                error = caught
                # Attempt to label a stored answer receipt as failed.
                if operation == 'answer':
                    # Keep receipt status failure separate from the original error.
                    try:
                        # Save the safe exception type while retaining the original response text.
                        self.store.fail_submission(execution.trace_id, type(caught).__name__)
                    # Handle a problem updating the receipt without masking the original error.
                    except Exception as journal_error:
                        # Record only the secondary error type.
                        logger.warning('submission_status_unavailable error_type=%s', type(journal_error).__name__)
                # Raise the original service error to the caller.
                raise
            # Always clean up the request-local execution and finalize its summary.
            finally:
                # Restore the parent execution or the empty previous value.
                CURRENT.reset(token)
                # Persist the end status and timing for this operation.
                execution.finish(error)
        # Return the traced implementation while preserving the service method's signature
        # metadata.
        return wrapped
    # Return the decorator configured for this operation name.
    return decorate
