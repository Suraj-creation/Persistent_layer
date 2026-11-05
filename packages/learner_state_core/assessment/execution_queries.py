"""One filter contract for execution lists and their measured summaries."""

# Both the detail view and aggregates use the same evidence of completeness.
# Older traces without an expected count stay unknown, unless a gap is provable.
TELEMETRY_STATUS_SQL = """CASE WHEN e.status='running'
    OR COALESCE(json_extract(e.metadata_json,'$.telemetry_write_failures'),0)>0
    OR (json_extract(e.metadata_json,'$.expected_stage_count') IS NOT NULL
        AND json_extract(e.metadata_json,'$.expected_stage_count') !=
            (SELECT count(*) FROM nlp_stages ts WHERE ts.trace_id=e.trace_id))
    OR EXISTS (SELECT 1 FROM nlp_stages ts WHERE ts.trace_id=e.trace_id
        AND json_extract(ts.payload_json,'$.status')='running')
    THEN 'incomplete'
    WHEN json_extract(e.metadata_json,'$.expected_stage_count') IS NOT NULL THEN 'complete'
    ELSE 'unknown' END"""


def execution_filter(learner_id, conversation_id=None, operation=None, status=None,
                     question_id=None, search=None, stage=None, error_only=False,
                     started_after=None, started_before=None):
    clauses, values = ['e.learner_id=?'], [learner_id]
    for key, value in [('conversation_id', conversation_id), ('operation', operation),
                       ('status', status), ('question_id', question_id)]:
        if value:
            clauses.append('e.' + key + '=?')
            values.append(value)
    if search:
        # instr is literal: '%' and '_' in IDs are not SQL wildcard operators.
        clauses.append('(instr(e.trace_id, ?) > 0 OR instr(COALESCE(e.question_id,\'\'), ?) > 0 OR instr(COALESCE(e.conversation_id,\'\'), ?) > 0)')
        values.extend([search] * 3)
    for value, operator in [(started_after, '>='), (started_before, '<=')]:
        if value:
            clauses.append('julianday(e.started_at) ' + operator + ' julianday(?)')
            values.append(value)
    if stage:
        clauses.append("EXISTS (SELECT 1 FROM nlp_stages s WHERE s.trace_id=e.trace_id AND json_extract(s.payload_json,'$.key')=?)")
        values.append(stage)
    if error_only:
        clauses.append("(e.status='failed' OR EXISTS (SELECT 1 FROM nlp_stages s WHERE s.trace_id=e.trace_id AND json_extract(s.payload_json,'$.status')='failed'))")
    return ' AND '.join(clauses), values


def summarize(conn, where, values):
    counts = dict(conn.execute(f'''SELECT count(*) total,
        COALESCE(sum(e.status='completed'),0) completed,
        COALESCE(sum(e.status='failed'),0) failed,
        COALESCE(sum(e.status='running'),0) running,
        avg(CASE WHEN e.status='completed' THEN e.duration_ms END) average_ms,
        COALESCE(sum(({TELEMETRY_STATUS_SQL})='incomplete'),0) incomplete_telemetry
        FROM nlp_executions e WHERE {where}''', values).fetchone())
    terminal = counts['completed'] + counts['failed']
    counts['success_rate'] = round(100 * counts['completed'] / terminal, 2) if terminal else None
    counts['failure_rate'] = round(100 * counts['failed'] / terminal, 2) if terminal else None
    counts['average_ms'] = round(counts['average_ms'], 2) if counts['average_ms'] is not None else None
    spans = conn.execute(f'''SELECT s.trace_id, json_extract(s.payload_json,'$.key') stage,
        json_extract(s.payload_json,'$.status') status, json_extract(s.payload_json,'$.duration_ms') duration,
        json_extract(s.payload_json,'$.attempt') attempt
        FROM nlp_stages s JOIN nlp_executions e ON s.trace_id=e.trace_id WHERE {where}''', values).fetchall()
    attempts, durations = {}, {'provider_ms': [], 'retrieval_ms': [], 'persistence_ms': []}
    failures = 0
    for span in spans:
        key = span['stage']
        if key in ('topic', 'generation', 'extraction'):
            group = (span['trace_id'], key)
            attempts[group] = max(attempts.get(group, 1), span['attempt'] or 1)
        if key == 'persistence' and span['status'] == 'failed':
            failures += 1
        metric = 'provider_ms' if key in ('topic', 'generation', 'extraction') else 'retrieval_ms' if key == 'retrieval' else 'persistence_ms' if key == 'persistence' else None
        if metric and span['status'] == 'completed' and span['duration'] is not None:
            durations[metric].append(span['duration'])
    counts.update({key: round(sum(items) / len(items), 2) if items else None for key, items in durations.items()})
    counts['retry_count'] = sum(attempt - 1 for attempt in attempts.values())
    counts['persistence_failures'] = failures
    return counts
