"""One filter contract for execution lists and their measured summaries."""

# Both the detail view and aggregates use the same evidence of completeness.
# Older traces without an expected count stay unknown, unless a gap is provable.
# Derive telemetry status from unfinished work, recorded write failures and mismatched or missing
# stage records.
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


# Build one set of SQL filters for both paged traces and aggregate results.
def execution_filter(learner_id, conversation_id=None, operation=None, status=None,
                     # Accept optional question, ID search and stage constraints.
                     question_id=None, search=None, stage=None, error_only=False,
                     # Accept optional time bounds for the trace start.
                     started_after=None, started_before=None):
    # Always limit visibility to the given learner ID.
    clauses, values = ['e.learner_id=?'], [learner_id]
    # Consider the supported exact-match fields.
    for key, value in [('conversation_id', conversation_id), ('operation', operation),
                       # Include status and question ID among the optional filters.
                       ('status', status), ('question_id', question_id)]:
        # Include a filter only when a value was supplied.
        if value:
            # Append the known SQL column with a placeholder.
            clauses.append('e.' + key + '=?')
            # Bind the value separately from the query.
            values.append(value)
    # Check whether the user requested a text search by ID.
    if search:
        # instr is literal: '%' and '_' in IDs are not SQL wildcard operators.
        # Search within execution, question and conversation IDs.
        clauses.append('(instr(e.trace_id, ?) > 0 OR instr(COALESCE(e.question_id,\'\'), ?) > 0 OR instr(COALESCE(e.conversation_id,\'\'), ?) > 0)')
        # Bind the same literal term for all three ID comparisons.
        values.extend([search] * 3)
    # Use an inclusive lower or upper bound when each date was supplied.
    for value, operator in [(started_after, '>='), (started_before, '<=')]:
        # Skip absent dates.
        if value:
            # Compare real UTC timestamps in SQLite instead of comparing formatted strings.
            clauses.append('julianday(e.started_at) ' + operator + ' julianday(?)')
            # Bind the requested time bound.
            values.append(value)
    # Optionally require an execution to contain a named stage.
    if stage:
        # Add a correlated lookup from this trace to its stages.
        clauses.append("EXISTS (SELECT 1 FROM nlp_stages s WHERE s.trace_id=e.trace_id AND json_extract(s.payload_json,'$.key')=?)")
        # Bind the requested stage name.
        values.append(stage)
    # Optionally show only executions or stages that actually failed.
    if error_only:
        # Match both terminal execution failures and failed steps that were later retried.
        clauses.append("(e.status='failed' OR EXISTS (SELECT 1 FROM nlp_stages s WHERE s.trace_id=e.trace_id AND json_extract(s.payload_json,'$.status')='failed'))")
    # Return the combined predicate and its bound arguments together.
    return ' AND '.join(clauses), values


# Compute aggregate metrics using the full filtered set.
def summarize(conn, where, values):
    # Count executions by status and get the mean duration of completed executions and the
    # incomplete count.
    counts = dict(conn.execute(f'''SELECT count(*) total,
        COALESCE(sum(e.status='completed'),0) completed,
        COALESCE(sum(e.status='failed'),0) failed,
        COALESCE(sum(e.status='running'),0) running,
        avg(CASE WHEN e.status='completed' THEN e.duration_ms END) average_ms,
        COALESCE(sum(({TELEMETRY_STATUS_SQL})='incomplete'),0) incomplete_telemetry
        FROM nlp_executions e WHERE {where}''', values).fetchone())
    # Count finished operations for the two outcome-rate denominators.
    terminal = counts['completed'] + counts['failed']
    # Compute the completed share only when at least one operation finished.
    counts['success_rate'] = round(100 * counts['completed'] / terminal, 2) if terminal else None
    # Compute the failed share using the same denominator.
    counts['failure_rate'] = round(100 * counts['failed'] / terminal, 2) if terminal else None
    # Round the mean only when SQLite found measured completed durations.
    counts['average_ms'] = round(counts['average_ms'], 2) if counts['average_ms'] is not None else None
    # Read actual stage durations and attempts for the filtered execution set.
    spans = conn.execute(f'''SELECT s.trace_id, json_extract(s.payload_json,'$.key') stage,
        json_extract(s.payload_json,'$.status') status, json_extract(s.payload_json,'$.duration_ms') duration,
        json_extract(s.payload_json,'$.attempt') attempt
        FROM nlp_stages s JOIN nlp_executions e ON s.trace_id=e.trace_id WHERE {where}''', values).fetchall()
    # Collect the highest attempt per provider stage and group timings by stage type.
    attempts, durations = {}, {'provider_ms': [], 'retrieval_ms': [], 'persistence_ms': []}
    # Start with no recorded persistence failures.
    failures = 0
    # Inspect each saved stage in the filtered set.
    for span in spans:
        # Read the stage type from the saved trace.
        key = span['stage']
        # Count retries on provider-backed topic, generation and extraction stages.
        if key in ('topic', 'generation', 'extraction'):
            # Group attempts by execution and stage name.
            group = (span['trace_id'], key)
            # Retain the highest attempt number recorded for that group.
            attempts[group] = max(attempts.get(group, 1), span['attempt'] or 1)
        # Count a failed persistence stage explicitly.
        if key == 'persistence' and span['status'] == 'failed':
            # Increment the observed write-stage failure count.
            failures += 1
        # Map observed stage names into provider, retrieval or persistence metrics.
        metric = 'provider_ms' if key in ('topic', 'generation', 'extraction') else 'retrieval_ms' if key == 'retrieval' else 'persistence_ms' if key == 'persistence' else None
        # Use only completed measurements with a saved duration.
        if metric and span['status'] == 'completed' and span['duration'] is not None:
            # Add that actual measurement to its metric group.
            durations[metric].append(span['duration'])
    # Compute the average for each group when there is a recorded measurement.
    counts.update({key: round(sum(items) / len(items), 2) if items else None for key, items in durations.items()})
    # Count the attempts beyond the initial attempt in each provider group.
    counts['retry_count'] = sum(attempt - 1 for attempt in attempts.values())
    # Include the number of failed persistence stages.
    counts['persistence_failures'] = failures
    # Return the aggregate record used by the inspector.
    return counts
