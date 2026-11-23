"""Additive assessment storage; legacy research records are never seeded or removed."""
# Store structured assessment records as JSON in SQLite text columns.
import json
# Fingerprint raw document content to avoid re-indexing identical notes.
import hashlib
# Use timezone-aware UTC timestamps for durable records.
from datetime import datetime, timezone
# Get unique identifiers for new documents and chunks.
from .models import new_id
# Reuse assessment NLP rules for document chunking, index text and keyword search.
from .nlp import chunk_text, normalize, search_terms


# Supply a single timestamp convention for assessment events.
def now():
    # Return the current UTC time as an ISO string.
    return datetime.now(timezone.utc).isoformat()


# Install the live assessment schema alongside the existing research tables.
def install_schema(conn):
    # Group the idempotent SQL schema statements for the connection.
    statements = [
        # Key current topic memory by learner plus topic and include a revision for writes.
        "CREATE TABLE IF NOT EXISTS assessment_topics (learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, definition_json TEXT NOT NULL, state_json TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(learner_id, topic_key))",
        # Save conversations with unique start request IDs per learner.
        "CREATE TABLE IF NOT EXISTS assessment_conversations (conversation_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, request_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload_json TEXT NOT NULL, UNIQUE(learner_id, request_id))",
        # Link each full question and private rubric to its conversation.
        "CREATE TABLE IF NOT EXISTS assessment_questions (question_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES assessment_conversations(conversation_id), payload_json TEXT NOT NULL)",
        # Save one accepted answer per conversation and request ID.
        "CREATE TABLE IF NOT EXISTS assessment_answers (turn_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES assessment_conversations(conversation_id), request_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload_json TEXT NOT NULL, UNIQUE(conversation_id, request_id))",
        # Track original submissions and distinguish received, failed and completed receipts.
        "CREATE TABLE IF NOT EXISTS assessment_submissions (trace_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES assessment_conversations(conversation_id), question_id TEXT NOT NULL REFERENCES assessment_questions(question_id), request_id TEXT NOT NULL, response_text TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('received','completed','failed')), error_code TEXT, turn_id TEXT REFERENCES assessment_answers(turn_id), created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
        # Speed up recovery of the most recent submission for a question.
        "CREATE INDEX IF NOT EXISTS assessment_submission_question ON assessment_submissions(conversation_id,question_id,created_at)",
        # Keep immutable state snapshots addressable by snapshot ID.
        "CREATE TABLE IF NOT EXISTS assessment_state_history (snapshot_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, payload_json TEXT NOT NULL)",
        # Store exact reference documents and deduplicate them in their own learner/topic/kind
        # scope.
        "CREATE TABLE IF NOT EXISTS assessment_documents (document_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, raw_text TEXT NOT NULL, text_hash TEXT NOT NULL, UNIQUE(learner_id, topic_key, kind, text_hash))",
        # Store material and learner-evidence passages with source links and provenance.
        "CREATE TABLE IF NOT EXISTS assessment_chunks (chunk_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, kind TEXT NOT NULL, source_id TEXT NOT NULL, content TEXT NOT NULL, metadata_json TEXT NOT NULL)",
        # Create a full-text index with Unicode tokenization and English Porter stemming.
        "CREATE VIRTUAL TABLE IF NOT EXISTS assessment_fts USING fts5(content, tokenize='porter unicode61')",
        # Index learner, topic and chunk kind for scoped searches.
        "CREATE INDEX IF NOT EXISTS assessment_scope ON assessment_chunks(learner_id, topic_key, kind)",
        # Index the conversation owner for workspace lists.
        "CREATE INDEX IF NOT EXISTS assessment_conversations_owner ON assessment_conversations(learner_id)",
    ]
    # Apply each schema statement in order.
    for statement in statements:
        # Execute the SQL through the database initialization connection.
        conn.execute(statement)


# Keep the SQL details behind methods used by the assessment service.
class AssessmentStore:
    # Receive the shared database manager.
    def __init__(self, db):
        # Keep access to its transaction-aware connection helpers.
        self.db = db

    # Read one learner's current memory for a normalized topic.
    def topic(self, learner_id, topic_key):
        # Use the current database connection if the caller already opened one.
        with self.db.connection() as conn:
            # Fetch the topic map, latest state and write revision using bound values.
            row = conn.execute("SELECT definition_json, state_json, revision FROM assessment_topics WHERE learner_id=? AND topic_key=?", (learner_id, topic_key)).fetchone()
            # Decode the saved JSON when this learner/topic pair exists.
            return {"definition": json.loads(row[0]), "state": json.loads(row[1]), "revision": row[2]} if row else None

    # Save both the current topic projection and the corresponding historical snapshot.
    def put_topic(self, learner_id, topic_key, definition, state, revision=0):
        # Reuse the caller's open transaction when present.
        with self.db.connection() as conn:
            # Insert a new topic or update the state and revision for an existing topic.
            conn.execute("INSERT INTO assessment_topics VALUES(?,?,?,?,?) ON CONFLICT(learner_id,topic_key) DO UPDATE SET state_json=excluded.state_json,revision=excluded.revision", (learner_id, topic_key, json.dumps(definition), json.dumps(state), revision))
            # Append a separate immutable record for the newly saved snapshot.
            conn.execute("INSERT INTO assessment_state_history VALUES(?,?,?,?)", (state["snapshot_id"], learner_id, topic_key, json.dumps(state)))

    # Read one conversation scoped to its learner.
    def conversation(self, conversation_id, learner_id):
        # Open or reuse a database connection for the lookup.
        with self.db.connection() as conn:
            # Require both conversation ID and learner ID to fetch the payload.
            row = conn.execute("SELECT payload_json FROM assessment_conversations WHERE conversation_id=? AND learner_id=?", (conversation_id, learner_id)).fetchone()
            # Return a decoded conversation or signal that it did not match.
            return json.loads(row[0]) if row else None

    # Look up the result of a prior create request.
    def creation(self, learner_id, request_id):
        # Read using the same connection as the caller when applicable.
        with self.db.connection() as conn:
            # Fetch the conversation and its original request fingerprint by retry ID.
            row = conn.execute("SELECT payload_json, request_hash FROM assessment_conversations WHERE learner_id=? AND request_id=?", (learner_id, request_id)).fetchone()
            # Provide both values so the service can detect a conflicting retry.
            return (json.loads(row[0]), row[1]) if row else None

    # Store the first record of a conversation.
    def create_conversation(self, payload, request_id, request_hash):
        # Write through the caller's database connection.
        with self.db.connection() as conn:
            # Save the identifiers, retry fingerprint and full JSON payload.
            conn.execute("INSERT INTO assessment_conversations VALUES(?,?,?,?,?,?)", (payload["conversation_id"], payload["learner_id"], payload["topic_key"], request_id, request_hash, json.dumps(payload)))

    # Persist changes such as a question pointer or status.
    def save_conversation(self, payload):
        # Reuse the enclosing service transaction.
        with self.db.connection() as conn:
            # Replace the conversation payload while preserving the row's identity and retry key.
            conn.execute("UPDATE assessment_conversations SET payload_json=? WHERE conversation_id=?", (json.dumps(payload), payload["conversation_id"]))

    # Build a list of a learner's saved conversations.
    def conversations(self, learner_id):
        # Read the rows with a scoped database query.
        with self.db.connection() as conn:
            # List matching conversations newest by insertion order.
            rows = conn.execute("SELECT payload_json FROM assessment_conversations WHERE learner_id=? ORDER BY rowid DESC", (learner_id,)).fetchall()
            # Decode each payload for use by the service.
            return [json.loads(row[0]) for row in rows]

    # Read the saved definition of a particular question.
    def question(self, question_id):
        # A waiting or finished conversation may not have a current question ID.
        if not question_id:
            # Return no question without querying for a missing ID.
            return None
        # Open or reuse a connection for a valid ID.
        with self.db.connection() as conn:
            # Fetch the full stored question, including private grading data.
            row = conn.execute("SELECT payload_json FROM assessment_questions WHERE question_id=?", (question_id,)).fetchone()
            # Decode the payload if this question exists.
            return json.loads(row[0]) if row else None

    # Store the accepted question and its provenance.
    def save_question(self, conversation_id, payload):
        # Use the same connection as the conversation activation write.
        with self.db.connection() as conn:
            # Insert the full question under its server-generated ID.
            conn.execute("INSERT INTO assessment_questions VALUES(?,?,?)", (payload["question_id"], conversation_id, json.dumps(payload)))

    # Read questions for one conversation in their original sequence.
    def questions(self, conversation_id):
        # Reuse the connection for the query.
        with self.db.connection() as conn:
            # Decode each row as SQLite returns question creation order.
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM assessment_questions WHERE conversation_id=? ORDER BY rowid", (conversation_id,))]

    # Read the topic-wide recent history for the novelty check.
    def topic_questions(self, learner_id, topic_key):
        # Join questions with their owning conversations on the same connection.
        with self.db.connection() as conn:
            # Restrict history to this learner and topic, taking up to 200 recent questions.
            rows = conn.execute("SELECT q.payload_json FROM assessment_questions q JOIN assessment_conversations c ON q.conversation_id=c.conversation_id WHERE c.learner_id=? AND c.topic_key=? ORDER BY q.rowid DESC LIMIT 200", (learner_id, topic_key)).fetchall()
            # Restore chronological order after selecting the latest rows.
            return [json.loads(row[0]) for row in reversed(rows)]

    # Look up an accepted answer by its retry key.
    def answer(self, conversation_id, request_id):
        # Use the current database view.
        with self.db.connection() as conn:
            # Scope the request ID to this conversation and read the original fingerprint.
            row = conn.execute("SELECT payload_json, request_hash FROM assessment_answers WHERE conversation_id=? AND request_id=?", (conversation_id, request_id)).fetchone()
            # Return the saved answer and its fingerprint when present.
            return (json.loads(row[0]), row[1]) if row else None

    # Insert a newly accepted answer with all of its evidence.
    def save_answer(self, conversation_id, request_id, request_hash, payload):
        # Reuse the transaction that also writes topic memory.
        with self.db.connection() as conn:
            # Save the turn ID, retry fingerprint and complete accepted payload.
            conn.execute("INSERT INTO assessment_answers VALUES(?,?,?,?,?)", (payload["turn_id"], conversation_id, request_id, request_hash, json.dumps(payload)))

    # Build this conversation's accepted response timeline.
    def answers(self, conversation_id):
        # Read through a single connection.
        with self.db.connection() as conn:
            # Return accepted turns in insertion order after decoding their JSON.
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM assessment_answers WHERE conversation_id=? ORDER BY rowid", (conversation_id,))]

    # Save raw input before it can be rejected by the model or validator.
    def receive_submission(self, cid, req, trace_id):
        # A receipt is not accepted evidence. Commit before calling the provider.
        # Open a connection for the receipt write.
        with self.db.connection() as conn:
            # Insert a received row with empty failure and accepted-turn fields.
            conn.execute("INSERT INTO assessment_submissions VALUES(?,?,?,?,?,'received',NULL,NULL,?,?)",
                         # Preserve request text verbatim and link it to the question and
                         # execution.
                         (trace_id, cid, req.question_id, req.request_id, req.response_text, now(), now()))

    # Connect a received attempt to an accepted turn.
    def complete_submission(self, trace_id, turn_id):
        # Called inside the same transaction as accepted evidence and state.
        # Write through the conversation's acceptance transaction.
        with self.db.connection() as conn:
            # Update the status, accepted turn reference and time for the exact receipt.
            conn.execute("UPDATE assessment_submissions SET status='completed',turn_id=?,updated_at=? WHERE trace_id=?", (turn_id, now(), trace_id))

    # Record why a received attempt failed while retaining its text.
    def fail_submission(self, trace_id, error_code):
        # Use a database connection for the failure update.
        with self.db.connection() as conn:
            # Change only a received receipt, so an already accepted answer cannot become failed.
            conn.execute("UPDATE assessment_submissions SET status='failed',error_code=?,updated_at=? WHERE trace_id=? AND status='received'", (error_code, now(), trace_id))

    # Find a submission the editor can recover for its current question.
    def pending_submission(self, cid, question_id):
        # Use the session's shared read connection when available.
        with self.db.connection() as conn:
            # Select the most recent incomplete receipt for this conversation and question.
            row = conn.execute("SELECT trace_id,question_id,response_text,status,created_at FROM assessment_submissions WHERE conversation_id=? AND question_id=? AND status!='completed' ORDER BY rowid DESC LIMIT 1", (cid, question_id)).fetchone()
            # Return its public receipt fields or no recovery record.
            return dict(row) if row else None

    # Persist one passage and index it for search.
    def add_chunk(self, learner_id, topic_key, kind, source_id, content, metadata):
        # Assign a citation or evidence chunk ID independently of the SQL row number.
        cid = new_id("chunk")
        # Use one connection for the raw passage and index entry.
        with self.db.connection() as conn:
            # Insert the original content and source metadata and capture its row number.
            cursor = conn.execute("INSERT INTO assessment_chunks VALUES(?,?,?,?,?,?,?)", (cid, learner_id, topic_key, kind, source_id, content, json.dumps(metadata)))
            # Index the normalized search text at the same row number for later joins.
            conn.execute("INSERT INTO assessment_fts(rowid,content) VALUES(?,?)", (cursor.lastrowid, normalize(content)))
        # Return the public chunk ID for provenance.
        return cid

    # Save a full reference document and index its passages once.
    def add_document(self, learner_id, topic_key, kind, title, text):
        # Hash the exact raw text so differences in case or formatting remain distinguishable.
        exact_hash = hashlib.sha256(text.encode()).hexdigest()
        # Use the same connection for deduplication and indexing.
        with self.db.connection() as conn:
            # Look for exactly this text in the same learner, topic and document kind.
            exists = conn.execute("SELECT document_id FROM assessment_documents WHERE learner_id=? AND topic_key=? AND kind=? AND text_hash=?", (learner_id, topic_key, kind, exact_hash)).fetchone()
            # Reuse an identical document if it was already saved.
            if exists:
                # Return its old ID so the conversation can pin it again.
                return exists[0]
            # Assign a new document ID for new material.
            did = new_id("source")
            # Store the full original text and its fingerprint.
            conn.execute("INSERT INTO assessment_documents VALUES(?,?,?,?,?,?,?)", (did, learner_id, topic_key, kind, title, text, exact_hash))
            # Walk overlapping raw passages so nearby text stays together in search results.
            for chunk in chunk_text(text):
                # Store each slice with its source document, title and raw bounds.
                self.add_chunk(learner_id, topic_key, kind, did, chunk["content"], {"start": chunk["start"], "end": chunk["end"], "title": title})
            # Return the document ID after indexing its passages.
            return did

    # List source metadata needed by the workspace and reports.
    def documents(self, learner_id, topic_key):
        # Scope the query to this learner and topic.
        with self.db.connection() as conn:
            # Return ID, kind, title and length without transferring the full reference text.
            return [dict(row) for row in conn.execute("SELECT document_id,kind,title,length(raw_text) AS character_count FROM assessment_documents WHERE learner_id=? AND topic_key=? ORDER BY rowid", (learner_id, topic_key))]

    # Retrieve reference passages or learner evidence from the local search index.
    def retrieve(self, learner_id, topic_key, query, evidence=False, limit=5, material_kind=None, source_ids=None):
        # Extract and quote appropriate keywords for FTS5.
        match = search_terms(query)
        # Select either learner evidence or material before ranking.
        kind_sql = "c.kind='learner_evidence'" if evidence else "c.kind!='learner_evidence'"
        # Collect extra parameters for material filters.
        filters = []
        # A conversation can choose which kind of reference is eligible.
        if not evidence and material_kind:
            # Add a bound filter for that exact material kind.
            kind_sql += ' AND c.kind=?'
            # Supply the selected kind as a query parameter.
            filters.append(material_kind)
        # Use pinned document IDs only for the factual material search.
        if not evidence and source_ids is not None:
            # An explicitly empty source selection permits no material.
            if not source_ids:
                # Return no hits when no document is allowed.
                return []
            # Build one placeholder per pinned document ID.
            kind_sql += ' AND c.source_id IN (' + ','.join('?' for _ in source_ids) + ')'
            # Bind the IDs separately from the SQL.
            filters.extend(source_ids)
        # Use the same connection for ranked search and any fallback.
        with self.db.connection() as conn:
            # Start with no matches.
            rows = []
            # Run full-text ranking only when there are usable query terms.
            if match:
                # Join FTS matches to scoped chunks and take the best BM25 ranks, breaking ties
                # with newer rows.
                rows = conn.execute(f"SELECT c.*, bm25(assessment_fts) AS rank FROM assessment_fts JOIN assessment_chunks c ON c.rowid=assessment_fts.rowid WHERE assessment_fts MATCH ? AND c.learner_id=? AND c.topic_key=? AND {kind_sql} ORDER BY rank,c.rowid DESC LIMIT ?", (match, learner_id, topic_key, *filters, limit)).fetchall()
            # Mark the normal ranking method for the result provenance.
            mode = "fts5_bm25"
            # Fall back only when full-text search produced no eligible rows.
            if not rows:
                # Label fallback hits as recent scoped passages.
                mode = "scoped_recent_fallback"
                # Keep the same learner, topic and source filters even when there is no keyword
                # match.
                rows = conn.execute(f"SELECT c.*, NULL AS rank FROM assessment_chunks c WHERE c.learner_id=? AND c.topic_key=? AND {kind_sql} ORDER BY c.rowid DESC LIMIT ?", (learner_id, topic_key, *filters, limit)).fetchall()
            # Return original content, provenance and retrieval method, with rank absent for
            # fallback hits.
            return [{"chunk_id": row["chunk_id"], "source_id": row["source_id"], "kind": row["kind"], "content": row["content"], "metadata": json.loads(row["metadata_json"]), "rank": row["rank"], "retrieval_method": mode} for row in rows]
