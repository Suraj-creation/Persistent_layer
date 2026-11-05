"""Additive assessment storage; legacy research records are never seeded or removed."""
import json
import hashlib
from datetime import datetime, timezone
from .models import new_id
from .nlp import chunk_text, normalize, search_terms


def now():
    return datetime.now(timezone.utc).isoformat()


def install_schema(conn):
    statements = [
        "CREATE TABLE IF NOT EXISTS assessment_topics (learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, definition_json TEXT NOT NULL, state_json TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(learner_id, topic_key))",
        "CREATE TABLE IF NOT EXISTS assessment_conversations (conversation_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, request_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload_json TEXT NOT NULL, UNIQUE(learner_id, request_id))",
        "CREATE TABLE IF NOT EXISTS assessment_questions (question_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES assessment_conversations(conversation_id), payload_json TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS assessment_answers (turn_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES assessment_conversations(conversation_id), request_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload_json TEXT NOT NULL, UNIQUE(conversation_id, request_id))",
        "CREATE TABLE IF NOT EXISTS assessment_submissions (trace_id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES assessment_conversations(conversation_id), question_id TEXT NOT NULL REFERENCES assessment_questions(question_id), request_id TEXT NOT NULL, response_text TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('received','completed','failed')), error_code TEXT, turn_id TEXT REFERENCES assessment_answers(turn_id), created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS assessment_submission_question ON assessment_submissions(conversation_id,question_id,created_at)",
        "CREATE TABLE IF NOT EXISTS assessment_state_history (snapshot_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, payload_json TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS assessment_documents (document_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, raw_text TEXT NOT NULL, text_hash TEXT NOT NULL, UNIQUE(learner_id, topic_key, kind, text_hash))",
        "CREATE TABLE IF NOT EXISTS assessment_chunks (chunk_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL, topic_key TEXT NOT NULL, kind TEXT NOT NULL, source_id TEXT NOT NULL, content TEXT NOT NULL, metadata_json TEXT NOT NULL)",
        "CREATE VIRTUAL TABLE IF NOT EXISTS assessment_fts USING fts5(content, tokenize='porter unicode61')",
        "CREATE INDEX IF NOT EXISTS assessment_scope ON assessment_chunks(learner_id, topic_key, kind)",
        "CREATE INDEX IF NOT EXISTS assessment_conversations_owner ON assessment_conversations(learner_id)",
    ]
    for statement in statements:
        conn.execute(statement)


class AssessmentStore:
    def __init__(self, db):
        self.db = db

    def topic(self, learner_id, topic_key):
        with self.db.connection() as conn:
            row = conn.execute("SELECT definition_json, state_json, revision FROM assessment_topics WHERE learner_id=? AND topic_key=?", (learner_id, topic_key)).fetchone()
            return {"definition": json.loads(row[0]), "state": json.loads(row[1]), "revision": row[2]} if row else None

    def put_topic(self, learner_id, topic_key, definition, state, revision=0):
        with self.db.connection() as conn:
            conn.execute("INSERT INTO assessment_topics VALUES(?,?,?,?,?) ON CONFLICT(learner_id,topic_key) DO UPDATE SET state_json=excluded.state_json,revision=excluded.revision", (learner_id, topic_key, json.dumps(definition), json.dumps(state), revision))
            conn.execute("INSERT INTO assessment_state_history VALUES(?,?,?,?)", (state["snapshot_id"], learner_id, topic_key, json.dumps(state)))

    def conversation(self, conversation_id, learner_id):
        with self.db.connection() as conn:
            row = conn.execute("SELECT payload_json FROM assessment_conversations WHERE conversation_id=? AND learner_id=?", (conversation_id, learner_id)).fetchone()
            return json.loads(row[0]) if row else None

    def creation(self, learner_id, request_id):
        with self.db.connection() as conn:
            row = conn.execute("SELECT payload_json, request_hash FROM assessment_conversations WHERE learner_id=? AND request_id=?", (learner_id, request_id)).fetchone()
            return (json.loads(row[0]), row[1]) if row else None

    def create_conversation(self, payload, request_id, request_hash):
        with self.db.connection() as conn:
            conn.execute("INSERT INTO assessment_conversations VALUES(?,?,?,?,?,?)", (payload["conversation_id"], payload["learner_id"], payload["topic_key"], request_id, request_hash, json.dumps(payload)))

    def save_conversation(self, payload):
        with self.db.connection() as conn:
            conn.execute("UPDATE assessment_conversations SET payload_json=? WHERE conversation_id=?", (json.dumps(payload), payload["conversation_id"]))

    def conversations(self, learner_id):
        with self.db.connection() as conn:
            rows = conn.execute("SELECT payload_json FROM assessment_conversations WHERE learner_id=? ORDER BY rowid DESC", (learner_id,)).fetchall()
            return [json.loads(row[0]) for row in rows]

    def question(self, question_id):
        if not question_id:
            return None
        with self.db.connection() as conn:
            row = conn.execute("SELECT payload_json FROM assessment_questions WHERE question_id=?", (question_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_question(self, conversation_id, payload):
        with self.db.connection() as conn:
            conn.execute("INSERT INTO assessment_questions VALUES(?,?,?)", (payload["question_id"], conversation_id, json.dumps(payload)))

    def questions(self, conversation_id):
        with self.db.connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM assessment_questions WHERE conversation_id=? ORDER BY rowid", (conversation_id,))]

    def topic_questions(self, learner_id, topic_key):
        with self.db.connection() as conn:
            rows = conn.execute("SELECT q.payload_json FROM assessment_questions q JOIN assessment_conversations c ON q.conversation_id=c.conversation_id WHERE c.learner_id=? AND c.topic_key=? ORDER BY q.rowid DESC LIMIT 200", (learner_id, topic_key)).fetchall()
            return [json.loads(row[0]) for row in reversed(rows)]

    def answer(self, conversation_id, request_id):
        with self.db.connection() as conn:
            row = conn.execute("SELECT payload_json, request_hash FROM assessment_answers WHERE conversation_id=? AND request_id=?", (conversation_id, request_id)).fetchone()
            return (json.loads(row[0]), row[1]) if row else None

    def save_answer(self, conversation_id, request_id, request_hash, payload):
        with self.db.connection() as conn:
            conn.execute("INSERT INTO assessment_answers VALUES(?,?,?,?,?)", (payload["turn_id"], conversation_id, request_id, request_hash, json.dumps(payload)))

    def answers(self, conversation_id):
        with self.db.connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM assessment_answers WHERE conversation_id=? ORDER BY rowid", (conversation_id,))]

    def receive_submission(self, cid, req, trace_id):
        # A receipt is not accepted evidence. Commit before calling the provider.
        with self.db.connection() as conn:
            conn.execute("INSERT INTO assessment_submissions VALUES(?,?,?,?,?,'received',NULL,NULL,?,?)",
                         (trace_id, cid, req.question_id, req.request_id, req.response_text, now(), now()))

    def complete_submission(self, trace_id, turn_id):
        # Called inside the same transaction as accepted evidence and state.
        with self.db.connection() as conn:
            conn.execute("UPDATE assessment_submissions SET status='completed',turn_id=?,updated_at=? WHERE trace_id=?", (turn_id, now(), trace_id))

    def fail_submission(self, trace_id, error_code):
        with self.db.connection() as conn:
            conn.execute("UPDATE assessment_submissions SET status='failed',error_code=?,updated_at=? WHERE trace_id=? AND status='received'", (error_code, now(), trace_id))

    def pending_submission(self, cid, question_id):
        with self.db.connection() as conn:
            row = conn.execute("SELECT trace_id,question_id,response_text,status,created_at FROM assessment_submissions WHERE conversation_id=? AND question_id=? AND status!='completed' ORDER BY rowid DESC LIMIT 1", (cid, question_id)).fetchone()
            return dict(row) if row else None

    def add_chunk(self, learner_id, topic_key, kind, source_id, content, metadata):
        cid = new_id("chunk")
        with self.db.connection() as conn:
            cursor = conn.execute("INSERT INTO assessment_chunks VALUES(?,?,?,?,?,?,?)", (cid, learner_id, topic_key, kind, source_id, content, json.dumps(metadata)))
            conn.execute("INSERT INTO assessment_fts(rowid,content) VALUES(?,?)", (cursor.lastrowid, normalize(content)))
        return cid

    def add_document(self, learner_id, topic_key, kind, title, text):
        exact_hash = hashlib.sha256(text.encode()).hexdigest()
        with self.db.connection() as conn:
            exists = conn.execute("SELECT document_id FROM assessment_documents WHERE learner_id=? AND topic_key=? AND kind=? AND text_hash=?", (learner_id, topic_key, kind, exact_hash)).fetchone()
            if exists:
                return exists[0]
            did = new_id("source")
            conn.execute("INSERT INTO assessment_documents VALUES(?,?,?,?,?,?,?)", (did, learner_id, topic_key, kind, title, text, exact_hash))
            for chunk in chunk_text(text):
                self.add_chunk(learner_id, topic_key, kind, did, chunk["content"], {"start": chunk["start"], "end": chunk["end"], "title": title})
            return did

    def documents(self, learner_id, topic_key):
        with self.db.connection() as conn:
            return [dict(row) for row in conn.execute("SELECT document_id,kind,title,length(raw_text) AS character_count FROM assessment_documents WHERE learner_id=? AND topic_key=? ORDER BY rowid", (learner_id, topic_key))]

    def retrieve(self, learner_id, topic_key, query, evidence=False, limit=5, material_kind=None, source_ids=None):
        match = search_terms(query)
        kind_sql = "c.kind='learner_evidence'" if evidence else "c.kind!='learner_evidence'"
        filters = []
        if not evidence and material_kind:
            kind_sql += ' AND c.kind=?'
            filters.append(material_kind)
        if not evidence and source_ids is not None:
            if not source_ids:
                return []
            kind_sql += ' AND c.source_id IN (' + ','.join('?' for _ in source_ids) + ')'
            filters.extend(source_ids)
        with self.db.connection() as conn:
            rows = []
            if match:
                rows = conn.execute(f"SELECT c.*, bm25(assessment_fts) AS rank FROM assessment_fts JOIN assessment_chunks c ON c.rowid=assessment_fts.rowid WHERE assessment_fts MATCH ? AND c.learner_id=? AND c.topic_key=? AND {kind_sql} ORDER BY rank,c.rowid DESC LIMIT ?", (match, learner_id, topic_key, *filters, limit)).fetchall()
            mode = "fts5_bm25"
            if not rows:
                mode = "scoped_recent_fallback"
                rows = conn.execute(f"SELECT c.*, NULL AS rank FROM assessment_chunks c WHERE c.learner_id=? AND c.topic_key=? AND {kind_sql} ORDER BY c.rowid DESC LIMIT ?", (learner_id, topic_key, *filters, limit)).fetchall()
            return [{"chunk_id": row["chunk_id"], "source_id": row["source_id"], "kind": row["kind"], "content": row["content"], "metadata": json.loads(row["metadata_json"]), "rank": row["rank"], "retrieval_method": mode} for row in rows]
