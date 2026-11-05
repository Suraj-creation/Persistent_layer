from __future__ import annotations

import json
import sqlite3
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from .connection import open_connection
from ..evidence.models import CognitiveEvidence, ConceptLink, CandidateInterpretation, Interaction
from ..state.models import (
    ConceptState,
    LearnerStateSnapshot,
    MisconceptionBelief,
    StateChangeEvent,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalDatabaseManager:
    """
    Persistent SQLite-backed storage manager for the Language-Grounded Learner State Platform.
    Ensures complete, durable persistence of all:
    - Learners
    - Raw Interactions & Dialogues
    - Extracted Cognitive Evidence
    - Persistent Concept States (Alpha, Beta, Mastery, Uncertainty)
    - Candidate Misconception Beliefs (Odds, Posteriors, Provenance)
    - State Audit Events & Mathematical Traces
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parents[3]
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "learner_state.db")
        else:
            self.db_path = db_path
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

    @contextmanager
    def _get_connection(self):
        active = getattr(self._local, "connection", None)
        if active is not None:
            yield active
            return
        conn = open_connection(self.db_path)
        self._local.connection = conn
        try:
            with conn:
                yield conn
        finally:
            self._local.connection = None
            conn.close()

    def connection(self):
        """Shared connection scope used by the assessment workspace."""
        return self._get_connection()

    @contextmanager
    def independent_connection(self):
        """Own a connection outside learner transactions for the trace journal."""
        conn = open_connection(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self):
        """Serialize state transitions across processes; rollback the entire event on failure."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            yield conn

    def _init_db(self):
        """Initializes tables and serializes additive schema migrations across workers."""
        with self._get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()

            # 1. Learners
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS learners (
                learner_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            );
            """)

            # 2. Interactions (Every conversational turn & answer submission)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                learner_id TEXT NOT NULL,
                item_id TEXT,
                item_text TEXT,
                item_concepts_json TEXT DEFAULT '[]',
                response_text TEXT,
                selected_option TEXT,
                correctness INTEGER,
                latency_seconds REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
            );
            """)

            # 3. Cognitive Evidence (Atomic extracted records)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_evidence (
                evidence_id TEXT PRIMARY KEY,
                interaction_id TEXT NOT NULL,
                learner_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_quote TEXT NOT NULL,
                is_verbatim_quote INTEGER NOT NULL,
                extracted_claim TEXT NOT NULL,
                polarity TEXT NOT NULL,
                expressed_uncertainty TEXT NOT NULL,
                category TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                hypothesis_description TEXT NOT NULL,
                hypothesis_confidence REAL NOT NULL,
                concept_links_json TEXT NOT NULL,
                extractor_model TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
            );
            """)

            # 4. Concept States (Per learner, per concept persistent Beta-Bayesian parameters)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_states (
                learner_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                concept_name TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                mastery_estimate REAL NOT NULL DEFAULT 0.5,
                uncertainty REAL NOT NULL DEFAULT 0.2887,
                evidence_sufficiency TEXT NOT NULL DEFAULT 'insufficient',
                interaction_count INTEGER NOT NULL DEFAULT 0,
                last_interaction_timestamp TEXT,
                PRIMARY KEY (learner_id, concept_id),
                FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
            );
            """)

            # 5. Candidate Misconception Beliefs
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS misconception_beliefs (
                learner_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                misconception_id TEXT NOT NULL,
                misconception_name TEXT NOT NULL,
                posterior_probability REAL NOT NULL DEFAULT 0.10,
                supporting_evidence_ids_json TEXT DEFAULT '[]',
                contradicting_evidence_ids_json TEXT DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'suspected',
                last_updated TEXT NOT NULL,
                PRIMARY KEY (learner_id, concept_id, misconception_id),
                FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
            );
            """)

            # 6. State Audit Events (Mathematical change trace)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS state_audit_events (
                event_id TEXT PRIMARY KEY,
                learner_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                triggering_evidence_id TEXT,
                prior_mastery REAL NOT NULL,
                new_mastery REAL NOT NULL,
                delta_mastery REAL NOT NULL,
                prior_uncertainty REAL,
                new_uncertainty REAL,
                applied_rule TEXT NOT NULL,
                change_summary TEXT NOT NULL,
                math_trace_json TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (learner_id) REFERENCES learners(learner_id)
            );
            """)


            for table in ("interactions", "cognitive_evidence", "state_audit_events"):
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                if "payload_json" not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN payload_json TEXT")
            # The improve branch stored history in learner_snapshots; training stored
            # one current row per learner. Migrate either layout without discarding
            # history, then keep an explicit current projection and history table.
            conn.execute("CREATE TABLE IF NOT EXISTS learner_snapshot_history (snapshot_id TEXT PRIMARY KEY, learner_id TEXT NOT NULL REFERENCES learners(learner_id), payload_json TEXT NOT NULL)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(learner_snapshots)")}
            if "snapshot_id" in columns:
                conn.execute("INSERT OR IGNORE INTO learner_snapshot_history SELECT snapshot_id, learner_id, payload_json FROM learner_snapshots ORDER BY rowid")
                conn.execute("DROP TABLE learner_snapshots")
            schema = """
                CREATE TABLE IF NOT EXISTS learner_snapshots (
                    learner_id TEXT PRIMARY KEY REFERENCES learners(learner_id), payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS learner_replay_baselines (
                    learner_id TEXT PRIMARY KEY REFERENCES learners(learner_id), snapshot_json TEXT NOT NULL,
                    excluded_evidence_ids_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS processed_requests (
                    learner_id TEXT NOT NULL REFERENCES learners(learner_id), request_key TEXT NOT NULL,
                    input_hash TEXT NOT NULL, response_json TEXT NOT NULL,
                    PRIMARY KEY(learner_id,request_key)
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, learner_id TEXT NOT NULL REFERENCES learners(learner_id),
                    interaction_id TEXT NOT NULL UNIQUE, payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieval_traces (
                    interaction_id TEXT PRIMARY KEY REFERENCES interactions(interaction_id),
                    learner_id TEXT NOT NULL REFERENCES learners(learner_id), payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS issued_probes (
                    learner_id TEXT NOT NULL REFERENCES learners(learner_id), parent_interaction_id TEXT NOT NULL,
                    probe_id TEXT NOT NULL, answered_interaction_id TEXT,
                    PRIMARY KEY(learner_id,parent_interaction_id,probe_id)
                );
                CREATE INDEX IF NOT EXISTS evidence_learner ON cognitive_evidence(learner_id);
                CREATE INDEX IF NOT EXISTS interaction_learner ON interactions(learner_id);
                CREATE INDEX IF NOT EXISTS audit_learner ON state_audit_events(learner_id);
            """
            for statement in schema.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute("CREATE TABLE IF NOT EXISTS learning_turns (learner_id TEXT NOT NULL REFERENCES learners(learner_id), request_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload_json TEXT NOT NULL, UNIQUE(learner_id,request_id))")
            conn.execute("CREATE INDEX IF NOT EXISTS snapshot_history_learner ON learner_snapshot_history(learner_id)")
            if "snapshot_id" in columns:
                conn.execute("INSERT OR IGNORE INTO learner_snapshots SELECT learner_id, payload_json FROM learner_snapshot_history WHERE rowid IN (SELECT MAX(rowid) FROM learner_snapshot_history GROUP BY learner_id)")
            # Preserve the training branch's existing current rows in history too.
            for row in conn.execute("SELECT learner_id,payload_json FROM learner_snapshots").fetchall():
                snapshot = LearnerStateSnapshot.model_validate_json(row[1])
                conn.execute("INSERT OR IGNORE INTO learner_snapshot_history VALUES(?,?,?)", (snapshot.snapshot_id,row[0],row[1]))
            from ..assessment.storage import install_schema
            from ..assessment.execution import install_schema as install_execution_schema
            from ..assessment.evaluation import install_schema as install_evaluation_schema
            install_schema(conn)
            install_execution_schema(conn)
            install_evaluation_schema(conn)

    # -------------------------------------------------------------
    # Learner Operations
    # -------------------------------------------------------------
    def ensure_learner(self, learner_id: str, name: Optional[str] = None) -> None:
        """Ensures learner exists in database."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            display_name = name or learner_id.replace("_", " ").title()
            cursor.execute(
                "INSERT OR IGNORE INTO learners (learner_id,name,created_at,updated_at) VALUES(?,?,?,?)",
                (learner_id,display_name,now,now),
            )

    def list_learners(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT learner_id, name, created_at, updated_at FROM learners ORDER BY created_at ASC")
            return [dict(row) for row in cursor.fetchall()]

    # -------------------------------------------------------------
    # Interaction & Evidence Persistence
    # -------------------------------------------------------------
    def save_interaction(self, interaction: Interaction) -> None:
        self.ensure_learner(interaction.learner_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO interactions (
                    interaction_id, learner_id, item_id, item_text, item_concepts_json,
                    response_text, selected_option, correctness, latency_seconds, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction.interaction_id,
                    interaction.learner_id,
                    interaction.item_id,
                    interaction.item_text,
                    json.dumps(interaction.item_concepts),
                    interaction.response_text,
                    interaction.selected_option,
                    interaction.correctness,
                    interaction.latency_seconds,
                    interaction.timestamp.isoformat(),
                ),
            )

            conn.execute("UPDATE interactions SET payload_json=? WHERE interaction_id=?", (interaction.model_dump_json(), interaction.interaction_id))

    def get_learner_interactions(self, learner_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM interactions WHERE learner_id = ? ORDER BY rowid ASC",
                (learner_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def save_evidence(self, evidence: CognitiveEvidence) -> None:
        self.ensure_learner(evidence.learner_id)
        interp = evidence.candidate_interpretation
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cognitive_evidence (
                    evidence_id, interaction_id, learner_id, source_type, source_quote,
                    is_verbatim_quote, extracted_claim, polarity, expressed_uncertainty,
                    category, hypothesis_id, hypothesis_description, hypothesis_confidence,
                    concept_links_json, extractor_model, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.evidence_id,
                    evidence.interaction_id,
                    evidence.learner_id,
                    evidence.source_type,
                    evidence.source_quote,
                    1 if evidence.is_verbatim_quote else 0,
                    evidence.extracted_claim,
                    evidence.polarity,
                    evidence.expressed_uncertainty,
                    interp.category,
                    interp.hypothesis_id,
                    interp.description,
                    interp.hypothesis_confidence,
                    json.dumps([cl.model_dump() for cl in evidence.concept_links]),
                    evidence.extractor_model,
                    evidence.timestamp.isoformat(),
                ),
            )

            conn.execute("UPDATE cognitive_evidence SET payload_json=? WHERE evidence_id=?", (evidence.model_dump_json(), evidence.evidence_id))

    def get_learner_evidence_ledger(self, learner_id: str) -> List[CognitiveEvidence]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM cognitive_evidence WHERE learner_id = ? ORDER BY rowid ASC",
                (learner_id,),
            )
            rows = cursor.fetchall()
            evidences = []
            for r in rows:
                if r["payload_json"]:
                    evidences.append(CognitiveEvidence.model_validate_json(r["payload_json"]))
                    continue
                concept_links = [
                    ConceptLink(**cl) for cl in json.loads(r["concept_links_json"])
                ]
                ev = CognitiveEvidence(
                    evidence_id=r["evidence_id"],
                    interaction_id=r["interaction_id"],
                    learner_id=r["learner_id"],
                    source_type=r["source_type"],
                    source_quote=r["source_quote"],
                    is_verbatim_quote=bool(r["is_verbatim_quote"]),
                    extracted_claim=r["extracted_claim"],
                    polarity=r["polarity"],
                    expressed_uncertainty=r["expressed_uncertainty"],
                    concept_links=concept_links,
                    candidate_interpretation=CandidateInterpretation(
                        hypothesis_id=r["hypothesis_id"],
                        category=r["category"],
                        description=r["hypothesis_description"],
                        hypothesis_confidence=r["hypothesis_confidence"],
                    ),
                    extractor_model=r["extractor_model"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                evidences.append(ev)
            return evidences

    # -------------------------------------------------------------
    # Persistent State Persistence (Snapshot & Belief Records)
    # -------------------------------------------------------------
    def save_concept_state(self, learner_id: str, cstate: ConceptState) -> None:
        self.ensure_learner(learner_id)
        alpha = getattr(cstate, "alpha", 1.0)
        beta = getattr(cstate, "beta", 1.0)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO concept_states (
                    learner_id, concept_id, concept_name, alpha, beta, mastery_estimate,
                    uncertainty, evidence_sufficiency, interaction_count, last_interaction_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    learner_id,
                    cstate.concept_id,
                    cstate.concept_name,
                    alpha,
                    beta,
                    cstate.mastery_estimate,
                    cstate.uncertainty,
                    cstate.evidence_sufficiency,
                    cstate.interaction_count,
                    cstate.last_interaction_timestamp.isoformat() if cstate.last_interaction_timestamp else None,
                ),
            )

            # Replace the concept's candidate set so removed hypotheses cannot reappear.
            cursor.execute("DELETE FROM misconception_beliefs WHERE learner_id=? AND concept_id=?", (learner_id, cstate.concept_id))
            cursor.execute("DELETE FROM learner_snapshots WHERE learner_id=?", (learner_id,))
            # Persist candidate misconceptions
            for mid, m in cstate.candidate_misconceptions.items():
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO misconception_beliefs (
                        learner_id, concept_id, misconception_id, misconception_name,
                        posterior_probability, supporting_evidence_ids_json,
                        contradicting_evidence_ids_json, status, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        learner_id,
                        cstate.concept_id,
                        m.misconception_id,
                        m.misconception_name,
                        m.posterior_probability,
                        json.dumps(m.supporting_evidence_ids),
                        json.dumps(m.contradicting_evidence_ids),
                        m.status,
                        m.last_updated.isoformat(),
                    ),
                )


    def get_learner_snapshot(self, learner_id: str) -> Optional[LearnerStateSnapshot]:
        self.ensure_learner(learner_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            saved = cursor.execute("SELECT payload_json FROM learner_snapshots WHERE learner_id=?", (learner_id,)).fetchone()
            if saved:
                return LearnerStateSnapshot.model_validate_json(saved[0])
            cursor.execute("SELECT * FROM concept_states WHERE learner_id = ?", (learner_id,))
            rows = cursor.fetchall()
            if not rows:
                return None

            legacy_evidence = self.get_learner_evidence_ledger(learner_id)
            gaps = set()
            for ev in legacy_evidence:
                for link in ev.concept_links:
                    if ev.candidate_interpretation.category == "prerequisite_gap":
                        gaps.add(link.concept_id)
                    elif ev.candidate_interpretation.category == "mastery":
                        gaps.discard(link.concept_id)
            concept_states: Dict[str, ConceptState] = {}
            for r in rows:
                cid = r["concept_id"]
                # Fetch misconceptions
                cursor.execute(
                    "SELECT * FROM misconception_beliefs WHERE learner_id = ? AND concept_id = ?",
                    (learner_id, cid),
                )
                mrows = cursor.fetchall()
                misconceptions: Dict[str, MisconceptionBelief] = {}
                for mr in mrows:
                    misconceptions[mr["misconception_id"]] = MisconceptionBelief(
                        misconception_id=mr["misconception_id"],
                        misconception_name=mr["misconception_name"],
                        posterior_probability=mr["posterior_probability"],
                        supporting_evidence_ids=json.loads(mr["supporting_evidence_ids_json"]),
                        contradicting_evidence_ids=json.loads(mr["contradicting_evidence_ids_json"]),
                        status=mr["status"],
                        last_updated=datetime.fromisoformat(mr["last_updated"]),
                    )

                cs = ConceptState(
                    concept_id=cid,
                    concept_name=r["concept_name"],
                    mastery_estimate=r["mastery_estimate"],
                    uncertainty=r["uncertainty"],
                    evidence_sufficiency=r["evidence_sufficiency"],
                    candidate_misconceptions=misconceptions,
                    last_interaction_timestamp=datetime.fromisoformat(r["last_interaction_timestamp"]) if r["last_interaction_timestamp"] else None,
                    interaction_count=r["interaction_count"],
                    informative_evidence_count=sum(1 for ev in legacy_evidence if ev.candidate_interpretation.category not in ("ambiguous", "guessing") and ev.candidate_interpretation.hypothesis_confidence >= .5 and any(link.concept_id == cid for link in ev.concept_links)),
                )
                setattr(cs, "alpha", r["alpha"])
                setattr(cs, "beta", r["beta"])
                concept_states[cid] = cs

            return LearnerStateSnapshot(
                learner_id=learner_id,
                snapshot_id="S_LEGACY_" + learner_id,
                timestamp=max((state.last_interaction_timestamp for state in concept_states.values() if state.last_interaction_timestamp), default=datetime(1970, 1, 1, tzinfo=timezone.utc)),
                as_of_interaction_id="LEGACY",
                concept_states=concept_states,
                active_prerequisite_gaps=sorted(gaps),
                policy_version="beta-bayesian-v2.0",
            )

    # -------------------------------------------------------------
    # State Audit Events & Mathematical Traces
    # -------------------------------------------------------------
    def save_audit_event(self, event: StateChangeEvent) -> None:
        self.ensure_learner(event.learner_id)
        math_trace = getattr(event, "math_trace", {})
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO state_audit_events (
                    event_id, learner_id, concept_id, triggering_evidence_id,
                    prior_mastery, new_mastery, delta_mastery,
                    applied_rule, change_summary, math_trace_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.learner_id,
                    event.concept_id,
                    event.triggering_evidence_id,
                    event.math_trace.get("prior_mean_mastery", 0.5) if isinstance(math_trace, dict) else 0.5,
                    event.math_trace.get("posterior_mean_mastery", 0.5) if isinstance(math_trace, dict) else 0.5,
                    event.delta_mastery,
                    event.applied_rule,
                    event.change_summary,
                    json.dumps(math_trace),
                    event.timestamp.isoformat(),
                ),
            )

            conn.execute("UPDATE state_audit_events SET payload_json=? WHERE event_id=?", (event.model_dump_json(), event.event_id))

    def get_learner_audit_events(self, learner_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM state_audit_events WHERE learner_id = ? ORDER BY rowid ASC",
                (learner_id,),
            )
            rows = cursor.fetchall()
            events = []
            for r in rows:
                item = dict(r)
                item["math_trace"] = json.loads(item.pop("math_trace_json", "{}"))
                events.append(item)
            return events

    def clear_learner_data(self, learner_id: str) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table in ("learner_replay_baselines", "issued_probes", "retrieval_traces", "conversation_turns", "processed_requests", "learner_snapshots", "learning_turns", "learner_snapshot_history"):
                cursor.execute(f"DELETE FROM {table} WHERE learner_id=?", (learner_id,))
            cursor.execute("DELETE FROM state_audit_events WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM misconception_beliefs WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM concept_states WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM cognitive_evidence WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM interactions WHERE learner_id = ?", (learner_id,))

    def save_snapshot(self, snapshot: LearnerStateSnapshot) -> None:
        self.ensure_learner(snapshot.learner_id)
        with self._get_connection() as conn:
            for state in snapshot.concept_states.values():
                self.save_concept_state(snapshot.learner_id, state)
            conn.execute("INSERT OR IGNORE INTO learner_snapshot_history VALUES(?,?,?)", (snapshot.snapshot_id, snapshot.learner_id, snapshot.model_dump_json()))
            conn.execute("UPDATE learners SET updated_at=? WHERE learner_id=?", (utc_now_iso(), snapshot.learner_id))
            conn.execute("INSERT INTO learner_snapshots VALUES(?,?) ON CONFLICT(learner_id) DO UPDATE SET payload_json=excluded.payload_json", (snapshot.learner_id, snapshot.model_dump_json()))

    def get_request(self, learner_id: str, request_key: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT input_hash,response_json FROM processed_requests WHERE learner_id=? AND request_key=?", (learner_id,request_key)).fetchone()
            return dict(row) if row else None

    def save_request(self, learner_id: str, request_key: str, input_hash: str, response: dict):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO processed_requests VALUES(?,?,?,?)", (learner_id,request_key,input_hash,json.dumps(response,sort_keys=True)))

    def save_turn(self, learner_id: str, interaction_id: str, turn: dict):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO conversation_turns(learner_id,interaction_id,payload_json) VALUES(?,?,?)", (learner_id,interaction_id,json.dumps(turn)))

    def get_turns(self, learner_id: str):
        with self._get_connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM conversation_turns WHERE learner_id=? ORDER BY sequence", (learner_id,))]

    def save_retrieval_trace(self, learner_id: str, interaction_id: str, trace: dict):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO retrieval_traces VALUES(?,?,?)", (interaction_id,learner_id,json.dumps(trace)))

    def get_retrieval_traces(self, learner_id: str):
        with self._get_connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM retrieval_traces WHERE learner_id=? ORDER BY rowid", (learner_id,))]

    def issue_probe(self, learner_id: str, parent_interaction_id: str, probe_id: str):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO issued_probes(learner_id,parent_interaction_id,probe_id) VALUES(?,?,?)", (learner_id,parent_interaction_id,probe_id))

    def get_issued_probe(self, learner_id: str, parent_interaction_id: str, probe_id: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM issued_probes WHERE learner_id=? AND parent_interaction_id=? AND probe_id=?", (learner_id,parent_interaction_id,probe_id)).fetchone()
            return dict(row) if row else None

    def answer_probe(self, learner_id: str, parent_interaction_id: str, probe_id: str, interaction_id: str):
        with self._get_connection() as conn:
            conn.execute("UPDATE issued_probes SET answered_interaction_id=? WHERE learner_id=? AND parent_interaction_id=? AND probe_id=?", (interaction_id,learner_id,parent_interaction_id,probe_id))

    def save_replay_baseline(self, snapshot: LearnerStateSnapshot, excluded_evidence_ids: list[str]):
        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO learner_replay_baselines VALUES(?,?,?)", (snapshot.learner_id,snapshot.model_dump_json(),json.dumps(excluded_evidence_ids)))

    def get_replay_baseline(self, learner_id: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT snapshot_json,excluded_evidence_ids_json FROM learner_replay_baselines WHERE learner_id=?", (learner_id,)).fetchone()
            return (LearnerStateSnapshot.model_validate_json(row[0]), set(json.loads(row[1]))) if row else None

    def get_turn(self, learner_id: str, request_id: str):
        with self.connection() as conn:
            row = conn.execute("SELECT payload_json, request_hash FROM learning_turns WHERE learner_id = ? AND request_id = ?", (learner_id, request_id)).fetchone()
            return (json.loads(row[0]), row[1]) if row else None

    def save_learning_turn(self, learner_id: str, request_id: str, request_hash: str, payload: dict):
        with self.connection() as conn:
            conn.execute("INSERT INTO learning_turns (learner_id, request_id, request_hash, payload_json) VALUES (?, ?, ?, ?)",
                         (learner_id, request_id, request_hash, json.dumps(payload)))

    def get_learning_turns(self, learner_id: str):
        with self.connection() as conn:
            rows = conn.execute("SELECT payload_json FROM learning_turns WHERE learner_id = ? ORDER BY rowid", (learner_id,)).fetchall()
            return [json.loads(r[0]) for r in rows]

    def get_snapshot_history(self, learner_id: str):
        with self.connection() as conn:
            rows = conn.execute("SELECT payload_json FROM learner_snapshot_history WHERE learner_id = ? ORDER BY rowid", (learner_id,)).fetchall()
            return [json.loads(r[0]) for r in rows]
