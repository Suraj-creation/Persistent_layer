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


# Manage one shared SQLite file while keeping the research and v2 tables distinct.
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

    # Resolve the storage location and initialize the database.
    def __init__(self, db_path: Optional[str] = None):
        # Use the standard path when the API did not pass an override.
        if db_path is None:
            # Find the repository root from this source file.
            project_root = Path(__file__).resolve().parents[3]
            # Build the path to the repository's data directory.
            data_dir = project_root / "data"
            # Create the directory on first run.
            data_dir.mkdir(parents=True, exist_ok=True)
            # Use learner_state.db as the default database file.
            self.db_path = str(data_dir / "learner_state.db")
        # If the caller supplied an override, honor that location.
        else:
            # Store the resolved choice for future connections.
            self.db_path = db_path
            # Create the configured database file's parent directory when needed.
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # Keep the active connection local to the thread serving the request.
        self._local = threading.local()
        # Install tables and apply any necessary additive migrations.
        self._init_db()

    # Define a managed connection scope.
    @contextmanager
    # Reuse one SQLite connection across nested storage calls in the same thread.
    def _get_connection(self):
        # Check for a connection already serving the outer operation.
        active = getattr(self._local, "connection", None)
        # A nested storage method may already be inside a transaction.
        if active is not None:
            # Let it use that connection so its writes commit with the rest.
            yield active
            # Leave commit and cleanup to the outer owner.
            return
        # Open a configured connection for a new top-level operation.
        conn = open_connection(self.db_path)
        # Make it available to any nested storage calls.
        self._local.connection = conn
        # Always clear the thread-local pointer and close the connection afterward.
        try:
            # Let SQLite commit on success or roll back on error.
            with conn:
                # Expose the connection to the work within the scope.
                yield conn
        # Clean up whether the work passed or raised.
        finally:
            # Remove the pointer so later requests cannot reuse a closed connection.
            self._local.connection = None
            # Release the SQLite connection.
            conn.close()

    # Provide the connection interface used by assessment and other callers.
    def connection(self):
        """Shared connection scope used by the assessment workspace."""
        # Delegate to the transaction-aware connection scope.
        return self._get_connection()

    # Expose a managed connection that does not reuse the active learner write.
    @contextmanager
    # Open an independent connection for best-effort observability writes.
    def independent_connection(self):
        """Own a connection outside learner transactions for the trace journal."""
        # Create the separate connection for this journal operation.
        conn = open_connection(self.db_path)
        # Clean up the journal connection after it is used.
        try:
            # Commit or roll back its own work independently.
            with conn:
                # Run the journal action against this connection.
                yield conn
        # Guarantee cleanup when the journal action fails.
        finally:
            # Close the connection used by the journal.
            conn.close()

    # Expose a managed write transaction for a multi-record learner update.
    @contextmanager
    # Group related writes so failure rolls back the whole operation.
    def transaction(self):
        """Serialize state transitions across processes; rollback the entire event on failure."""
        # Use the connection nested save calls will share.
        with self._get_connection() as conn:
            # Acquire the write lock before reading or replacing the current state.
            conn.execute("BEGIN IMMEDIATE")
            # Run the caller's database changes within that write transaction.
            yield conn

    # Initialize storage for existing and new local databases.
    def _init_db(self):
        """Initializes tables and serializes additive schema migrations across workers."""
        # Use one database connection during schema setup.
        with self._get_connection() as conn:
            # Serialize schema decisions with an immediate write transaction.
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
            # Apply the research and learner table schema statement by statement.
            for statement in schema.split(";"):
                if statement.strip():
                    conn.execute(statement)
            # Install request-keyed accepted turn storage for the compatibility workspace.
            conn.execute("CREATE TABLE IF NOT EXISTS learning_turns (learner_id TEXT NOT NULL REFERENCES learners(learner_id), request_id TEXT NOT NULL, request_hash TEXT NOT NULL, payload_json TEXT NOT NULL, UNIQUE(learner_id,request_id))")
            # Index snapshot history by learner for trajectory lookup.
            conn.execute("CREATE INDEX IF NOT EXISTS snapshot_history_learner ON learner_snapshot_history(learner_id)")
            # Detect an older schema in which the latest state was stored with history.
            if "snapshot_id" in columns:
                # Recover the most recent historic state as the initial current projection.
                conn.execute("INSERT OR IGNORE INTO learner_snapshots SELECT learner_id, payload_json FROM learner_snapshot_history WHERE rowid IN (SELECT MAX(rowid) FROM learner_snapshot_history GROUP BY learner_id)")
            # Preserve the training branch's existing current rows in history too.
            # Walk existing current rows to preserve them in history as well.
            for row in conn.execute("SELECT learner_id,payload_json FROM learner_snapshots").fetchall():
                # Validate old state JSON before using its ID in history.
                snapshot = LearnerStateSnapshot.model_validate_json(row[1])
                # Save each old snapshot only if it has not already been copied.
                conn.execute("INSERT OR IGNORE INTO learner_snapshot_history VALUES(?,?,?)", (snapshot.snapshot_id,row[0],row[1]))
            # Load the separate live-assessment schema installer.
            from ..assessment.storage import install_schema
            # Load the NLP execution journal schema installer.
            from ..assessment.execution import install_schema as install_execution_schema
            # Load the expert-annotation evaluation schema installer.
            from ..assessment.evaluation import install_schema as install_evaluation_schema
            # Create the v2 learner, document, question, answer and state tables.
            install_schema(conn)
            # Create the execution journal tables beside accepted data.
            install_execution_schema(conn)
            # Create the research evaluation tables.
            install_evaluation_schema(conn)

    # -------------------------------------------------------------
    # Learner Operations
    # -------------------------------------------------------------
    # Ensure the learner row exists before saving related fraction research records.
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
    # Save the original fraction-workspace interaction and learner links.
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

    # Load original interactions for research and diagnostic history.
    def get_learner_interactions(self, learner_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM interactions WHERE learner_id = ? ORDER BY rowid ASC",
                (learner_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    # Persist the quoted evidence and its structured interpretation.
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

    # Rebuild research evidence records from the saved ledger rows.
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
    # Persist per-concept evidence masses and candidate misconception beliefs.
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


    # Load the current fraction learner state, with support for older databases.
    def get_learner_snapshot(self, learner_id: str) -> Optional[LearnerStateSnapshot]:
        # Ensure the research learner can own the requested records.
        self.ensure_learner(learner_id)
        with self._get_connection() as conn:
            # Prepare a cursor for the current or migrated state lookup.
            cursor = conn.cursor()
            # Prefer the current snapshot payload when it has already been saved.
            saved = cursor.execute("SELECT payload_json FROM learner_snapshots WHERE learner_id=?", (learner_id,)).fetchone()
            # Use the JSON projection as the authoritative current state.
            if saved:
                # Validate and reconstruct the typed snapshot.
                return LearnerStateSnapshot.model_validate_json(saved[0])
            # Fall back to the older per-concept state tables if needed.
            cursor.execute("SELECT * FROM concept_states WHERE learner_id = ?", (learner_id,))
            rows = cursor.fetchall()
            # A learner with no projected or legacy state has no saved snapshot.
            if not rows:
                return None

            # Inspect historical evidence to reconstruct unresolved prerequisite flags.
            legacy_evidence = self.get_learner_evidence_ledger(learner_id)
            gaps = set()
            # Walk historical observations in the order returned by the ledger.
            for ev in legacy_evidence:
                for link in ev.concept_links:
                    # Add an active flag where the older evidence showed a prerequisite gap.
                    if ev.candidate_interpretation.category == "prerequisite_gap":
                        gaps.add(link.concept_id)
                    # Remove the flag when later mastery of that concept appears.
                    elif ev.candidate_interpretation.category == "mastery":
                        gaps.discard(link.concept_id)
            # Rebuild typed concept entries and their saved beliefs.
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
                # Restore each misconception belief and both sets of evidence references.
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

                # Reconstruct the concept state from its legacy projection and ledger.
                cs = ConceptState(
                    concept_id=cid,
                    concept_name=r["concept_name"],
                    mastery_estimate=r["mastery_estimate"],
                    uncertainty=r["uncertainty"],
                    evidence_sufficiency=r["evidence_sufficiency"],
                    candidate_misconceptions=misconceptions,
                    last_interaction_timestamp=datetime.fromisoformat(r["last_interaction_timestamp"]) if r["last_interaction_timestamp"] else None,
                    interaction_count=r["interaction_count"],
                    # Recount informative evidence while excluding ambiguity and guessing.
                    informative_evidence_count=sum(1 for ev in legacy_evidence if ev.candidate_interpretation.category not in ("ambiguous", "guessing") and ev.candidate_interpretation.hypothesis_confidence >= .5 and any(link.concept_id == cid for link in ev.concept_links)),
                )
                # Restore the stored positive evidence mass.
                setattr(cs, "alpha", r["alpha"])
                # Restore the stored negative evidence mass.
                setattr(cs, "beta", r["beta"])
                concept_states[cid] = cs

            # Return a compatible snapshot marked with its legacy policy and identity.
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
    # Persist each research policy change and its mathematical trace.
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

    # Read sandbox audit events for inspection or replay checks.
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

    # Publish a fraction snapshot to its current and historical storage views.
    def save_snapshot(self, snapshot: LearnerStateSnapshot) -> None:
        # Ensure the snapshot has an owner row.
        self.ensure_learner(snapshot.learner_id)
        # Use one connection for the snapshot and associated concept projections.
        with self._get_connection() as conn:
            # Walk the current concept states carried by this snapshot.
            for state in snapshot.concept_states.values():
                # Save the denormalized concept records used by earlier callers.
                self.save_concept_state(snapshot.learner_id, state)
            # Append the immutable snapshot payload once.
            conn.execute("INSERT OR IGNORE INTO learner_snapshot_history VALUES(?,?,?)", (snapshot.snapshot_id, snapshot.learner_id, snapshot.model_dump_json()))
            # Update the learner's last activity time.
            conn.execute("UPDATE learners SET updated_at=? WHERE learner_id=?", (utc_now_iso(), snapshot.learner_id))
            # Replace the learner's current projection with the new snapshot.
            conn.execute("INSERT INTO learner_snapshots VALUES(?,?) ON CONFLICT(learner_id) DO UPDATE SET payload_json=excluded.payload_json", (snapshot.learner_id, snapshot.model_dump_json()))

    # Read a sandbox idempotency result before processing an interaction again.
    def get_request(self, learner_id: str, request_key: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT input_hash,response_json FROM processed_requests WHERE learner_id=? AND request_key=?", (learner_id,request_key)).fetchone()
            return dict(row) if row else None

    # Save the sandbox response and request fingerprint for retries.
    def save_request(self, learner_id: str, request_key: str, input_hash: str, response: dict):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO processed_requests VALUES(?,?,?,?)", (learner_id,request_key,input_hash,json.dumps(response,sort_keys=True)))

    # Save a sandbox tutor or diagnostic turn for conversation history.
    def save_turn(self, learner_id: str, interaction_id: str, turn: dict):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO conversation_turns(learner_id,interaction_id,payload_json) VALUES(?,?,?)", (learner_id,interaction_id,json.dumps(turn)))

    # Read sandbox turns in order.
    def get_turns(self, learner_id: str):
        with self._get_connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM conversation_turns WHERE learner_id=? ORDER BY sequence", (learner_id,))]

    # Save the cited store result for this sandbox interaction.
    def save_retrieval_trace(self, learner_id: str, interaction_id: str, trace: dict):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO retrieval_traces VALUES(?,?,?)", (interaction_id,learner_id,json.dumps(trace)))

    # Read the earlier search results for the same learner.
    def get_retrieval_traces(self, learner_id: str):
        with self._get_connection() as conn:
            return [json.loads(row[0]) for row in conn.execute("SELECT payload_json FROM retrieval_traces WHERE learner_id=? ORDER BY rowid", (learner_id,))]

    # Record that this learner was issued this probe for that interaction.
    def issue_probe(self, learner_id: str, parent_interaction_id: str, probe_id: str):
        with self._get_connection() as conn:
            conn.execute("INSERT INTO issued_probes(learner_id,parent_interaction_id,probe_id) VALUES(?,?,?)", (learner_id,parent_interaction_id,probe_id))

    # Verify that a diagnostic answer belongs to an issued probe.
    def get_issued_probe(self, learner_id: str, parent_interaction_id: str, probe_id: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM issued_probes WHERE learner_id=? AND parent_interaction_id=? AND probe_id=?", (learner_id,parent_interaction_id,probe_id)).fetchone()
            return dict(row) if row else None

    # Link the accepted interaction that answered a probe.
    def answer_probe(self, learner_id: str, parent_interaction_id: str, probe_id: str, interaction_id: str):
        with self._get_connection() as conn:
            conn.execute("UPDATE issued_probes SET answered_interaction_id=? WHERE learner_id=? AND parent_interaction_id=? AND probe_id=?", (interaction_id,learner_id,parent_interaction_id,probe_id))

    # Store the preserved starting point for replay across a policy migration.
    def save_replay_baseline(self, snapshot: LearnerStateSnapshot, excluded_evidence_ids: list[str]):
        with self._get_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO learner_replay_baselines VALUES(?,?,?)", (snapshot.learner_id,snapshot.model_dump_json(),json.dumps(excluded_evidence_ids)))

    # Retrieve the starting snapshot and the earlier evidence already accounted for.
    def get_replay_baseline(self, learner_id: str):
        with self._get_connection() as conn:
            row = conn.execute("SELECT snapshot_json,excluded_evidence_ids_json FROM learner_replay_baselines WHERE learner_id=?", (learner_id,)).fetchone()
            return (LearnerStateSnapshot.model_validate_json(row[0]), set(json.loads(row[1]))) if row else None

    # Check whether a compatibility request has already saved a learning turn.
    def get_turn(self, learner_id: str, request_id: str):
        with self.connection() as conn:
            row = conn.execute("SELECT payload_json, request_hash FROM learning_turns WHERE learner_id = ? AND request_id = ?", (learner_id, request_id)).fetchone()
            return (json.loads(row[0]), row[1]) if row else None

    # Store the compatibility turn and its request fingerprint together.
    def save_learning_turn(self, learner_id: str, request_id: str, request_hash: str, payload: dict):
        with self.connection() as conn:
            conn.execute("INSERT INTO learning_turns (learner_id, request_id, request_hash, payload_json) VALUES (?, ?, ?, ?)",
                         (learner_id, request_id, request_hash, json.dumps(payload)))

    # Read compatibility turns in insertion order to recover the current task.
    def get_learning_turns(self, learner_id: str):
        with self.connection() as conn:
            rows = conn.execute("SELECT payload_json FROM learning_turns WHERE learner_id = ? ORDER BY rowid", (learner_id,)).fetchall()
            return [json.loads(r[0]) for r in rows]

    # Load immutable fraction snapshots for inspection and replay comparison.
    def get_snapshot_history(self, learner_id: str):
        with self.connection() as conn:
            rows = conn.execute("SELECT payload_json FROM learner_snapshot_history WHERE learner_id = ? ORDER BY rowid", (learner_id,)).fetchall()
            return [json.loads(r[0]) for r in rows]
