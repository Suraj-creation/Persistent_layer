from __future__ import annotations

import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
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

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency and write safety
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Initializes tables if they do not exist."""
        with self._get_connection() as conn:
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

            conn.commit()

    # -------------------------------------------------------------
    # Learner Operations
    # -------------------------------------------------------------
    def ensure_learner(self, learner_id: str, name: Optional[str] = None) -> None:
        """Ensures learner exists in database."""
        now = utc_now_iso()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT learner_id FROM learners WHERE learner_id = ?", (learner_id,))
            if not cursor.fetchone():
                display_name = name or learner_id.replace("_", " ").title()
                cursor.execute(
                    "INSERT INTO learners (learner_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (learner_id, display_name, now, now),
                )
                conn.commit()

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
                INSERT OR REPLACE INTO interactions (
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
            conn.commit()

    def get_learner_interactions(self, learner_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM interactions WHERE learner_id = ? ORDER BY timestamp ASC",
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
                INSERT OR REPLACE INTO cognitive_evidence (
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
            conn.commit()

    def get_learner_evidence_ledger(self, learner_id: str) -> List[CognitiveEvidence]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM cognitive_evidence WHERE learner_id = ? ORDER BY timestamp ASC",
                (learner_id,),
            )
            rows = cursor.fetchall()
            evidences = []
            for r in rows:
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

            conn.commit()

    def get_learner_snapshot(self, learner_id: str) -> Optional[LearnerStateSnapshot]:
        self.ensure_learner(learner_id)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM concept_states WHERE learner_id = ?", (learner_id,))
            rows = cursor.fetchall()
            if not rows:
                return None

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
                )
                setattr(cs, "alpha", r["alpha"])
                setattr(cs, "beta", r["beta"])
                concept_states[cid] = cs

            return LearnerStateSnapshot(
                learner_id=learner_id,
                as_of_interaction_id="LATEST",
                concept_states=concept_states,
                active_prerequisite_gaps=[],
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
            conn.commit()

    def get_learner_audit_events(self, learner_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM state_audit_events WHERE learner_id = ? ORDER BY timestamp ASC",
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
            cursor.execute("DELETE FROM state_audit_events WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM misconception_beliefs WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM concept_states WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM cognitive_evidence WHERE learner_id = ?", (learner_id,))
            cursor.execute("DELETE FROM interactions WHERE learner_id = ?", (learner_id,))
            conn.commit()
