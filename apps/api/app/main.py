from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
from pydantic import BaseModel, Field, ConfigDict, model_validator
import uuid
from packages.learner_state_core.pipeline import run_turn, ensure_snapshot
from packages.learner_state_core.observability import LearningTurn
from packages.learner_state_core.diagnosis.policy import INITIAL_ITEM, ITEM_BANK

from packages.learner_state_core.evidence.models import CognitiveEvidence
from packages.learner_state_core.evidence.extractor import (
    RuleBasedEvidenceExtractor,
    InstructorEvidenceExtractor,
)
from packages.learner_state_core.grounding.taxonomy import FRACTION_TAXONOMY
from packages.learner_state_core.state.models import LearnerStateSnapshot, StateChangeEvent
from packages.learner_state_core.state.engine import BayesianLearnerStateEngine
from packages.learner_state_core.db import LocalDatabaseManager
from packages.learner_state_core.diagnosis.prober import (
    FRACTION_DIAGNOSTIC_BANK,
)
from packages.learner_state_core.assessment.models import StartConversation, AnswerRequest, IdentityRequest, SourceUpdateRequest
from packages.learner_state_core.assessment.provider import AssessmentProvider
from packages.learner_state_core.assessment.service import AssessmentService
from packages.learner_state_core.assessment.reports import build_report
from packages.learner_state_core.assessment.execution import ExecutionStore, catalog
from packages.learner_state_core.assessment.research import research_status

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="Language-Grounded Learner State Modeling Platform",
    version="2.0.0",
    description="Transforms natural-language student interactions into structured Cognitive Evidence and maintains persistent, uncertainty-aware learner states.",
)

# Enable CORS for web console
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Azure OpenAI client
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

azure_client = None
if AZURE_ENDPOINT and AZURE_API_KEY and os.getenv("LEARNER_EXTRACTOR", "auto") != "rules":
    try:
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            timeout=20.0,
            max_retries=0,
        )
        print(f"Azure extraction configured: {AZURE_DEPLOYMENT}")
    except Exception as e:
        print(f"Warning: Failed to initialize Azure OpenAI: {e}")

# Persistent database manager & core domain services
db_manager = LocalDatabaseManager(os.getenv("LEARNER_DB_PATH"))
state_engine = BayesianLearnerStateEngine()
rule_extractor = RuleBasedEvidenceExtractor()
instructor_extractor = InstructorEvidenceExtractor(client=azure_client, model_name=AZURE_DEPLOYMENT)
assessment_provider = AssessmentProvider(azure_client, AZURE_DEPLOYMENT)


def assessments():
    return AssessmentService(db_manager, assessment_provider)


@app.get("/api/v2/status")
def assessment_status():
    return {"model_configured": assessment_provider.configured, "model": assessment_provider.model_name if assessment_provider.configured else None,
            "storage": "SQLite", "retrieval": "FTS5 / BM25", "mode": "saved_state_grounded_generation"}


@app.get("/api/v2/conversations")
def list_conversations(learner_id: str):
    return [{k: v for k, v in c.items() if k not in ("baseline_state", "ending_state")} for c in assessments().store.conversations(learner_id)]


@app.post("/api/v2/conversations")
def create_assessment(req: StartConversation):
    return assessments().create(req)


@app.get("/api/v2/conversations/{conversation_id}")
def get_assessment(conversation_id: str, learner_id: str):
    return assessments().session(conversation_id, learner_id)


@app.post("/api/v2/conversations/{conversation_id}/answers")
def answer_assessment(conversation_id: str, req: AnswerRequest):
    return assessments().answer(conversation_id, req)


@app.post("/api/v2/conversations/{conversation_id}/next")
def next_assessment_question(conversation_id: str, req: IdentityRequest):
    return assessments().ensure_question(conversation_id, req.learner_id)


@app.post("/api/v2/conversations/{conversation_id}/finish")
def finish_assessment(conversation_id: str, req: IdentityRequest):
    return assessments().finish(conversation_id, req.learner_id)


@app.post('/api/v2/conversations/{conversation_id}/sources')
def update_assessment_sources(conversation_id: str, req: SourceUpdateRequest):
    return assessments().update_sources(conversation_id, req)


@app.post("/api/v2/conversations/{conversation_id}/resume")
def resume_assessment(conversation_id: str, req: IdentityRequest):
    return assessments().resume(conversation_id, req.learner_id)


@app.get("/api/v2/conversations/{conversation_id}/report")
def assessment_report(conversation_id: str, learner_id: str):
    return build_report(assessments(), conversation_id, learner_id)


@app.get('/api/v2/observability/traces')
def list_nlp_executions(learner_id: str = Query(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$'),
                        conversation_id: str | None = None, question_id: str | None = None,
                        operation: str | None = Query(default=None, pattern=r'^(create|answer|generation)$'),
                        status: str | None = Query(default=None, pattern=r'^(running|completed|failed)$'),
                        limit: int = Query(default=30, ge=1, le=100), before: int | None = Query(default=None, ge=1)):
    store = ExecutionStore(db_manager)
    return {**store.list(learner_id, conversation_id, operation, status, question_id, limit, before),
            'summary': store.summary(learner_id, conversation_id)}


@app.get('/api/v2/observability/traces/{trace_id}')
def get_nlp_execution(trace_id: str, learner_id: str = Query(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')):
    result = ExecutionStore(db_manager).get(learner_id, trace_id)
    if result is None:
        raise HTTPException(404, 'Execution not found for this learner.')
    return result


@app.get('/api/v2/observability/techniques')
def nlp_techniques():
    return {'items': catalog(), 'scope': 'Current v2 assessment pipeline. Installed libraries do not imply use.'}


@app.get('/api/v2/observability/records')
def nlp_saved_records(learner_id: str = Query(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$'),
                      conversation_id: str | None = None, limit: int = Query(default=30, ge=1, le=100),
                      before: int | None = Query(default=None, ge=1)):
    return ExecutionStore(db_manager).records(learner_id, conversation_id, limit, before)


@app.get('/api/v2/observability/records/{turn_id}')
def nlp_saved_record(turn_id: str, learner_id: str = Query(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')):
    record = ExecutionStore(db_manager).record(learner_id, turn_id)
    if record is None:
        raise HTTPException(404, 'Saved response not found for this learner.')
    return record


@app.get('/api/v2/observability/research')
def nlp_research_status(learner_id: str = Query(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$'), run_id: str | None = None):
    try:
        return research_status(db_manager, learner_id, run_id)
    except ValueError:
        raise HTTPException(404, 'Evaluation run not found for this learner.') from None

class LearnerRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    learner_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=120)


class ProcessInteractionRequest(LearnerRequest):
    item_id: str = Field(min_length=1, max_length=120)
    item_text: str = Field(min_length=1, max_length=2000)
    item_concepts: List[str] = Field(default_factory=lambda: ["fraction_addition_unlike_denominators"])
    response_text: Optional[str] = Field(None, max_length=8000)
    selected_option: Optional[str] = Field(None, max_length=1000)
    correctness: Optional[int] = Field(None, ge=0, le=1)
    latency_seconds: Optional[float] = Field(None, ge=0)

    @model_validator(mode="after")
    def require_observation(self):
        if not self.response_text and not self.selected_option:
            raise ValueError("Provide an explanation or selected answer.")
        return self


class ChatRequest(LearnerRequest):
    message: str = Field(min_length=1, max_length=8000)
    item_id: Optional[str] = Field(None, max_length=120)
    item_text: Optional[str] = Field(None, max_length=2000)
    selected_option: Optional[str] = Field(None, max_length=1000)
    # Backwards compatible input; never used as trusted memory or model instructions.
    chat_history: List[dict] = Field(default_factory=list, max_length=50)


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "learner-state-platform",
        "version": "2.0.0",
        "azure_openai_configured": azure_client is not None,
        "provider_connectivity_verified": False,
        "azure_deployment": AZURE_DEPLOYMENT if azure_client else None,
        "concepts_loaded": len(FRACTION_TAXONOMY),
        "diagnostic_items_loaded": len(ITEM_BANK),
        "storage": "sqlite",
        "reply_mode": "state_policy",
        "extraction_mode": "pretrained_model" if azure_client else "rule_based",
    }


@app.get("/api/v1/taxonomy")
def get_curriculum_taxonomy():
    """Returns the full curriculum taxonomy including Knowledge Components and known misconceptions."""
    return FRACTION_TAXONOMY


@app.get("/api/v1/audit")
def get_dataset_audits():
    """Returns dataset suitability audit reports for benchmark datasets."""
    from packages.learner_state_core.data.eedi_loader import EediDataLoader
    from packages.learner_state_core.data.assistments_loader import FoundationalAssistLoader

    eedi_audit = EediDataLoader().audit()
    assist_audit = FoundationalAssistLoader().audit()

    return {
        "eedi_diagnostic_questions": eedi_audit,
        "foundational_assistments": assist_audit,
    }


@app.get("/api/v1/benchmarks")
def get_benchmark_results():
    """Returns the latest 3-tier comparative benchmark results."""
    import json
    from pathlib import Path
    results_path = Path("experiments/benchmark_results.json")
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "pending", "message": "Benchmark has not been run yet."}


@app.get("/api/v1/learners")
def list_learners():
    """Returns list of all active learners from persistent database."""
    records = db_manager.list_learners()
    return [r["learner_id"] for r in records]


@app.post("/api/v1/interactions/process", response_model=LearningTurn)
def process_student_interaction(req: ProcessInteractionRequest):
    return run_turn(req, db_manager, state_engine, instructor_extractor, rule_extractor)


@app.post("/api/v1/chat", response_model=LearningTurn)
def chat_with_tutor(req: ChatRequest):
    return run_turn(req, db_manager, state_engine, instructor_extractor, rule_extractor, conversational=True)


@app.get("/api/v1/learners/{learner_id}/session")
def get_learner_session(learner_id: str):
    with db_manager.transaction():
        snapshot = ensure_snapshot(db_manager, state_engine, learner_id)
        turns = db_manager.get_turns(learner_id)
        linked = {turn["evidence"]["evidence_id"] for turn in turns}
        legacy_evidence = [ev for ev in db_manager.get_learner_evidence_ledger(learner_id) if ev.evidence_id not in linked]
        return {"learner_id": learner_id, "state": snapshot, "turns": turns, "legacy_evidence": legacy_evidence,
                "active_question": turns[-1]["recommended_diagnostic_probe"] if turns else INITIAL_ITEM}


@app.get("/api/v1/learners/{learner_id}/traces")
def get_learner_traces(learner_id: str):
    return [turn["trace"] for turn in db_manager.get_turns(learner_id)]


@app.get("/api/v1/learners/{learner_id}/snapshots")
def get_learner_snapshots(learner_id: str):
    return db_manager.get_snapshot_history(learner_id)


@app.get("/api/v1/learners/{learner_id}/state", response_model=LearnerStateSnapshot)
def get_learner_state(learner_id: str):
    """Returns the persistent state snapshot for a given learner from SQLite."""
    with db_manager.transaction():
        return ensure_snapshot(db_manager, state_engine, learner_id)


@app.get("/api/v1/learners/{learner_id}/evidence", response_model=List[CognitiveEvidence])
def get_learner_evidence_ledger(learner_id: str):
    """Returns the complete immutable Cognitive Evidence ledger for a learner from SQLite."""
    return db_manager.get_learner_evidence_ledger(learner_id)


@app.get("/api/v1/learners/{learner_id}/events", response_model=List[StateChangeEvent])
def get_learner_state_events(learner_id: str):
    """Returns the complete audit log of state change events explaining 'why did the state change' from SQLite."""
    rows = db_manager.get_learner_audit_events(learner_id)
    events = []
    for r in rows:
        events.append(
            StateChangeEvent(
                event_id=r["event_id"],
                learner_id=r["learner_id"],
                concept_id=r["concept_id"],
                prior_snapshot_id=r.get("prior_snapshot_id"),
                new_snapshot_id=r.get("new_snapshot_id", ""),
                triggering_evidence_id=r["triggering_evidence_id"] or "",
                delta_mastery=r["delta_mastery"],
                prior_uncertainty=r.get("prior_uncertainty"),
                new_uncertainty=r.get("new_uncertainty"),
                change_summary=r["change_summary"],
                applied_rule=r["applied_rule"],
                math_trace=r.get("math_trace", {}),
                timestamp=datetime.fromisoformat(r["timestamp"]),
                prior_misconception_prob=r.get("prior_misconception_prob"),
                new_misconception_prob=r.get("new_misconception_prob"),
            )
        )
    return events


@app.post("/api/v1/learners/{learner_id}/replay")
def replay_learner_trajectory(learner_id: str):
    """
    Replays the entire sequence of evidence records from SQLite from scratch,
    proving that state is a pure deterministic projection over the evidence ledger.
    """
    ledger = db_manager.get_learner_evidence_ledger(learner_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="No evidence records found in database for this learner.")

    concept_ids = list(FRACTION_TAXONOMY.keys())
    trajectory = state_engine.replay_trajectory(learner_id, concept_ids, ledger)

    return {
        "learner_id": learner_id,
        "total_interactions_replayed": len(ledger),
        "final_snapshot": trajectory[-1][0],
        "replay_status": "SUCCESS_IDENTICAL_PROJECTION" if trajectory[-1][0].concept_states == db_manager.get_learner_snapshot(learner_id).concept_states and trajectory[-1][0].active_prerequisite_gaps == db_manager.get_learner_snapshot(learner_id).active_prerequisite_gaps else "PROJECTION_MISMATCH",
    }


@app.get("/api/v1/learners/{learner_id}/audit_report")
def get_learner_audit_report(learner_id: str):
    """
    Generates a deterministic mathematical audit report detailing:
    - How each claim was extracted from raw student text
    - Exact Beta distribution parameters (alpha, beta, E[L], sigma)
    - Step-by-step mathematical derivation for every state transition
    - Candidate misconception odds updates & contradiction discounts
    """
    snapshot = db_manager.get_learner_snapshot(learner_id)
    evidence_ledger = db_manager.get_learner_evidence_ledger(learner_id)
    audit_events = db_manager.get_learner_audit_events(learner_id)
    interactions = db_manager.get_learner_interactions(learner_id)

    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Learner {learner_id} not found in database.")

    # Build Markdown Report
    lines = [
        f"# Deterministic Cognitive State Audit Report: Learner `{learner_id}`",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Database Engine:** SQLite 3 (WAL Mode, Persistent on Disk)",
        f"**Mathematical Formulation:** Weighted Beta evidence updates & heuristic misconception tracking",
        "",
        "## 1. Executive Cognitive Profile",
        "| Concept | Mastery $E[L]$ | Uncertainty $\\sigma$ | $\\alpha$ | $\\beta$ | Sufficiency | Candidate Misconceptions |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for cid, cs in snapshot.concept_states.items():
        misc_str = ", ".join([f"{m.misconception_name} ({m.posterior_probability:.0%})" for m in cs.candidate_misconceptions.values()]) or "None"
        alpha = getattr(cs, "alpha", 1.0)
        beta = getattr(cs, "beta", 1.0)
        lines.append(
            f"| `{cid}` | **{cs.mastery_estimate * 100:.1f}%** | {cs.uncertainty * 100:.1f}% | {alpha:.2f} | {beta:.2f} | `{cs.evidence_sufficiency}` | {misc_str} |"
        )

    lines.extend([
        "",
        "## 2. Interaction & Cognitive Evidence Provenance",
        f"Total Recorded Interactions: **{len(interactions)}** | Extracted Evidence Units: **{len(evidence_ledger)}**",
        "",
    ])

    for idx, ev in enumerate(evidence_ledger, 1):
        interp = ev.candidate_interpretation
        lines.extend([
            f"### Evidence #{idx}: `{ev.evidence_id}`",
            f"- **Source Quote (Verbatim):** `\"{ev.source_quote}\"`",
            f"- **Extracted Atomic Claim:** {ev.extracted_claim}",
            f"- **Category:** `{interp.category}` | **Hypothesis:** `{interp.hypothesis_id}` ({interp.description})",
            f"- **Extractor Model:** `{ev.extractor_model}` | **Hypothesis Confidence:** {interp.hypothesis_confidence:.0%}",
            f"- **Grounding:** {', '.join([cl.concept_id for cl in ev.concept_links])}",
            "",
        ])

    lines.extend([
        "## 3. Mathematical State Transition Ledger (Step-by-Step Derivations)",
        "Every change in learner state is mathematically derived from the evidence ledger:",
        "",
    ])

    for idx, event in enumerate(audit_events, 1):
        mt = event.get("math_trace", {})
        lines.extend([
            f"#### Step {idx}: Rule `{event['applied_rule']}` on Concept `{event['concept_id']}`",
            f"- **Summary:** {event['change_summary']}",
            f"- **Delta Mastery:** {event['delta_mastery']:+.4f} (Prior: {event['prior_mastery']:.3f} -> New: {event['new_mastery']:.3f})",
        ])
        if mt:
            lines.extend([
                f"- **Beta Conjugate Math:** $\\alpha: {mt.get('prior_alpha')} \\to {mt.get('posterior_alpha')}$ ($\\Delta \\alpha = {mt.get('delta_alpha')}$), $\\beta: {mt.get('prior_beta')} \\to {mt.get('posterior_beta')}$ ($\\Delta \\beta = {mt.get('delta_beta')}$)",
                f"- **Epistemic Uncertainty:** $\\sigma_{{prior}} = {mt.get('prior_uncertainty_sd')} \\to \\sigma_{{post}} = {mt.get('posterior_uncertainty_sd')}$",
            ])
            if mt.get("misconception_likelihood_ratio"):
                lines.append(f"- **Misconception Bayes Factor $\\Lambda$:** {mt.get('misconception_likelihood_ratio')} ($P_{{prior}}={mt.get('prior_misconception_prob')} \\to P_{{post}}={mt.get('new_misconception_prob')}$)")
        lines.append("")

    report_markdown = "\n".join(lines)

    return {
        "learner_id": learner_id,
        "report_markdown": report_markdown,
        "total_interactions": len(interactions),
        "total_evidence": len(evidence_ledger),
        "total_audit_events": len(audit_events),
        "snapshot": snapshot,
    }
