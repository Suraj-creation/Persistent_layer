# Conversation learning implementation plan

Goal: Replace the demo-facing fixed fractions workflow with real topic-based adaptive conversations, local retrieval, explainable learner-state updates and per-conversation report cards.

User direction: Topic plus optional notes; clearly label topic-only model knowledge. Remove Research Studio and seeded profiles/content. Add unobtrusive anchored guidance. Learning improvement must be checked through subsequent learner responses, not inferred from showing coaching.

Architecture: Keep existing research data intact. Add conversation-scoped assessment tables, a reusable per-learner/topic state, immutable answer evidence and snapshots, and FTS5 retrieval of learning material and prior evidence. The LLM proposes curriculum, rubric-bound evidence and novel questions. Deterministic validation, updates, target selection, concurrency checks, report construction and persistence govern the loop.

## Deliverables

- [x] Tests: no seeded conversations, restart/resume, grounded generation, exact rubric/quote validation, contradictions, repetition damping, cross-conversation topic memory, isolation, atomicity, stale writes and reports.
- [x] NLP/retrieval: preserve raw text; NFKC normalization, token and sentence spans, keyword extraction, overlapping source chunks, Porter/Unicode FTS5 and BM25 with source references and explicit source kinds.
- [x] Core: dynamic topic maps; criterion-level extraction; weighted Beta updates, confidence holds, misconception provenance, deterministic adaptive plans, state-driven generated questions and targeted coaching.
- [x] Backend: additive SQLite tables, typed API, idempotent conversation creation/answering, report endpoints and safe provider failure/retry behavior.
- [x] UI: empty-first setup; saved conversation sidebar; guided assessment chat; state/evidence/decision inspector; conversation history and structured reports; anchored contextual tour.
- [x] Validation: isolated tests, frontend build, live-provider multi-turn flow, desktop/mobile browser checks, existing-data preservation and updated documentation.

## State contract

Per concept store alpha/beta, estimate, standard deviation, informative observations, exact evidence references, active hypotheses and missing criteria. Effective weight depends on extraction confidence, response informativeness and duplicate evidence. Ambiguous/low-confidence evidence holds the numeric state. Discount only explicitly contradicted hypotheses in the same concept. Store inputs, formulas and before/after snapshots. Every report labels estimates as uncalibrated and describes observed evidence of progress rather than asserting measured learning gains.

## Generation contract

The planner selects concept, intent, difficulty, reasons and evidence references from SQLite. Retrieval selects material/evidence scoped to the learner/topic. Persist and hash the exact generation context; never pass browser history or an unpersisted response to question generation. Validate target IDs, citations, rubric and novelty before activating a question. Invalid extraction leaves state unchanged. Question-generation failure preserves the committed answer and exposes a generation-only retry; no static question is invented.

## UI contract

No sample learners, prefilled answers, question arrays or seeded conversations. The browser owns only a local anonymous identity and display preferences; assessment content comes from API records. Reports include question, expected focus, original answer, evidence, criteria, state computation, retrieval, next-question rationale, concept progress and remaining gaps. The tour highlights real controls, can be dismissed/restarted and does not obstruct answering.

## Verification evidence

- 61 backend tests passed with isolated storage and fixture models. Legacy pyBKT emits NumPy deprecation warnings.
- TypeScript/Vite production build passed; mechanical UI detector reported no findings.
- Synthetic live Azure flows verified both supplied-notes NLP assessment and topic-only SQL assessment, saved-state generation, and retrieval of prior learner evidence.
- Browser checks covered empty-first setup, absence of demo profiles/Research Studio, generated questions, exact evidence, question-scoped draft restoration, reports, history, desktop/mobile/tablet layout, guided highlighting, all report details printed/restored, and no JavaScript exceptions.
- Existing local database migrated on a temporary SQLite backup with all original record counts retained. Running v2 status endpoint returned HTTP 200 in saved_state_grounded_generation mode.
