# Product transformation audit — 3 November 2026

Checkpoint: `4eb32a9` on `main`; ongoing work: `improve`.

## Architecture and source inventory

React 19/Vite renders one workspace with practice, conversation history, reports and NLP observability views (`apps/web/src/components/AssessmentWorkspace.tsx`). The question editor, state/evidence inspector, reports and execution inspectors are separate components under `components/assessment`. Fetch clients live under `src/api`. Browser identity and unsent drafts use local storage; identity is not authentication.

FastAPI (`apps/api/app/main.py`) delegates v2 assessment operations to `assessment/service.py`. Structured model contracts are in `models.py`, Azure/Instructor calls in `provider.py`, lexical NLP in `nlp.py`, deterministic state/planning in `state.py`, SQLite/FTS5 persistence in `storage.py`, execution journaling in `execution.py`, and reports/research evaluation in their respective modules. Legacy v1 research routes remain supported. Database connections close deterministically; transactions use WAL, foreign keys and `BEGIN IMMEDIATE`. Migrations are additive.

## Capability matrix at checkpoint

| Capability | Frontend | Backend / persistence | Observability | Verification / classification |
|---|---|---|---|---|
| Assessment | AssessmentWorkspace, AssessmentChat, SourceRecovery, SavedDrafts | service.create/answer/ensure_question; conversations, questions, answers | create, answer, generation executions | API and browser harnesses exist; fresh regression pending |
| NLP preprocessing | ExecutionInspectors | nlp.analyze; original text and offsets saved in answer payload | preprocessing stage | deterministic offset/negation tests; implemented |
| Retrieval | LearnerInspector, ExecutionInspectors | storage.retrieve; scoped FTS5/BM25, documents, chunks | query, ranking, fallback and material references | scope and source recovery tests; implemented |
| AI extraction | feedback and evidence panels | provider.evaluate; Pydantic plus deterministic contracts | extraction attempts, model, usage | fixture failures/retries covered; live recheck pending |
| Cognitive Evidence | LearnerInspector, ExecutionInspectors | service.validate_observation; immutable answer evidence JSON | evidence, quote/rubric validation, grounding | provenance and rejection tests; implemented |
| Learner state | LearnerInspector; report concept rows | state.integrate; topic state and immutable state_history | before/after, weights, damping, held estimates | duplicate, abstention, contradiction tests; implemented |
| Adaptive planning | Why next? inspector; next question workflow | state.plan_next, provider.generate_question; question context persisted | planning/generation/validation/persistence | recovery and long-session tests; implemented |
| Reports | ConversationReport; print and JSON export | reports.build_report reads saved evidence | linked turn/execution IDs | completion and reload tests; implemented |
| Observability | NlpObservability, ExecutionInspectors | execution journal, stages, evaluation runs | best effort, retention bounded | partially implemented: filter/summary inconsistency and missing telemetry classification |

## Baseline and prioritized findings

- Frontend: 7 existing tests pass. Production build passes; main JavaScript is 587.69 kB (174.97 kB gzip), triggering the 500 kB warning. Reports and diagnostics are eagerly imported.
- Backend baseline: 107 tests passed with 32 existing pyBKT/NumPy deprecation warnings. Final regression results are recorded separately in `PRODUCT_TRANSFORMATION_REPORT.md`.
- P1: failed extraction retains text in a browser draft and best-effort trace, but has no durable submission record independent of successful evidence. Preserve rejected submissions without admitting them to learner state.
- P1: a successful business operation can have missing stage writes and still display an apparently complete trace. Track telemetry write failures and missing/running stages separately from business status.
- P1: summary totals ignore operation/status/question filters. Average duration weights status groups by all rows, including rows without duration. Use the same filtered population and only completed timed records.
- P2: explorer lacks date, execution ID, stage and error filters; its existing scoping must remain learner-bound.
- P2: styles.css and workspace-theme.css overlap and carry conflicting indigo/teal values. Centralize semantic colors and dimensions, reduce decorative emphasis, and retain readable state/evidence panels.
- P2: database exceptions have no explicit safe JSON response contract. Add request correlation and safe database/service error handling without leaking payloads or credentials.
- P2: history/report selection has no search/status/date controls. Add useful local filtering over real saved conversations.

## Visual critique before changes

The saved desktop capture in `docs/verification/workspace-feedback/question-desktop.png` agrees with the current component/CSS structure. The question is dominant and the actions are clear. However the saturated sidebar, active-item marker, boxed memory concepts and repeated explanation blocks compete with the question. Diagnostic controls occupy too much vertical space and summary labels are less precise than their data warrants. Keep the dominant question and secondary evidence pane, lighten navigation, use aligned rows for peer records, and put detailed qualifications behind disclosures. Preserve the existing mobile question/memory switch.

## Constraints

No model-generated success fallback; no production data resets; no fabricated operational metrics. FTS5/BM25 is lexical, learner estimates and planner weights are uncalibrated heuristics, and fixture/live synthetic checks are not learner studies. This remains a local research application without authenticated multi-user security. No new infrastructure or framework migration.
