# Integrated NLP Observability Implementation Plan

**Goal:** Inspect real learner-language processing, failures, evidence and persistent state without adding unrelated NLP demonstrations.

**Architecture:** Add an independent, best-effort SQLite execution ledger to the existing v2 assessment pipeline. Ordered stage records are committed independently of learner transactions, so rollbacks remain inspectable. A learner-scoped, paginated API feeds a polling React workspace using the incumbent typography, colors and sidebar.

## Inspection and implementation map

| Capability | Existing implementation | Assessment and change |
|---|---|---|
| Research context | Root research reference and PRD | Reviewed; no separate presentation file exists in the workspace. Preserve evidence/inference distinction and six unproven hypotheses. |
| Frontend | AssessmentWorkspace, chat, inspector, reports | Working v2 flow; extend navigation with a separate trace workspace. Preserve unsent drafts. |
| API | FastAPI main.py, AssessmentService | Working v2 topic, question, answer and report APIs; add scoped read-only trace endpoints. |
| Storage | LocalDatabaseManager, AssessmentStore | SQLite WAL, transactional answer/evidence/state bundles, optimistic snapshots and idempotency. Add tables/indexes without rewriting records. |
| NLP | assessment/nlp.py | NFKC, regex spans, sentence segmentation, lexical frequency, negation/uncertainty and fingerprints. Instrument actual outputs; preserve mathematical symbol spans. |
| Retrieval | AssessmentStore.retrieve | Real FTS5 Porter/unicode61 and BM25 with scoped recency fallback; expose ranks as ranks, not probabilities. |
| Model | AssessmentProvider | Azure + Instructor structured topic/question/evidence inference; record model, contract, usage, duration and bounded attempts. Do not claim access to model-internal tokenizer or reasoning. |
| Grounding | Saved question concept and topic graph | Constrained model association, not embedding retrieval. Dynamic map is model-authored and unverified, even when derived from supplied notes. |
| State/diagnosis | assessment/state.py | Deterministic criterion-weighted Beta updates, duplicate damping, contradiction weakening and rule-based adaptive planning. Uncalibrated, no information gain. |
| Existing traces | Turn extraction_trace, question generation_trace | Committed successful summaries only; add live stage lifecycle, failed attempts and persistence outcomes. Never reconstruct unrecorded historical timings. |
| Legacy research | pipeline.py, taxonomy, matcher, baselines, dataset loaders | Separate v1 compatibility flow. Legacy matcher embeddings are not used by current assessment UI. Preserve this distinction. |
| Evaluation | experiments/run_benchmark.py and stored JSON | Legacy results lack run provenance and compare current-text classification to future prediction. Display as historical/non-comparable, exclude unsupported improvements. Add reproducible operational audit metrics, keep research validity metrics pending. |
| Operations | uvicorn, Vite proxy, tests | Local research app, no production authentication or deploy configuration. Add explicit local access boundary and configurable trace retention. |

## Design decisions

Use existing light Operate surface, DM Sans, indigo accent and border tokens. Execution list + stage timeline + expanded input/output is the primary composition. Research and technique views are tabs, not extra navigation destinations. Refero Lightdash style d0f65d12-a8e6-4631-99f7-bb7cdcd5b6c5 and Weights & Biases execution-list patterns informed density and expandable detail; incumbent application is visual authority.

Only existing useful methods are exposed. Mathematical token spans extend actual preprocessing. No POS/dependency/NER/vector models are added to inflate syllabus coverage. Retrieval has lexical representations, not dense semantic vectors. Provider retry internals are not fabricated.

## Tasks (inline execution)

- [x] Add failing integration tests for successful/historical traces, cross-profile isolation, retries, failures, live running states, concurrent execution, transactional rollback, retention, pagination and truthful evaluation.
- [x] Implement execution ledger, technique registry and independent stage recording with safe error codes and capped payloads.
- [x] Instrument topic creation, generation, retrieval, preprocessing, extraction, grounding, validation, state computation and committed persistence. Link evidence and turn records to traces.
- [x] Add scoped trace APIs, operational scorecards and research hypothesis matrix. Historical records without stage events stay explicitly legacy. Added reproducible supplied-annotation diagnosis evaluation and run provenance.
- [x] Add responsive workspace with polling/backoff/abort, filters, pagination, technique lookup, transformation/evidence/state inspectors and JSON export. Permit navigation during model processing.
- [x] Run Python tests, frontend tests/build and browser integration on isolated synthetic interactions. Record outcomes and fix regressions.
- [x] Generate technical justification report including syllabus categories A–N, actual execution walkthrough, version/license sources, limitations and acceptance audit.

## Verification

Run `python -m pytest tests -q`, `npm test`, `npm run build`. Browser tests submit through the actual API to an isolated SQLite database with an explicitly labeled fixture provider; a separately labeled live-provider check can verify configured inference. No synthetic tests establish H1–H6. Never place fixture interactions in the user's learner database.
