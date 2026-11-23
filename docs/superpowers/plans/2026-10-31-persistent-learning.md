# Persistent learning implementation plan

Goal: Implement the attached research brief's Language → Evidence → Learner State → Diagnosis → Updated Evidence loop in the existing local fractions prototype.

Architecture: SQLite is the source of truth for complete snapshots, immutable evidence, conversations, and decision traces. A pretrained model may propose evidence; deterministic validation, state integration, and question selection control what is persisted and shown. Student UI leads with a researcher evidence inspector alongside it.

## Constraints

- Preserve existing local data; use additive SQLite schema migrations.
- No training, new cloud database, or unbounded autonomous tutor.
- Confidence values and question-selection likelihoods are heuristic research parameters, not calibrated measurements.
- This remains a single-user local research prototype; learner switching is not authentication.

## Tasks

- [x] Add regression tests for durable snapshots, student isolation, grounded follow-ups, idempotency, rollback, and fabricated extraction.
- [x] Add transactional storage, snapshot history, persistent conversations and decision traces.
- [x] Consolidate interaction processing; validate quotes, canonical concepts and answer grading; choose curated follow-ups from database state.
- [x] Build a responsive student workspace, saved profile switching, learner-state inspector and researcher audit views.
- [x] Verify backend tests, production frontend build and desktop/mobile interaction paths. Document setup and research limits.

## Acceptance

A student can answer, receive a state-selected follow-up, answer that exact question, reload, and continue. Another student sees independent history. Every decision exposes the exact saved snapshot, evidence IDs, extraction mode/fallback, timings and selected item; database failures leave no partial learner update.

## Verification evidence

- Backend: 39 tests passed using isolated offline storage. Existing pyBKT emits NumPy deprecation warnings.
- Frontend: TypeScript and Vite production build passed.
- Browser: answer, saved follow-up, trace, reload, second response, learner isolation, replay, desktop/mobile/tablet, profile-creation race protection, evaluation outputs; no JavaScript exceptions.
- Provider: synthetic Azure extraction succeeded with no fallback; reply mode remained state_policy.
- Legacy migration: verified on a temporary SQLite backup of the existing local database; original evidence and projection rows retained.
