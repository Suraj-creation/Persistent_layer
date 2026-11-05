# Product transformation implementation plan

**Goal:** Close audited product reliability and observability gaps on `improve`, refine the existing interface, and verify the real persisted workflow.

**Architecture:** Keep React/Vite, FastAPI and SQLite. Add durable rejected-submission storage separately from accepted evidence. Share execution query filters between list and aggregate statistics. Keep telemetry best effort and expose its incompleteness.

**Spec:** `docs/PRODUCT_TRANSFORMATION_AUDIT.md` and the user's complete transformation brief.

- [x] Save existing progress and switch to `improve`.
- [x] Inspect source, contracts, schemas, UI capture and existing verification.
- [x] Finish baseline backend tests and record results.
- [x] Add semantic design tokens, improve navigation/record density and lazy-load reports/observability; verify build and viewports.
- [x] Add durable submission recovery and safe API errors; test failed extraction, rollback, retry, ownership and immutable history using temporary SQLite databases.
- [x] Add shared execution filters, truthful latency/retry/failure summaries and explicit incomplete telemetry detection; test persisted events and filtered aggregates.
- [x] Connect search/date/stage/error filters and truthful metrics to the execution UI; retain current evidence/state inspectors and export.
- [x] Add history/report filtering and verify loading/error/empty/keyboard states.
- [x] Run frontend and backend regression, fixture browser checks, one live assessment cycle, and SQL integrity/relationship checks. Keep artifacts separate from prior verification.
- [x] Document exact results, remaining gaps and production limits; commit verified improvements locally.

Validation uses existing pytest, Node tests and Python Playwright harnesses. No additional runtime dependencies are needed. A passing UI or HTTP response alone is insufficient: check SQLite answer/question/evidence/snapshot relationships and retained failed submissions directly.

Final evidence and limitations: `docs/PRODUCT_TRANSFORMATION_REPORT.md`. The final local commit records this implementation and its verification artifacts.
