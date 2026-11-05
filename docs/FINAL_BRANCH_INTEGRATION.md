# Final branch integration

Date: 2026-11-05. This is a software integration, not a new training experiment.

## Branch composition

`final` starts from `improve` at `7395a1e` and merges `training` at `105fe56` with both parents retained. The source branches are preserved.

The improved assessment workspace remains the default screen. Its conversation flow, criterion-based learner memory, reports, source recovery, durable submissions, drafts, guide, mathematical rendering and NLP observability remain in place. Its component styles, tokens and assessment components are retained from `improve`.

The new **Training & retrieval** sidebar view (`/#training`) loads the retained training tutor, lab and researcher tools. The training screens use locally compiled, scoped Tailwind utilities; they require no CDN and do not reset assessment styling. The initial assessment bundle does not load the research components.

## Retained training work

- Aggregate ASSISTments BKT checkpoints, fitted skill parameters, two student holdout splits and comparison metrics.
- Dataset audits, licensing/access notes, training scripts, experiment reports and slide deliverables.
- Fraction evidence extraction, cautious guessing/negation handling, deterministic state and audit IDs, targeted misconception counterevidence and replay.
- Atomic interaction/evidence/state/event/receipt writes; diagnostic issuance checks and server-side answer grading; retry-safe saved conversations.
- Persistent graph/FTS/hash-vector retrieval, five query modes, citations, relationship traces, namespace isolation and abstention.

BKT is an offline correctness baseline and is not connected to live assessment estimates. MAP is still untrained because authorized training data is absent. Eedi research graph data is not individual learner evidence. No new fitted weights, improved paper benchmark scores or learning gains are claimed by this merge.

## API integration

| Surface | Contract |
| --- | --- |
| Primary assessment workspace | `/api/v2/*`; improved topic/conversation APIs and observability are retained. |
| Training research sandbox | `/api/v1/*`; training reports, graph retrieval, fraction chat/lab, diagnostic answers, evidence and replay are retained. Writes use the `Idempotency-Key` header. |
| Older improve fraction conversation protocol | Write routes `/api/workspace/v1/chat` and `/api/workspace/v1/interactions/process`; body `request_id`, saved active question and `LearningTurn` trace contract are retained separately. |
| Legacy session/trace/history reads | `/api/workspace/v1/learners/{id}/session`, `/traces`, `/snapshots`, with their original `/api/v1/learners/{id}/…` aliases. Shared state/evidence/events/replay reads remain under `/api/v1`. |

The fraction POST protocols have different request and response formats. Namespacing the older improve protocol avoids interpreting training requests using a different grading or retry policy. External callers of improve's former v1 POST routes must use the workspace prefix. The improved frontend already uses v2, so its requests are unchanged.

Azure configuration remains available for v2 generation. The fraction sandbox defaults to rules and deterministic replies, with its separate optional Azure flags. Provider calls have bounded timeouts and no SDK retries. Safe JSON transport errors and request correlation apply to both surfaces. Neither research checkpoints nor provider configuration are exposed as executable downloads through the API.

## Persistence and migration

The branches used incompatible `learner_snapshots` schemas. The final schema stores a current projection in `learner_snapshots` and immutable history in `learner_snapshot_history`.

- Existing improve histories are copied in commit order and the latest snapshot per learner becomes the current projection.
- Existing training current snapshots and their full JSON are retained and also recorded in history.
- Migration is transactional, repeatable and serialized across concurrent startup workers. A current projection deliberately invalidated by a standalone concept update is not resurrected from stale history.
- Both receipt formats keep separate turn tables and explicit persistence methods; nested assessment and research writes share a transaction-aware connection scope.
- Observability uses a separately owned connection so rejected or rolled-back learner operations can remain inspectable. Its journal remains best effort.
- Assessment source/evidence retrieval stays scoped by learner, topic and conversation source IDs. Public curriculum and Eedi research graph namespaces remain separate.

Database paths support `LEARNER_DB_PATH` and the training-compatible `LEARNER_STATE_DB_PATH` (the latter takes precedence). Local databases, raw data, backups and generated browser artifacts are ignored by Git. Verification uses disposable databases and synthetic fixture inference; pytest collection is isolated before API imports.

## Verification on the combined code

| Check | Result |
| --- | --- |
| Full Python suite | **210 passed**; 32 upstream pyBKT/NumPy deprecation warnings. |
| Added integration coverage | Six checks: both historical snapshot schemas with concurrent/repeated startup; joint snapshot/history rollback; standalone concept invalidation across restart; distinct conversation receipt formats; assessment and graph retrieval coexistence. Included in the full suite. |
| Frontend unit tests | **7 passed**. |
| TypeScript and production Vite build | **Passed**. Research UI, reports, observability and math retain separate lazy bundles. |
| Playwright integrated flows | **7 passed**: default workspace/style retention, lab/diagnostics/replay, persisted chat and learner isolation, research/retrieval/audit views, mobile error handling, transport retry, truncated-JSON retry. |
| Improved workspace browser harness | **16 checks passed**; responsive widths 1440, 1024, 768, 390 and 320 px; no JavaScript errors. |
| NLP observability browser harness | **16 checks passed**; no JavaScript errors. |
| Dependency and patch checks | `uv lock --check`, `git diff --check`, and frontend install audit passed (zero npm vulnerabilities at installation). |
| Local app smoke checks | API healthy; Vite proxy to API port 8002 works. No model is currently configured, so new v2 assessments require provider setup. |

Machine-readable counts and command logs are in [the verification summary](verification/final-merge/verification-summary.json) and its directory.

Browser evidence: [workspace checks](verification/final-merge/workspace/results.json), [observability execution](verification/final-merge/observability/fixture-execution.json), and screenshots in the same directories. Fixture inference checks software contracts; it does not establish model quality or educational outcomes. Production hosted-model behavior was not re-evaluated by this merge.

## Run

```bash
.venv/bin/python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8002
```

In another terminal:

```bash
cd apps/web
npm ci
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open <http://localhost:5173>. Research tools are at <http://localhost:5173/#training>. Vite defaults to API port 8002; `API_PROXY_TARGET` can override it. The app remains a local single-user research prototype, with the research limits documented in the README and experiment reports.
