# Product transformation report

Verified 3 November 2026. Checkpoint: `4eb32a9` on `main`. Implementation branch: `improve`.

## 1. Executive summary

The existing React/FastAPI/SQLite product now has a coherent light workspace, searchable conversation/report collections, a filtered execution explorer with consistent metrics, durable recovery of rejected submissions, safe correlated API errors, and explicit separation of business success from incomplete telemetry. Existing evidence validation, state-update formulas and adaptive planning were preserved.

Final verification passed: **117 backend tests, 7 frontend client/state tests, 16 new browser checks, 16 existing fixture browser checks, and one live configured-model assessment cycle with 16 checks**. The live run committed a real answer, evidence, snapshot and next question to a disposable SQLite database. All model content used in verification is synthetic; none is research evaluation data.

This is suitable for continued local research and supervised evaluation. It is not ready for public, authenticated multi-user deployment or validated educational measurement.

## 2. Frontend improvements

| Surface | Implemented behavior |
|---|---|
| Application shell | Shared semantic colors, typography sizes, spacing and dimensions in `tokens.css`; quiet navigation; clear active section; compact header; skip-to-workspace link and visible keyboard focus |
| Assessment | Dominant question/editor, quieter memory rows, concise setup, recovered-submission notice; original failed text restored from SQLite when no browser draft exists |
| Evidence and learner state | Existing exact quotes, criteria, hypotheses, estimates, uncertainty and calculation disclosures remain connected to saved records; shared theme improves density without changing research semantics |
| Conversations and reports | Real title/topic search, status filter and updated-since date, matching counts and clear-filter action; All conversations/All reports navigation; fixed cancelled report request leaving permanent loading |
| Observability | Execution/conversation/question ID lookup, UTC date range, stage and error/recovered-retry filters; measured performance disclosure; explicit telemetry completeness |
| Loading and failures | Reports, diagnostics and math load on demand; pending states remain readable; a failed view bundle leaves navigation available and offers reload; math falls back to literal text |
| Guide | Escape dismisses the nonmodal guide, focus enters its controls and returns to the trigger |

The initial JavaScript bundle decreased from **587.69 kB / 174.97 kB gzip** to **285.89 kB / 87.08 kB gzip**. Math (262.84 kB), diagnostics (35.02 kB) and reports (10.53 kB) load separately. This is an initial-transfer improvement, not a claim that total application code or model latency halved. No runtime dependency was added.

Taste's framed-sheet direction (seed **1793011816**, existing palette/type retained) governed the redesign, followed by Impeccable critique, UI/UX Pro Max guidance and Emil Kowalski principles from `design-motion-principles`. UI/UX Pro Max's search script was missing; its written guidance was used, without claiming a generated search result. No external imagery or invented learner records were introduced.

Final Taste browser inspection: **0 failures, 0 warnings** for setup and assessment at 1440, 768 and 390px. Impeccable's detector returned an empty finding list. Taste source lint returned **0 failures, 6 warnings**: shadow declarations include `none`, an overridden selection style and the actual floating guide; monospace is used for technical identifiers; em dashes include missing-data marks. These are not hidden as passing lint without warnings. The browser harness also checks 1024 and 320px; setup and execution views had no horizontal overflow at any tested width.

Visual self-critique, not a user-study score: hierarchy 8/10, consistency 8/10, product identity 7/10, interaction clarity 8/10. The main question is clearly dominant and technical detail is disclosed on demand. Dense diagnostics still require more reading than the learning view. The product keeps system typography and a restrained identity rather than adding decorative assets.

Screenshots: [setup](verification/product-transformation/taste-setup/desktop-fold.jpg), [assessment](verification/product-transformation/taste-assessment/desktop-fold.jpg), [mobile assessment](verification/product-transformation/taste-assessment/mobile-fold.jpg), [execution explorer](verification/product-transformation/browser/execution-desktop.png), [report](verification/product-transformation/browser/report-desktop.png), [recovery](verification/product-transformation/browser/recovered-submission.png). Every fold/contact-sheet image from the final Taste runs was visually inspected.

## 3. Backend improvements

- Added `assessment_submissions` through the existing additive schema installer. A valid active-question submission is persisted before inference with original text, request/question/conversation/trace IDs and received/completed/failed status. Invalid extracted evidence never enters accepted answers or state history. Completing the receipt shares the accepted-answer transaction; failed attempts remain recoverable.
- Session responses expose the latest pending submission for the current unanswered question. Browser drafts take precedence over recovered text. Unsent edits remain browser-local; the server does not claim to save a request that never reached it or whose receipt write failed.
- Added HTTP request correlation through `X-Request-ID` and execution metadata. Unexpected SQLite failures return safe JSON with HTTP 503; other unexpected errors return safe HTTP 500. Logs record request ID, method and error type without raw learner text, provider bodies, credentials or filesystem messages.
- Extracted shared parameterized execution filters and aggregate queries. Literal ID search, dates, stage/status/error filters use the same population for rows and summaries. Pagination does not change the total. Naive date inputs are treated as UTC and reversed ranges are rejected.
- Kept existing WAL transactions, optimistic revision checks, replay protection, fixed retry limits and provider timeouts. No production reset, formula change, new framework or background job was introduced.

The [API inventory](verification/product-transformation/api-inventory.json) records all **31 application routes: 16 v2 and 15 retained v1 routes**, including handlers and annotated input/response models. V2 assessment routes continue delegating to services. Expected validation/not-found/conflict/provider failures retain their existing status contracts. Most v2 responses are assembled dictionaries constrained by internal Pydantic models and frontend types; they are not all declared as public FastAPI response models. The legacy research API remains supported but is not the frontend's assessment path. Identity scoping is verified, but it is not authentication.

## 4. Observability improvements

Executions expose `execution_id` as the stable trace identity, plus HTTP/business request IDs and the existing question, conversation, turn, evidence, snapshot and parent references. The UI navigates from a saved response to its actual execution and from stages to their safe inputs/outputs.

| Actual journal stage | Observable work |
|---|---|
| input | Validated active question and original response |
| topic | Structured concept-map construction and validation |
| preprocessing | Normalization, original offsets, token/sentence spans and lexical cues |
| retrieval | Scoped query, FTS5/BM25 passages/ranks and explicit recent-passage fallback |
| planning | Saved-state objective, target concept, prerequisite/repair/recheck policy |
| generation | Model question formulation, attempt, usage and source context |
| extraction | Structured model interpretation, attempt, usage and latency |
| grounding | Assigned-concept identity constraint |
| validation | Quote/rubric/source/hypothesis checks and rejected attempts |
| evidence | Validated Cognitive Evidence linked to its source response |
| state_update | Before/after values, weights, duplicate damping, held estimates and hypothesis changes |
| persistence | Accepted-answer/evidence/snapshot transaction; completion recorded after commit |

These are real grouped operations, not 32 fabricated independent timings. Request correlation connects the HTTP boundary; browser action/render events and separate API response spans are not persisted. Context preparation and fine-grained checks are visible within grouped stage payloads, not assigned invented durations.

Rates use finished executions. Execution-duration means use completed timed executions. Provider/retrieval/persistence means use completed timed stages; missing durations are excluded. Retry totals count additional topic/generation/extraction attempts. Persistence failures are actual failed persistence stages. Filters apply to all these metrics.

Detail and summary share one completeness predicate: running execution, recorded write failure, missing expected stage or unfinished stage means incomplete. Historical traces without expected-stage metadata remain unknown unless a gap can be proved. Business success with incomplete telemetry is explicitly distinct from business failure. Entirely missing journal records cannot be counted, so **100% trace coverage is not claimed**. Existing raw-trace retention remains separate from durable answers/state history.

## 5. Test results

| Test | Type | Result | Evidence |
|---|---|---|---|
| Full backend suite | Deterministic unit/API/SQLite integration | **117 passed**, 32 existing pyBKT/NumPy deprecation warnings | [output](verification/product-transformation/backend-tests.txt) |
| Frontend contracts and draft state | Node client/state tests | **7 passed** | [output](verification/product-transformation/frontend-tests.txt) |
| Rendered UI workflows | Chromium browser, fixture provider | **16 checks passed**, no unexpected JavaScript errors | [results](verification/product-transformation/browser/results.json) |
| Existing observability workflow | Chromium browser, fixture provider | **16 checks passed**, no JavaScript errors | [fixture record](verification/product-transformation/fixture/fixture-execution.json) |
| Successful real assessment | Chromium, configured Azure model | **16 checks passed**, no JavaScript errors | [live record](verification/product-transformation/live/live-execution.json), execution `nlp_bf000fd7855d479e8d106dd627f45bd7` |
| Duplicate HTTP replay and repeated answer damping | Fixture API plus direct SQL | Replay adds no answer; repeated explanation has reduced weight | [duplicate scenario](verification/product-transformation/state-integrity.json) |
| Insufficient evidence / abstention | Fixture API plus persisted snapshots | Alpha/beta unchanged | [abstention scenario](verification/product-transformation/state-integrity.json) |
| Contradictory evidence | Fixture API plus persisted snapshots | Later evidence linked to earlier hypotheses | [contradiction scenario](verification/product-transformation/state-integrity.json) |
| Validation/provider/storage/telemetry failures | API integration and browser faults | No invalid accepted evidence; rollback, safe errors, retries and recovery verified | `test_product_reliability.py`, `test_nlp_observability.py`, `test_question_recovery.py`; browser recovery artifact |
| Reload and adaptive next question | Fixture and live browser | New question references committed state; survives reload | fixture/live records above |
| Production build | TypeScript and Vite | **Passed**, no oversized-chunk warning | [output](verification/product-transformation/frontend-build.txt) |
| Visual inspection | Taste + screenshots | **0 failures / 0 warnings** for both inspected views | [setup](verification/product-transformation/taste-setup.txt), [assessment](verification/product-transformation/taste-assessment.txt) |
| Static design checks | Taste / Impeccable | Taste 0 failures / 6 explained warnings; Impeccable no findings | [lint](verification/product-transformation/taste-lint.txt), [detector](verification/product-transformation/impeccable-detect.json) |
| Production frontend dependencies | npm advisory audit | **0 known vulnerabilities reported** | [audit](verification/product-transformation/npm-audit.json) |

There is no separate shallow-render/JSDOM component suite: rendered interaction coverage is provided by Chromium, while Node tests cover client/state contracts. The live check was performed once; subsequent UI/failure regressions used fixtures and did not spend model credits. The live cycle preceded the final telemetry-completeness and keyboard/bundle-recovery fixes; those changes were verified by the final deterministic/browser runs. Independent code review found two defects (cancelled report loading and incomplete telemetry totals); both were reproduced, fixed and re-reviewed without remaining critical findings in that scope.

## 6. Database verification

Direct SQL inspected `assessment_conversations`, `assessment_questions`, `assessment_answers`, `assessment_state_history`, `assessment_submissions`, `assessment_documents`, `assessment_chunks`, `nlp_executions` and `nlp_stages`. Evidence is stored in the immutable answer payload, rather than a separate v2 evidence table. Checks matched the actual question/conversation, source interaction/evidence IDs, saved snapshot payload, completed receipt/turn and execution status. `PRAGMA integrity_check` returned `ok`; `PRAGMA foreign_key_check` returned no violations.

| Run | Answers | Questions | Snapshots | Submission receipts | Executions | Stages |
|---|---:|---:|---:|---:|---:|---:|
| New browser failure/recovery cycle | 1 | 2 | 2 | 2 (failed + completed) | 5 | 30 |
| Live configured-model cycle | 1 | 2 | 2 | 1 completed | 4 | 23 |
| Existing local database, read-only inspection | 14 | 20 | 19 | 0 (no retrospective fabrication) | 38 | 231 |

Live persisted turn: `turn_1e0ef63d640c4a82b5d6746459463a96`; evidence: `evidence_f50c463895d147d5972c9f9b3efd4bf1`; resulting snapshot: `state_34fb10fb7410494191b04b7556bfc6f1`.

The additional state-integrity run verified **five accepted answer bundles** across duplicate, abstention and contradiction scenarios. Its artifact retains actual before/after snapshot IDs and state calculations. In the contradiction scenario the snapshots are `state_de9db760061e4c4ea314083b6fbb80af` and `state_b9ea382e91334250a0af28dfb3df1e7a`. The [existing-database audit](verification/product-transformation/existing-database.json) was opened with SQLite `mode=ro`, confirmed WAL and exposed only counts. Historical learner records were not deleted or reset.

## 7. Remaining gaps

- Public hosting requires authenticated identity, authorization beyond client-supplied learner IDs, rate limiting, deployment configuration and a backup/restore/privacy policy. The supported mode remains loopback/local use. Raw submitted text is intentionally durable; trace retention does not delete it.
- Best-effort telemetry cannot prove an execution exists when every journal write fails. There is no browser event store, distributed tracing service, alerting backend or independent timing for every substep.
- Confidence, evidence weights and planning heuristics remain uncalibrated. The checks establish provenance and implementation behavior, not semantic truth, psychometric validity or measured learning gains. Expert annotation, calibration and longitudinal evaluation remain required.
- Browser coverage is Chromium on Windows at five viewport widths. No Safari/Firefox, screen-reader certification, touch-device laboratory or high-volume load/soak test is claimed.
- Existing broad stylesheets still have legacy overrides despite shared semantic roles. Further cleanup is possible without changing the verified behavior. Existing pyBKT deprecation warnings remain.
- No deployment, remote push, external monitoring, automated operational backup or independent security penetration test was performed. Failed receipts cannot be durable when their own storage write fails; unsent drafts remain browser-local.
- The npm advisory audit covered production frontend dependencies at this run. It is not a Python dependency audit, development-tool audit or proof that undisclosed vulnerabilities do not exist.

## 8. Readiness assessment

| Area | Evidence-based assessment |
|---|---|
| Design and UX | Coherent working interface; inspected responsive layouts and keyboard recovery; formal user testing still absent |
| Frontend functionality | Major saved-data workflows and introduced controls pass browser checks; initial bundle materially smaller |
| Backend reliability | Deterministic regression, transaction rollback, retry/idempotency and durable failure recovery verified |
| NLP correctness | Structured provenance/validation and state invariants verified; model interpretation remains uncertain |
| Data integrity | Direct SQLite relationships, immutable snapshots and existing-data integrity verified |
| Observability | Actual stages, filtered metrics, error attempts and incomplete telemetry inspectable; coverage is best effort |
| Test coverage | Strong deterministic/API/SQLite coverage plus fixture and one real-model browser cycle; no blanket claim that every browser/platform or production failure is covered |
| Security | Appropriate only within the documented local trust boundary; multi-user public deployment is blocked by missing authentication/authorization |
| Production readiness | Ready for continued local research and supervised evaluation; public production and educational efficacy claims remain blocked by the gaps above |

Reproduce with the commands in `README.md`; all new verification artifacts are under `docs/verification/product-transformation/`.
