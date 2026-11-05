# Question recovery and workspace readability

Goal: recover blocked question generation, preserve learner evidence exactly, and make the learning and observability workflows legible.

Evidence: the reported conversation retrieves only `Class 6th`; every model attempt cites learner-evidence chunk IDs and fails `unsupported_citations`. The state snapshot remains identical across retries. The creation trace misleadingly reports completed although its generation child failed.

Constraints: preserve existing uncommitted work and learner data; no invented questions, evidence, or success metrics. Synthetic browser and live-provider checks use temporary databases.

- [x] Add failing API regressions for short notes, source recovery, failed creation traces, citation repair, and unchanged state after retries and observability reads.
- [x] Validate material before inference; add a conversation-scoped source recovery endpoint, precise safe failure codes, and schema-constrained source citations. Verify the regression tests and existing Python suite.
- [x] Add clear generation/source recovery states and topic-memory provenance; apply the design contract in `DESIGN.md` across learning, reports, and diagnostics.
- [x] Extend browser verification to require the next question, reload/persistent state, generation outage/recovery, finish/resume, and desktop/mobile readability. Run fixture and real configured-model checks.
- [x] Restart the local application with verified code and record commands, results, and any remaining limits in `docs/verification/question-recovery/RESULTS.md`.
