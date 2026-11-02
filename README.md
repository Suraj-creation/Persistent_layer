# Language-Grounded Learner State Modeling

A local, text-based adaptive learning platform for the research loop:

**Language → Evidence → Persistent Learner State → Diagnosis → New Evidence**

Start with a topic you have already studied and optionally paste your study notes. The system asks a question, interprets your explanation against a saved rubric, updates evidence-supported topic memory, and generates a new question from that saved state. Every conversation has a structured report.

## Run

Python 3.10+ and Node.js/npm are required.

```powershell
python -m pip install -e .
python -m uvicorn apps.api.app.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd apps/web
npm ci
npm run dev
```

Open [the learning workspace](http://localhost:5173). The browser proxies API requests to port 8000. [API documentation](http://127.0.0.1:8000/docs) includes the conversation endpoints under `/api/v2`.

The application starts empty: no seeded learner profiles, sample conversations, prefilled answers or fixed question bank. Your browser creates a local learner identifier; the sidebar shows only conversations created for that identifier. Previous research data stays on disk and is not inserted into this new workflow.

## Model configuration

Preserve an existing `.env`, or use `.env.example` to configure Azure OpenAI. The existing deployment remains supported. `LEARNER_EXTRACTOR=auto` enables the model; `rules` disables cloud calls and keeps saved conversations/reports readable, but cannot generate new assessments. `LEARNER_DB_PATH` optionally overrides the local database path; otherwise it is `data/learner_state.db` resolved from the repository.

New assessments require a configured model. A failed model call never inserts a canned question or invented evidence. If evidence extraction fails validation, the explanation stays in the editor and learner memory is unchanged. If next-question generation fails after a valid answer has been saved, the answer remains saved and the UI retries only generation.

Study notes need at least 8 words of subject material; a class or grade label alone cannot ground an assessment. This is a minimum-size check, not a factual or relevance check. Older conversations with insufficient notes show **Update study material**, which accepts replacement notes or an explicit switch to a **model-authored, unverified reference**. A generated question can cite only retrieved reference passages. Learner evidence is retained separately as diagnostic context.

Each conversation pins its selected source documents. Updating notes in another conversation cannot silently change an existing question's evaluation material. The **Saved topic memory** panel labels evidence carried over from earlier conversations. Question generation, generation retries and observability reads do not update learner estimates or resubmit answers. Previous questions and feedback can be expanded above the current question.

With supplied notes, retrieval uses passages from those notes. Without notes, the model writes a reference overview which is saved as **model-authored and unverified**. Supplied notes are also not independently fact-checked. Model calls send the selected source passages, learner-state context and current explanation to the configured provider.

## What the NLP pipeline does

See the [workspace and NLP review](docs/NLP_WORKSPACE_REVIEW.md) for the current techniques, the purpose of each observability inspector, viewport fixes, executed checks and research limits.

1. Preserve the original source text and original learner response in SQLite.
2. Separately compute Unicode NFKC/whitespace normalization, original-text token and sentence spans, lexical keywords, uncertainty and negation cues, and a response fingerprint. Case-sensitive source changes remain distinct.
3. Use the pretrained model to extract an exact quote, a candidate conceptual claim, a concept link and criterion-level evidence against the stored question rubric. Validate quotation offsets, concept identity, rubric partitions, citations and hypothesis references in deterministic code.
4. Update a structured per-concept state with weighted positive/negative evidence. Ambiguous or low-confidence evidence holds the numerical estimate. Repeated identical explanations are heavily damped. Missing criteria persist until explicitly rechecked; unrelated success cannot erase them.
5. Persist the observation, computation, full before/after states and retrieval provenance atomically.
6. Reload topic memory and retrieve relevant passages and prior learner evidence with local **SQLite FTS5 / BM25**. English Porter stemming and Unicode tokenization provide lightweight lexical retrieval; this is not a dense embedding search or the separate LightRAG library.
7. A deterministic planner selects the next concept, question kind, repair/transfer objective and any earlier criteria to recheck. The model formulates a fresh question from that stored plan and retrieved context. It does not receive browser-supplied chat history as memory.

Coaching can explain an idea from the retrieved material, but **it never increases the learner estimate by itself**. A later learner response must provide new evidence. See [the state and retrieval specification](docs/LEARNER_STATE_PIPELINE.md) for formulas and limitations.

## Persistent memory and reports

SQLite stores dynamic topic maps, source documents/chunks, searchable evidence, topic states, immutable snapshots, conversations, generated questions with private rubrics, answers and full decision contexts. Topic memory is reused across new conversations by the same local learner about the same normalized topic.

The inspector exposes the state calculation, quoted evidence, missing criteria, extraction model, retrieval passages, generation objective, snapshot ID and context hash. JSON exports preserve the underlying records. The question's private reference answer is hidden until that question has been answered, including in report exports.

Reports contain concept estimates, evidence of addressed gaps, unresolved criteria, unassessed concepts and a complete question/response timeline. Each transition records why the next question was formulated. A report can be viewed during a conversation, completed, printed, exported and revisited. The contextual Guide highlights the actual controls and can be dismissed or restarted.

**Submit & next question** saves the explanation and continues with a fresh question; there is no three-answer limit. Repairs focus on a remaining criterion and vary the question format after the first guided check. **Submit & finish** saves and assesses the current explanation, then completes the conversation without generating an unused next question. **Finish & view report** completes immediately using submitted evidence only. Unsent drafts stay on this browser, associated with their original question, and can be resumed later. If newer topic memory requires a different question, earlier drafts remain available to view and copy rather than being attached to that new question. The inspector includes a glossary explaining the scores and their uncalibrated settings. Mathematical notation in generated questions and feedback is rendered locally.

## Verify

### NLP Observability

Open **NLP Observability** in the sidebar, including while an explanation is processing. Execution traces update automatically every 1.5 seconds, with backoff when the API is unavailable. Filter by conversation, operation, status or question, then inspect actual stage inputs/outputs, retries, transformations, source-grounded evidence and state changes. Completed persistence is recorded only after the transaction commits. Failed evidence extraction and rolled-back writes remain inspectable. The trace journal is best effort and does not abort the learning pipeline when telemetry storage is unavailable.

**Saved responses** includes complete persisted records from earlier conversations. Earlier records have no invented stage events; **NLP techniques** links the actual v2 algorithms to their source implementations. Private question rubrics/reference answers and credential configuration are omitted from traces. Long arrays/payloads are bounded; inspect or export the complete saved response for full data.

Raw execution traces default to 30-day retention (`NLP_TRACE_RETENTION_DAYS`), cleaned on new executions and trace reads. This does not delete learner evidence or state history. Trace and record APIs use the current local learner profile; this remains a loopback, single-user app, not authenticated multi-tenant hosting.

The **Research scorecard** distinguishes measured quote/state integrity from untested research metrics and H1–H6. Historical benchmark values are marked non-comparable because their prediction horizons differ. To record a reproducible diagnosis evaluation against your own expert annotations:

```powershell
python experiments/evaluate_assessment_annotations.py --learner-id YOUR_ID --annotations labels.json
```

See the script docstring for the JSON schema. The run saves label/record hashes, model IDs, per-class scores and actual eligible-label counts; select it in the scorecard. It does not invent baselines, future predictions, calibration results, or learning gains. Annotations and their provenance must be supplied by the evaluator.

The [NLP implementation report](docs/NLP_CONCEPTS_IMPLEMENTATION_AND_JUSTIFICATION.md) documents the complete concept selection, architecture, research gaps and verified execution walkthrough.

```powershell
python -m pytest tests -q
cd apps/web
npm test
npm run build
```

Tests use temporary databases and fixture model responses, never the real learner database or cloud inference. Separate synthetic live-provider and browser checks verify the integrated flow. Existing v1 research APIs and baseline experiments remain for compatibility; the frontend uses only v2 and contains no demo assessment content.

For reproducible browser verification, install the optional Playwright test dependency/browser (`python -m pip install playwright`, `python -m playwright install chromium`), build the frontend, then run `python experiments/verify_nlp_observability.py`. Add `--live` to exercise the configured model using synthetic learner text in a temporary database. Artifacts under `docs/verification/nlp-observability/` are verification evidence, not research datasets or scored learning outcomes.

The observability browser check also requires a distinct visible next question, tied to the committed learner snapshot, and verifies question/state persistence after reload. Run `python experiments/verify_question_recovery.py` (and optionally `--live`) for server-side generation failure, repeated retries, legacy short-note recovery, both source choices, finish/report/resume and mobile checks. These scripts use disposable databases. See [question recovery verification](docs/verification/question-recovery/RESULTS.md) for the executed checks and screenshots.

## Research limits

This is a local, single-user research prototype. A browser identifier is not authentication; keep the API bound to loopback. The existing baseline experiments and sample datasets do not validate this adaptive assessment implementation.

Confidence, update weights and misconception scores are uncalibrated heuristics. Quote validation proves provenance, not semantic correctness. The model can misinterpret a valid quote or generate imperfect educational material. Local retrieval is lexical, so it may fall back to recent passages when terminology differs; every fallback is recorded.

An estimate rising is not proof of learning. Reports flag evidence of progress only when an earlier missing criterion is explicitly rechecked and addressed in a distinct later question, without an identical repeated response. These are not psychometrically equated questions or measured causal learning gains. Evaluate extraction, diagnosis, calibration and longitudinal consistency against annotated data and suitable baselines before making research claims.
