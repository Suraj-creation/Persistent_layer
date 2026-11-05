# Research and Implementation Assessment

## Executive assessment

The intended project is a research-oriented learner-state system. It asks whether a student's explanations and interaction history can provide more useful evidence about conceptual understanding than correctness alone. The central output is not an unconstrained tutor answer: it is a traceable record of what the learner said, a candidate interpretation tied to educational concepts, and an appropriately cautious update to a persistent learner representation.

The repository has a working fraction-focused application prototype and a useful evidence/state data model. It has an exploratory real-data correctness benchmark and a locally audited public curriculum graph, but it does **not** have a validated language model for student explanations, a general math retrieval system, or a validated persistent-state estimator. The browser benchmark's 48 rows are generated from hand-authored patterns; those scores are software-demo results only.

The best first supervised task is response-level **MAP Category:Misconception ranking** from the MAP dataset, using question text, selected answer, and student explanation. This is a small, conventional fitting task (TF-IDF plus logistic regression), not pretraining or state-model training. MAP does not contain learner history, so it cannot train or validate temporal learner-state updates. The data is not present locally, and its Kaggle access rules require acceptance and restrict use to non-commercial research. Training must wait until an authorized copy is available.

## 1. Research understanding

### Research problem and questions

Correctness does not identify why an answer is right or wrong. A wrong answer may reflect a stable conceptual rule, a one-off calculation slip, partial understanding, guessing, or missing context. A correct answer may also conceal flawed reasoning. The system should therefore retain observed language and outcomes, propose evidence-backed interpretations, track uncertainty and contradiction, and request another response when competing explanations remain plausible.

The defensible research contribution is initially a **system and evaluation framework**, not a claim that this repository has discovered a new cognitive model. Useful research questions are:

1. Does question-aware explanation text improve response-level misconception ranking over majority and lexical baselines?
2. Does a structured evidence pipeline preserve the learner's exact words, correctly link concepts, and distinguish misconceptions from slips on expert-reviewed examples?
3. On separate longitudinal data, do language-derived features improve next-response prediction over correctness-only BKT without turning prediction into an unsupported claim about cognition?
4. Can a targeted follow-up distinguish competing hypotheses, and are its answer and subsequent state update actually recorded and evaluated?

No empirical claim about improved learning, retention, transfer, diagnosis, or intervention effectiveness is supported by the current fixture.

### Intended and current flow

**Target:** learner input → structure-preserving text handling → candidate Cognitive Evidence → curriculum/misconception grounding → auditable state update → Repair / Diagnose / Verify / Transfer decision → recorded follow-up evidence.

**Current runtime:** React fraction form or chat → synchronous FastAPI endpoint → Azure OpenAI Instructor extractor when configured, otherwise hand-written phrase rules → SQLite interaction/evidence/current-state/audit writes → fraction-only probe selector → optional Azure-generated tutor response. The actual application path does not use the MAP trainer, does not use the optional semantic matcher to ground extracted claims, and does not ingest an external education knowledge base.

## 2. Source-audited implementation inventory

| Area | Status | Evidence and assessment |
| --- | --- | --- |
| Product purpose | Implemented as a clear prototype goal | README and PRD describe language-grounded learner state. The implementation is currently one fractions case study. |
| Web application | Partially implemented | React/Vite student and researcher views exist. The student portal submits a fixed fraction item; the researcher view shows stored evidence and state. It is not a general curriculum workflow. |
| API | Partially implemented | FastAPI has health, taxonomy, audit, benchmark, learner, process, chat, replay, and audit-report routes. The request pipeline is synchronous. There are no background jobs. |
| External language model | Partially implemented | `main.py` initializes an Azure OpenAI client from environment variables and falls back to rules. This is hosted inference, not open-source model use or training. `openai` and `python-dotenv` are now declared in the project dependencies. |
| Evidence schema | Implemented structurally; semantic validation is partial | Pydantic records interaction/learner IDs, timestamp, quote, claim, polarity, uncertainty, concept links, candidate interpretation, model/prompt/schema version and review status. The quote check removes punctuation before comparison, so it is not an exact-span check and can erase mathematical distinctions. No code path appears to validate concept IDs against the taxonomy before persistence. |
| Rule extractor | Implemented as a narrow baseline | It matches fixed English phrases for a fraction denominator-addition misconception, calculation slip, or mastery. It is not a trained model and will not generalize to unseen wording or other math topics. |
| Prompt/model evidence extraction | Experimental | Instructor returns the schema through Azure; quote validation is attempted, but semantic faithfulness, category correctness, and confidence calibration are not measured. An LLM hypothesis remains an unverified candidate. |
| Text/math preprocessing | Missing beyond whitespace cleanup | There is no dedicated math tokenizer, symbolic parser, sentence segmenter, or documented Unicode normalization pipeline. The MAP trainer preserves input strings and uses word and character TF-IDF; character features retain more punctuation, but this is not mathematical-symbol understanding. |
| Taxonomy | Partially implemented | A hand-authored fraction taxonomy has four concepts and a few misconceptions. It is not the Eedi curriculum graph or a broad concept hierarchy. |
| Embeddings | Experimental and disconnected from live extraction | `SemanticTaxonomyMatcher` can lazily load `all-MiniLM-L6-v2`, or fall back to token overlap. It compares claims to a small taxonomy; it is not an indexed document retriever, BM25 search, or a validated grounding model. It recomputes concept embeddings on each match. |
| Retrieval / FTS5 / BM25 | Missing | No FTS5, BM25, vector index, reranker, or LightRAG integration is present in the application source. |
| Persistence | Partially implemented | SQLite stores learners, interactions, evidence, latest per-concept state, misconception beliefs, and audit events. The schema does not store a versioned snapshot history; the event table omits prior/new snapshot IDs. Writes occur in separate calls, not one transaction across the whole process. The schema does not persist all `Interaction` model fields such as attempt number or hint-used. |
| Weighted Beta-style state | Implemented as an unvalidated heuristic | It initializes alpha=beta=1 and applies hand-set increments for slips, misconceptions, and mastery. These are not estimated from learner data. Some field names retain the word `posterior`, but the values are heuristic scores, not calibrated probabilities. Repeated evidence can increment repeatedly. |
| Contradiction logic | Partially improved; candidate targeting remains broad | Mastery updates now skip resolved/dismissed candidates, with regression coverage. Current evidence polarity is concept-level, so any mastery evidence still discounts every active misconception candidate on that concept. Add a candidate-specific counterevidence target before treating this as production diagnosis. |
| Replay | Deterministic for identical state/evidence inputs; persistence is incomplete | Transition snapshot and event IDs are derived stably from state/evidence content, duplicate evidence IDs in a replay ledger are rejected, and the duplicate method was removed. SQLite still lacks persistent snapshot history and transactional cross-process event idempotency. |
| BKT | Separate wrapper exists | `ClassicalBKTBaseline` wraps pyBKT for correctness sequences. It is a suitable separate baseline when run on longitudinal skill data; it cannot classify free-text misconceptions. The synthetic benchmark result is not evidence of its generalization. |
| Active diagnosis | Partial prototype | Two hand-authored fraction diagnostic items have fixed response likelihoods. The selector reduces multiple candidate misconceptions to an average probability over a mastery-vs-misconception pair; option likelihoods are not normalized as a full outcome distribution. The selected question is a hypothesis probe, not an optimized or validated intervention. |
| Follow-up persistence | Missing | The student UI only stores the selected probe option in React state, then claims it resolves uncertainty and updates the learner belief. It does not send that answer to the backend or create new evidence. |
| NLP observability | Partial | The researcher view displays evidence, quote, candidate interpretation, and state events. It does not show actual normalization, tokens, model inputs, retrieval candidates, validation traces, or processing timings. UI claims need to remain tied to real backend traces. |
| Dataset ingestion | Split between synthetic fixtures and real audited sources | The Eedi and FoundationalASSIST response loaders still create synthetic fixtures when no path is passed. A separate ingestion script now fetches and audits the pinned CC BY Eedi misconception graph; this graph is not wired into live retrieval. Real MAP's schema is not the row schema expected by the Eedi fixture loader, and MAP has no chronological learner history. |
| Benchmarking | Synthetic demo only | The saved 48-row output is generated from repeated hand-written response templates. TF-IDF can memorize those templates. Tier 1 misconception F1 is assigned as zero because pyBKT has no misconception head, not measured. Tier 3's AUC is lower and ECE higher than Tier 1 in the fixture. |
| Training | Prepared, not executed | A local MAP trainer has been added. It consumes a local authorized `train.csv`, makes five-fold grouped out-of-fold predictions, saves a checksum, class labels, metrics, and a fitted final model. There is no MAP data file in this checkout, so no fit or score has been produced. |
| Tests and deployment | Core tests/build pass; research limits remain | The full Python suite passes (31 tests), and the React/Vite production build passes. pyBKT emits 32 NumPy deprecation warnings on its pure-Python path. The package declares `python-dotenv` and `openai`; compatible NumPy/scikit-learn ranges are bounded. No MAP fit has run. |

Key source locations: `apps/api/app/main.py`; `apps/web/src/components/StudentPortal.tsx`; `apps/web/src/components/ResearcherStudio.tsx`; `packages/learner_state_core/evidence/models.py`; `packages/learner_state_core/evidence/extractor.py`; `packages/learner_state_core/state/engine.py`; `packages/learner_state_core/diagnosis/prober.py`; `packages/learner_state_core/db/database.py`; `packages/learner_state_core/data/eedi_loader.py`; `packages/learner_state_core/data/assistments_loader.py`; `packages/learner_state_core/grounding/matcher.py`; `packages/learner_state_core/baselines/bkt.py`; `experiments/run_benchmark.py`; `experiments/train_map_misconception_classifier.py`; `pyproject.toml`.

## 3. Dataset audit and recommendations

Do not merge separate releases as if their records came from the same students. Use each release for the task its fields support and connect it to domain taxonomy only through documented construct IDs or labels.

| Dataset | License / access | Data and capability | Recommended role | Limits |
| --- | --- | --- | --- | --- |
| [MAP: Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) | Kaggle rules apply; CC BY-NC 4.0 and non-commercial research/education use. Participants must accept the rules before data access; do not redistribute the raw data. | 52,463 explanations across 15 multiple-choice items in the associated benchmark paper; expert-assigned Category and Misconception labels; a mixture of authentic and generated examples. Competition input contains question, selected answer, and explanation. | **Primary NLP training/evaluation set** for ranked response-level candidate labels. | No learner sequence or usable learner-held-out split in the competition CSV; only 15 questions; synthetic and authentic examples must be separated if release metadata permits; does not annotate exact evidence spans or prove persistent cognition. |
| [ASSISTments 2009–2010 Skill Builder](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data) | Official page says free to use and requests precise dataset citation. Review its PII guidance and the exact corrected release; no standardized CC license is stated on the page. | Student/problem/skill IDs, chronological `order_id`, first-response correctness, attempts, hint counts, assignment metadata. | **Secondary correctness KT benchmark** for pyBKT and sequence prediction; use chronological or student-level splits. | No rich explanations; hint and attempt totals do not preserve action-level sequence. Multi-skill rows can duplicate a student-problem; the source links a corrected version. |
| [FoundationalASSIST](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST) | CC BY-NC 4.0; gated access request and Responsible Use Guidelines. Access is restricted for some jurisdictions/researchers; inspect current terms before acquisition. | 1,722,169 interactions, 5,000 students, 3,395 problems, 224 skills, problem text/options and first answer. | Later correctness/answer-type sequence analysis, problem grounding, and a larger separate KT baseline if access is approved. | First answer only, no later attempts; `discrete_score` includes support/hint effects; no verified misconception labels; possible typed PII is acknowledged by maintainers. |
| [Eedi Misconceptions Graph v1.0](https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph) | CC BY 4.0 with attribution; public CSV release. | 19,294 weighted misconception–construct links; 2,258 constructs; 8,117 misconception nodes; subject/topic hierarchy. | **Knowledge-grounding source**. Pin version, preserve IDs and attribution, load as explicit graph/table and evaluate retrieval. | No diagnostic questions, distractors, student responses, or student data. It is not training data for learner classification or temporal state. |
| [Eedi Question-Anchored Tutoring Dialogues-2k](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k) | CC BY-NC 4.0. | English conversation turns with user/intervention/question IDs and tutor TalkMovePrediction labels; public viewer shows 68.7K rows. | Optional later dialogue-context and tutor-move research. | Talk-move labels are not learner misconception labels, validated cognitive state, or intervention outcomes. Do not join its users to MAP or ASSISTments users. |

The primary/secondary/grounding/dialogue recommendation is therefore **MAP / ASSISTments 2009–10 (or gated FoundationalASSIST) / Eedi Misconceptions Graph / Eedi dialogues**. The MAP competition rules currently block local training until an authorized participant accepts them and provides the data. FoundationalASSIST is also gated and has country/access restrictions, so it is not the fallback for the immediate classifier.

## 4. Model and training decision

### What to fit first

The project needs a small supervised baseline to test whether text is useful for the response-level misconception task. Fit **word and character TF-IDF plus class-balanced logistic regression** on the MAP target `Category:Misconception`, using `QuestionText`, `MC_Answer`, and `StudentExplanation` as context. Return ranked labels (top 3), not a definitive diagnosis. Keep the student's original response unchanged alongside all derived features.

The trainer currently uses a five-fold `StratifiedGroupKFold` grouped by normalized full model input, reports aggregated out-of-fold top-1, MAP@3, macro/per-label F1, Brier, ECE and label coverage, then refits a final artifact on all training rows. The probability metrics are explicitly uncalibrated diagnostics. This split tests explanations for questions represented in training. It does **not** establish unseen-question transfer; add a separate leave-question-out experiment and report class coverage before making that claim. The 15-item benchmark is too narrow for broad curriculum generalization.

This conventional model fitting is the requested first training step. It is cheap, inspectable and CPU-capable; the GPU is not required. It trains a task classifier from labeled data, not a foundation model.

### What not to train from MAP

Do not train the persistent-state update from MAP rows. The competition training file has no ordered history per learner. Do not train an exact-quote extractor from MAP labels: they mark response-level classes, not evidence spans. Do not interpret classifier confidence as the probability that the child has a misconception.

### Pretrained inference and optional fine-tuning

| Component | Choice now | Training need | Decision |
| --- | --- | --- | --- |
| Candidate label ranking | TF-IDF + logistic regression | Fit supervised class weights on MAP | First baseline, after access is authorized. |
| Structured extraction | Existing Instructor/Azure path | Inference only; not open-source | Keep as an optional comparator, validate against expert-reviewed evidence before state integration. |
| Concept similarity | Existing `all-MiniLM-L6-v2` path | Inference only | Use only as a candidate matcher; similarity scores are not calibrated grounding confidence. |
| Small pretrained encoder | DistilRoBERTa-base is a candidate (82M parameters, Apache 2.0) | Fine-tune only if TF-IDF shows a meaningful, repeatable gap | Optional second-stage experiment; identical splits, top-3 metrics, class-level errors, and latency. Do not make it a dependency now. |
| Larger instruction model | No new local model selected | Inference only would be the initial mode | Avoid adding a second LLM runtime until latency, cost, extraction fidelity and local GPU-memory needs are measured. |
| Persistent state updater | Existing transparent weighted Beta-style heuristic | No supervised fitting from MAP | Preserve it for audit, correct its logic only with documented behavior and targeted tests; do not claim calibrated posterior. |
| BKT | Existing pyBKT wrapper | Fit parameters on a separate longitudinal dataset | Keep as correctness-only comparator; do not apply it to MAP. |

Fine-tuning is not necessary to make the initial prototype function. It is only justified after an ablation shows that the fitted lexical baseline or existing inference pipeline is insufficient. Training from scratch is out of scope.

## 5. NLP, evidence, and math handling

The current extraction contract should be retained but treated as a hypothesis record. Before integrating a classifier into the API:

1. Preserve raw math text and record any normalized form separately; never strip operators or punctuation from model input.
2. Validate exact quote offsets against the original string. The current punctuation-stripping containment test can confuse `1/2` with `12`; it is not sufficient evidence validation.
3. Validate concept and misconception IDs against a versioned taxonomy/graph. Keep out-of-taxonomy predictions as abstentions or review candidates.
4. Save model version, prompt version, taxonomy release, raw interaction ID and timestamp with each evidence record.
5. Measure structural validity separately from semantic faithfulness. Use expert-reviewed examples for quote accuracy, claim faithfulness, concept mapping, category distinction and abstention behavior.
6. Keep all candidate interpretations explicitly tentative. A classifier output is not a verified fact about the learner.

The initial classifier's `Question:`, `Selected answer:`, and `Student explanation:` fields make context explicit. Whitespace normalization is minimal and mathematics-preserving, but the vectorizer is still a lexical baseline, not a symbolic math parser.

## 6. Grounding and retrieval

The repository currently has a static fraction taxonomy plus optional embedding similarity. A separate script now fetches and structurally audits the Eedi graph at a pinned revision, but there is still no FTS5/BM25 index, document chunker, vector database, retrieval adapter, or retrieval evaluation. The next grounding step is an expert-reviewed Eedi-to-app concept crosswalk and lexical/embedding retrieval comparison.

Keep **domain knowledge** (concept hierarchy, definitions, misconception–construct edges, worked examples) separate from **individual learner state** (evidence ledger, current estimates and uncertainty). The graph should ground candidate evidence; it must not assert what a particular student knows.

LightRAG is an experiment, not a prerequisite. Its official repository describes entity/relation extraction, multiple chunking and retrieval modes, LLM/embedding configuration, and default in-memory KV/vector/graph storage intended for small-scale evaluation rather than production. A small, known taxonomy graph does not need LLM-generated graph extraction. If later tested, compare it with lexical and vector baselines on retrieval precision@k, recall@k, MRR/NDCG, grounding correctness, latency and indexing cost. Human-review generated edges.

## 7. Persistent state, diagnosis, and experiment design

The current mastery update is Beta-shaped: starts with alpha=beta=1 and adds fixed, confidence-weighted increments by category. It is not the weighted novelty/token formula described in earlier research notes, nor is it fitted from data. Focused regression coverage now checks replay IDs, duplicate replay evidence, learner identity, repeated discounts to resolved beliefs, and mastery updates. The next work is candidate-specific counterevidence, state invariants, persistence/idempotency, and validation of transition weights against reviewed longitudinal examples.

The current diagnosis prober's assumptions need an explicit outcome model: all answer-option probabilities should form a valid distribution under each hypothesis; candidate misconception hypotheses should remain separate; selection should account for item difficulty, prior probes, and observed outcomes. Until then, call the choices **candidate diagnostic probes**, not optimal interventions. Persist each chosen probe, learner answer, evidence interpretation and subsequent state event through an API path before claiming a closed learning loop.

For experiments, report only metrics that match the task:

- **MAP response labels:** MAP@3, top-1 accuracy, per-class precision/recall/F1, confusion matrix, label coverage, macro-F1, Brier/ECE only with explicit uncalibrated/calibrated status; include majority baseline.
- **New wording on known items:** grouped duplicate-safe cross-validation (the trainer's current split).
- **Unseen items:** leave-question-out evaluation; report unseen-label coverage and do not score impossible classes as ordinary misses without an explicit protocol.
- **Evidence quality:** exact-span correctness, claim faithfulness, concept-link precision, uncertainty/abstention quality, expert agreement.
- **Longitudinal prediction:** student-held-out and chronological next-answer AUC, log loss/Brier and calibration on ASSISTments sequences. This tests prediction only, not cognition or learning impact.
- **Retrieval:** precision@k, recall@k, MRR/NDCG and grounding correctness.
- **Operations:** end-to-end latency, memory, inference cost, failure/retry rates and persistence reliability.

Keep splits, release version, checksum, preprocessing, label map, software versions and data-use terms with each run. Do not use a random synthetic archetype split as research evidence.

## 8. Prioritized roadmap and acceptance criteria

| Priority | Work | Acceptance criteria |
| --- | --- | --- |
| 0. Data readiness | User obtains MAP through accepted Kaggle terms and places `train.csv` in the ignored local data folder. Review current rules at acquisition time. | Authorized local file, schema audit, checksum, no raw data in Git, intended use is non-commercial research/education. |
| 1. MAP baseline | Run the prepared five-fold grouped TF-IDF/logistic-regression trainer; add majority baseline and a separate leave-question-out evaluation before broad generalization claims. | Reproducible out-of-fold metrics and per-label coverage; save exact release/checksum/config; no uncalibrated probabilities used as learner-state truth. |
| 2. Model comparison | Only if baseline gaps justify it, compare DistilRoBERTa fine-tuning on the identical split. | Repeated controlled improvement in MAP@3 and meaningful class-level gains without worse calibration/latency; retain the simpler model otherwise. |
| 3. Evidence validation | Build an adjudicated sample with exact spans, category, concept, candidate misconception, uncertainty, and abstention labels. | Quote offsets verify exactly; human agreement and error analysis reported; schema validity not confused with semantic quality. |
| 4. State audit | Deterministic replay IDs, duplicate-ledger checks, learner identity checks, and inactive-belief discount protection have focused regression tests. Next add candidate-specific counterevidence targets, persistent snapshot metadata, and transactional idempotency. | Regression suite passes; state probabilities remain documented as heuristic until externally calibrated. |
| 5. Separate KT benchmark | ASSISTments BKT is evaluated on two student-level seeds against prevalence and opportunity-count baselines. Next add at least five seeds, per-skill paired metrics, and a defensible temporal split. | No cross-dataset student joins; duplicate events audited; next-response prediction reported separately from misconception diagnosis. |
| 6. Grounding | Pinned Eedi graph v1.0 was downloaded and structurally audited with CC BY attribution. Next evaluate lexical and embedding retrieval and the app-taxonomy crosswalk. | Versioned graph IDs; expert query set; retrieval metrics and human audit; no LightRAG unless a measured gap justifies it. |
| 7. Closed-loop product | Persist a learner's probe answer and generate follow-up evidence; add Repair/Diagnose/Verify/Transfer policy gradually. | Every UI state change maps to a real API event, evidence record and state event; no hardcoded observability values. |
| 8. System verification | Run unit/API tests, frontend type/build checks, then controlled data experiments. | Report which checks actually ran, environment/dependency versions, failures, and limits; never equate tests with research validation. |

## 9. Work completed and current blocker

- Read the project README, PRD, research document and benchmark report; inspected the source architecture, data loaders, trainer path, runtime API, persistence, state update, diagnosis, UI and available tests.
- Added `DATA_AND_TRAINING_PLAN.md` and a local MAP classification trainer. Corrected the default benchmark and researcher UI to label its numbers as synthetic and to stop describing heuristic scores as calibrated findings.
- Added a local ASSISTments BKT trainer and trained a correctness-only model on the official corrected 2009–10 Skill Builder release. The final run used an 80/20 student split, eight fraction/percent skills, 39,757 training events, and 9,901 held-out events. Held-out AUC was 0.6673, accuracy 0.7428, Brier 0.1805, and uncalibrated ECE 0.0088. This tests next-response prediction only and is not integrated into the product state engine. See `experiments/assistments_bkt_report.md`; the raw data remains ignored, while aggregate model artifacts are included in the `training` branch.
- Repeated the correctness experiment on a second student-level split and added smoothed skill and opportunity-count baselines. Across two seeds, mean BKT AUC was 0.6639 versus 0.5653 for skill plus opportunity logistic regression; the small split count is exploratory only. The second aggregate model checkpoint, parameter table, and robustness metrics are recorded on the branch.
- Downloaded and structurally audited the Eedi Misconceptions Graph v1.0 at a pinned revision under CC BY 4.0. All 19,294 edge rows joined to the 2,258-construct catalogue, and misconception weights summed to 100 within tolerance. Raw CSVs remain ignored; an aggregate, attributed audit and file checksums are tracked. No student responses were used.
- Repaired deterministic replay identifiers and repeated discounts to inactive beliefs; updated traces and documentation to label hand-set state scores as heuristics, not calibrated Bayesian posteriors. Concept-level mastery still discounts all active candidates, so per-hypothesis contradiction targeting is outstanding.
- The MAP trainer has not been executed. There is no MAP data in the checkout, so no real-data misconception classifier or score exists.
- The browser benchmark remains a synthetic software demonstration; the auxiliary BKT result is recorded separately and is not a benchmark of the proposed language-grounded diagnosis pipeline.
- The first ASSISTments attempt exposed costly `pyBKT` parallel-pool behavior; the trainer now uses serial fitting and a declared support threshold. The model artifacts record the exact dataset SHA-256 and package versions.
- Focused state, dataset-ingestion, and trainer checks were added. The expanded full Python suite passes: 31 passed, with 32 existing pyBKT NumPy deprecation warnings. `npm run build --prefix apps/web` passes. NumPy remains bounded below 2.0 because pyBKT's fallback relies on behavior removed in NumPy 2.
- No synthetic fixture was treated as training or research data. Raw data and unselected generated artifacts remain ignored; the reviewed aggregate BKT checkpoints, parameters, and metrics are tracked.
- I have not accepted MAP's binding competition rules. To proceed with the actual fit, the user must sign in to Kaggle, accept the rules themselves, confirm non-commercial research/education use, and place the authorized `train.csv` at `data/raw/map/train.csv`. Once it is present, run `python experiments/train_map_misconception_classifier.py --confirm-map-data-access-authorization`.

## Sources

- MAP benchmark paper: <https://aclanthology.org/2026.aimecon-main.33/>
- MAP competition and rules: <https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings> and <https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/rules>
- FoundationalASSIST card and responsible-use terms: <https://huggingface.co/datasets/ASSISTments/FoundationalASSIST>
- ASSISTments 2009–2010 schema and data terms: <https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data>
- Corrected one-row-per-student/problem release and first-action coding: <https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010> and <https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets>
- Eedi Misconceptions Graph card and data dictionary: <https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph>
- Eedi tutoring-dialogue card: <https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k>
- DistilRoBERTa model card: <https://huggingface.co/distilbert/distilroberta-base>
- Hugging Face sequence-classification guide: <https://huggingface.co/docs/transformers/main/tasks/sequence_classification>
- LightRAG repository: <https://github.com/HKUDS/LightRAG>
