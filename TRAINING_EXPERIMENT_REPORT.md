# Training and Experiment Report

**Repository:** Persistent_layer

**Report date:** 2026-11-05

**Scope:** Codebase purpose, data selection, implementation changes, training runs, evaluation, results, reproducibility, and remaining research work.

## 1. Executive summary

This project is a fraction-focused prototype for maintaining a cautious, auditable representation of a learner. Its intended path is to preserve a learner's words and answer, derive candidate evidence tied to math concepts, and update a learner state as new observations arrive. Its research question is whether reasoning language adds useful and faithful evidence beyond binary correctness.

At the start of the training work, the repository did not contain a trained explanation classifier or a data-fitted persistent state updater. Its built-in benchmark used 48 synthetic examples generated from four fixed response archetypes. The current extractor remains rule-based when the optional Azure-backed extraction path is unavailable, and state updates use hand-set heuristics.

Work in this branch adds two distinct training paths:

1. **Executed:** a correctness-only pyBKT model trained and evaluated on the public ASSISTments 2009–2010 corrected Skill Builder data, narrowed to fraction/percent skills. It predicts later correctness for held-out learners. Its result is an exploratory knowledge-tracing baseline, not misconception diagnosis.
2. **Prepared, waiting on data authorization:** a response-level MAP `Category:Misconception` classifier using word/character TF-IDF and logistic regression. The MAP file is absent; its competition rules require each user to accept the rules and restrict use. No MAP training or score is claimed.

The report distinguishes synthetic demonstrations, actual training, and planned experiments so none can be mistaken for evidence about the live product.

## 2. Project purpose and current implementation

The intended system should preserve evidence rather than turn a single answer into an overconfident cognitive label. A useful model should propose candidate interpretations from the item, selected answer, and explanation. The evidence ledger and state updater should remain auditable, should represent uncertainty and contradiction, and should allow later responses to revise earlier hypotheses.

Current runtime remains a prototype; the retrieval and persistence continuation is documented in [RAG_AND_REGRESSION_REPORT.md](RAG_AND_REGRESSION_REPORT.md). The current architecture is: a React fraction interface calls a FastAPI service; evidence extraction uses hand-written phrase rules or an optional Azure Instructor path; SQLite stores interactions, evidence, current learner state, and audit events; and a small fraction-only diagnostic prober selects follow-up items. The app does not currently consume the trained BKT artifact or the prepared MAP model path. The optional semantic matcher compares claims with a small static taxonomy; it is not a production retrieval system.

Important implementation findings recorded in the assessment include:

- The default Eedi and FoundationalASSIST loaders return synthetic fixtures when no local file is supplied. Their demo rows are not the corresponding releases.
- State updates are deterministic hand-set heuristics; the Beta-shaped quantities and misconception scores are not calibrated Bayesian posteriors or learned transitions.
- This continuation stopped repeat discounts to resolved/dismissed misconceptions and added a regression check. Replay snapshot/event IDs are repeatable for the same prior state and evidence, and duplicate evidence IDs in a replay ledger are rejected. The evidence schema now also names explicit counterevidence targets, and only those candidate scores are discounted.
- The state engine rejects cross-learner evidence and repeated concept links in one evidence record. The SQLite API path now commits the interaction, evidence, full snapshot, events, retrieval trace, and idempotency receipt atomically under a write transaction.
- Diagnostic responses now persist through a server-scored endpoint tied to the issued learner/item/parent event. Answer choices alone remain unverified reasoning evidence.
- Live tutoring now retrieves from a persistent SQLite FTS5, hashed-vector, and prerequisite graph index of the app curriculum. The audited Eedi graph is separately indexed for researcher retrieval; it is not automatically mapped to learner concepts.

These findings are detailed in [RESEARCH_AND_IMPLEMENTATION_ASSESSMENT.md](RESEARCH_AND_IMPLEMENTATION_ASSESSMENT.md). The data/model choices and their task boundaries are also summarized in [DATA_AND_TRAINING_PLAN.md](DATA_AND_TRAINING_PLAN.md).

## 3. Dataset inventory and task fit

| Dataset or source | Use/status | What its labels and fields support | Constraints and limits |
| --- | --- | --- | --- |
| ASSISTments 2009–2010 corrected Skill Builder | **Downloaded and used for the BKT run.** Local raw CSV is Git-ignored. | Student, problem, skill, chronological order, first-action type, and correctness. Supports a separate next-response correctness baseline. | The official page says the release is free to use and asks users to cite the exact dataset page. It has no verified misconception labels or detailed explanation text. Historical skills are not mapped to this app's taxonomy. |
| MAP: Charting Student Math Misunderstandings | **Selected for the first language-to-misconception classifier; not trained.** No authorized file is present. | Question text, selected answer, student explanation, category, and misconception label. The paper reports 52,463 explanations over 15 items with expert labels and a mixture of authentic and synthetic responses. | It has no learner chronology. Kaggle rules require acceptance before access and limit use to non-commercial purposes. No third-party mirror was used. |
| FoundationalASSIST | Researched; not downloaded or trained. | Longitudinal correctness/skill sequences can support a separate KT task. | No verified misconception labels; the release has access/responsible-use conditions. Short/numeric answer strings are not equivalent to reasoning explanations. |
| Eedi Misconceptions Graph v1.0 | **Downloaded locally and audited** at pinned revision `2ae0839…`; CC BY 4.0. Raw CSVs are Git-ignored. | 19,294 weighted misconception↔construct edges; 2,258 constructs; 8,117 misconception IDs; four-subject curriculum hierarchy. Supports grounding and retrieval research. | No questions, distractors, answers, or learner data. It cannot directly train a response classifier or learner-state updater. Mapping to the prototype taxonomy remains unreviewed. |
| Eedi Question-Anchored Tutoring Dialogues-2k | Researched; not used. | Dialogue and tutor-talk-move studies. | Tutor turn labels are not student misconception labels or ground truth for learner state. |
| Built-in Eedi-style fraction fixture | **Used only by the existing synthetic benchmark.** | Exercises code paths and output presentation. | Generated examples reuse a handful of hand-authored wording templates; labels and wording are coupled by construction. Not an empirical dataset. |

The official ASSISTments pages describe the 2009–10 release as free to use, define `first_action` codes (`0` attempt, `1` hint, `2` scaffolding), and ask for citation of the exact release page. The MAP paper describes the response-level task; the competition rules govern data access and usage. See the source list at the end.

## 4. Experiment A: existing synthetic pipeline demonstration

The checked-in benchmark uses 48 generated examples from 12 synthetic learner IDs, four fraction items, and four fixed response archetypes. The learner split does not prevent response-template leakage because the same archetype wording recurs across learners.

| Metric | Tier 1: pyBKT | Tier 2: TF-IDF text classifier | Tier 3: evidence + heuristic state |
| --- | ---: | ---: | ---: |
| AUC-ROC | 0.9062 | 1.0000 | 0.5938 |
| Accuracy | 0.7500 | 1.0000 | 0.6667 |
| Brier score | 0.1653 | 0.0958 | 0.2233 |
| ECE | 0.1161 | 0.2938 | 0.2744 |
| Misconception F1 | 0.0000* | 1.0000 | 0.5000 |

`*` Tier 1 does not produce a misconception label; the benchmark assigns F1 = 0.0 by construction. It is not a measured diagnostic comparison.

These scores are retained as a software demonstration only. Tier 2's perfect score reflects synthetic-template leakage, and its ECE is poor. Tier 3's AUC remains below Tier 1. The refreshed policy produces AUC 0.5938 and accuracy 0.6667 on this fixture; Tier 2 uses current-response text while Tiers 1 and 3 use history, so cross-tier deltas are not fair model comparisons. Consequently, these numbers do not show that the proposed evidence/state pipeline improves prediction or produces calibrated uncertainty. The benchmark code, JSON, report, and researcher UI were updated to label these limitations accurately.

## 5. Experiment B: ASSISTments correctness-only BKT

### Objective and target

This run tests whether a conventional knowledge-tracing model predicts a held-out learner's later first-attempt correctness from previous responses. It is the longitudinal-state experiment; it does not use language and does not test misconception diagnosis.

The target is binary `correct` for original problems where `first_action == 0` (attempt). Hint-first and scaffolding-first interactions are excluded because the release can record help-seeking as incorrect. Rows without a user, skill, problem, order, or binary outcome are excluded. Events are sorted by `order_id`, the release's chronological problem-log key. Duplicate `(user_id, problem_id, order_id)` events are rejected.

### Dataset audit and split

| Stage | Rows / learners / skills |
| --- | ---: |
| Raw corrected CSV | 346,860 rows |
| Original problems | 275,458 rows |
| Original problems with `first_action == 0` | 249,799 rows |
| Rows with valid user, skill, problem, order, and binary target | 228,250 rows |
| Rows after `fraction|percent` skill-name filter | 49,715 rows across 10 skills |
| Rows after support threshold (≥100 rows and ≥20 learners per skill) | 49,658 rows across 8 skills |
| Duplicate user/problem/order events in model rows | 0 |

The support threshold removed 57 records from two rare labels: Equivalent Fractions (14 rows) and Percent Discount (43 rows). This avoids fitting the main comparison on extremely sparse skill sequences.

The learner-level split is deterministic with seed 42 and a 20% holdout. Training has 39,757 events from 1,762 learners; test has 9,901 events from 441 previously unseen learners. All eight test skills appear in training and all 9,901 test events were scored. Earlier interactions within each held-out learner's sequence are available to predict that learner's later interactions; held-out learners never contribute to parameter fitting.

### Model and configuration

- Implementation: `pyBKT` 1.4.3, standard non-forgetting BKT, one initialization, seed 42.
- `parallel=False`: pyBKT's default parallel path repeatedly created large process pools inside EM iterations and was impractical at this scale.
- The output probabilities come from pyBKT's one-step prediction path. The comparison baseline assigns each test row the training-set correctness prevalence.
- No hyperparameter tuning, multi-seed analysis, temporal split, or post-hoc probability calibration was performed.

### Held-out results

| Metric | BKT | Constant training-prevalence baseline |
| --- | ---: | ---: |
| Test events | 9,901 | 9,901 |
| ROC AUC | 0.6673 | 0.5000 |
| Accuracy at 0.5 | 0.7428 | 0.7365 |
| Brier score | 0.1805 | 0.1942 |
| 10-bin ECE (uncalibrated) | 0.0088 | 0.0114 |

Per-skill ROC AUC ranges from 0.5880 to 0.7610. The `Percents` test group has only 21 events, so its metrics are unstable. The other per-skill metrics are in `models/assistments_2009_2010_bkt/metrics.json`. The model has 40 finite fitted parameters; no estimated parameter (excluding the intentionally fixed zero forgetting values) is at a 0/1 boundary. Eight zero forgetting entries are expected in this non-forgetting model.

Interpretation is deliberately narrow: on this particular historical release and one learner holdout, BKT beats a constant prevalence baseline on AUC, accuracy, and Brier score. This does not establish transfer to the product's learner population, validate the app's state updates, or show that the model detects a learner's reasoning.

### Provenance and artifacts

- Dataset SHA-256: `162ef8d2d28bcbfea6591a282994062bd8d5eaa00636544292a0d268dca6e5da`
- Recorded versions: Python 3.12; pyBKT 1.4.3; NumPy 1.26.4; pandas 2.3.3; scikit-learn 1.7.2.
- Dataset and artifacts: raw CSV remains Git-ignored; trained artifacts are small aggregate model parameters, not learner event sequences.
- Model: `models/assistments_2009_2010_bkt/model.pkl`
- Parameters: `models/assistments_2009_2010_bkt/parameters.csv`
- Machine-readable metrics, per-skill results, support audit, hash, split, versions: `models/assistments_2009_2010_bkt/metrics.json`
- Narrative methods/results: [ASSISTments BKT run report](experiments/assistments_bkt_report.md)

The model pickle was inspected: it contains the pyBKT configuration and fitted per-skill parameter arrays, not learner IDs or training interaction rows. The raw CSV is not included in the branch.

### Repeated student-split comparison and simple baselines

The follow-up evaluation reuses the tracked seed-42 model only when its training-data SHA-256 matches the local CSV, and fits a separate seed-7 BKT model on that split's training learners. It compares both BKT runs against constant training prevalence, smoothed per-skill prevalence, and logistic regression using skill identity plus the held-out learner's within-skill opportunity count. These baselines use training labels only; the opportunity feature is sequence position, not future correctness. No per-learner predictions are written.

| Model | Mean ROC AUC | Split SD | Mean accuracy | Mean Brier | Mean 10-bin ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Non-forgetting BKT | 0.6639 | 0.0048 | 0.7449 | 0.1798 | 0.0088 |
| Constant training prevalence | 0.5000 | 0.0000 | 0.7396 | 0.1926 | 0.0074 |
| Smoothed skill prevalence | 0.5531 | 0.0065 | 0.7397 | 0.1911 | 0.0122 |
| Skill + opportunity logistic | 0.5653 | 0.0016 | 0.7397 | 0.1904 | 0.0091 |

Across these two paired splits, BKT's mean AUC exceeds the skill/opportunity baseline by 0.0986 and its mean Brier score is lower by 0.0105. Two seeds are an exploratory robustness check, not a confidence interval or significance test. ECE remains bin-dependent and uncalibrated. Per-seed metrics, row counts, model source, artifact hashes, and provenance are in `models/assistments_2009_2010_bkt/robustness.json`; the seed-7 model and parameter CSV are tracked as well. This still tests correctness prediction only.

## 6. Experiment C: open Eedi graph ingestion and audit

The public Eedi v1.0 graph was fetched from a pinned Hugging Face revision under CC BY 4.0. The ingestion script validates both CSV schemas, expected row counts, unique construct IDs, every edge's construct join, construct-name consistency, percentage bounds, and that every misconception's edge weights sum to 100 within 0.1 percentage points. The audit passed for 19,294 edges and 2,258 constructs, covering 8,117 misconceptions, 236 subtopics, 50 topics, and four subjects.

The raw CSVs remain under ignored `data/raw/eedi_misconceptions_graph/`. The checked-in `experiments/eedi_graph_audit.json` contains aggregate counts, the pinned revision, attribution, and file SHA-256 values. This is **curriculum grounding data**, not learner-level training or validation data; the code does not automatically map Eedi IDs into app taxonomy IDs. The [Eedi dataset card](https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph/blob/2ae0839debf73c0ce7dd6578d8ea8b99e7be8631/README.md) confirms the CC BY 4.0 terms and that the release omits diagnostic questions and learner responses; its [data dictionary](https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph/blob/2ae0839debf73c0ce7dd6578d8ea8b99e7be8631/docs/data-dictionary.md) defines the weighted edge fields.

## 7. Prepared but unexecuted experiment: MAP misconception classifier

The response-level training target is the joint `Category:Misconception` label. Input text combines the question, selected multiple-choice answer, and student explanation. This keeps `True_Correct`, `Neither`, and misconception classes distinct rather than reducing them to correctness alone.

The prepared baseline uses:

- Word TF-IDF n-grams (1–2) plus character-within-word n-grams (3–5), with whitespace cleanup that preserves math symbols.
- Class-balanced logistic regression (`C=4.0`, `lbfgs`) to produce a ranked set of labels and class scores.
- Five-fold `StratifiedGroupKFold` out-of-fold validation, grouping exact normalized model inputs so duplicates cannot land across folds. Conflicting labels for an identical input are removed and counted; exact duplicates are deduplicated.
- Metrics: MAP@3, top-1 accuracy, macro-F1, per-label precision/recall/F1/support, label coverage, multiclass Brier, and top-label ECE. Confidence metrics are explicitly marked uncalibrated.
- A final model fit on all cleaned rows, plus labels, data checksum, split, hyperparameters, versions, and metrics.

This code is ready, but it was **not run**. There is no `data/raw/map/train.csv` in the workspace. The Kaggle rules require each participant to accept the terms and restrict the data to non-commercial uses. The script neither downloads the data nor accepts the rules. It requires the user-confirmation flag only after the user has accepted the rules. No MAP score or trained misconception artifact exists.

MAP is also unsuitable for training the persistent state updater: the competition rows do not provide learner identity or chronological interaction histories. Response-level classification and longitudinal state estimation remain separate experiments.

## 8. Implementation and research-document changes

- Added a detailed source/code audit, dataset plan, BKT run report, and this combined experiment report.
- Added the MAP trainer and the ASSISTments BKT trainer, with provenance and local artifact outputs.
- Added a two-split ASSISTments robustness evaluator with constant, skill-prevalence, and opportunity-count baselines; saved the seed-7 BKT checkpoint and parameter table.
- Added a pinned Eedi graph downloader and structural audit with local raw-data handling, CC BY attribution, checksums, and aggregate-only tracked output.
- Integrated the research record into Researcher Studio through `/api/v1/training-report`. The page presents the ASSISTments task and filtering, repeated split metrics and baselines, per-skill results, fitted skill parameters, dataset/model checksums, Eedi graph audit, and MAP's prepared-but-untrained status. It keeps the synthetic fixture in its separate benchmark tab and states that none of these artifacts feed the live state updater. The API returns aggregate values and parameter tables; it does not deserialize checkpoints or return raw learner rows or predictions.
- Made state transitions and audit event IDs reproducible for identical state/evidence inputs; stopped repeat discounts to inactive beliefs and removed a duplicate replay implementation. Candidate-specific counterevidence targeting remains open because the evidence contract is concept-level.
- Updated state math traces to identify evidence-weighted Beta masses and misconception scores as heuristics rather than calibrated Bayesian posteriors.
- Added focused tests for MAP schema/deduplication/ranking and ASSISTments event filters, duplicate rejection, student splits, skill selection, and support thresholds.
- Corrected descriptions of the synthetic benchmark in the benchmark script/report/results and Researcher Studio; removed unsupported calibration and replay claims.
- Labeled default Eedi and FoundationalASSIST fixtures as synthetic and adjusted replay API wording to match actual behavior.
- Isolated API tests in temporary SQLite files and renamed the manual HTTP conversation driver so pytest no longer collects it as a test module requiring an undeclared `httpx` dependency.
- Added explicit runtime dependencies and bounded NumPy/scikit-learn versions for pyBKT compatibility; updated the lockfile.
- Added ignored-path rules for raw data and generated models. The branch will include the small trained BKT model, parameters, and metric artifact; it will not include raw dataset rows.

## 9. Verification

- Python: `.venv/bin/python -m pytest -q` → **102 passed**, with 32 existing pyBKT NumPy deprecation warnings. Added API coverage for the training report, checksum and aggregate fields, missing/malformed-artifact behavior, and launch-directory independence.
- Frontend: `npm run build --prefix apps/web` → passed (`tsc -b` and Vite production build). Six isolated Playwright browser flows also pass, including diagnostics, chat reload, research retrieval, mobile failure handling, and retry after a committed response is lost. The training-report API includes both splits, eight per-skill metric rows, eight parameter rows, and verified checkpoint/parameter checksums.
- Patch whitespace: `git diff --check` → passed.
- Real-data training/evaluation: ASSISTments seed 42 was trained earlier; seed 7 was fit for a second learner holdout. The robustness script checksum-checks the seed-42 artifact before reuse and stores aggregate metrics without per-learner predictions.
- MAP training: not run because authorized data is absent and the terms have not been accepted by the user.

Tests verify software behavior and fixture handling; they do not count as educational validation or evidence that the model improves learning.

## 10. Next experiments, in order

1. Expand ASSISTments to at least five learner seeds and a time-based evaluation if event chronology supports a defensible cutoff; add per-skill paired metrics and learner-cluster resampling intervals. The current two-seed summary reports only descriptive split SD.
2. Run the MAP baseline only after the user accepts the current rules and supplies the authorized file. Add leave-question-out validation and unseen-label coverage; do not claim generalization from known-item folds alone.
3. Evaluate graph-to-app concept mapping on an expert-reviewed set, then build the retrieval adapter. The graph remains grounding data, not learner-level training data.
4. Build a human-reviewed response sample with exact evidence spans, concept links, category, misconception, and abstention labels. MAP's class label does not provide evidence offsets.
5. Add explicit per-hypothesis counterevidence targets, transactional event idempotency, and persistent snapshot metadata to the SQLite API path; validate policy weights against reviewed longitudinal trajectories.
6. Keep response classification, correctness prediction, evidence faithfulness, and learning impact separate. Nothing in this branch establishes that the approach beats the cited papers or improves student learning; only compare after datasets, splits, and target definitions are aligned.

## 11. Sources

- [ASSISTments 2009–2010 dataset page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data)
- [Corrected ASSISTments Skill Builder download page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010)
- [ASSISTments field interpretation](https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets)
- [MAP benchmark paper](https://aclanthology.org/2026.aimecon-main.33/)
- [MAP competition](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) and [competition rules](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/rules)
- [FoundationalASSIST dataset card](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST)
- [Eedi Misconceptions Graph](https://huggingface.co/datasets/Eedi/Eedi-Misconceptions-Graph)
- [Eedi Question-Anchored Tutoring Dialogues-2k](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k)
