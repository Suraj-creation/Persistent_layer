# Data and Training Plan

## Research objective

The project's core object is a learner representation: preserve a learner's words and outcome, turn them into concept-grounded candidate evidence with provenance, and revise a learner's uncertain state as later evidence arrives. A useful model should therefore propose a candidate interpretation from the current question, answer, and explanation. It should not overwrite the evidence ledger or directly assert a persistent diagnosis.

## What this repository currently does

- The app is a single-topic fraction prototype. Its rule extractor relies on hand-written phrase matches; its optional Instructor path calls an Azure OpenAI deployment.
- `EediDataLoader()` and `FoundationalAssistLoader()` create synthetic fixtures when no data path is supplied. The local ASSISTments 2009–10 CSV remains ignored; the selected aggregate BKT model, parameter table, and metrics are included in the `training` branch.
- The current benchmark runs those synthetic fixtures. Its 48-row, 12-student result is a demonstration of code paths only; it is not evidence about real learners or model quality.
- The state engine is deterministic, but its increments are hand-chosen heuristics rather than parameters learned from longitudinal learner outcomes. The misconception-probability update is also a heuristic. It must not be described as a validated Bayesian posterior.
- The benchmark fixtures do not match the real MAP schema or the current FoundationalASSIST schema. The real MAP competition data has question, selected answer, explanation, category, and misconception fields, but no learner sequence. FoundationalASSIST has student sequences and first answers, but does not provide verified misconception labels.

## Dataset choices

| Dataset | Use in this project | What it cannot establish | Access and license constraints |
| --- | --- | --- | --- |
| [MAP: Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) | Primary supervised signal for a candidate-misconception classifier. The associated paper describes 52,463 explanations across 15 math items, expert labels, and a mix of authentic and synthetic explanations. | It does not provide a longitudinal learner history, and 15 items are too few to establish broad transfer across curricula. It does not annotate exact evidence spans. | Competition rules state CC BY-NC 4.0, non-commercial use only, and that data users must accept the rules and prevent access by people who have not accepted them. Do not redistribute the raw data. Obtain the data through an authorized account after the rules are accepted. |
| [ASSISTments 2009–2010 Skill Builder](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | Free-to-use longitudinal correctness/skill data for a separate BKT next-response baseline; the official page requests citation. | No free-form reasoning or verified misconception labels. Its historical skill tags are not mapped to the app's taxonomy. It cannot validate misconception diagnosis. | Use the corrected one-row-per-student/problem file linked from the official page. Keep raw data local and cite the exact release. Filter hint-first and scaffolding rows when targeting first-attempt correctness. |
| [FoundationalASSIST](https://huggingface.co/datasets/ASSISTments/FoundationalASSIST) | Later correctness-only and response-prediction baselines with student sequence history, question text, skills, and first answers. | An answer string is often a numeric or short answer, not a reasoning explanation. The release does not contain verified misconception labels, so it cannot train the proposed misconception classifier by itself. | CC BY-NC 4.0 with responsible-use and access conditions. The dataset card notes access review, missing-row/version updates, and restrictions for some researchers. Review its current card before acquisition. |
| [Eedi Question-Anchored Tutoring Dialogues-2k](https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k) | A later source for dialogue handling and tutor-talk-move analysis; its dialogues can inform how to preserve turns and context. | Tutor talk-move labels are not learner misconception labels or a ground truth for learner-state updates. | CC BY-NC-SA 4.0; intended for non-commercial research. Do not treat its labels as student-cognition labels. |
| [Eedi 2024 Mining Misconceptions](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics) | Optional source of question/distractor-to-misconception mappings for taxonomy alignment and diagnostic-item metadata. | It is not a dataset of learner explanations or longitudinal interactions. | Competition data rules apply. Review the specific release and terms before use. |

The MAP competition is the best first source for the language-to-candidate-hypothesis component. Its rules are non-commercial and bind access to accepted participants. This checkout has no MAP data and no Kaggle credentials. Do not fetch a third-party mirror to avoid those access terms.

### Completed supplementary BKT run

The official ASSISTments 2009–10 corrected Skill Builder release was used for a separate correctness-only baseline. The run retained original problems with `first_action == 0` (attempt), filtered fraction/percent skill names to at least 100 events and 20 learners, and split learners 80/20 before fitting. It trained on 39,757 events and evaluated on 9,901 events from 441 held-out learners across eight skills. Aggregate held-out metrics were AUC 0.6673, accuracy 0.7428, Brier 0.1805, and uncalibrated 10-bin ECE 0.0088, against a constant-prevalence baseline of AUC 0.5000, accuracy 0.7365, Brier 0.1942, ECE 0.0114. This is an exploratory one-split knowledge-tracing result, not a misconception classifier or proof of transfer. See [the run report](experiments/assistments_bkt_report.md) and the aggregate artifacts in `models/assistments_2009_2010_bkt/`.

## What to train

### Stage 1: supervised candidate-interpretation baseline

Train a compact, local multi-class classifier on MAP's labeled training rows.

- **Input:** question text, selected multiple-choice answer, and the student's explanation.
- **Target:** the joint `Category:Misconception` label used by the benchmark. This preserves distinctions among correct reasoning, neither/vague explanations, and misconception types.
- **First model:** word and character TF-IDF features with logistic regression. It is inexpensive, reproducible, and creates a clear baseline before spending GPU time.
- **Possible next model:** fine-tune a small pretrained encoder (for example DistilRoBERTa) only if the baseline and data audit justify it. This is task-specific classifier training, not foundation-model pretraining.
- **Inference contract:** return ranked candidate labels and probabilities as proposals. Preserve the student's full original text verbatim and validate any quote spans separately. Do not call a class probability a calibrated probability until calibration is measured and corrected on held-out data.

The initial trainer is [experiments/train_map_misconception_classifier.py](experiments/train_map_misconception_classifier.py). It requires an explicit flag confirming that the person running it has already accepted the MAP data-access rules. It does not download the data.

### Stage 2: evidence extraction quality

MAP labels identify a response category and misconception, not the exact phrase supporting that interpretation. A separate human-reviewed subset is needed to evaluate exact quote spans, claim faithfulness, polarity, concept links, and ambiguity. LLM-generated labels cannot be the only reference used to score that same model.

### Stage 3: persistent state and temporal updates

Do not train the persistent state updater on MAP: its rows do not give learner identity or chronological history. Keep the updater explicit and replayable until a suitable longitudinal source and an external state target are available. Fit or tune update strengths only from repeated, validated learner trajectories or an adjudicated longitudinal study. Evaluate contradictory evidence, sparse histories, and overreaction to isolated errors separately from text classification.

The ASSISTments run is an initial correctness-only KT baseline. FoundationalASSIST may later support a separate sequence audit if its current access conditions are satisfied. Neither source supplies ground-truth misconception trajectories. Immediate next-answer prediction is not proof of misconception diagnosis, retention, transfer, or learning impact.

## Evaluation design

1. Record the exact dataset release, access date, license, row count, field mapping, and file checksum. Measure missingness and report authentic and synthetic MAP examples separately when the release identifies them.
2. Normalize exact duplicate model inputs before splitting. Keep repeated identical inputs in one split group and report conflicting duplicate labels instead of silently choosing one.
3. The MAP release does not expose a learner ID, so learner-level splitting is not possible. Use grouped stratified validation for new explanations on known items, then separately report leave-question-out results and the percentage of validation labels that were never seen in training. Do not claim broad item or curriculum generalization from the 15-item benchmark.
4. Report MAP@3, top-1 accuracy, macro-F1, per-label precision/recall, multiclass Brier score, and top-label ECE. Report uncertainty and class coverage; calibrate only on held-out data.
5. Compare against majority, TF-IDF, and a pretrained-encoder classifier on identical MAP splits. A knowledge-tracing baseline belongs on longitudinal interaction data, not on the MAP item-level rows. For KT, repeat the ASSISTments evaluation across seeds or temporal splits and compare opportunity-count baselines before integrating anything.
6. Integrate the best validated candidate model into the evidence extractor only after checking human-reviewed extraction fidelity, abstention behavior, and concept alignment.

## Known implementation gaps to address in order

1. Obtain an authorized real MAP release and implement an explicit dataset audit. Keep raw files local and ignored by Git.
2. Train the Stage 1 baseline and save the data checksum, split definition, label map, dependency versions, metrics, and model artifact together.
3. Update the canonical evidence schema/taxonomy mapping to represent MAP categories without conflating `Neither`, a calculation slip, a misconception, and mastery.
4. Validate evidence extraction and quote fidelity with human annotations.
5. Repair and calibrate the state-update math, make replay snapshots reproducible, and evaluate on real temporal evidence before making learner-state claims.
6. Replace synthetic benchmark claims with real, leakage-aware results. Defer active diagnosis and intervention-effectiveness claims until the required item-response data and outcomes exist.
