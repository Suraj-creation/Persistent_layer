# ASSISTments 2009–2010 BKT Baseline

## Purpose

This is a supplementary, correctness-only knowledge-tracing baseline for the project's learner-state work. It tests whether a standard BKT model can predict later first-attempt correctness from earlier responses on the same released skill. It does not classify misconceptions from student explanations and is not integrated into the app's current fraction taxonomy.

## Data and preparation

- Source: [official ASSISTments 2009–2010 Skill Builder page](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010), using the corrected one-row-per-student/problem CSV linked there. The page says the data is free to use and requests citation of the precise dataset page.
- The official [data interpretation page](https://sites.google.com/site/assistmentsdata/an-explanation-on-how-to-interpret-our-data-sets) defines `first_action` as `0 = attempt`, `1 = hint`, `2 = scaffolding`; `correct` is first-attempt correctness, with hint/scaffolding use otherwise recorded as incorrect. The trainer therefore keeps `original == 1`, `first_action == 0`, binary `correct`, and nonmissing user, skill, problem, and order fields.
- `order_id` is sorted within student before modeling. Duplicate student/problem/order events are rejected rather than silently deduplicated.
- The raw file has 346,860 rows. After the event filters and metadata checks, 228,250 rows remained. Selecting fraction/percent skill names and requiring at least 100 events and 20 students kept 49,658 events across eight skills.
- Split: deterministic 80/20 student holdout, seed 42. This produced 39,757 training events from 1,762 students and 9,901 test events from 441 unseen students. All eight test skills were represented in training; no test events were skipped for unknown skills.
- BKT was fit serially with one initialization and no forgetting. Serial mode avoids `pyBKT`'s default creation of a new process pool during each EM iteration.

## Held-out result

| Metric | BKT | Constant training-prevalence baseline |
| --- | ---: | ---: |
| Test events | 9,901 | 9,901 |
| ROC AUC | 0.6673 | 0.5000 |
| Accuracy at 0.5 | 0.7428 | 0.7365 |
| Brier score | 0.1805 | 0.1942 |
| 10-bin ECE, uncalibrated | 0.0088 | 0.0114 |

These are single-split exploratory scores. The micro-average pools skills, so frequent skills contribute more. Per-skill results are in `models/assistments_2009_2010_bkt/metrics.json`; the `Percents` test subset has only 21 rows, so its metric is especially unstable. ECE is bin-dependent and no post-hoc probability calibration was applied.

## Artifacts and reproducibility

- Training code: `experiments/train_assistments_bkt.py`
- Metrics, per-skill results, data checksum, versions: `models/assistments_2009_2010_bkt/metrics.json`
- Fitted parameters: `models/assistments_2009_2010_bkt/parameters.csv`
- Serialized `pyBKT` model: `models/assistments_2009_2010_bkt/model.pkl`
- The raw data remains ignored by Git. The selected aggregate model, parameter table, and metrics are included in the `training` branch; the metrics record the input SHA-256 so the run can be reproduced with the same release.

Run with the default local data path:

```bash
python experiments/train_assistments_bkt.py
```

## Limits and next step

This result supports only same-release next-response prediction on these historical ASSISTments skills. The skills are not mapped to the app's current concept IDs, and correctness prediction is not evidence of misconception diagnosis, retention, transfer, or intervention benefit. The next KT iteration should use repeated student-level splits or temporal validation, test a better-supported item/skill alignment, compare with a simple opportunity-count baseline, and inspect parameter stability across seeds. The primary response-to-misconception classifier still requires an authorized MAP training file and accepted competition rules.
