# Synthetic 3-Tier Pipeline Demonstration — Not a Research Benchmark

> **Validity warning:** The default loader generated hand-authored archetypes. These values do not measure performance on real student data and must not be cited as research findings.

## Dataset: Synthetic Eedi-style Fraction Fixture
- **Total Interactions:** 48
- **Students:** 12 | **Items:** 4
- **Text Availability:** 100.0%
- **Misconception Labeling Rate:** 50.0%

---

## Benchmark Results Matrix

| Metric | Tier 1: Correctness-Only (pyBKT) | Tier 2: Language-Aware (Direct Text) | Tier 3: Proposed Persistent Evidence State | Delta (Tier 3 vs. Tier 1) |
| :--- | :---: | :---: | :---: | :---: |
| **AUC-ROC** | 0.9062 | 1.0000 | **0.5938** | +-0.3124 |
| **Accuracy** | 0.7500 | 1.0000 | **0.6667** | +-0.0833 |
| **Brier Score** (lower is better) | 0.1653 | 0.0958 | **0.2233** | 0.0580 |
| **ECE (Calibration Error)** (lower is better) | 0.1161 | 0.2938 | **0.2744** | --0.1583 |
| **Misconception Precision** | 0.0000 | 1.0000 | **0.5000** | +0.5000 |
| **Misconception Recall** | 0.0000 | 1.0000 | **0.5000** | +0.5000 |
| **Binary Misconception F1** | 0.0000 | 1.0000 | **0.5000** | **+0.5000** |

---

## Key Findings
1. The tiers run against repeated, hand-authored response templates, so the split does not test generalization to new wording or real learners.
2. Tier 3 AUC is **0.5938** versus Tier 1's **0.9062**; Tier 3 ECE is **0.2744**. These fixture metrics do not show improved prediction or calibrated probabilities.
3. Tier 1's zero misconception F1 is assigned because pyBKT emits no misconception class; it is not a measured head-to-head diagnosis result.
4. Tier 2 uses the current explanation. Tiers 1 and 3 use previous responses. The tasks differ and their score deltas are not a fair model comparison.
5. This run uses state policy `evidence-heuristic-v3.0`, rule extractor `rule-based-baseline-v3`, and seed `42`.

For the real-data training and evaluation plan, see `DATA_AND_TRAINING_PLAN.md`.
