# Training branch — one-slide summary

**Purpose** — Build a fraction-focused learner-state prototype that preserves reasoning as evidence, and test correctness prediction separately.

**Trained** — Correctness-only pyBKT on corrected ASSISTments 2009–2010: **49,658** first attempts, **8** fraction/percent skills. Two 80/20 student holdouts (seeds 42, 7; **9,901 / 9,366** test events). The model uses no explanation text and is not wired into app state.

**Mean results across 2 splits** — AUC **0.6639 ± 0.0048** · accuracy **0.7449** · Brier **0.1798 ± 0.0009** · 10-bin ECE **0.0088** (uncalibrated). Skill + opportunity logistic: AUC **0.5653**, Brier **0.1904**. Two splits are descriptive, not a confidence interval or significance test.

**Other work and limits** — Eedi graph audited: **19,294** links, **2,258** constructs (grounding only). MAP explanation classifier is prepared but **not trained**; authorized data is absent. No real diagnosis or learning benefit has been validated.

**Next** — Train MAP after authorized access; evaluate evidence faithfulness and state updates separately.

*Sources: `TRAINING_EXPERIMENT_REPORT.md`; `experiments/assistments_bkt_report.md`; `models/assistments_2009_2010_bkt/robustness.json`; `experiments/eedi_graph_audit.json`.*
