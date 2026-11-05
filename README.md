# Language-Grounded Learner State Modeling Platform
> **For Misconception-Aware Adaptive Learning**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-cyan.svg)](https://react.dev)
[![pyBKT](https://img.shields.io/badge/pyBKT-1.4.3-orange.svg)](https://github.com/CAHLR/pyBKT)

## 1. Problem Statement & Motivation

> **Authoritative Statement:** We aim to develop and evaluate a language-grounded learner-state modeling platform that transforms students' natural-language interactions and learning history into structured, concept-grounded, uncertainty-aware evidence. This evidence is used to maintain a persistent representation of learners' conceptual understanding, identify possible misconceptions and prerequisite gaps, and support more informed diagnostic and adaptive learning decisions.

### Why Correctness Alone Fails (The Fractions Example)
Consider two students attempting $\frac{1}{2} + \frac{1}{3}$:
- **Student A ($\frac{2}{5}$):** *"I added the top numbers and the bottom numbers because we add both parts."* $\rightarrow$ **Deep systematic rule misconception** (`denominator_addition_rule`), requiring conceptual instruction on fraction units.
- **Student B ($\frac{4}{6}$):** *"I know the denominators must match, so I tried to find a common denominator, but I made a calculation mistake."* $\rightarrow$ **Calculation slip** (`calculation_slip`), requiring an arithmetic check, **not** a misconception flag.

In conventional Knowledge Tracing (BKT, DKT), both students receive `correctness = 0` and are treated identically. The platform's goal is to preserve the student's language as **Cognitive Evidence** and use that evidence to maintain an uncertain learner state. The current extractor is rule-based and the current state updates use hand-set heuristics; they are not a trained evidence model or validated Bayesian posterior.

---

## 2. The Core Closed Loop

$$\mathbf{Language} \longrightarrow \mathbf{Evidence} \longrightarrow \mathbf{Learner\ State} \longrightarrow \mathbf{Diagnosis} \longrightarrow \mathbf{Updated\ Evidence}$$

```
Learner Interaction ──▶ NLP Interpretation ──▶ Cognitive Evidence ──▶ Persistent State Update ──▶ Active Diagnosis ──▶ Targeted Probe
(Text & Metadata)       (Rules / optional LLM) (Quotes & Polarity)    (Heuristic prototype)      (EIG Item Selector)    (Tests candidate explanations)
```

---

## 3. Reuse vs. Build: Open-Source Codebases & Literature

| Component | Codebase / Literature | Role |
| :--- | :--- | :--- |
| **Knowledge Tracing Engine** | `pyKT` (NeurIPS 2022) | Standard deep KT comparator models (DKT, AKT, SAKT) & fold splitters |
| **Bayesian Knowledge Tracing** | `pyBKT` (MIT) | Fast individualized BKT baseline with slip/guess parameters |
| **Longitudinal correctness data** | [ASSISTments 2009–2010 Skill Builder](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) | Free-to-use historical skill sequences; supports correctness-only KT, not misconception supervision |
| **Cognitive Diagnosis** | `EduCDM` (BigData USTC) | Standard psychometric CDMs (DINA, NeuralCDM, IRT) |
| **Longitudinal correctness data** | `FoundationalASSIST` (2026) | 1.72M interactions with items, answers, and KCs; useful for later sequence baselines, but not verified misconception supervision |
| **Response-level misconception labels** | `MAP: Charting Student Math Misunderstandings` | Student explanations with expert misconception/category labels; non-commercial access rules apply and the release has no learner history |
| **Question-Aware KT Reference** | `SQKT` (ACL 2025) | Question integration methodology |
| **LLM Profiling Comparator** | `CIKT` (EMNLP 2025) | Analyst-Predictor textual learner profiling baseline |
| **Structured Output Engine** | `instructor` + Pydantic v2 | Guaranteed JSON Schema extraction from pretrained LLMs |
| **Semantic Concept Matcher** | `sentence-transformers` | Zero-shot KC embedding alignment |

---

## 4. Repository Structure

```
NLP/
├── apps/
│   ├── web/                         # React 19 + TypeScript + Vite UI
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── Navbar.tsx       # Switcher between Student and Researcher view
│   │   │   │   ├── StudentPortal.tsx# Problem solver, explanation box, adaptive probe
│   │   │   │   └── ResearcherStudio.tsx # Audit, Evidence Ledger, Trajectories, Benchmarks
│   │   │   ├── api/client.ts        # Typed API client
│   │   │   └── types/index.ts       # TypeScript data contracts
│   │   └── package.json
│   └── api/                         # FastAPI Application Gateway
│       └── app/main.py              # REST routes (/process, /state, /audit, /benchmarks, /replay)
├── packages/
│   └── learner_state_core/          # Shared Core Python Library
│       ├── data/                    # Eedi & FoundationalASSIST loaders, leakage-safe splitters
│       ├── evidence/                # CognitiveEvidence schemas & Instructor/Rule extractors
│       ├── grounding/               # Curriculum taxonomies & SemanticTaxonomyMatcher
│       ├── state/                   # BayesianLearnerStateEngine (slips, misconceptions, replay)
│       ├── diagnosis/               # ActiveDiagnosisProber (Expected Information Gain)
│       └── baselines/               # ClassicalBKTBaseline (pyBKT wrapper)
├── experiments/
│   ├── run_benchmark.py             # 3-tier comparative evaluation script
│   ├── benchmark_results.json       # JSON benchmark metrics
│   └── benchmark_report.md          # Generated benchmark report
├── tests/                           # 15 unit and integration tests
├── pyproject.toml                   # Python package definition
└── README.md
```

---

## 5. Quickstart Guide

### Current Research Status

The checked-in benchmark is a synthetic software demonstration, not a result on real student data. A separate exploratory BKT baseline has been trained on an ignored local copy of ASSISTments 2009–2010. There is still no trained misconception classifier or learned state updater. See [DATA_AND_TRAINING_PLAN.md](DATA_AND_TRAINING_PLAN.md) for the research plan, dataset limits, and access requirements.

### Run Backend Tests
```bash
python -m pytest tests
```

### Run the Synthetic Pipeline Demonstration
```bash
python experiments/run_benchmark.py
```
This currently uses generated fraction archetypes and must not be cited as an empirical benchmark. It compares:
- **Tier 1:** Correctness-Only Knowledge Tracing (`pyBKT`)
- **Tier 2:** Language-Aware Baseline (Direct Text Features / TF-IDF)
- **Tier 3:** Heuristic Structured Evidence + Persistent Learner State

### Train the First MAP Misconception Baseline

The MAP dataset is non-commercial and its competition rules require account holders to accept the terms before accessing the data. After you have accepted those rules and placed an authorized `train.csv` at `data/raw/map/train.csv`, run:

```bash
python experiments/train_map_misconception_classifier.py --confirm-map-data-access-authorization
```

The trainer writes a local model and validation report to `models/map_misconception_classifier/`; raw data and model artifacts are ignored by Git.

### Train the ASSISTments BKT Baseline

The separate correctness-only baseline uses the official corrected 2009–2010 Skill Builder file. Its page says the data is free to use and requests citation. Place the CSV at `data/raw/assistments/assistments_2009_2010_corrected.csv`, then run:

```bash
python experiments/train_assistments_bkt.py
```

By default it selects fraction/percent skills with at least 100 events and 20 learners, then evaluates on a deterministic student-level holdout. It predicts next-response correctness, not misconceptions. Metrics, parameters, and a serialized model are written to `models/assistments_2009_2010_bkt/`. See [the run report](experiments/assistments_bkt_report.md).

For the repeated-split comparison against constant, smoothed skill-prevalence, and opportunity-count baselines, run:

```bash
.venv/bin/python experiments/evaluate_assistments_bkt_robustness.py
```

The default check uses student split seeds 42 and 7. It verifies the seed-42 artifact against the local CSV checksum, fits and saves the seed-7 aggregate model, and writes metrics to `models/assistments_2009_2010_bkt/robustness.json`. The two-split spread is descriptive only and is not a confidence interval.

### Audit the public Eedi misconception graph

The separate graph release is licensed CC BY 4.0 and contains curriculum constructs and weighted misconception links, not student responses. Fetch and validate the pinned version with:

```bash
.venv/bin/python experiments/ingest_eedi_misconceptions_graph.py --download
```

Raw graph CSVs remain under the Git-ignored `data/raw/` directory. Only the aggregate audit, source revision, attribution, and file checksums are tracked in `experiments/eedi_graph_audit.json`. See [the training pipeline plan](DATA_AND_TRAINING_PLAN.md).

### Start Backend API Server
```bash
uvicorn apps.api.app.main:app --reload --port 8000
```
Interactive API Docs available at `http://localhost:8000/docs`.

### Start Frontend Console (React 19 + TypeScript + Vite)
```bash
cd apps/web
npm run dev
```
Open `http://localhost:5173` to interact with:
- **Student Learning Portal:** Solve problems, provide reasoning, and trigger real-time Cognitive Evidence extraction and adaptive diagnostic probing.
- **Researcher Studio:** Inspect dataset audits, browse the evidence ledger, review learner-state trajectories with "Why Changed?" events, and view the synthetic pipeline demonstration.
