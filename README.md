# Language-Grounded Learner State Modeling Platform
> **For Misconception-Aware Adaptive Learning**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green.svg)](https://fastapi.tiangolo.com)
[![React 19](https://img.shields.io/badge/React-19.0-cyan.svg)](https://react.dev)
[![pyBKT](https://img.shields.io/badge/pyBKT-1.4.3-orange.svg)](https://github.com/CAHLR/pyBKT)
[![Tests Passing](https://img.shields.io/badge/Tests-15%20Passed-brightgreen.svg)]()

## 1. Problem Statement & Motivation

> **Authoritative Statement:** We aim to develop and evaluate a language-grounded learner-state modeling platform that transforms students' natural-language interactions and learning history into structured, concept-grounded, uncertainty-aware evidence. This evidence is used to maintain a persistent representation of learners' conceptual understanding, identify possible misconceptions and prerequisite gaps, and support more informed diagnostic and adaptive learning decisions.

### Why Correctness Alone Fails (The Fractions Example)
Consider two students attempting $\frac{1}{2} + \frac{1}{3}$:
- **Student A ($\frac{2}{5}$):** *"I added the top numbers and the bottom numbers because we add both parts."* $\rightarrow$ **Deep systematic rule misconception** (`denominator_addition_rule`), requiring conceptual instruction on fraction units.
- **Student B ($\frac{4}{6}$):** *"I know the denominators must match, so I tried to find a common denominator, but I made a calculation mistake."* $\rightarrow$ **Calculation slip** (`calculation_slip`), requiring an arithmetic check, **not** a misconception flag.

In conventional Knowledge Tracing (BKT, DKT), both students receive `correctness = 0` and are treated identically. Our platform decodes the language signal, mints an immutable **Cognitive Evidence** record, and updates a persistent Bayesian belief state.

---

## 2. The Core Closed Loop

$$\mathbf{Language} \longrightarrow \mathbf{Evidence} \longrightarrow \mathbf{Learner\ State} \longrightarrow \mathbf{Diagnosis} \longrightarrow \mathbf{Updated\ Evidence}$$

```
Learner Interaction ──▶ NLP Interpretation ──▶ Cognitive Evidence ──▶ Persistent State Update ──▶ Active Diagnosis ──▶ Targeted Probe
(Text & Metadata)       (Instructor + LLM)     (Quotes & Polarity)    (Bayesian Integrator)      (EIG Item Selector)    (Resolves Slip vs Bug)
```

---

## 3. Reuse vs. Build: Open-Source Codebases & Literature

| Component | Codebase / Literature | Role |
| :--- | :--- | :--- |
| **Knowledge Tracing Engine** | `pyKT` (NeurIPS 2022) | Standard deep KT comparator models (DKT, AKT, SAKT) & fold splitters |
| **Bayesian Knowledge Tracing** | `pyBKT` (MIT) | Fast individualized BKT baseline with slip/guess parameters |
| **Cognitive Diagnosis** | `EduCDM` (BigData USTC) | Standard psychometric CDMs (DINA, NeuralCDM, IRT) |
| **Language-Rich Educational Data** | `FoundationalASSIST` (2026) | 1.72M student interactions with problem stems, text, and KCs |
| **Ground-Truth Misconceptions** | `Eedi Diagnostic Questions` | Math multiple-choice items with distractor-targeted misconceptions and explanations |
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

### Run Backend Tests (15 Passed)
```bash
python -m pytest tests
```

### Run 3-Tier Comparative Benchmark
```bash
python experiments/run_benchmark.py
```
Outputs evaluation comparing:
- **Tier 1:** Correctness-Only Knowledge Tracing (`pyBKT`)
- **Tier 2:** Language-Aware Baseline (Direct Text Features / TF-IDF)
- **Tier 3:** Proposed Structured Evidence + Persistent Bayesian State

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
- **Researcher Studio:** Inspect dataset audits, browse the immutable evidence ledger, inspect Bayesian state trajectories with "Why Changed?" events, and view the 3-tier benchmark matrix.
