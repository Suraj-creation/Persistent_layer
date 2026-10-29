# Language-Grounded Persistent Learner State Modeling

## Comprehensive Research Reference

**Working title:** Language-Grounded Learner State Modeling for
Misconception-Aware Adaptive Learning\
**Domain:** NLP, educational data mining, knowledge tracing, learner
modeling\
**Purpose:** A unified guide to the research problem, motivation, gap,
four core papers, proposed system, experimental plan, and references.

------------------------------------------------------------------------

# 1. Executive Summary

This project investigates whether a learner's natural-language
interactions can be transformed into reliable, structured evidence about
evolving conceptual understanding and maintained as a persistent,
uncertainty-aware learner state. Conventional Knowledge Tracing (KT)
commonly uses sequences of questions, skills, and correctness labels to
estimate mastery and predict future performance. These signals are
useful but incomplete: identical outcomes can arise from different
reasoning, misconceptions, missing prerequisites, slips, guessing, or
ambiguous wording. Likewise, a correct answer may reflect understanding,
memorization, or chance.

Learners' questions, explanations, written responses, reasoning,
requests for hints, and interaction patterns may provide additional
clues. The project therefore places NLP at the center of a pipeline that
processes language, grounds it in educational concepts, extracts
**Cognitive Evidence**, and updates a learner representation over time.
The representation should distinguish observations from interpretations,
preserve the source of each inference, and express uncertainty rather
than treating an LLM's output as fact.

The longer-term system could use this state to identify uncertainty,
select diagnostic questions, and support personalized interventions. The
immediate research contribution should remain experimentally testable:
determine whether structured, language-grounded evidence and explicit
state updates add measurable value beyond correctness-only KT and
simpler text-feature baselines.

**Core pipeline:**\
`Learner interactions → NLP interpretation → Cognitive Evidence → State update → Diagnosis/prediction → Next question or intervention → New evidence`

------------------------------------------------------------------------

# 2. Research Problem

## 2.1 Formal problem statement

Traditional educational systems and knowledge-tracing models largely
infer learner knowledge from observable outcomes such as correctness,
question IDs, skill labels, attempts, and interaction history. Although
these signals support prediction of future performance, they provide an
incomplete account of a learner's evolving conceptual state: identical
outcomes may arise from different reasoning processes, prerequisite
gaps, misconceptions, guessing, or careless errors, while correct
responses do not necessarily establish durable understanding. Learners'
natural-language questions, explanations, written responses, and
reasoning can contain diagnostic information about what they understand,
misunderstand, and remain uncertain about, yet this information is often
used as a task-specific predictive input rather than preserved as
structured evidence linked to concepts and updated over time. This
research investigates how NLP and language models can transform
heterogeneous linguistic and behavioral interactions into **structured,
concept-grounded Cognitive Evidence** and integrate that evidence into a
**persistent, probabilistic, and provenance-aware learner-state
representation** that tracks competence, candidate misconceptions,
reasoning patterns, and uncertainty across learning interactions. It
further investigates whether this representation can support more
informative diagnosis and, in later stages, adaptive question and
intervention selection, compared with conventional correctness-based
knowledge tracing.

## 2.2 Central research question

**Can natural-language learner interactions be transformed into
reliable, structured cognitive evidence that enables a persistent,
uncertainty-aware representation of a learner's evolving conceptual
state, and can this representation improve diagnosis and prediction
beyond conventional correctness-based knowledge tracing?**

Two parts must be tested: 1. **Representation:** Does language-derived
evidence capture useful information beyond correctness and metadata? 2.
**Utility:** Does maintaining and updating that evidence improve
measurable tasks such as misconception detection, calibration,
future-response prediction, retention, or transfer?

The second part is a hypothesis, not an assumption.

## 2.3 Why correctness alone is insufficient

Two students answer a fraction question incorrectly.

-   A: "I added the top numbers and the bottom numbers."
-   B: "I knew I needed a common denominator, but multiplied
    incorrectly."

Both receive `incorrect = 0`, but A may hold a rule-level misconception
while B may have made an arithmetic slip. Similarly, a correct response
may come from a sound explanation or a guess. Correctness is an outcome
observation; cognition is a hidden state inferred from multiple
observations.

## 2.4 What the project is---and is not

The research object is the **learner representation and its update
mechanism**, not primarily a chat interface. The questions are: what can
NLP extract, how should it be represented and grounded, how should
new/conflicting evidence update the state, and does the state help
downstream tasks?

An LLM is one possible language-interpretation component. It is not
itself the learner model and does not directly observe a student's mind.
The project is not simply "build an AI tutor," "use an LLM to teach," or
"add text to KT."

------------------------------------------------------------------------

# 3. Essential Concepts

## Knowledge Tracing

KT estimates changing knowledge/skill mastery from sequential learning
interactions, often predicting the probability of a correct response to
a future item. Classical examples include Bayesian Knowledge Tracing
(BKT); deep methods include recurrent, memory-based, attention, and
transformer models.

## Learner modeling

A broader task of representing learner characteristics relevant to
learning---such as mastery, gaps, strategies, misconceptions, and
support needs. This project focuses on evolving conceptual knowledge.

## Latent state and belief state

True understanding is not directly observable in ordinary logs, so it is
a **latent state**. A **belief state** is the system's current estimate
of that hidden state. For example, store
`P(candidate misconception: denominator addition)=0.72`, not
`misconception=true`. The probability is the model's uncertainty, not a
literal measurement of the mind, and must be validated.

## Cognitive Evidence

The proposed intermediate representation between raw interaction and
state inference. It should record what was observed/extracted without
prematurely turning it into a definitive diagnosis.

Illustrative record:

``` yaml
evidence_id: E_104
learner_id: pseudonymous_17
interaction_id: I_208
timestamp: 2026-09-20T10:15:00
source_type: student_explanation
raw_text: "I added the top and bottom numbers."
concept_links:
  - concept: fraction_addition
    confidence: 0.91
claim: "Learner describes adding numerators and denominators independently."
candidate_interpretation:
  hypothesis: denominator_addition_rule
  confidence: 0.78
polarity: supports
extractor_version: model_x
review_status: unverified
```

This is an illustrative schema, not a validated ontology. Keep raw
observations, extracted claims, inferred hypotheses, and provenance
separate.

## Provenance

The origin and processing history of evidence: source interaction,
text/answer, timestamp, model version, transformation, and human review.
It allows teachers and researchers to inspect why a state claim exists
and supports auditing and correction.

## Persistence and longitudinal modeling

A persistent state is retained and updated across interactions.
Persistence is more than saving a profile: the system needs an explicit
policy for new, old, contradictory, weak, or time-sensitive evidence.
Longitudinal evaluation respects time order and tests changes across a
learning trajectory.

## Misconception versus error

An error is an incorrect outcome. A misconception is a candidate
systematic misunderstanding. One wrong response rarely proves a
misconception. Repeated patterns, explanations, distractor choices, and
diagnostic follow-up may strengthen or weaken a hypothesis. Use
"candidate misconception" and "evidence consistent with" until
validated.

## Uncertainty and calibration

Uncertainty describes incomplete or ambiguous evidence. Calibration asks
whether confidence matches empirical reliability. If predictions labeled
near 90% confidence are correct only 60% of the time, the model is
overconfident. Possible metrics include Brier score, expected
calibration error (ECE), and negative log-likelihood.

## Concept grounding / Knowledge Components

Grounding links language and responses to a curriculum taxonomy:
concepts, skills, knowledge components (KCs), prerequisites, or learning
objectives. Use documented curriculum labels or expert-validated
mappings; do not accept an LLM-invented concept graph as truth without
review.

## Retention and transfer

Retention is whether knowledge remains available after a delay. Transfer
is applying it in a new context. Neither is established by immediate
next-answer prediction alone; both require appropriate delayed or
novel-task assessments.

------------------------------------------------------------------------

# 4. Research Gap

Existing work has established knowledge tracing, language-aware KT, rich
educational datasets, question/concept representation learning,
LLM-generated learner profiles, misconception analysis, and adaptive
assessment as active areas. SQKT demonstrates that student questions can
improve programming KT; FoundationalASSIST preserves large-scale
question and response text; Q-MCKT models question- and concept-level
acquisition with contrastive learning and an IRT-based prediction layer;
and CIKT uses an iterative Analyst--Predictor framework to generate and
refine learner profiles. It would therefore be inaccurate to claim that
NLP has never been used in KT, that learner profiles do not exist, or
that interpretable representations are absent.

**Proposed gap:** how to transform heterogeneous learner language and
behavior into explicit, concept-grounded evidence records, maintain a
persistent probabilistic state with uncertainty and provenance, and
evaluate whether that state supports diagnosis and downstream decisions
across time. The precise novelty must be established through a
systematic review of adjacent work and operationalized as testable
differences from the closest methods---not asserted as an absolute
absence of prior work.

The project does not claim novelty for NLP, LLMs, embeddings, KT,
misconception detection, knowledge graphs, RAG, active learning, or
tutoring individually. The contribution under investigation is their
integration around an evidence-centric persistent learner-state
representation.

------------------------------------------------------------------------

# 5. Four Core Research Papers

These papers play different roles and use different tasks, datasets, and
metrics; their numerical results are not directly interchangeable.

## 5.1 SQKT --- ACL 2025

**Title:** "Knowledge Tracing in Programming Education Integrating
Students' Questions"\
**Authors:** Doyoun Kim, Suin Kim, Yohan Jo\
**Venue:** Proceedings of ACL 2025, pp. 27703--27718\
**Paper:** https://aclanthology.org/2025.acl-long.1343/\
**Code:** https://github.com/holi-lab/SQKT

**Problem:** Programming KT is difficult, and conventional models may
overlook questions students ask while learning. Questions can expose
confusion or misconceptions not evident from completion labels.

**Approach:** SQKT (Students' Question-based Knowledge Tracing)
incorporates student questions and automatically extracted skill
information into a model predicting performance on subsequent
programming problems. It creates semantically rich representations from
learning history and question-related information.

**Reported contribution/result:** Improved prediction across Python
courses of varying difficulty; the paper reports a **33.1
percentage-point absolute AUC improvement over baselines in its
in-domain experiments**. This is specific to its setting and should not
be generalized to other tasks or datasets.

**Relevance:** Supports the premise that learner language contains
useful information beyond conventional interaction signals. It is a
motivation and language-aware baseline.

**Remaining distinction to investigate:** SQKT primarily targets
future-performance prediction. Our study asks whether language can be
converted into atomic, concept-grounded evidence with provenance and
maintained as an explicit state over time. Confirm exact method overlap
before making an absolute novelty claim.

**Use in our work:** Compare correctness-only KT, text/embedding
features, and structured evidence; do not treat its reported AUC gain as
a target guaranteed on our dataset.

## 5.2 FoundationalASSIST --- 2026

**Title:** "FoundationalASSIST: An Educational Dataset for Foundational
Knowledge Tracing and Pedagogical Grounding of LLMs"\
**Authors:** Worden et al.\
**Paper:** https://arxiv.org/abs/2602.00070\
**Dataset:**
https://huggingface.co/datasets/ASSISTments/FoundationalASSIST

**Problem:** Many educational datasets expose question/skill IDs and
binary correctness while omitting problem text, actual responses, and
answer-option selections. That limits language-based learner modeling
and error analysis.

**Dataset contribution:** Approximately **1.72 million interactions,
5,000 students, 3,395 problems, and 462 knowledge components**, with
problem text, answer options, actual student responses, distractors, and
interaction metadata. The paper describes ASSISTments data spanning
January 2019--July 2024; check the exact dataset release/version and
schema before use.

**Benchmark/finding:** The work evaluates LLMs on knowledge tracing and
pedagogical-grounding tasks, distinguishing solving a problem from
predicting a particular learner's response. Reported findings include
optimistic bias in some models and weak item-discrimination judgments.
The earlier slide cited **56.2% best LLM KT accuracy versus 51.3%
trivial baseline**; those figures belong only to the paper's specific
setup/metric, not a universal statement about LLMs. Do not compare these
accuracy figures directly with SQKT's AUC.

**Relevance:** Supplies language-rich data that may support evidence
extraction, response analysis, and concept grounding.

**Limitation:** Rich data is not itself a cognitive-state model. Actual
answer strings may be numeric/short and not detailed explanations. Check
whether the chosen release contains genuine questions/explanations
before claiming that it supports reasoning extraction. Misconception
labels, update algorithms, and validity of inferences are not
automatically supplied.

**Use in our work:** Audit fields, missingness, time structure,
licensing, and KC alignment; establish correctness-only and
language-aware baselines; annotate a subset if reliable
misconception/evidence labels are absent.

## 5.3 Q-MCKT

**Title:** "A Question-centric Multi-experts Contrastive Learning
Framework for Improving the Accuracy and Interpretability of Deep
Sequential Knowledge Tracing Models"\
**Authors:** Hengyuan Zhang, Zitao Liu, Chenming Shang, Dawei Li, Yong
Jiang\
**Preprint:** https://arxiv.org/abs/2403.07322\
**Code:** https://github.com/rattlesnakey/Q-MCKT

**Problem:** Deep KT may overfit question embeddings when some questions
have few interactions. Black-box predictions can also be difficult for
teachers to interpret. Question-level and concept-level information can
be complementary.

**Approach:** Q-MCKT includes an interaction encoder; multi-expert
question knowledge acquisition (MQKA); multi-expert concept knowledge
acquisition (MCKA); fine-grained question-centric contrastive learning
for less-interactive questions; and an IRT-based prediction layer.

**Contribution:** Explicitly models question- and concept-level
acquisition states, uses a mixture-of-experts approach for robust
representations, contrastive learning for sparse questions, and IRT to
make prediction outputs more interpretable. Evaluated on four public
educational datasets against multiple KT methods.

**Relevance:** Informs our representation design, concept/question
granularity, and interpretable prediction baselines.

**Remaining distinction to investigate:** Its central focus is
sequential KT and interpretable acquisition scores. Our focus is
language-derived evidence, provenance, uncertainty over candidate
misconceptions, and persistent evidence updates. Confirm overlap in the
full paper before stating exclusions categorically.

**Use in our work:** Consider it as a strong KT comparator if
reproducible; use its concept/question distinction to inform state
variables and define interpretability evaluation carefully.

## 5.4 CIKT --- EMNLP 2025

**Title:** "CIKT: A Collaborative and Iterative Knowledge Tracing
Framework with Large Language Models"\
**Authors:** Runze Li, Siyu Wu, Jun Wang, Wei Zhang\
**Venue:** Proceedings of EMNLP 2025, pp. 19321--19334\
**Reference:** Attached `CIKT.pdf`; verify the official ACL Anthology
landing page by exact title before final submission.

**Problem:** Direct use of general LLMs for KT can struggle to produce
structured, explainable learner representations and lacks task-specific
refinement. Traditional methods also face challenges with long-range
dependencies.

**Approach:** Two components cooperate: - **Analyst:** generates
dynamic, explainable textual profiles from learner history. -
**Predictor:** uses profiles to forecast future performance. The Analyst
is iteratively refined based on Predictor performance, and the Predictor
is retrained using improved profiles. The paper also studies
distillation and ablations.

**Reported evidence:** CIKT reports improvements over traditional KT and
direct LLM baselines on several datasets. In a blind forced-choice human
evaluation of 100 initial/refined profile pairs from CIKT-Qwen2.5-7B on
ASSIST2009, refined profiles were preferred in **82%** of cases;
reported inter-annotator agreement was **Cohen's κ = 0.85**.

**Limitation:** The paper says it did not incorporate question-stem text
due to backbone context-window constraints, limiting fine-grained
content-aware modeling. It focuses on binary response judgments and
reports ACC/F1 in its described evaluation rather than AUC.

**Relevance:** Closest of these four to dynamic textual learner
profiling and iterative refinement.

**Potential distinction:** A generated profile is not necessarily an
explicit probabilistic evidence store. Our study can compare free-form
summaries with atomic evidence records that retain concept links,
timestamps, confidence, and source interactions, and test whether these
improve traceability and state quality.

**Use in our work:** Treat CIKT-style profiling as a comparator; inspect
profile drift, omissions, contradictions, and traceability.

------------------------------------------------------------------------

# 6. Synthesis of the Literature

  ------------------------------------------------------------------------
  Paper                   Establishes /            Motivation for our
                          contributes              project
  ----------------------- ------------------------ -----------------------
  SQKT                    Student questions can    Test the information
                          improve programming KT   value of learner
                          prediction               language

  FoundationalASSIST      Large-scale data can     Evaluate
                          preserve actual text and language-grounded
                          responses                evidence extraction

  Q-MCKT                  Question/concept-level   Model concepts
                          acquisition and          explicitly and compare
                          interpretable prediction representation designs
                          matter                   

  CIKT                    LLM-generated profiles   Compare summaries with
                          can be iteratively       persistent, traceable
                          refined and useful       evidence
  ------------------------------------------------------------------------

The coherent story is: language can matter; data can preserve it;
representation and interpretability matter; dynamic profiles are
feasible; and the next question for this project is whether **atomic,
grounded, uncertainty-aware evidence maintained persistently** improves
inference and decisions across time.

This is motivation, not proof that no prior work integrates these
elements. A proper novelty review should also cover dialogue KT,
misconception detection, Bayesian learner models, adaptive assessment,
educational knowledge graphs, and state-space approaches.

------------------------------------------------------------------------

# 7. Proposed Architecture

## 7.1 Interaction inputs

Potential inputs include learner questions, explanations, written
answers, selected distractors, correctness, question text/options, KC
tags, hints, attempts, timestamps, and response latency. Data
availability varies. Track missingness explicitly.

## 7.2 NLP layer

Possible tasks: - Clean/normalize text while retaining original input. -
Encode semantics (TF-IDF, sentence embeddings, transformer encoders). -
Classify interaction intent (question, explanation, help request,
uncertainty statement). - Extract claims and reasoning steps. - Detect
negation, contradiction, and ambiguity. - Predict candidate
misconception categories. - Link evidence to concepts/KCs. - Estimate
extraction confidence.

LLM outputs should use constrained schemas and be validated against
human-annotated examples. Fluency is not evidence of correctness.

## 7.3 Cognitive Evidence layer

Keep four kinds of information distinct: 1. **Observed:** original text,
answer, selected option, correctness, time. 2. **Extracted:** claims,
concepts, reasoning relations, expressed uncertainty. 3. **Inferred:**
candidate misconception, mastery implication, probability. 4.
**Provenance:** source, model/version, timestamp, transformations,
reviewer status.

Do not overwrite original observations with generated interpretations.

## 7.4 Concept grounding

Map evidence to a documented skill/KC taxonomy. Start with
dataset-provided tags and manually inspect a sample. Add prerequisite or
related-concept graph edges only when sourced or validated.

## 7.5 Persistent learner state

Potential per-concept fields: - Mastery/competence estimate - Candidate
misconception probabilities - Supporting and contradicting evidence
links - Confidence/uncertainty - Recent trajectory and time since
evidence - Evidence sufficiency indicator - Provenance -
Retention/transfer estimates only when appropriate outcomes exist

The update policy is central: decide how evidence strength, recency,
contradiction, and missingness affect state. Start with transparent
rules or a probabilistic model, then compare more complex approaches.

## 7.6 Decision layer (later)

If state inference is validated, use it to identify uncertain concepts,
select diagnostic questions, or recommend practice/explanations.
Information-gain question selection is an option, but requires a
validated item bank with known concept coverage and diagnostic
properties. Intervention effectiveness requires outcome evaluation.

**Closed loop:**

``` text
Interaction
   ↓
NLP interpretation
   ↓
Cognitive Evidence
   ↓
Concept grounding + provenance
   ↓
Persistent probabilistic state update
   ↓
Prediction / uncertainty / diagnosis
   ↓
Diagnostic question or intervention (later)
   ↓
New interaction → update
```

------------------------------------------------------------------------

# 8. Experimental Blueprint

## Phase 0 --- Dataset/task audit

1.  Inspect schema, version, license, terms, and missingness.
2.  Confirm IDs, timestamps, sequence ordering, question text, answer
    text, options, correctness, KCs, hints.
3.  Distinguish detailed utterances from short/numeric answer strings.
4.  Determine whether validated misconception labels exist.
5.  Define target population, concepts, and tasks.
6.  Use student-level and/or chronological splits to prevent leakage.

**Feasibility warning:** FoundationalASSIST preserves actual responses,
but that does not guarantee it contains free-form explanations or
student questions. If the chosen data lacks them, narrow the NLP claim
or add a suitable dialogue/explanation dataset and annotation effort.

## Phase 1 --- Baselines

  -----------------------------------------------------------------------
  Baseline                Inputs                  Purpose
  ----------------------- ----------------------- -----------------------
  B0 trivial/majority     Training-set class      Sanity-check reference
                          prevalence              

  B1 classical KT         Skill/question IDs +    Conventional comparison
                          correctness             

  B2 deep KT              Sequential interaction  Strong predictive
                          features                baseline

  B3 text-only            Problem/response        Raw language value
                          embeddings              

  B4 language-aware KT    Text + skills + history Language-augmented
                                                  prediction

  B5 profile-based        CIKT-style profiles, if Summary-based
                          reproducible            comparison

  Proposed                Structured evidence +   Test core hypothesis
                          explicit update         
  -----------------------------------------------------------------------

For binary correctness, consider AUC, accuracy, F1, and calibration. For
extraction, use claim/evidence and concept-linking precision/recall/F1.
For misconception labels, report per-class metrics and class imbalance.

## Phase 2 --- Evidence extraction

Create a human-reviewed annotation subset and define a rubric for
claims, reasoning steps, concept links, explicit uncertainty, and
whether a misconception interpretation is supported, contradicted, or
ambiguous. Use multiple annotators on a subset if possible; report
agreement and adjudication. Avoid using LLM-generated labels as the sole
ground truth for evaluating that same model.

## Phase 3 --- Ablations

Compare: 1. Correctness/IDs only. 2. Raw text or embeddings. 3. Text +
metadata. 4. Structured evidence without persistence. 5. Persistent
state without uncertainty. 6. Persistent state with uncertainty. 7. Full
state with provenance and contradiction handling.

## Phase 4 --- Longitudinal evaluation

At each time step, only use evidence available up to that point.
Evaluate future-response prediction, calibration over time, state
stability/responsiveness, contradictory evidence handling, sparse
evidence behavior, and delayed retention/transfer only if suitable
outcomes exist.

## Phase 5 --- Active diagnosis (optional)

Compare random selection, fixed curriculum order, highest-uncertainty
concept, and information-gain selection. Measure interactions needed to
reach a defined diagnostic criterion and state quality under a fixed
interaction budget. Simulated learners can help debug, but do not equate
simulation with real learning evidence.

## Phase 6 --- Intervention (ambitious)

Test state-conditioned practice or explanation selection only after
state inference is validated. Claims of improved learning require a
credible controlled design and appropriate pre/post or delayed outcomes.
Offline prediction alone does not establish causal impact.

------------------------------------------------------------------------

# 9. Research Questions and Hypotheses

-   **RQ1 Language value:** Does learner language add information beyond
    correctness and metadata?
-   **RQ2 Structured evidence:** Does explicit evidence improve
    inference versus embeddings or free-form profiles?
-   **RQ3 Misconceptions:** Can the system identify candidate
    misconceptions and distinguish them from slips or prerequisite gaps?
-   **RQ4 Uncertainty:** Is confidence calibrated, and can the system
    identify insufficient evidence?
-   **RQ5 Persistence:** Do explicit updates improve temporal coherence
    or prediction versus recomputing profiles independently?
-   **RQ6 Active diagnosis:** Can information-gain questioning reduce
    diagnostic interaction cost?
-   **RQ7 Intervention:** Does state-conditioned intervention improve
    retention/transfer in controlled evaluation?

Possible hypotheses: language helps at least some tasks; structured
evidence improves traceability and may improve misconception
classification; uncertainty supports selective prediction; persistence
improves temporal coherence; active selection reduces interactions to a
target diagnostic quality. All require empirical testing.

------------------------------------------------------------------------

# 10. Metrics: Interpretation and Cautions

  -----------------------------------------------------------------------
  Metric                  What it measures        Caution
  ----------------------- ----------------------- -----------------------
  AUC-ROC                 Ranking of correct vs   Not calibration; same
                          incorrect outcomes      task/split needed

  Accuracy                Correct class decisions Threshold and imbalance
                                                  sensitive

  Precision/Recall/F1     Positive-class          Define positive class
                          detection               and averaging

  Brier score             Probabilistic forecast  Combines calibration
                          quality                 and discrimination

  ECE                     Calibration gap         Depends on binning
                                                  choice

  NLL                     Probabilistic           Penalizes confident
                          prediction quality      errors

  Concept-link F1         Evidence grounding      Needs gold concept
                          quality                 annotations

  Extraction F1           Claims/reasoning        Needs a clear
                          extraction              annotation rubric

  State agreement/error   Quality of state        Needs an external
                          estimate                target or expert rubric

  Stability/revision      State dynamics          Too stable ignores
                                                  evidence; too volatile
                                                  overreacts

  Diagnostic cost         Efficiency of diagnosis Define target
                                                  quality/stop rule

  Retention/transfer      Delayed or novel-task   Requires suitable
  score                   performance             assessment data
  -----------------------------------------------------------------------

Higher AUC alone does not prove cognition understanding,
interpretability, or learning improvement. Each claim needs its own
measure.

------------------------------------------------------------------------

# 11. Technology and Model Choices

  ------------------------------------------------------------------------------
  Component               Candidate methods              Purpose
  ----------------------- ------------------------------ -----------------------
  Preprocessing           Python, pandas,                Inspect and normalize
                          regex/tokenizers               data

  Classical NLP           TF-IDF + logistic regression   Simple, reproducible
                                                         baseline

  Semantic encoding       Sentence Transformers /        Dense text
                          pretrained encoder             representations

  KT                      BKT, DKT, transformer KT       Conventional sequence
                                                         baselines

  Evidence extraction     Schema-constrained LLM or      Structured candidate
                          classifier                     evidence

  Concept grounding       KC                             Curriculum alignment
                          mapping/classifier/retrieval   

  State update            Rules, Bayesian/state-space    Explicit temporal
                          model                          updates

  Storage                 JSONL/Parquet; database later  Evidence ledger and
                                                         state snapshots

  Visualization           Python plots/dashboard         Inspect state
                                                         trajectories

  Graph/RAG               Optional validated concept     Prerequisites and
                          resources                      grounded content
  ------------------------------------------------------------------------------

**API vs local LLM:** An API is easier to start but brings cost,
privacy, rate limits, and model-version drift. A local open-weight model
gives more control/data locality but requires compute and inference
engineering. Small encoders/classifiers may be enough for some tasks. Do
not make the project depend on one proprietary model; log model
versions, prompts, parameters, and dates. Review consent, institutional
policy, data minimization, and provider terms before transmitting
learner data externally.

------------------------------------------------------------------------

# 12. Risks and Responsible Research

1.  **Construct validity:** cognition is latent; define each state
    variable and its validation target.
2.  **Overinterpretation:** language is ambiguous; maintain candidate
    hypotheses rather than definitive diagnoses.
3.  **Label scarcity:** annotate a subset or narrow the task.
4.  **Dataset mismatch:** numeric answers are not explanations.
5.  **Leakage:** preserve chronology and student boundaries.
6.  **Confounding:** item difficulty, hints, order, and opportunity
    affect outcomes.
7.  **Language fairness:** verbosity, dialect, fluency, and willingness
    to explain are not equivalent to understanding.
8.  **Privacy:** pseudonymize, minimize, restrict access, and follow
    dataset/institutional rules.
9.  **Human oversight:** allow inspection and correction; present as
    decision support.
10. **Causal claims:** offline prediction does not prove intervention
    effectiveness.
11. **Model drift:** version all models and prompts.
12. **Scope:** active diagnosis, interventions, retention, transfer,
    graphs, and agents are extensions; prioritize core representation.

------------------------------------------------------------------------

# 13. Recommended Minimum Viable Study

**Core contribution:** Extract concept-grounded Cognitive Evidence from
available learner text and update a persistent learner state with
explicit uncertainty and provenance.

Minimum experiments: 1. Dataset audit and leakage-safe split. 2.
Correctness-only KT baseline. 3. Text-only and text-plus-metadata
baselines. 4. Evidence extraction evaluation on a human-annotated
subset. 5. Non-persistent versus persistent state comparison. 6.
Calibration and evidence-traceability evaluation. 7. Error analysis for
ambiguous, contradictory, and sparse evidence.

Defer full adaptive intervention trials, reinforcement learning,
large-scale graph construction, and retention/transfer claims unless
data and resources support them.

------------------------------------------------------------------------

# 14. Worked Example

Target concept: fraction addition.

**Interaction 1:** Problem `1/2 + 1/3`; student answer `2/5`; incorrect.

The system records the outcome and extracts a possible pattern
consistent with independently adding numerators and denominators. With
one response, the misconception probability should remain cautious.

**Interaction 2:** Student says, "I added the top and bottom
separately."

This explicit explanation supports the candidate misconception. The
evidence record links the quote, interaction, timestamp, concept, and
extractor version.

**Diagnostic step:** Ask a question that distinguishes a rule-level
misconception from an arithmetic slip. It should be selected for
diagnostic value, not merely because it is next.

**Interaction 3:** Student explains the need for common denominators and
solves a related item correctly.

The new evidence may weaken the earlier hypothesis. The system should
revise the estimate while retaining both supporting and contradicting
evidence in its history. Exact probability updates must be designed and
validated; the example does not prescribe numerical values.

------------------------------------------------------------------------

# 15. Professor-Facing Explanation

> Conventional knowledge tracing estimates student mastery mainly from
> interaction histories and correctness labels. But students' questions,
> explanations, and actual responses can reveal why they are struggling
> and which concepts they misunderstand. Recent work has demonstrated
> the value of student questions, introduced language-rich educational
> data, improved question/concept representations, and generated dynamic
> learner profiles. Our research investigates whether these signals can
> be transformed into explicit Cognitive Evidence and maintained as a
> persistent, uncertainty-aware, concept-grounded learner state with
> traceable sources. We will compare this representation against
> correctness-only KT, text-feature models, and profile-based
> approaches, and evaluate whether it improves learner-state inference
> and diagnosis. Adaptive question selection and intervention are later
> extensions whose value must be demonstrated experimentally.

**One-sentence distinction:** We are not merely asking whether language
improves next-answer prediction; we are asking whether language can
become auditable evidence in a learner state that is explicitly updated
and evaluated across time.

------------------------------------------------------------------------

# 16. References

## Four core papers

1.  Kim, Doyoun; Kim, Suin; Jo, Yohan (2025). "Knowledge Tracing in
    Programming Education Integrating Students' Questions." *Proceedings
    of ACL 2025*, 27703--27718.\
    Paper: https://aclanthology.org/2025.acl-long.1343/\
    Code: https://github.com/holi-lab/SQKT

2.  Worden et al. (2026). "FoundationalASSIST: An Educational Dataset
    for Foundational Knowledge Tracing and Pedagogical Grounding of
    LLMs."\
    Paper: https://arxiv.org/abs/2602.00070\
    Dataset:
    https://huggingface.co/datasets/ASSISTments/FoundationalASSIST

3.  Zhang, Hengyuan; Liu, Zitao; Shang, Chenming; Li, Dawei; Jiang,
    Yong. "A Question-centric Multi-experts Contrastive Learning
    Framework for Improving the Accuracy and Interpretability of Deep
    Sequential Knowledge Tracing Models."\
    Preprint: https://arxiv.org/abs/2403.07322\
    Code: https://github.com/rattlesnakey/Q-MCKT

4.  Li, Runze; Wu, Siyu; Wang, Jun; Zhang, Wei (2025). "CIKT: A
    Collaborative and Iterative Knowledge Tracing Framework with Large
    Language Models." *Proceedings of EMNLP 2025*, 19321--19334.\
    Use the attached `CIKT.pdf`; verify the official Anthology record by
    exact title before final submission.

## Foundational / adjacent references for the next review pass

5.  Corbett, A. T., & Anderson, J. R. (1994). "Knowledge tracing:
    Modeling the acquisition of procedural knowledge." *User Modeling
    and User-Adapted Interaction*, 4, 253--278. Foundational BKT
    reference.
6.  Piech et al. (2015). "Deep Knowledge Tracing." *NeurIPS 2015.*
    Foundational deep sequential KT reference.
7.  Scarlatos, Baker, and Lan. "Exploring Knowledge Tracing in
    Tutor-Student Dialogues using LLMs." Relevant dialogue-based learner
    evidence; verify exact version and publication metadata before
    citation.
8.  Pardos & Heffernan (2010/2011), work on individualized Bayesian
    Knowledge Tracing. Verify exact title/year/version before final
    bibliography.

**Bibliography note:** Cite the exact version read (conference paper,
preprint, or dataset card), preserve author order, and verify DOI/URLs
before submission. The four core papers are complementary; their scores
and metrics are not directly interchangeable.

------------------------------------------------------------------------

# 17. Final Research Position

The strongest defensible framing is: **an evidence-centric,
language-grounded, persistent learner-state investigation---not a new
LLM tutor and not a claim that prior KT ignores all language.** The
contribution depends on demonstrating that structured evidence, explicit
updates, uncertainty, and provenance provide value over strong
conventional and language-aware baselines.

Prioritize validity and reproducibility in this order:

`language quality → evidence extraction → concept grounding → state update → leakage-safe evaluation → diagnosis → intervention`

If extraction or grounding is unreliable, the resulting state may sound
convincing while being wrong. The first priority is therefore to
validate the evidence and state---not to build the largest architecture.
