# Question recovery verification

Verified locally on 1 November 2026. These are implementation checks using synthetic test data, not research evidence of learning gains or diagnosis accuracy.

## Cause of the reported failure

The saved conversation had `Class 6th` as its study notes. Retrieval excluded the earlier model reference because notes took precedence. The model repeatedly cited learner-evidence chunks as factual references, causing `unsupported_citations` on both generation attempts. The frontend received a saved conversation without an active question and displayed only a generic retry message. The parent creation trace incorrectly showed completed while its generation child had failed.

The learner snapshot was unchanged across those failed attempts. The two observations in the inspector came from the earlier conversation on the same topic. The redesigned inspector explicitly labels that inherited memory.

## Changes checked

- Minimum 8-word material check before new notes are saved, with an actionable recovery path for older short-note records. This is a size check, not a claim that notes are accurate or relevant.
- Model question schemas restrict citations to the exact retrieved reference IDs. Service validation independently checks citations, target, question kind, rubric skills and novelty.
- Separate safe failure codes and retry feedback; failed creation is represented as failed in the execution journal.
- Conversation-specific source IDs are pinned before generation. Source replacement in one conversation cannot change another question's assessment context. Legacy questions retain their recorded provenance. New conversations exclude grade-only legacy notes.
- Generation retries, source recovery and observability reads preserve learner state and submitted responses. Only accepted learner evidence updates the state.
- A reference-backed question was generated in the user's original blocked conversation. Its state and answer list were compared before/after and matched exactly. A SQLite backup was created in the local temporary directory before recovery; `existing-conversation.json` records the invariant checks.
- A new `DESIGN.md` contract drives dark navigation, high-contrast prose, teal primary actions, separated working/diagnostic panels, and collapsible previous feedback.

## Executed checks

| Check | Result |
| --- | --- |
| `python -m pytest tests -q` | 99 tests passed in 111.42 seconds; 32 existing third-party deprecation warnings |
| `npm test` in `apps/web` | 5 tests passed |
| `npm run build` in `apps/web` | TypeScript and Vite production build passed |
| `python experiments/verify_question_recovery.py` | 11 fixture browser checks passed |
| `python experiments/verify_question_recovery.py --live` | 11 real configured-model browser checks passed |
| `python experiments/verify_nlp_observability.py --live --output docs/verification/question-recovery/observability` | 16 real configured-model browser checks passed |
| Running app frontend, API status and frontend API proxy | All returned HTTP 200 after restart |
| Original conversation in desktop/mobile browser | Active question visible, two earlier explanations explicitly labeled, no horizontal mobile overflow, no browser JavaScript errors |
| Independent source-isolation review | Five temporary-database cases passed, including legacy and current conversations |

The recovery browser checks cover invalid new notes without domain writes, a distinct next question, generation outages after a committed answer, repeated retries, reload, replacement notes, model-reference recovery, observability reads, finish/report/resume, and mobile typography/overflow. The observability checks cover actual trace stages and inputs/outputs, source/state linkage, a visible next question tied to the committed snapshot, historical records, outage recovery, research-metric labeling, cross-profile isolation, and desktop/mobile rendering.

Manifests: [fixture checks](fixture/checks.json), [live recovery checks](live/checks.json), [live observability execution](observability/live-execution.json).

## Screenshots

- [Recovered original workspace — desktop](recovered-workspace-desktop.png)
- [Recovered original workspace — mobile](recovered-workspace-mobile.png)
- [Source recovery controls](live/source-recovery-desktop.png)
- [Observability](live/observability-desktop.png)

## Limits

The full Python suite emits existing NumPy conversion deprecation warnings from the third-party pyBKT baseline. Vite warns about the existing large JavaScript bundle, primarily including mathematical rendering. These checks establish the tested runtime behavior; model content remains unverified and cloud availability can still fail. Such failures preserve evidence and remain inspectable and recoverable.
