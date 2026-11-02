# Extended assessment verification

The runner submits every answer through the real browser controls. React, the API, SQLite, retrieval, validation and state integration execute normally. Each run uses a disposable database and a separate synthetic browser profile. The existing learner database is not used for these assessments.

## Completed verification

| Run | Saved answers | Saved evidence | State snapshots | JavaScript errors |
| --- | ---: | ---: | ---: | ---: |
| Simulated inference, real browser/API/storage | 207 | 207 | 208 | 0 |
| Historical live-model run, completed before the live-budget instruction | 12 | 12 | 13 | 0 |

The simulated run passed all six diagnostic categories, repeated-answer damping, low-confidence/ambiguity holds, reload/draft checks after 1, 3, 10, 30, 100, 160 and 200 answers, report export and continuation beyond completion. The final database contains 207 answers, 208 questions (including the next unanswered question), 208 snapshots and 207 retrievable evidence chunks. There is no fixed application question-count cap.

The 66 affected backend tests passed. No additional live model run was started after the user's instruction to limit live testing to four or five questions; the 207-answer test used fixture inference throughout.

```powershell
python experiments/verify_extended_assessment.py --answers 205
python experiments/verify_extended_assessment.py --live --answers 3
```

Build `apps/web` before running. Each command submits two additional answers to check submit-and-finish, report export, reload, resume, and continuation beyond the requested run length. Live mode defaults to three normal submissions plus two finish/resume submissions (five questions total), and rejects larger runs before initializing the provider. A separate guard allows at most five question-generation attempts. Longer runs must use fixture mode, which makes no cloud model calls. This test budget does not limit the application itself.

`fixture/results.json` and `live/results.json` contain the final status, every saved turn/evidence/snapshot identifier, category, evidence weight, retry count, reload checkpoints and database totals. Screenshots show questions, completed reports and state inspectors. The recorded 12-answer live result predates the user's request to cap future live checks at five; it is retained as historical verification evidence. No additional live run was started after that request.

## What is checked

- Submit & next question is clicked for every normal answer; a genuinely different question is displayed afterward and uses the newly committed state snapshot.
- Answer count, immutable earlier answers and unique evidence/snapshot IDs agree with persisted SQLite rows after every submission.
- Raw learner text, quoted evidence offsets, token spans and sentence spans match the original response.
- Met/missing criteria form a valid partition of the current saved question rubric.
- Alpha/beta updates, duplicate damping, confidence/ambiguity holds, estimates and uncertainty match independently calculated values. Other concepts are unchanged by an unrelated concept's update.
- Evidence and persistence trace stages link the actual evidence and saved snapshot. Separate evidence chunks remain retrievable.
- Reloads preserve the active question, state, submitted-answer count and unsent draft. No reload is counted as new evidence.
- Text transformations, Cognitive Evidence and State changes display the saved record. A browser-downloaded complete-record export equals the database record.
- Finishing after three answers, resuming, completing the long assessment, exporting the full report, reloading the completed session and continuing do not introduce an automatic question-count stop or duplicate answers.
- The full browser run produces no JavaScript runtime errors.

## Test data and interpretation

Fixture inference deliberately covers understanding, partial reasoning, misconception, prerequisite gap, slip, ambiguity, low-confidence holds and an identical repeated answer. It uses varied source-grounded text samples so a fixed four-question fixture cannot itself impose a fake generation limit.

Live inference uses the configured Azure model and synthetic addition answers: uncertainty, an incorrect multiplication rule, then answers based on the saved private reference answer with an explicit correction of that rule. This tests the application and persistence, not independent learner performance or model diagnostic accuracy. Private answers are accessed by the isolated test runner; the learner UI still hides them until submission.

A finite run cannot prove operation for infinitely many questions. Code inspection found no fixed answer-count cap. The bounded 30-question model prompt history and 200-question novelty window are context limits, not completion limits. Longer sessions still consume storage, browser memory and larger history payloads; provider/network outages remain external failure conditions.

## Defects found and corrected

The initial live run saved seven answers but paused on next-question generation after both candidates changed the planner's assigned target. The question schema now constrains the exact concept and question kind as well as source citations.

A further live run exposed near-duplicate generation: generic retry feedback could produce the same question structure again. Retry feedback now names the rejected and conflicting prompts and provides a different task structure compatible with the same planned question kind. Novelty, source, rubric and persistence checks remain enabled. The generation contract is versioned `saved-state-question-v2`.

The long-run harness also encountered Chromium's debug response-body cache limit around 146 answers. The application had saved the answer; the test recorder could no longer retrieve the large response body. The long fixture run now checks a fresh persisted session read and an exact match to the question visibly rendered by React. It does not impose a debug-cache limit on the application or skip browser submissions.

The runner disposes Playwright API response buffers after reading them, so repeated full-history checks do not accumulate response bodies in the test driver. Only read-only verification GETs retry transport resets; answer submissions are never automatically replayed by the harness.
