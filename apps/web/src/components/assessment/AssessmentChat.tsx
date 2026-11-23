import {
  Activity,
  ArrowRight,
  Check,
  FileText,
  LoaderCircle,
  Search,
  ShieldCheck,
} from "lucide-react";
import { AssessmentSession } from "../../types/assessment";
import { displayDate, displayProse, readable } from "./utils";
import { MathText } from "./MathText";
import { SourceRecovery, SourceMode } from "./SourceRecovery";

export function AssessmentChat({
  session,
  draft,
  busy,
  onDraft,
  onSubmit,
  onInspect,
  onNext,
  onReport,
  onResume,
  onSources,
}: {
  session: AssessmentSession;
  draft: string;
  busy: boolean;
  onDraft: (value: string) => void;
  onSubmit: (finishAfterAnswer?: boolean) => void;
  onInspect: (
    index: number | null,
    tab: "state" | "evidence" | "decision",
  ) => void;
  onNext: () => void;
  onReport: () => void;
  onResume: () => void;
  onSources: (mode: SourceMode, notes?: string) => void;
}) {
  const question = session.active_question;
  const generationCode =
    session.conversation.generation_failure_code ||
    session.conversation.last_error;
  const sourceBlocked = generationCode === "insufficient_source_material";
  const material = (
    question || session.turns[session.turns.length - 1]?.question
  )?.generation_trace.context.retrieval.materials;
  return (
    <div className="assessment-chat">
      <div className="session-intro">
        <span className="tutor-mark">
          <Activity size={20} />
        </span>
        <div>
          <small className="source-mode">
            {material?.some((source) => source.kind === "study_notes")
              ? "Reference: your study notes"
              : material?.length
                ? "Reference: model-authored material · unverified"
                : "Preparing source-grounded questions"}
          </small>
        </div>
      </div>
      {session.turns.length > 0 && (
        <details className="conversation-history">
          <summary>
            Previous questions & feedback{" "}
            <span>
              {session.turns.length} saved{" "}
              {session.turns.length === 1 ? "explanation" : "explanations"}
            </span>
          </summary>
          <div
            className="conversation-stream"
            aria-label="Assessment conversation"
          >
            {session.turns.map((turn, index) => (
              <article className="assessment-exchange" key={turn.turn_id}>
                <div className="past-question">
                  <div className="message-heading">
                    <span>Learning guide · {readable(turn.question.kind)}</span>
                    <time dateTime={turn.timestamp}>
                      {displayDate(turn.timestamp)}
                    </time>
                  </div>
                  <h3>
                    <MathText text={turn.question.prompt} />
                  </h3>
                  <details>
                    <summary>What was expected?</summary>
                    <p>
                      <MathText text={turn.question.expected_focus} />
                    </p>
                  </details>
                </div>
                <div className="learner-answer">
                  <span className="speaker-label">Your explanation</span>
                  <p>{turn.response_text}</p>
                </div>
                <div className="answer-feedback">
                  <span className="tutor-mark">
                    <Activity size={17} />
                  </span>
                  <div>
                    <p>
                      <MathText text={displayProse(turn.evidence.feedback)} />
                    </p>
                    {turn.evidence.coaching && (
                      <div className="coaching-text">
                        <strong>Revisit this idea</strong>
                        <p>
                          <MathText
                            text={displayProse(turn.evidence.coaching)}
                          />
                        </p>
                        <small>
                          Based on retrieved material · checked through a later
                          answer
                        </small>
                      </div>
                    )}
                    <div className="feedback-actions">
                      <span>
                        <Check size={13} />
                        {turn.state_changes.weight
                          ? "Evidence saved"
                          : "Saved · estimate held"}
                      </span>
                      <button onClick={() => onInspect(index, "evidence")}>
                        <Search size={14} />
                        Inspect this answer
                      </button>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </details>
      )}
      {session.conversation.status === "completed" ? (
        <section className="completed-assessment">
          <ShieldCheck size={25} />
          <h2>This conversation is saved.</h2>
          <p>
            Your report connects each explanation with the evidence and state
            changes it produced. You can continue checking your understanding
            whenever you’re ready.
          </p>
          <div className="button-row">
            <button className="primary-button" onClick={onReport}>
              <FileText size={16} />
              View report
            </button>
            <button
              className="secondary-button"
              onClick={onResume}
              disabled={busy}
            >
              Continue assessment
              <ArrowRight size={15} />
            </button>
          </div>
        </section>
      ) : question ? (
        <section
          className="active-assessment"
          data-guide="question"
          aria-labelledby="active-question-heading"
        >
          <div className="active-question-meta">
            <span>
              Question {session.turns.length + 1} · {readable(question.kind)}
            </span>
            <button
              className="text-button"
              onClick={() => onInspect(null, "decision")}
            >
              <ShieldCheck size={14} />
              Why this question?
            </button>
          </div>
          {question.learning_support && (
            <div className="learning-support">
              <strong>A useful idea to revisit</strong>
              <p>
                <MathText text={displayProse(question.learning_support)} />
              </p>
              <small>
                Retrieved source citations are available in “Why this question?”
              </small>
            </div>
          )}
          <h2 id="active-question-heading">
            <MathText text={question.prompt} />
          </h2>
          <details className="expected-focus" data-guide="expectation">
            <summary>What is expected from my answer?</summary>
            <p>
              <MathText text={question.expected_focus} />
            </p>
          </details>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              onSubmit();
            }}
          >
            <label htmlFor="assessment-response">Explain your thinking</label>
            <textarea
              id="assessment-response"
              disabled={busy}
              value={draft}
              onChange={(event) => onDraft(event.target.value)}
              onKeyDown={(event) => {
                if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                  event.preventDefault();
                  onSubmit();
                }
              }}
              rows={3}
              maxLength={8000}
              placeholder="Work through the question in your own words. Include any step you are unsure about."
              required
            />
            <div className="composer-footer">
              <span>
                <ShieldCheck size={14} />
                Draft saved locally · Ctrl+Enter to submit
              </span>
              <div className="submission-buttons">
                <button
                  type="button"
                  className="secondary-button"
                  disabled={busy || !draft.trim()}
                  onClick={() => onSubmit(true)}
                >
                  <FileText size={15} />
                  Submit & finish
                </button>
                <button
                  className="primary-button"
                  disabled={busy || !draft.trim()}
                >
                  {busy ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    <ArrowRight size={16} />
                  )}
                  {busy
                    ? "Saving evidence & preparing your question…"
                    : "Submit & next question"}
                </button>
              </div>
            </div>
          </form>
        </section>
      ) : (
        <section
          className="question-retry"
          aria-labelledby="generation-status-heading"
          aria-busy={busy}
        >
          <Activity size={26} />
          <h2 id="generation-status-heading">
            {busy
              ? "Preparing your next question…"
              : sourceBlocked
                ? "Add material to continue."
                : "Question generation is paused."}
          </h2>
          <p role={busy ? "status" : "alert"}>
            {busy
              ? "Retrieving saved topic memory and checking a source-grounded question. Your estimates stay unchanged during generation."
              : sourceBlocked
                ? "The saved notes contain too little subject material to ground a question. A class or grade label alone is not enough. Update the study material below."
                : generationCode === "unsupported_citations"
                  ? "The generated question cited material outside the allowed reference sources and was rejected. Retry, or update the study material below."
                  : generationCode === "missing_recheck_criteria"
                    ? "The generated question did not recheck the required skills and was rejected. Retry to generate a valid question."
                    : session.conversation.last_error === "state_changed_retry"
                      ? "Your topic memory changed in another conversation. Generate a question from the updated state."
                      : session.conversation.last_error ===
                          "generation_unavailable"
                        ? "The question model could not complete the request. Check the model connection and retry."
                        : "The next question could not pass generation checks. Retry generation, or update the study material below."}
          </p>
          <p className="generation-preserved">
            {session.turns.length} explanations saved in this conversation.
            Retrying does not resubmit answers or update learner estimates.
          </p>
          {!sourceBlocked && (
            <button className="primary-button" onClick={onNext} disabled={busy}>
              {busy ? (
                <LoaderCircle size={16} className="spin" />
              ) : (
                <ArrowRight size={16} />
              )}
              {busy ? "Generating question…" : "Generate next question"}
            </button>
          )}
          <SourceRecovery busy={busy} onUpdate={onSources} />
        </section>
      )}
      <p className="assessment-footnote">
        A question or coaching message never raises your estimate. Only new
        evidence in your own explanation can change it.
      </p>
    </div>
  );
}
