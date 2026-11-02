import { FormEvent, lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ArrowRight,
  BookOpen,
  ChevronRight,
  Database,
  FileText,
  HelpCircle,
  History,
  LoaderCircle,
  MessageSquare,
  Plus,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import * as api from "../api/assessment";
import {
  AssessmentSession,
  Conversation,
  ConversationReportData,
} from "../types/assessment";
import { AssessmentChat } from "./assessment/AssessmentChat";
import { GuidedTour } from "./assessment/GuidedTour";
import { RenderBoundary } from "./RenderBoundary";
import { InspectorTab, LearnerInspector } from "./assessment/LearnerInspector";
import { displayDate, displayProse, exportJSON } from "./assessment/utils";
import { MathText } from "./assessment/MathText";
import { readDrafts, saveDraft, SavedDraft } from "./assessment/drafts";
import { SavedDrafts } from "./assessment/SavedDrafts";
import { SourceMode } from "./assessment/SourceRecovery";

type View = "practice" | "history" | "reports" | "observability";

const ConversationReport = lazy(() => import("./assessment/ConversationReport").then(m => ({ default: m.ConversationReport })));
const NlpObservability = lazy(() => import("./assessment/NlpObservability").then(m => ({ default: m.NlpObservability })));

function localIdentity() {
  let id = localStorage.getItem("learner-state:identity:v2");
  if (!id) {
    id = `learner_${crypto.randomUUID().replace(/-/g, "")}`;
    localStorage.setItem("learner-state:identity:v2", id);
  }
  return id;
}

export function AssessmentWorkspace() {
  const [learner] = useState(localIdentity);
  const [view, setView] = useState<View>("practice");
  const [activeId, setActiveId] = useState<string | null>(() =>
    localStorage.getItem("learner-state:conversation:v2"),
  );
  const [items, setItems] = useState<Conversation[]>([]);
  const [session, setSession] = useState<AssessmentSession | null>(null);
  const [report, setReport] = useState<ConversationReportData | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [topic, setTopic] = useState("");
  const [notes, setNotes] = useState("");
  const [draft, setDraft] = useState("");
  const [savedDrafts, setSavedDrafts] = useState<SavedDraft[]>([]);
  const [tab, setTab] = useState<InspectorTab>("state");
  const [selected, setSelected] = useState<number | null>(null);
  const [model, setModel] = useState<{
    model_configured: boolean;
    model: string | null;
  } | null>(null);
  const [guide, setGuide] = useState(
    () => !localStorage.getItem("learner-state:guide:v2"),
  );
  const [reload, setReload] = useState(0);
  const [reportLoading, setReportLoading] = useState(false);
  const [practicePanel, setPracticePanel] = useState<"question" | "memory">(
    "question",
  );
  const revealGuideTarget = useCallback((target: string) => {
    if (target === "inspector") setPracticePanel("memory");
    else if (target === "question" || target === "expectation")
      setPracticePanel("question");
  }, []);
  const requestRef = useRef<{ key: string; id: string } | null>(null);
  const scrollToQuestion = useRef(false);
  const currentId = useRef(activeId);
  currentId.current = activeId;

  useEffect(() => {
    const controller = new AbortController();
    api
      .conversations(learner, controller.signal)
      .then(setItems)
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      });
    api
      .getStatus()
      .then(setModel)
      .catch(() => setModel(null));
    return () => controller.abort();
  }, [learner, reload]);

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setSelected(null);
    setDraft("");
    setSavedDrafts(activeId ? readDrafts(activeId) : []);
    setReport(null);
    requestRef.current = null;
    if (!activeId) {
      setSession(null);
      setLoading(false);
      localStorage.removeItem("learner-state:conversation:v2");
      return;
    }
    localStorage.setItem("learner-state:conversation:v2", activeId);
    setLoading(true);
    setSession(null);
    api
      .loadConversation(activeId, learner, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setSession(data);
          scrollToQuestion.current = data.conversation.status !== "completed";
          const drafts = readDrafts(activeId);
          setSavedDrafts(drafts);
          const saved = drafts.find(
            (item) =>
              item.questionId ===
              (data.active_question?.question_id ||
                data.conversation.current_question_id),
          );
          if (saved) setDraft(saved.text);
          else if (data.pending_submission) setDraft(data.pending_submission.response_text);
        }
      })
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [activeId, learner, reload]);

  useEffect(() => {
    if (view !== "reports" || !activeId) {
      setReportLoading(false);
      return;
    }
    const controller = new AbortController();
    setReportLoading(true);
    setReport(null);
    api
      .loadReport(activeId, learner, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setReport(data);
      })
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setReportLoading(false);
      });
    return () => controller.abort();
  }, [view, activeId, learner, session?.conversation.version]);

  useEffect(() => {
    if (
      !scrollToQuestion.current ||
      busy ||
      loading ||
      view !== "practice" ||
      !session
    )
      return;
    scrollToQuestion.current = false;
    requestAnimationFrame(() => {
      const target = document.querySelector(
        ".active-assessment, .question-retry, .completed-assessment",
      );
      target?.scrollIntoView({
        block: "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
      });
    });
  }, [
    busy,
    loading,
    view,
    session?.active_question?.question_id,
    session?.conversation.status,
  ]);

  function updateSession(data: AssessmentSession) {
    setSession(data);
    setPracticePanel("question");
    setSelected(null);
    setItems((old) => [
      data.conversation,
      ...old.filter(
        (c) => c.conversation_id !== data.conversation.conversation_id,
      ),
    ]);
  }

  function requestId(payload: unknown) {
    const key = JSON.stringify(payload);
    if (requestRef.current?.key !== key)
      requestRef.current = { key, id: crypto.randomUUID() };
    return requestRef.current.id;
  }

  function newAssessment() {
    if (busy) return;
    setActiveId(null);
    setSession(null);
    setView("practice");
    setError("");
    setTopic("");
    setNotes("");
    setDraft("");
  }

  async function start(event: FormEvent) {
    event.preventDefault();
    if (busy || !topic.trim()) return;
    const payload = { learner_id: learner, topic: topic.trim(), notes };
    setBusy(true);
    setError("");
    setGuide(false);
    try {
      const data = await api.startConversation({
        ...payload,
        request_id: requestId(payload),
      });
      setActiveId(data.conversation.conversation_id);
      updateSession(data);
      setView("practice");
      setTopic("");
      setNotes("");
      requestRef.current = null;
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not start the assessment. Your input is still here.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function submit(finishAfterAnswer = false) {
    if (busy || !session?.active_question || !draft.trim()) return;
    const id = session.conversation.conversation_id;
    const payload = {
      learner_id: learner,
      question_id: session.active_question.question_id,
      response_text: draft,
      finish_after_answer: finishAfterAnswer,
    };
    setBusy(true);
    setError("");
    try {
      const data = await api.submitAnswer(id, {
        ...payload,
        request_id: requestId(payload),
      });
      if (currentId.current === id) {
        updateSession(data);
        setDraft("");
        setSavedDrafts(
          saveDraft(id, {
            questionId: payload.question_id,
            questionPrompt: session.active_question.prompt,
            text: "",
          }),
        );
        requestRef.current = null;
        scrollToQuestion.current = !finishAfterAnswer;
        if (finishAfterAnswer) setView("reports");
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not save this explanation. Your draft is still here.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function action(kind: "next" | "finish" | "resume") {
    if (!activeId || busy) return;
    const id = activeId;
    if (kind === "finish" && session?.active_question && draft.trim()) {
      setSavedDrafts(
        saveDraft(id, {
          questionId: session.active_question.question_id,
          questionPrompt: session.active_question.prompt,
          text: draft,
        }),
      );
    }
    setBusy(true);
    setError("");
    try {
      const data = await (
        kind === "finish"
          ? api.finishConversation
          : kind === "resume"
            ? api.resumeConversation
            : api.nextQuestion
      )(id, learner);
      if (currentId.current === id) {
        updateSession(data);
        if (kind === "resume") {
          const drafts = readDrafts(id);
          setSavedDrafts(drafts);
          setDraft(
            drafts.find(
              (item) => item.questionId === data.active_question?.question_id,
            )?.text || "",
          );
        }
        scrollToQuestion.current = kind !== "finish";
        setView(kind === "finish" ? "reports" : "practice");
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "The request could not be completed. Please retry.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function changeSources(mode: SourceMode, notes = "") {
    if (!activeId || busy) return;
    const id = activeId;
    setBusy(true);
    setError("");
    try {
      const data = await api.updateSources(id, learner, mode, notes);
      if (currentId.current === id) {
        updateSession(data);
        scrollToQuestion.current = true;
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Study material could not be updated. Your saved evidence is unchanged.",
      );
    } finally {
      setBusy(false);
    }
  }

  function inspect(index: number | null, inspectorTab: InspectorTab) {
    setSelected(index);
    setTab(inspectorTab);
    setPracticePanel("memory");
    if (window.innerWidth <= 1180)
      requestAnimationFrame(() =>
        document.querySelector('[data-guide="inspector"]')?.scrollIntoView({
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)")
            .matches
            ? "instant"
            : "smooth",
          block: "start",
        }),
      );
  }

  return (
    <div
      className={`workspace workspace-${view}${view === "practice" ? (session ? " workspace-session" : " workspace-setup") : ""}`}
    >
      <a className="skip-link" href="#workspace-content">Skip to workspace</a>
      <aside className="sidebar">
        <a
          href="#"
          className="brand"
          onClick={(event) => {
            event.preventDefault();
            setView("practice");
          }}
        >
          <span className="brand-icon">
            <Activity size={21} />
          </span>
          <span>
            LearnerState<span className="brand-dot">.</span>
          </span>
        </a>
        <button
          className="new-assessment-button"
          onClick={newAssessment}
          disabled={busy}
        >
          <Plus size={17} />
          New assessment
        </button>
        <nav className="main-nav" aria-label="Main navigation">
          {(
            [
              { id: "practice", label: "Learning workspace", icon: BookOpen },
              { id: "history", label: "Conversations", icon: History },
              { id: "reports", label: "Reports", icon: FileText },
              {
                id: "observability",
                label: "NLP Observability",
                icon: Activity,
              },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              disabled={busy && item.id !== "observability"}
              data-guide={item.id === "reports" ? "reports" : undefined}
              className={view === item.id ? "nav-item active" : "nav-item"}
              aria-current={view === item.id ? "page" : undefined}
              onClick={() => {
                setView(item.id);
                setGuide(false);
              }}
            >
              <item.icon size={17} />
              <span>{item.label}</span>
              {view === item.id && <ChevronRight size={14} />}
            </button>
          ))}
        </nav>
        <div className="conversation-nav" data-guide="conversations">
          <h2>Saved conversations</h2>
          {items.length ? (
            <div className="conversation-list">
              {items.map((item) => (
                <button
                  disabled={busy}
                  key={item.conversation_id}
                  className={
                    activeId === item.conversation_id
                      ? "conversation-item selected"
                      : "conversation-item"
                  }
                  onClick={() => {
                    setActiveId(item.conversation_id);
                    setView("practice");
                    setPracticePanel("question");
                    setGuide(false);
                  }}
                >
                  <MessageSquare size={15} />
                  <span>
                    {item.title}
                    <small>
                      {item.answered_count}{" "}
                      {item.answered_count === 1
                        ? "explanation"
                        : "explanations"}{" "}
                      ·{" "}
                      {item.status === "completed"
                        ? "Completed"
                        : "In progress"}
                    </small>
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <p className="sidebar-empty">
              Your conversations appear here after you start an assessment.
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <Database size={17} />
          <p>
            Your local learning profile
            <small>Saved in your local SQLite database</small>
          </p>
        </div>
      </aside>
      <div className="main-shell" id="workspace-content" tabIndex={-1}>
        <header className="topbar">
          <div className="breadcrumb">
            Your learning <ChevronRight size={14} />
            <strong>
              {view === "observability"
                ? "NLP Observability"
                : view === "reports"
                  ? "Reports"
                  : view === "history"
                    ? "Conversations"
                    : session
                      ? session.conversation.title
                      : "New assessment"}
            </strong>
          </div>
          <div className="topbar-actions">
            <span className="connection-status">
              <span
                className={`status-dot ${model?.model_configured ? "" : "offline"}`}
              />
              {model?.model_configured
                ? "Model configured · local memory"
                : model
                  ? "Model setup needed"
                  : "Connecting to local API"}
            </span>
            <button
              className="help-button"
              onClick={() => {
                setView("practice");
                setGuide(true);
              }}
              aria-label="Show workspace guide"
            >
              <HelpCircle size={18} />
              <span>Guide</span>
            </button>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            <div>
              <strong>We couldn’t complete that request.</strong>
              <p>{error}</p>
            </div>
            <button
              className="secondary-button"
              disabled={busy}
              onClick={() => setReload((value) => value + 1)}
            >
              <RefreshCw size={15} />
              Reload saved state
            </button>
          </div>
        )}
        <RenderBoundary key={view}>
        <Suspense fallback={<div className="loading-state" role="status"><LoaderCircle className="spin" size={24} /><p>Opening {view === "observability" ? "execution records" : "your report"}…</p></div>}>
        {view === "observability" ? (
          <NlpObservability
            learner={learner}
            conversations={items}
            onPractice={() => setView("practice")}
          />
        ) : loading ? (
          <div className="loading-state" role="status">
            <LoaderCircle className="spin" size={26} />
            <p>Retrieving your saved conversation…</p>
          </div>
        ) : view === "practice" ? (
          session ? (
            <>
              <div className="page-heading">
                <div>
                  <h1>{session.conversation.title}</h1>
                  <p>
                    {session.conversation.answered_count}{" "}
                    {session.conversation.answered_count === 1
                      ? "explanation"
                      : "explanations"}{" "}
                    saved · Topic memory persists
                  </p>
                </div>
                <div className="workspace-actions">
                  {session.conversation.status !== "completed" && (
                    <button
                      disabled={busy}
                      className="secondary-button"
                      title="Completes using submitted evidence. Unsent drafts stay on this device."
                      onClick={() => action("finish")}
                    >
                      <FileText size={15} />
                      Finish & view report
                    </button>
                  )}
                  <button
                    className="text-button"
                    onClick={() =>
                      exportJSON(
                        session,
                        `${session.conversation.conversation_id}.json`,
                      )
                    }
                  >
                    <ArrowDownToLine size={15} />
                    Export record
                  </button>
                </div>
              </div>
              <nav className="practice-panels" aria-label="Assessment panels">
                <button
                  aria-pressed={practicePanel === "question"}
                  onClick={() => setPracticePanel("question")}
                >
                  Current question
                </button>
                <button
                  aria-pressed={practicePanel === "memory"}
                  onClick={() => setPracticePanel("memory")}
                >
                  Saved topic memory
                </button>
              </nav>
              <div className={`assessment-layout show-${practicePanel}`}>
                <main className="learning-column">
                  {session.pending_submission && <p className="recovered-submission" role="status">Your previous submission is saved. It has not changed learner state; review the explanation below and retry when ready.</p>}
                  <SavedDrafts
                    drafts={savedDrafts.filter(
                      (item) =>
                        item.questionId !==
                        session.conversation.current_question_id,
                    )}
                  />
                  <AssessmentChat
                    session={session}
                    draft={draft}
                    busy={busy}
                    onDraft={(value) => {
                      setDraft(value);
                      if (activeId && session.active_question)
                        setSavedDrafts(
                          saveDraft(activeId, {
                            questionId: session.active_question.question_id,
                            questionPrompt: session.active_question.prompt,
                            text: value,
                          }),
                        );
                    }}
                    onSubmit={submit}
                    onInspect={inspect}
                    onNext={() => action("next")}
                    onReport={() => setView("reports")}
                    onResume={() => action("resume")}
                    onSources={changeSources}
                  />
                </main>
                <LearnerInspector
                  session={session}
                  selected={selected}
                  tab={tab}
                  onTab={setTab}
                />
              </div>
            </>
          ) : (
            <main className="assessment-setup">
              <div className="setup-intro">
                <span className="setup-mark">
                  <Activity size={29} />
                </span>
                <h1>Explore your understanding</h1>
                <p>
                  Choose a topic you’ve studied. Explain your thinking and let
                  the next question follow your reasoning.
                </p>
              </div>
              <form className="setup-form" onSubmit={start}>
                <div className="setup-field" data-guide="topic">
                  <label htmlFor="assessment-topic">Topic to assess</label>
                  <input
                    id="assessment-topic"
                    value={topic}
                    onChange={(event) => setTopic(event.target.value)}
                    disabled={busy}
                    maxLength={180}
                    minLength={2}
                    required
                    placeholder="Name the topic or concept you have studied"
                  />
                  <p>Choose a focused concept or skill.</p>
                </div>
                <div className="setup-field" data-guide="notes">
                  <label htmlFor="assessment-notes">
                    Your study notes <span>Optional</span>
                  </label>
                  <textarea
                    id="assessment-notes"
                    value={notes}
                    onChange={(event) => setNotes(event.target.value)}
                    disabled={busy}
                    rows={3}
                    maxLength={60000}
                    placeholder="Paste the material you learned from. Questions and feedback will retrieve relevant passages from it."
                  />
                  <p>
                    Add at least 8 words of subject material. Without notes, the
                    model uses an unverified reference.
                  </p>
                </div>
                <button
                  className="primary-button start-button"
                  disabled={busy || topic.trim().length < 2}
                >
                  {busy ? (
                    <LoaderCircle size={18} className="spin" />
                  ) : (
                    <ArrowRight size={18} />
                  )}
                  {busy
                    ? "Building topic memory & first question…"
                    : "Start my assessment"}
                </button>
              </form>
              <div className="setup-assurance">
                <ShieldCheck size={17} />
                <p>
                  Only your submitted explanations update learner estimates.
                </p>
              </div>
            </main>
          )
        ) : view === "history" ? (
          <main className="history-page">
            <div className="page-heading">
              <div>
                <h1>Conversations</h1>
                <p>
                  Revisit your questions, explanations, and evidence.
                </p>
              </div>
              <button
                className="secondary-button"
                disabled={busy}
                onClick={newAssessment}
              >
                <Plus size={16} />
                New assessment
              </button>
            </div>
            {session && <button className="text-button collection-back" onClick={() => setActiveId(null)}>All conversations</button>}
            {session ? (
              <>
                <div className="conversation-detail-heading">
                  <h2>{session.conversation.title}</h2>
                  <button
                    className="text-button"
                    onClick={() => setView("practice")}
                  >
                    Open conversation
                    <ArrowRight size={15} />
                  </button>
                </div>
                {session.turns.length ? (
                  session.turns.map((turn, index) => (
                    <article className="history-entry" key={turn.turn_id}>
                      <div>
                        <span>
                          Question {index + 1} ·{" "}
                          {turn.question.kind.replace(/_/g, " ")}
                        </span>
                        <time>{displayDate(turn.timestamp)}</time>
                      </div>
                      <h3>
                        <MathText text={turn.question.prompt} />
                      </h3>
                      <blockquote>{turn.response_text}</blockquote>
                      <p>
                        <MathText text={displayProse(turn.evidence.feedback)} />
                      </p>
                      <div className="history-entry-actions">
                        <span>
                          {turn.state_changes.weight
                            ? "Evidence incorporated"
                            : "Estimate held · insufficient evidence"}
                        </span>
                        <button
                          className="text-button"
                          onClick={() => {
                            setView("practice");
                            inspect(index, "evidence");
                          }}
                        >
                          Inspect state change
                          <ArrowRight size={14} />
                        </button>
                      </div>
                    </article>
                  ))
                ) : (
                  <EmptyState
                    title="No explanations submitted yet."
                    text="The conversation is ready. Your first explanation will begin its evidence history."
                    action={() => setView("practice")}
                    label="Open conversation"
                  />
                )}
              </>
            ) : items.length ? (
              <ConversationCollection
                items={items}
                onOpen={(id) => {
                  setActiveId(id);
                  setView("history");
                }}
              />
            ) : (
              <EmptyState
                title="Your history starts with a question."
                text="There are no saved conversations in this browser profile yet."
                action={newAssessment}
                label="Start an assessment"
              />
            )}
          </main>
        ) : (
          <main className="reports-page">
            <SavedDrafts
              drafts={savedDrafts.filter(
                (item) =>
                  item.questionId !== session?.conversation.current_question_id,
              )}
            />
            {draft.trim() &&
              session?.conversation.conversation_id === activeId && (
                <aside className="draft-notice" role="status">
                  <FileText size={20} />
                  <div>
                    <strong>Your unsent explanation is kept as a draft.</strong>
                    <p>
                      This report includes submitted explanations only. Your
                      draft has not been graded or added to learner state.
                    </p>
                    <details>
                      <summary>View saved draft</summary>
                      <p className="saved-draft-text">{draft}</p>
                    </details>
                  </div>
                  <button
                    className="secondary-button"
                    disabled={busy}
                    onClick={() =>
                      session.conversation.status === "completed"
                        ? action("resume")
                        : ((scrollToQuestion.current = true),
                          setView("practice"))
                    }
                  >
                    Resume draft
                    <ArrowRight size={15} />
                  </button>
                </aside>
              )}
            {activeId && <button className="text-button collection-back" onClick={() => setActiveId(null)}>All reports</button>}
            {reportLoading ? (
              <div className="loading-state" role="status">
                <LoaderCircle className="spin" size={25} />
                <p>Building the report from saved evidence…</p>
              </div>
            ) : report ? (
              <ConversationReport
                report={report}
                busy={busy}
                onContinue={() =>
                  report.conversation.status === "completed"
                    ? action("resume")
                    : ((scrollToQuestion.current = true), setView("practice"))
                }
              />
            ) : (
              <>
                <div className="page-heading">
                  <div>
                    <h1>Learning reports</h1>
                    <p>
                      A report for every conversation, based on its saved
                      evidence.
                    </p>
                  </div>
                </div>
                {items.length ? (
                  <ConversationCollection
                    items={items}
                    onOpen={(id) => setActiveId(id)}
                  />
                ) : (
                  <EmptyState
                    title="A report needs your explanations."
                    text="Start an assessment to build a report of questions, evidence, state changes, and what still needs checking."
                    action={newAssessment}
                    label="Start an assessment"
                  />
                )}
              </>
            )}
          </main>
        )}
        </Suspense>
        </RenderBoundary>
      </div>
      <GuidedTour
        onTarget={revealGuideTarget}
        active={guide && !busy && !loading && view === "practice"}
        hasSession={!!session}
        onClose={() => {
          setGuide(false);
          localStorage.setItem("learner-state:guide:v2", "seen");
        }}
      />
    </div>
  );
}

function EmptyState({
  title,
  text,
  action,
  label,
}: {
  title: string;
  text: string;
  action: () => void;
  label: string;
}) {
  return (
    <div className="empty-state">
      <FileText size={28} />
      <h2>{title}</h2>
      <p>{text}</p>
      <button className="secondary-button" onClick={action}>
        {label}
        <ArrowRight size={15} />
      </button>
    </div>
  );
}
function ConversationCollection({
  items,
  onOpen,
}: {
  items: Conversation[];
  onOpen: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("");
  const [from, setFrom] = useState("");
  const filtered = items.filter(item => item.title.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()) && (!status || item.status === status) && (!from || item.updated_at.slice(0, 10) >= from));
  return (
    <div className="conversation-collection">
      <div className="collection-filters">
        <label>Search conversations<input type="search" value={search} onChange={e => setSearch(e.target.value)} placeholder="Find a topic" /></label>
        <label>Conversation status<select value={status} onChange={e => setStatus(e.target.value)}><option value="">All conversations</option><option value="active">In progress</option><option value="needs_question">Waiting for a question</option><option value="completed">Completed</option></select></label>
        <label>Updated since (UTC)<input type="date" value={from} onChange={e => setFrom(e.target.value)} /></label>
      </div>
      <p className="collection-count" role="status">{filtered.length} of {items.length} conversations</p>
      {!filtered.length && <p className="collection-count">No conversations match. <button className="text-button" onClick={() => { setSearch(""); setStatus(""); setFrom(""); }}>Clear filters</button></p>}
      {filtered.map((item) => (
        <button
          key={item.conversation_id}
          onClick={() => onOpen(item.conversation_id)}
        >
          <MessageSquare size={19} />
          <span>
            <strong>{item.title}</strong>
            <small>
              {item.answered_count}{" "}
              {item.answered_count === 1 ? "explanation" : "explanations"} ·{" "}
              {displayDate(item.updated_at)}
              {" · "}{item.status === "completed" ? "Completed" : item.status === "needs_question" ? "Waiting for a question" : "In progress"}
            </small>
          </span>
          <ChevronRight size={18} />
        </button>
      ))}
    </div>
  );
}
