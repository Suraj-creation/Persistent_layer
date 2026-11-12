import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Search,
} from "lucide-react";
import * as api from "../../api/observability";
import { AssessmentTurn, Conversation } from "../../types/assessment";
import { displayDate, exportJSON } from "./utils";
import { ExecutionInspectors } from "./ExecutionInspectors";
import { initialExecution } from "./executionInspection";

type Panel = "timeline" | "text" | "evidence" | "state";
type Tab = "executions" | "records" | "techniques" | "research";

export function NlpObservability({
  learner,
  conversations,
  onPractice,
}: {
  learner: string;
  conversations: Conversation[];
  onPractice: () => void;
}) {
  const [tab, setTab] = useState<Tab>("executions");
  const [conversation, setConversation] = useState("");
  const [operation, setOperation] = useState("");
  const [status, setStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [lookup, setLookup] = useState("");
  const [stageFilter, setStageFilter] = useState("");
  const [errorOnly, setErrorOnly] = useState(false);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [before, setBefore] = useState<number | undefined>();
  const [cursorStack, setCursorStack] = useState<(number | undefined)[]>([]);
  const [page, setPage] = useState<api.ExecutionPage | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const [execution, setExecution] = useState<api.ExecutionDetail | null>(null);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");
  const [live, setLive] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [panel, setPanel] = useState<Panel>("timeline");
  const [techniques, setTechniques] = useState<api.Technique[]>([]);
  const [search, setSearch] = useState("");
  const [research, setResearch] = useState<api.Research | null>(null);
  const [dimension, setDimension] = useState("");
  const [evaluationRun, setEvaluationRun] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    api
      .getTechniques(controller.signal)
      .then((data) => setTechniques(data.items))
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    if (tab !== "research") return;
    const controller = new AbortController();
    api
      .getResearch(learner, controller.signal, evaluationRun)
      .then(setResearch)
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      });
    return () => controller.abort();
  }, [learner, tab, refresh, evaluationRun]);

  useEffect(() => {
    if (tab !== "executions") return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    async function poll() {
      try {
        const result = await api.listExecutions(
          learner,
          {
            conversation_id: conversation,
            operation,
            status,
            question_id: question.trim(),
            search: lookup.trim(),
            stage: stageFilter,
            error_only: errorOnly ? "true" : undefined,
            started_after: from ? `${from}T00:00:00Z` : undefined,
            started_before: to ? `${to}T23:59:59.999Z` : undefined,
            before,
            limit: 20,
          },
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setPage(result);
        const id = selectedRef.current || initialExecution(result.items);
        if (id) {
          const detail = await api.getExecution(learner, id, controller.signal);
          if (controller.signal.aborted) return;
          setExecution(detail);
          if (!selectedRef.current) setSelected(id);
        } else setExecution(null);
        setError("");
        setLastUpdated(new Date().toLocaleTimeString());
        failures = 0;
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Live updates unavailable. Retrying automatically.",
          );
          failures++;
        }
      } finally {
        if (!controller.signal.aborted && live)
          timer = setTimeout(poll, Math.min(15000, 1500 * 2 ** failures));
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [
    learner,
    conversation,
    operation,
    status,
    question,
    lookup, stageFilter, errorOnly, from, to,
    before,
    selected,
    live,
    refresh,
    tab,
  ]);

  function changeFilter(action: () => void) {
    action();
    setBefore(undefined);
    setCursorStack([]);
    setSelected(null);
    setExecution(null);
    setPage(null);
  }

  function inspectLatestResponse() {
    changeFilter(() => {
      setConversation(execution?.conversation_id ?? conversation);
      setOperation("answer");
      setStatus("");
      setQuestion("");
      setLookup(""); setStageFilter(""); setErrorOnly(false); setFrom(""); setTo("");
    });
  }

  const matching = techniques.filter((t) =>
    `${t.name} ${t.technique} ${t.purpose}`
      .toLowerCase()
      .includes(search.toLowerCase()),
  );
  const summary = page?.summary;
  return (
    <main className={`nlp-page nlp-page-${tab}`}>
      <div className="page-heading">
        <div>
          <h1>NLP Observability</h1>
          <p>
            Inspect responses, validation, and committed learner state.
          </p>
        </div>
        <div className="nlp-heading-actions">
          <button className="secondary-button" onClick={onPractice}>
            Learning workspace <ChevronRight size={15} />
          </button>
          <button
            className="text-button"
            aria-label="Refresh observability"
            onClick={() => setRefresh((n) => n + 1)}
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </div>
      <nav className="nlp-tabs" aria-label="Observability views">
        {(["executions", "records", "techniques", "research"] as const).map(
          (t) => (
            <button
              key={t}
              aria-current={tab === t ? "page" : undefined}
              className={tab === t ? "active" : ""}
              onClick={() => {
                setTab(t);
                setError("");
              }}
            >
              {t === "executions"
                ? "Execution traces"
                : t === "records"
                  ? "Saved responses"
                  : t === "techniques"
                    ? "NLP techniques"
                    : "Research scorecard"}
            </button>
          ),
        )}
      </nav>
      {error && (
        <div className="nlp-alert" role="alert">
          <strong>Updates interrupted.</strong> {error} Saved records remain
          available.{" "}
          <button
            className="text-button"
            onClick={() => setRefresh((n) => n + 1)}
          >
            Retry now
          </button>
        </div>
      )}
      {tab === "executions" ? (
        <>
          <div className="nlp-summary" aria-label="Recorded execution totals">
            <span>
              <strong>{summary?.total ?? "—"}</strong> executions
            </span>
            <span>
              <strong>{summary?.completed ?? "—"}</strong> completed
            </span>
            <span>
              <strong>{summary?.failed ?? "—"}</strong> failed
            </span>
            <span>
              <strong>{summary?.running ?? "—"}</strong> unfinished
            </span>
            <span>
              <strong>
                {summary?.average_ms == null
                  ? "—"
                  : `${Math.round(summary.average_ms).toLocaleString()} ms`}
              </strong>{" "}
              completed mean
            </span>
            <span><strong>{summary?.success_rate == null ? "—" : `${summary.success_rate}%`}</strong> success rate</span>
          </div>
          <details className="nlp-performance">
            <summary>Performance & reliability</summary>
            <dl>
              <div><dt>AI call mean</dt><dd>{duration(summary?.provider_ms)}</dd></div>
              <div><dt>Retrieval mean</dt><dd>{duration(summary?.retrieval_ms)}</dd></div>
              <div><dt>Commit mean</dt><dd>{duration(summary?.persistence_ms)}</dd></div>
              <div><dt>Failure rate</dt><dd>{summary?.failure_rate == null ? "—" : `${summary.failure_rate}%`}</dd></div>
              <div><dt>Provider retries</dt><dd>{summary?.retry_count ?? "—"}</dd></div>
              <div><dt>Persistence failures</dt><dd>{summary?.persistence_failures ?? "—"}</dd></div>
              <div><dt>Incomplete telemetry</dt><dd>{summary?.incomplete_telemetry ?? "—"}</dd></div>
            </dl>
            <p className="nlp-caption">Rates use finished executions. Latency uses completed, timed stages; missing measurements are excluded. Incomplete telemetry includes running executions and known gaps. Older traces may have unknown completeness.</p>
          </details>
          <div className="nlp-filters">
            <label>Execution lookup<input type="search" value={lookup} placeholder="Execution, question or conversation ID" onChange={e => changeFilter(() => setLookup(e.target.value))} /></label>
            <label>
              Conversation
              <select
                value={conversation}
                onChange={(e) =>
                  changeFilter(() => setConversation(e.target.value))
                }
              >
                <option value="">All conversations</option>
                {conversations.map((c) => (
                  <option key={c.conversation_id} value={c.conversation_id}>
                    {c.title} · {displayDate(c.created_at)}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Execution
              <select
                value={operation}
                onChange={(e) =>
                  changeFilter(() => setOperation(e.target.value))
                }
              >
                <option value="">All operations</option>
                <option value="answer">Learner response</option>
                <option value="generation">Question generation</option>
                <option value="create">Topic setup</option>
              </select>
            </label>
            <label>
              Status
              <select
                value={status}
                onChange={(e) => changeFilter(() => setStatus(e.target.value))}
              >
                <option value="">All statuses</option>
                {["running", "completed", "failed"].map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
            <label className="nlp-live">
              <input
                type="checkbox"
                checked={live}
                onChange={(e) => setLive(e.target.checked)}
              />{" "}
              Live updates
            </label>
          </div>
          <details className="nlp-advanced-filters">
            <summary>Stage, error & date filters{(stageFilter || errorOnly || from || to) ? " · active" : ""}</summary>
            <div className="nlp-filters">
              <label>Pipeline stage<select value={stageFilter} onChange={e => changeFilter(() => setStageFilter(e.target.value))}><option value="">All stages</option>{techniques.map(t => <option key={t.key} value={t.key}>{t.name}</option>)}</select></label>
              <label>From (UTC)<input type="date" value={from} max={to || undefined} onChange={e => changeFilter(() => setFrom(e.target.value))} /></label>
              <label>Through (UTC)<input type="date" value={to} min={from || undefined} onChange={e => changeFilter(() => setTo(e.target.value))} /></label>
              <label className="nlp-live"><input type="checkbox" checked={errorOnly} onChange={e => changeFilter(() => setErrorOnly(e.target.checked))} />Errors and recovered retries</label>
              <button className="text-button" onClick={() => changeFilter(() => {setLookup(""); setConversation(""); setOperation(""); setStatus(""); setQuestion(""); setStageFilter(""); setFrom(""); setTo(""); setErrorOnly(false);})}>Clear filters</button>
            </div>
          </details>
          <p className="nlp-caption" role="status">
            {lastUpdated
              ? `${live ? "Following every 1.5 seconds" : "Updates paused"} · last received ${lastUpdated}`
              : "Loading recorded executions…"}{" "}
            · Totals follow all filters. Duration includes nested work.
          </p>
          {page && !page.items.length ? (
            <div className="empty-state">
              <Activity size={30} />
              <h2>No executions match these filters.</h2>
              <p>
                Submit an explanation in the learning workspace to record the
                real pipeline. Earlier conversations retain their saved evidence
                in Conversations; stage events begin with this update.
              </p>
              <button className="secondary-button" onClick={onPractice}>
                Open learning workspace
              </button>
            </div>
          ) : (
            <div className="nlp-execution-layout">
              <section
                className="nlp-execution-list"
                aria-label="Recorded NLP executions"
              >
                {page?.items.map((item) => (
                  <button
                    className={selected === item.trace_id ? "selected" : ""}
                    key={item.trace_id}
                    onClick={() => {
                      setSelected(item.trace_id);
                      setExecution(null);
                      setPanel("timeline");
                    }}
                  >
                    <div>
                      <strong>
                        {item.operation === "answer"
                          ? "Learner response"
                          : item.operation === "generation"
                            ? "Question generation"
                            : "Topic setup"}
                      </strong>
                      <Status value={item.status} />
                    </div>
                    <small>{displayDate(item.started_at)}</small>
                    <code>{item.trace_id.slice(-12)}</code>
                    <small>
                      {item.duration_ms === null
                        ? "Awaiting completion"
                        : `${Math.round(item.duration_ms).toLocaleString()} ms`}
                    </small>
                  </button>
                ))}
                <div className="nlp-pagination">
                  <button
                    className="text-button"
                    disabled={!cursorStack.length}
                    aria-label="Previous execution page"
                    onClick={() => {
                      setBefore(cursorStack[cursorStack.length - 1]);
                      setCursorStack((s) => s.slice(0, -1));
                      setSelected(null);
                      setExecution(null);
                    }}
                  >
                    <ChevronLeft size={16} />
                  </button>
                  <span>Page {cursorStack.length + 1}</span>
                  <button
                    className="text-button"
                    disabled={!page?.has_more}
                    aria-label="Next execution page"
                    onClick={() => {
                      setCursorStack((s) => [...s, before]);
                      setBefore(page?.next_cursor ?? undefined);
                      setSelected(null);
                      setExecution(null);
                    }}
                  >
                    <ChevronRight size={16} />
                  </button>
                </div>
              </section>
              <section
                className="nlp-execution-detail"
                aria-label="Execution detail"
              >
                {execution ? (
                  <>
                    <div className="nlp-detail-heading">
                      <div>
                        <h2>
                          {execution.operation === "answer"
                            ? "Response processing"
                            : execution.operation === "generation"
                              ? "Question generation"
                              : "Topic setup"}
                        </h2>
                        <code>{execution.trace_id}</code>
                      </div>
                      <button
                        className="text-button"
                        onClick={() =>
                          exportJSON(execution, `${execution.trace_id}.json`)
                        }
                      >
                        <ArrowDownToLine size={15} /> Export trace
                      </button>
                    </div>
                    <div className="nlp-execution-meta">
                      <Status value={execution.status} />
                      <span>{execution.stages.length} recorded stages</span>
                      <span>Telemetry {execution.telemetry_status}</span>
                      {execution.question_id && (
                        <code title={execution.question_id}>
                          Question {execution.question_id.slice(-12)}
                        </code>
                      )}
                    </div>
                    {execution.incomplete_warning && (
                      <p className="nlp-notice">
                        {execution.incomplete_warning}
                      </p>
                    )}
                    {execution.metadata.parent_trace_id != null && (
                      <button
                        className="text-button nlp-parent"
                        onClick={() => {
                          setSelected(
                            String(execution.metadata.parent_trace_id),
                          );
                          setExecution(null);
                        }}
                      >
                        Inspect parent execution
                      </button>
                    )}
                    {typeof execution.metadata.source_trace_id === "string" && (
                      <button
                        className="text-button nlp-parent"
                        onClick={() => {
                          setSelected(
                            String(execution.metadata.source_trace_id),
                          );
                          setExecution(null);
                        }}
                      >
                        Inspect original response execution
                      </button>
                    )}
                    <nav
                      className="nlp-subtabs"
                      aria-label="Execution inspectors"
                    >
                      {(["timeline", "text", "evidence", "state"] as const).map(
                        (p) => (
                          <button
                            className={panel === p ? "active" : ""}
                            aria-pressed={panel === p}
                            key={p}
                            onClick={() => setPanel(p)}
                          >
                            {p === "timeline"
                              ? "Stage timeline"
                              : p === "text"
                                ? "Text transformations"
                                : p === "evidence"
                                  ? "Cognitive Evidence"
                                  : "State changes"}
                          </button>
                        ),
                      )}
                    </nav>
                    {panel === "timeline" ? (
                      <ExecutionPanel execution={execution} panel={panel} />
                    ) : (
                      <ExecutionInspectors
                        execution={execution}
                        panel={panel}
                        learner={learner}
                        onResponse={inspectLatestResponse}
                      />
                    )}
                  </>
                ) : (
                  <p className="nlp-waiting">
                    {selected
                      ? "Loading this execution…"
                      : "Select an execution to inspect its recorded stages."}
                  </p>
                )}
              </section>
            </div>
          )}
          <p className="nlp-caption">
            Local profile scope · Trace retention defaults to 30 days · Private
            reference answers and provider credentials are omitted. A local
            browser profile is not authentication.
          </p>
        </>
      ) : tab === "records" ? (
        <SavedResponses
          learner={learner}
          refresh={refresh}
          conversations={conversations}
          onTrace={(id) => {
            setSelected(id);
            setExecution(null);
            setTab("executions");
          }}
        />
      ) : tab === "techniques" ? (
        <>
          <div className="nlp-technique-intro">
            <p>
              Only methods executed by the assessment pipeline appear here. A
              model’s internal subword tokenizer, attention, or embeddings are
              not exposed as local operations.
            </p>
            <label className="nlp-search">
              <Search size={16} />
              <input
                aria-label="Search NLP techniques"
                placeholder="Search techniques or their purpose"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </label>
          </div>
          <div className="nlp-techniques">
            {matching.map((t) => (
              <article key={t.key}>
                <h2>{t.name}</h2>
                <p className="nlp-method">{t.technique}</p>
                <p>{t.purpose}</p>
                <code>{t.implementation}</code>
                <details>
                  <summary>Why this method?</summary>
                  <p>{t.justification}</p>
                </details>
                {execution?.stages.some((s) => s.key === t.key) ? (
                  <details>
                    <summary>Actual output from selected execution</summary>
                    <Json
                      value={execution.stages
                        .filter((s) => s.key === t.key)
                        .map((s) => ({
                          status: s.status,
                          input: s.input,
                          output: s.output,
                        }))}
                    />
                  </details>
                ) : (
                  <p className="nlp-caption">
                    Select an execution containing this method to inspect an
                    actual result.
                  </p>
                )}
              </article>
            ))}
          </div>
          {!matching.length && (
            <p className="nlp-notice">
              No implemented technique matches your search.
            </p>
          )}
        </>
      ) : research ? (
        <>
          <p className="nlp-notice">{research.notice}</p>
          <div className="nlp-filters">
            <label>
              Dimension
              <select
                value={dimension}
                onChange={(e) => setDimension(e.target.value)}
              >
                <option value="">All research dimensions</option>
                {[...new Set(research.scorecard.map((r) => r.dimension))].map(
                  (d) => (
                    <option key={d}>{d}</option>
                  ),
                )}
              </select>
            </label>
            <label>
              Dataset
              <select>
                <option>{research.dataset}</option>
              </select>
            </label>
            <label>
              Approach
              <select>
                <option>{research.model}</option>
              </select>
            </label>
            <label>
              Evaluation run
              <select
                value={evaluationRun}
                onChange={(e) => setEvaluationRun(e.target.value)}
              >
                <option value="">
                  {research.evaluation_runs.length
                    ? "Operational telemetry / pending research"
                    : "No labelled v2 evaluation runs"}
                </option>
                {research.evaluation_runs.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.dataset} · {r.sample_size} records ·{" "}
                    {displayDate(r.created_at)}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {research.selected_evaluation != null && (
            <details>
              <summary>
                Selected evaluation configuration, labels and record provenance
              </summary>
              <Json value={research.selected_evaluation} />
            </details>
          )}
          <div className="nlp-table-scroll">
            <table className="nlp-scorecard">
              <caption>
                Research score matrix · {research.sample_size} saved responses ·{" "}
                {displayDate(research.generated_at)}
              </caption>
              <thead>
                <tr>
                  <th>Dimension / metric</th>
                  <th>Correctness baseline</th>
                  <th>Text baseline</th>
                  <th>Proposed</th>
                  <th>Improvement</th>
                  <th>Evidence / status</th>
                </tr>
              </thead>
              <tbody>
                {research.scorecard
                  .filter((r) => !dimension || r.dimension === dimension)
                  .map((r) => (
                    <tr key={r.metric}>
                      <th>
                        <small>{r.dimension}</small>
                        {r.metric}
                      </th>
                      <td>{metric(r.baseline)}</td>
                      <td>{metric(r.text_baseline)}</td>
                      <td>{metric(r.proposed)}</td>
                      <td>{metric(r.improvement)}</td>
                      <td>
                        <strong>{r.status}</strong>
                        <p>{r.evidence}</p>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <h2 className="nlp-section-title">
            Six hypotheses, six open research tests
          </h2>
          <div className="nlp-hypotheses">
            {research.hypotheses.map((h) => (
              <details key={h.id}>
                <summary>
                  <strong>{h.id}</strong> {h.hypothesis}
                  <span>Not tested</span>
                </summary>
                <dl>
                  <dt>Implemented capability</dt>
                  <dd>{h.capability}</dd>
                  <dt>Baseline</dt>
                  <dd>{h.baseline}</dd>
                  <dt>Required data</dt>
                  <dd>{h.required_data}</dd>
                  <dt>Evaluation</dt>
                  <dd>{h.evaluation}</dd>
                </dl>
              </details>
            ))}
          </div>
          <details className="nlp-legacy">
            <summary>
              Inspect legacy benchmark artifact · historical, non-comparable
            </summary>
            <p>{research.legacy_benchmark.status}</p>
            <ul>
              {research.legacy_benchmark.limitations?.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
            <code>
              {research.legacy_benchmark.artifact} · SHA-256{" "}
              {research.legacy_benchmark.sha256}
            </code>
            <Json value={research.legacy_benchmark.results} />
          </details>
        </>
      ) : (
        <p className="nlp-waiting" role="status">
          Reading the research scorecard…
        </p>
      )}
    </main>
  );
}

function metric(value: number | null) {
  return value === null ? "—" : value.toFixed(4);
}
function duration(value: number | null | undefined) {
  return value == null ? "—" : `${Math.round(value).toLocaleString()} ms`;
}
function Status({ value }: { value: string }) {
  return <span className={`nlp-status ${value}`}>{value}</span>;
}
function Json({ value }: { value: unknown }) {
  return (
    <pre className="nlp-json">
      {JSON.stringify(value, null, 2) ?? "No output recorded."}
    </pre>
  );
}

function ExecutionPanel({
  execution,
  panel,
}: {
  execution: api.ExecutionDetail;
  panel: Panel;
}) {
  const stages = execution.stages;
  if (panel === "timeline")
    return (
      <div className="nlp-timeline">
        {stages.length ? (
          stages.map((s) => (
            <details key={s.stage_id} className={`nlp-stage ${s.status}`}>
              <summary>
                <span className="nlp-stage-number">{s.sequence}</span>
                <span>
                  <strong>{s.name}</strong>
                  <small>
                    {s.technique}
                    {s.attempt > 1 ? ` · Retry ${s.attempt - 1}` : ""}
                  </small>
                </span>
                <Status value={s.status} />
                <span className="nlp-stage-duration">
                  {s.duration_ms === null
                    ? "In progress"
                    : `${Math.round(s.duration_ms)} ms`}
                </span>
              </summary>
              <div className="nlp-stage-content">
                <code>{s.implementation}</code>
                <p className="nlp-caption">
                  {displayDate(s.started_at)} ·{" "}
                  {s.completed_at
                    ? `Completed ${displayDate(s.completed_at)}`
                    : "No completion recorded"}{" "}
                  · {s.stage_id}
                </p>
                <div className="nlp-io">
                  <div>
                    <h3>Actual input</h3>
                    <Json value={s.input} />
                  </div>
                  <div>
                    <h3>Actual output</h3>
                    <Json value={s.output} />
                  </div>
                </div>
                <details>
                  <summary>Configuration and causal links</summary>
                  <Json
                    value={{
                      parameters: s.parameters,
                      parent_stage_id: s.parent_stage_id,
                      previous_stage_id: s.previous_stage_id,
                    }}
                  />
                </details>
                {s.error && (
                  <div role="note">
                    <h3>Recorded failure</h3>
                    <Json value={s.error} />
                    <p>
                      Provider error bodies are excluded. Later attempts appear
                      as separate stages.
                    </p>
                  </div>
                )}
              </div>
            </details>
          ))
        ) : (
          <p className="nlp-notice">
            No NLP stages executed in this request. It may have reused a saved
            result or failed before processing.
          </p>
        )}
      </div>
    );
  return null;
}

function SavedResponses({
  learner,
  conversations,
  refresh,
  onTrace,
}: {
  learner: string;
  conversations: Conversation[];
  refresh: number;
  onTrace: (id: string) => void;
}) {
  const [conversation, setConversation] = useState("");
  const [before, setBefore] = useState<number | undefined>();
  const [page, setPage] = useState<{
    items: api.SavedResponse[];
    has_more: boolean;
    next_cursor: number | null;
  } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [record, setRecord] = useState<AssessmentTurn | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    setPage(null);
    api
      .getRecords(
        learner,
        { conversation_id: conversation, before, limit: 20 },
        controller.signal,
      )
      .then(setPage)
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      });
    return () => controller.abort();
  }, [learner, conversation, before, refresh]);
  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setError("");
    setRecord(null);
    api
      .getRecord(learner, selected, controller.signal)
      .then(setRecord)
      .catch((err) => {
        if (!controller.signal.aborted) setError(err.message);
      });
    return () => controller.abort();
  }, [learner, selected, refresh]);
  return (
    <>
      <p className="nlp-notice">
        Inspect complete saved responses, including those created before stage
        tracing was introduced. Missing historical stage events are never
        reconstructed. Trace retention does not delete these learner records.
      </p>
      {error && (
        <p className="nlp-alert" role="alert">
          {error}
        </p>
      )}
      <div className="nlp-filters">
        <label>
          Conversation
          <select
            value={conversation}
            onChange={(e) => {
              setConversation(e.target.value);
              setBefore(undefined);
              setSelected(null);
              setRecord(null);
            }}
          >
            <option value="">All conversations</option>
            {conversations.map((c) => (
              <option key={c.conversation_id} value={c.conversation_id}>
                {c.title}
              </option>
            ))}
          </select>
        </label>
        <button
          className="text-button"
          disabled={!before}
          onClick={() => setBefore(undefined)}
        >
          Newest responses
        </button>
      </div>
      <div className="nlp-record-list">
        {page?.items.map((item) => (
          <button
            key={item.turn_id}
            className={selected === item.turn_id ? "selected" : ""}
            onClick={() => setSelected(item.turn_id)}
          >
            <strong>{item.question}</strong>
            <small>
              {displayDate(item.timestamp)} · Evidence{" "}
              {item.evidence_id.slice(-12)} ·{" "}
              {item.trace_id ? "Trace linked" : "Recorded summary only"}
            </small>
          </button>
        ))}
      </div>
      {!page && !error ? (
        <p className="nlp-waiting">Reading saved responses…</p>
      ) : page && !page.items.length ? (
        <p className="nlp-notice">No submitted responses in this scope.</p>
      ) : page ? (
        <button
          className="secondary-button"
          disabled={!page.has_more}
          onClick={() => {
            setBefore(page.next_cursor ?? undefined);
            setSelected(null);
            setRecord(null);
          }}
        >
          Older responses <ChevronRight size={15} />
        </button>
      ) : null}
      {record && (
        <section className="nlp-saved-response">
          <div className="nlp-detail-heading">
            <h2>Saved response record</h2>
            <button
              className="text-button"
              onClick={() => exportJSON(record, `${record.turn_id}.json`)}
            >
              <ArrowDownToLine size={15} /> Export complete record
            </button>
          </div>
          <p className="nlp-caption">
            {record.turn_id} · Snapshot {record.state_after.snapshot_id}
          </p>
          <div className="nlp-io">
            <div>
              <h3>Original learner response</h3>
              <blockquote>{record.response_text}</blockquote>
            </div>
            <div>
              <h3>Extracted claim and candidate interpretation</h3>
              <p>{record.evidence.claim}</p>
              <p>
                {record.evidence.category} · uncalibrated confidence{" "}
                {record.evidence.confidence}
              </p>
            </div>
          </div>
          {record.extraction_trace.trace_id ? (
            <button
              className="secondary-button"
              onClick={() => onTrace(record.extraction_trace.trace_id!)}
            >
              Inspect linked execution
            </button>
          ) : (
            <p className="nlp-notice">
              No stage trace was recorded for this historical response. The
              original saved extraction summary and state are available below.
            </p>
          )}
          {Object.entries({
            nlp: record.nlp,
            evidence: record.evidence,
            state_before: record.state_before,
            state_after: record.state_after,
            state_changes: record.state_changes,
            extraction_trace: record.extraction_trace,
          }).map(([key, value]) => (
            <details key={key}>
              <summary>{key.replace(/_/g, " ")}</summary>
              <Json value={value} />
            </details>
          ))}
        </section>
      )}
    </>
  );
}
