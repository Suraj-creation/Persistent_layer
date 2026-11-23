import { Database, FileText, Search, ShieldCheck } from "lucide-react";
import { AssessmentSession, SourceChunk } from "../../types/assessment";
import { exportJSON, percent, readable } from "./utils";
import { MathText } from "./MathText";

export type InspectorTab = "state" | "evidence" | "decision";

export function SourceList({ chunks }: { chunks: SourceChunk[] }) {
  return (
    <div className="retrieval-sources">
      {chunks.map((chunk) => (
        <details key={chunk.chunk_id}>
          <summary>
            {chunk.kind === "study_notes"
              ? "Your study notes"
              : chunk.kind === "learner_evidence"
                ? "Prior learner evidence"
                : "Model-authored reference · unverified"}
          </summary>
          <p>{chunk.content}</p>
          <dl className="technical-details">
            <div>
              <dt>Chunk</dt>
              <dd>
                <code>{chunk.chunk_id}</code>
              </dd>
            </div>
            <div>
              <dt>Retrieval</dt>
              <dd>{chunk.retrieval_method}</dd>
            </div>
            {chunk.rank !== null && (
              <div>
                <dt>BM25 rank</dt>
                <dd>
                  {chunk.rank.toPrecision(3)}
                  <small>Lower is more relevant; not confidence.</small>
                </dd>
              </div>
            )}
          </dl>
        </details>
      ))}
    </div>
  );
}

export function LearnerInspector({
  session,
  selected,
  tab,
  onTab,
}: {
  session: AssessmentSession;
  selected: number | null;
  tab: InspectorTab;
  onTab: (tab: InspectorTab) => void;
}) {
  const turn =
    selected === null
      ? session.turns[session.turns.length - 1]
      : session.turns[selected];
  const state =
    selected === null ? session.state : turn?.state_after || session.state;
  const question =
    selected === null
      ? session.active_question || turn?.question
      : turn?.question;
  const trace = question?.generation_trace;
  const concepts = Object.values(state.concepts);
  const checked = concepts.filter((c) => c.informative_count > 0).length;
  const earlierCount = Math.max(0, concepts.reduce((total, c) => total + c.observation_count, 0) - session.turns.length);
  const tabs: { id: InspectorTab; label: string }[] = [
    { id: "state", label: "Learner state" },
    { id: "evidence", label: "Evidence" },
    { id: "decision", label: "Why next?" },
  ];
  return (
    <aside
      className="learner-inspector"
      data-guide="inspector"
      aria-label="Learning evidence inspector"
    >
      <div className="inspector-heading">
        <Search size={19} />
        <div>
          <h2>Saved topic memory</h2>
          <p>
            {selected === null
              ? "Evidence across conversations on this topic"
              : "After the selected explanation"}
          </p>
        </div>
      </div>
      {selected === null && <div className="memory-provenance">
        <strong>{earlierCount ? `Includes ${earlierCount} earlier ${earlierCount === 1 ? "explanation" : "explanations"}` : "Built from submitted explanations"}</strong>
        <p>{session.turns.length} submitted here. Generating questions and viewing traces do not change these estimates.</p>
      </div>}
      <div
        className="inspector-tabs"
        role="tablist"
        aria-label="Evidence inspector"
      >
        {tabs.map((item, index) => (
          <button
            key={item.id}
            id={`inspector-${item.id}`}
            role="tab"
            aria-selected={tab === item.id}
            aria-controls="inspector-content"
            tabIndex={tab === item.id ? 0 : -1}
            onClick={() => onTab(item.id)}
            onKeyDown={(event) => {
              const delta =
                event.key === "ArrowRight"
                  ? 1
                  : event.key === "ArrowLeft"
                    ? -1
                    : 0;
              if (delta) {
                event.preventDefault();
                const next =
                  tabs[(index + delta + tabs.length) % tabs.length].id;
                onTab(next);
                document.getElementById(`inspector-${next}`)?.focus();
              }
            }}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div
        className="inspector-content"
        id="inspector-content"
        role="tabpanel"
        aria-labelledby={`inspector-${tab}`}
      >
        <details className="metric-explainer">
          <summary>What do the scores and settings mean?</summary>
          <p>
            These are research estimates, not grades. The confidence values and
            update settings have not been experimentally calibrated against
            labeled learner data.
          </p>
          <dl>
            <div>
              <dt>Rubric coverage</dt>
              <dd>
                The proportion of this question’s criteria supported by your
                submitted explanation. Missing evidence does not automatically
                mean a permanent misconception.
              </dd>
            </div>
            <div>
              <dt>Model confidence</dt>
              <dd>
                The model’s own confidence in its interpretation. It is not a
                validated probability that the interpretation is correct.
              </dd>
            </div>
            <div>
              <dt>Effective weight</dt>
              <dd>
                How much this answer contributes after confidence, text
                informativeness, and repetition are considered. Zero means the
                numerical estimate is held.
              </dd>
            </div>
            <div>
              <dt>α and β</dt>
              <dd>
                Accumulated weighted support and missing/contradictory evidence.
                Both start at 1, which is a neutral prior rather than a 50% test
                score.
              </dd>
            </div>
            <div>
              <dt>Evidence estimate</dt>
              <dd>
                α ÷ (α + β). It summarizes the stored observations; it does not
                measure all of your understanding.
              </dd>
            </div>
            <div>
              <dt>Uncertainty, σ</dt>
              <dd>
                The spread of the evidence distribution. A smaller spread means
                a narrower estimate, not necessarily greater mastery.
              </dd>
            </div>
          </dl>
        </details>
        {tab === "state" && (
          <>
            <div className="memory-summary">
              <Database size={15} />
              <span>
                {checked} of {concepts.length} concepts have informative
                evidence
              </span>
            </div>
            <p className="small-note">
              Estimates describe evidence, not grades. Unassessed concepts
              retain a neutral prior.
            </p>
            <div className="memory-concepts">
              {concepts.map((concept) => (
                <article className="memory-concept" key={concept.concept_id}>
                  <div>
                    <h3>{concept.name}</h3>
                    <span>
                      {concept.informative_count
                        ? percent(concept.estimate)
                        : "Unassessed"}
                    </span>
                  </div>
                  {concept.informative_count > 0 && (
                    <div className="evidence-track">
                      <span style={{ width: percent(concept.estimate) }} />
                    </div>
                  )}
                  <p>
                    {concept.observation_count} observations ·{" "}
                    {concept.informative_count} informative
                    {concept.informative_count > 0 &&
                      ` · σ ${concept.uncertainty.toFixed(3)}`}
                  </p>
                  {concept.missing_criteria.length > 0 && (
                    <details>
                      <summary>What still needs checking</summary>
                      <ul>
                        {concept.missing_criteria.map((criterion, i) => (
                          <li key={i}>{criterion}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                  {Object.values(concept.hypotheses).map((h) => (
                    <div className="hypothesis-record" key={h.hypothesis_id}>
                      <span className="category-label">{h.status}</span>
                      <p>{h.description}</p>
                      <small>
                        {h.supporting_evidence_ids.length} supporting ·{" "}
                        {h.contradicting_evidence_ids.length} contradicting
                        observations
                      </small>
                    </div>
                  ))}
                </article>
              ))}
            </div>
            <details className="state-method">
              <summary>How is this state calculated?</summary>
              <p>
                Each concept starts at α = 1, β = 1. The model checks your
                explanation against the saved question’s criteria.
              </p>
              <p>
                Weight = extraction confidence × response informativeness ×
                novelty. Met criteria add weighted support to α; missing
                criteria add weighted evidence to β. A calculation slip receives
                a smaller penalty.
              </p>
              <p>
                Estimate = α / (α + β). Ambiguous or low-confidence observations
                hold the estimate. Repeated identical explanations contribute
                much less. Coaching contributes nothing.
              </p>
              <p>
                The uncertainty shown is the Beta distribution’s standard
                deviation. These values are research heuristics, not calibrated
                measurements of knowledge.
              </p>
            </details>
            <div className="saved-state-reference">
              <span>Saved snapshot</span>
              <code>{state.snapshot_id}</code>
            </div>
          </>
        )}
        {tab === "evidence" &&
          (turn ? (
            <>
              <div className="memory-summary">
                <ShieldCheck size={15} />
                Exact source span validated
              </div>
              <h3>Your original words</h3>
              <blockquote className="source-quote">
                “{turn.evidence.source_quote}”
              </blockquote>
              <h3>Extracted interpretation</h3>
              <span className="category-label">
                {readable(turn.evidence.category)}
              </span>
              <p>
                <MathText text={turn.evidence.claim} />
              </p>
              <p className="small-note">
                Confidence {percent(turn.evidence.confidence)} · uncalibrated.
                The quote is verified; the interpretation can still be wrong.
              </p>
              <h3>What the question was checking</h3>
              <div className="criterion-list">
                {turn.question.criteria?.map((criterion) => (
                  <div key={criterion.criterion_id}>
                    <span
                      className={
                        turn.evidence.met_criteria.includes(
                          criterion.criterion_id,
                        )
                          ? "criterion-met"
                          : "criterion-missing"
                      }
                    >
                      {turn.evidence.met_criteria.includes(
                        criterion.criterion_id,
                      )
                        ? "Supported"
                        : "Not yet evidenced"}
                    </span>
                    <p>
                      <MathText text={criterion.description} />
                    </p>
                  </div>
                ))}
              </div>
              <details className="state-method">
                <summary>Text processing & extraction</summary>
                <p>
                  Original text is preserved. Normalization uses Unicode NFKC
                  and whitespace normalization; token and sentence offsets refer
                  to the original response.
                </p>
                <dl className="technical-details">
                  <div>
                    <dt>Tokens / sentences</dt>
                    <dd>
                      {turn.nlp.token_count} / {turn.nlp.sentences.length}
                    </dd>
                  </div>
                  <div>
                    <dt>Keywords</dt>
                    <dd>{turn.nlp.keywords.join(", ")}</dd>
                  </div>
                  <div>
                    <dt>Quote offsets</dt>
                    <dd>
                      {turn.evidence.source_start}–{turn.evidence.source_end}
                    </dd>
                  </div>
                  <div>
                    <dt>Extractor</dt>
                    <dd>{turn.evidence.extractor_model}</dd>
                  </div>
                  <div>
                    <dt>Evidence</dt>
                    <dd>
                      <code>{turn.evidence.evidence_id}</code>
                    </dd>
                  </div>
                </dl>
                <h4>Normalized text</h4>
                <p>{turn.nlp.normalized_text}</p>
              </details>
              <details className="state-method">
                <summary>Exact state update for this answer</summary>
                <p>{turn.state_changes.formula}</p>
                <dl className="technical-details">
                  <div>
                    <dt>Rubric coverage</dt>
                    <dd>{percent(turn.state_changes.coverage)}</dd>
                  </div>
                  <div>
                    <dt>Effective weight</dt>
                    <dd>{turn.state_changes.weight.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>α</dt>
                    <dd>
                      {turn.state_changes.before_alpha.toFixed(3)} →{" "}
                      {turn.state_changes.after_alpha.toFixed(3)}
                    </dd>
                  </div>
                  <div>
                    <dt>β</dt>
                    <dd>
                      {turn.state_changes.before_beta.toFixed(3)} →{" "}
                      {turn.state_changes.after_beta.toFixed(3)}
                    </dd>
                  </div>
                  <div>
                    <dt>Estimate</dt>
                    <dd>
                      {percent(turn.state_changes.before_estimate)} →{" "}
                      {percent(turn.state_changes.after_estimate)}
                    </dd>
                  </div>
                </dl>
              </details>
              <button
                className="secondary-button full-width"
                onClick={() => exportJSON(turn, `${turn.turn_id}.json`)}
              >
                Export evidence & computation
              </button>
            </>
          ) : (
            <InspectorEmpty text="Submit an explanation to inspect its exact words, rubric evidence, and state calculation." />
          ))}
        {tab === "decision" &&
          (trace ? (
            <>
              <div className="memory-summary">
                <ShieldCheck size={15} />
                Generated from saved state
              </div>
              <h3>Why this question?</h3>
              <p>{trace.plan.reason}</p>
              <ol className="decision-steps">
                <li>
                  <strong>Read memory</strong>
                  <p>The server retrieved this topic snapshot from SQLite.</p>
                  <code>{trace.state_snapshot_id}</code>
                </li>
                <li>
                  <strong>Choose a target</strong>
                  <p>
                    {
                      session.definition.concepts.find(
                        (c) => c.concept_id === trace.plan.concept_id,
                      )?.name
                    }{" "}
                    · {readable(trace.plan.intent)} · {trace.plan.difficulty}
                  </p>
                  {trace.plan.required_criteria.length > 0 && (
                    <p>
                      {trace.plan.required_criteria.length} earlier missing{" "}
                      {trace.plan.required_criteria.length === 1
                        ? "criterion is"
                        : "criteria are"}{" "}
                      being rechecked in a fresh task.
                    </p>
                  )}
                </li>
                <li>
                  <strong>Retrieve evidence & material</strong>
                  <p>
                    {trace.context.retrieval.materials.length} material chunks ·{" "}
                    {trace.context.retrieval.evidence.length} learner-evidence
                    chunks. Retrieval is limited to your topic memory.
                  </p>
                </li>
                <li>
                  <strong>Formulate & validate</strong>
                  <p>
                    {trace.model} formulated the question against this saved
                    plan. Target, citations, rubric, and novelty were validated
                    before activation.
                  </p>
                </li>
              </ol>
              <h3>Retrieved material</h3>
              <SourceList chunks={trace.context.retrieval.materials} />
              {trace.context.retrieval.evidence.length > 0 && (
                <>
                  <h3>Retrieved learner evidence</h3>
                  <SourceList chunks={trace.context.retrieval.evidence} />
                </>
              )}
              <dl className="technical-details">
                <div>
                  <dt>Generation time</dt>
                  <dd>{(trace.duration_ms / 1000).toFixed(1)} s</dd>
                </div>
                <div>
                  <dt>Input hash · SHA-256</dt>
                  <dd>
                    <code>{trace.context_sha256}</code>
                  </dd>
                </div>
              </dl>
              <button
                className="secondary-button full-width"
                onClick={() => exportJSON(trace, `${trace.trace_id}.json`)}
              >
                Export exact generation context
              </button>
            </>
          ) : (
            <InspectorEmpty text="Once a question is generated, its saved-state plan, retrieval sources, and input hash appear here." />
          ))}
      </div>
      <div className="inspector-footer">
        <Database size={13} />
        Local SQLite memory · evidence stays traceable
      </div>
    </aside>
  );
}

function InspectorEmpty({ text }: { text: string }) {
  return (
    <div className="inspector-empty">
      <FileText size={26} />
      <p>{text}</p>
    </div>
  );
}
