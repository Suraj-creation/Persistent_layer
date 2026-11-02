import { MathText } from "./MathText";
import {
  ArrowDownToLine,
  ArrowRight,
  FileText,
  Printer,
  ShieldCheck,
} from "lucide-react";
import { ConversationReportData } from "../../types/assessment";
import { SourceList } from "./LearnerInspector";
import {
  displayDate,
  displayProse,
  exportJSON,
  percent,
  readable,
} from "./utils";

function printCompleteReport() {
  const sections = Array.from(
    document.querySelectorAll<HTMLDetailsElement>(
      ".conversation-report details",
    ),
  );
  const previous = sections.map((section) => section.open);
  sections.forEach((section) => {
    section.open = true;
  });
  window.addEventListener(
    "afterprint",
    () =>
      sections.forEach((section, index) => {
        section.open = previous[index];
      }),
    { once: true },
  );
  window.print();
}

export function ConversationReport({
  report,
  onContinue,
  busy,
}: {
  report: ConversationReportData;
  onContinue: () => void;
  busy: boolean;
}) {
  return (
    <article className="conversation-report">
      <header className="report-header">
        <div>
          <span className="report-status">
            {report.conversation.status === "completed"
              ? "Completed conversation"
              : "Report in progress"}
          </span>
          <h1>{report.conversation.title}</h1>
          <p>
            {displayDate(report.conversation.created_at)} ·{" "}
            {report.answered_count}{" "}
            {report.answered_count === 1 ? "explanation" : "explanations"} ·{" "}
            {report.concept_progress.length} concepts in scope
          </p>
        </div>
        <div className="report-actions">
          <button
            className="secondary-button"
            onClick={() => exportJSON(report, `${report.report_id}.json`)}
          >
            <ArrowDownToLine size={15} />
            Export JSON
          </button>
          <button className="secondary-button" onClick={printCompleteReport}>
            <Printer size={15} />
            Print report
          </button>
        </div>
      </header>
      <section className="report-overview">
        <h3>What the evidence suggests</h3>
        <p>
          {report.evidence_of_progress
            ? "A later explanation addressed an earlier missing criterion in a fresh question. The linked responses below show that evidence of progress."
            : report.answered_count
              ? "This conversation has recorded evidence, but it has not yet demonstrated a later explanation addressing an earlier gap. Continue with fresh questions to check understanding."
              : "There are no submitted explanations in this conversation yet. No new conclusions about understanding can be drawn."}
        </p>
        <div className="report-facts">
          <span>
            <strong>
              {report.concept_progress.filter((c) => c.observations > 0).length}
            </strong>{" "}
            concepts checked here
          </span>
          <span>
            <strong>{report.remaining_gaps.length}</strong> need another check
          </span>
          <span>
            <strong>{report.unassessed_concepts.length}</strong> not checked
            here
          </span>
        </div>
        <p className="report-limitation">
          <ShieldCheck size={15} />
          {report.interpretation}
        </p>
      </section>
      <section className="report-section">
        <h3>Concept report card</h3>
        <p className="section-description">
          Before and after values include the topic memory available at the
          time. “This conversation” adds only updates caused by explanations in
          this conversation.
        </p>
        <div className="table-scroll">
          <table className="report-table">
            <thead>
              <tr>
                <th scope="col">Concept</th>
                <th scope="col">Before</th>
                <th scope="col">After</th>
                <th scope="col">This conversation</th>
                <th scope="col">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {report.concept_progress.map((c) => (
                <tr key={c.concept_id}>
                  <th scope="row">
                    {c.name}
                    <small>
                      {c.evidence_of_progress
                        ? "Earlier gap addressed"
                        : c.observations
                          ? "Further verification may help"
                          : "Not assessed here"}
                    </small>
                  </th>
                  <td>{c.observations ? percent(c.before_estimate) : "—"}</td>
                  <td>{c.observations ? percent(c.after_estimate) : "—"}</td>
                  <td>
                    {c.observations
                      ? `${c.conversation_delta >= 0 ? "+" : ""}${(c.conversation_delta * 100).toFixed(1)} pp`
                      : "—"}
                  </td>
                  <td>
                    {c.informative_observations} informative / {c.observations}{" "}
                    total<small>σ {c.uncertainty.toFixed(3)}</small>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small-note">
          These percentages are weighted evidence estimates, not exam marks or
          calibrated knowledge probabilities.
        </p>
      </section>
      <section className="report-section">
        <h3>What to revisit next</h3>
        {report.remaining_gaps.length ? (
          <div className="report-gaps">
            {report.remaining_gaps.map((c) => (
              <article key={c.concept_id}>
                <h4>{c.name}</h4>
                {c.missing_criteria.length ? (
                  <ul>
                    {c.missing_criteria.map((criterion, i) => (
                      <li key={i}>{criterion}</li>
                    ))}
                  </ul>
                ) : (
                  <p>
                    {c.informative_observations > 0
                      ? "New positive evidence is recorded, but an earlier candidate belief still needs an explicit check."
                      : "The available explanation is still too limited or uncertain to verify the concept."}
                  </p>
                )}
                {c.hypotheses
                  .filter((h) => h.status !== "weakened")
                  .map((h) => (
                    <p className="candidate-belief" key={h.hypothesis_id}>
                      <strong>Candidate belief:</strong> {h.description}{" "}
                      <span>
                        ({h.supporting_evidence_ids.length} supporting
                        observations; {h.status})
                      </span>
                    </p>
                  ))}
                {c.supporting_quotes.length > 0 && (
                  <blockquote>
                    “{c.supporting_quotes[c.supporting_quotes.length - 1].quote}
                    ”
                  </blockquote>
                )}
              </article>
            ))}
          </div>
        ) : (
          <p>
            {report.answered_count
              ? "No unresolved criteria are currently recorded in the concepts checked here. This is not a claim of complete mastery; unassessed concepts and transfer still need checking."
              : "Submit an explanation before the platform can identify evidence-supported gaps."}
          </p>
        )}
        {report.unassessed_concepts.length > 0 && (
          <p className="small-note">
            Not checked in this conversation:{" "}
            {report.unassessed_concepts.map((c) => c.name).join(", ")}.
          </p>
        )}
      </section>
      <section className="report-section">
        <h3>The question → evidence → state loop</h3>
        <p className="section-description">
          Each entry preserves the actual question and response, the rubric
          check, the computation, and the saved-state reasoning used to
          formulate the next question.
        </p>
        {report.timeline.length === 0 && (
          <div className="empty-inline">
            <FileText size={23} />
            <p>Your submitted explanations will form this timeline.</p>
          </div>
        )}
        {report.timeline.map((turn, index) => (
          <details
            className="report-turn"
            key={turn.turn_id}
            open={index === report.timeline.length - 1}
          >
            <summary>
              <span>
                Question {index + 1} · {readable(turn.question.kind)}
              </span>
              <strong>
                <MathText text={turn.question.prompt} />
              </strong>
              <span className="category-label">
                {readable(turn.evidence.category)}
              </span>
            </summary>
            <div className="report-turn-body">
              <h4>Expected focus</h4>
              <p>
                <MathText text={turn.question.expected_focus} />
              </p>
              <h4>Your response</h4>
              <blockquote>{turn.response_text}</blockquote>
              <h4>Extracted evidence</h4>
              <p>
                <MathText text={turn.evidence.claim} />
              </p>
              <p className="small-note">
                Exact quote: “{turn.evidence.source_quote}” ·{" "}
                {percent(turn.evidence.confidence)} uncalibrated confidence ·{" "}
                <code>{turn.evidence.evidence_id}</code>
              </p>
              <div className="criterion-list">
                {turn.question.criteria?.map((c) => (
                  <div key={c.criterion_id}>
                    <span
                      className={
                        turn.evidence.met_criteria.includes(c.criterion_id)
                          ? "criterion-met"
                          : "criterion-missing"
                      }
                    >
                      {turn.evidence.met_criteria.includes(c.criterion_id)
                        ? "Supported"
                        : "Not yet evidenced"}
                    </span>
                    <p>
                      <MathText text={c.description} />
                    </p>
                  </div>
                ))}
              </div>
              <h4>How persistent state changed</h4>
              <p>
                α: {turn.state_changes.before_alpha.toFixed(3)} →{" "}
                {turn.state_changes.after_alpha.toFixed(3)}; β:{" "}
                {turn.state_changes.before_beta.toFixed(3)} →{" "}
                {turn.state_changes.after_beta.toFixed(3)}. Estimate:{" "}
                {percent(turn.state_changes.before_estimate)} →{" "}
                {percent(turn.state_changes.after_estimate)}.
              </p>
              <p className="small-note">
                Weight {turn.state_changes.weight.toFixed(3)} · rubric coverage{" "}
                {percent(turn.state_changes.coverage)} · novelty factor{" "}
                {turn.state_changes.novelty_factor} ·{" "}
                {readable(turn.state_changes.rule)}.
              </p>
              <p className="formula-text">{turn.state_changes.formula}</p>
              <h4>Support offered</h4>
              <p>
                <MathText
                  text={
                    displayProse(turn.evidence.coaching) ||
                    "No additional coaching was recorded."
                  }
                />
              </p>
              <h4>How the next question was framed</h4>
              <p>{turn.next_question_reason}</p>
              {turn.next_question && (
                <p className="next-question-text">
                  <MathText text={turn.next_question.prompt} />
                </p>
              )}
              <details className="report-technical">
                <summary>Inspect NLP, retrieval and exact provenance</summary>
                <p>
                  Unicode normalization, {turn.nlp.token_count} original-text
                  tokens, {turn.nlp.sentences.length} sentence spans; keywords:{" "}
                  {turn.nlp.keywords.join(", ")}.
                </p>
                <p>
                  Extraction read snapshot{" "}
                  <code>{turn.extraction_trace.state_snapshot_id}</code>; update
                  produced <code>{turn.state_after.snapshot_id}</code>.
                </p>
                <SourceList
                  chunks={turn.extraction_trace.retrieval.materials}
                />
                {turn.next_question && (
                  <>
                    <p>
                      Next-question input hash:{" "}
                      <code>
                        {turn.next_question.generation_trace.context_sha256}
                      </code>
                    </p>
                    <p>
                      Next-question state:{" "}
                      <code>
                        {turn.next_question.generation_trace.state_snapshot_id}
                      </code>
                    </p>
                  </>
                )}
                <button
                  className="text-button"
                  onClick={() => exportJSON(turn, `${turn.turn_id}.json`)}
                >
                  Download this complete transition
                  <ArrowDownToLine size={14} />
                </button>
              </details>
            </div>
          </details>
        ))}
      </section>
      <section className="report-section report-method">
        <h3>Method and limits</h3>
        <dl>
          {Object.entries(report.state_method).map(([key, value]) => (
            <div key={key}>
              <dt>{readable(key)}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        <h4>Reference material</h4>
        {report.source_documents.map((source) => (
          <p key={source.document_id}>
            {source.title} · {source.character_count.toLocaleString()}{" "}
            characters
            {source.kind === "model_reference" && (
              <strong> · Model-authored; not independently verified</strong>
            )}
          </p>
        ))}
      </section>
      <div className="report-bottom">
        <button className="primary-button" disabled={busy} onClick={onContinue}>
          {report.conversation.status === "completed"
            ? "Continue this assessment"
            : "Return to conversation"}
          <ArrowRight size={16} />
        </button>
      </div>
    </article>
  );
}
