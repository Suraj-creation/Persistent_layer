import { useEffect, useState } from "react";
import * as api from "../../api/observability";
import type {
  AssessmentTurn,
  ExtractedEvidence,
  TopicState,
} from "../../types/assessment";
import { completedOutput, inspectTokens } from "./executionInspection";
import { exportJSON } from "./utils";

export type InspectorPanel = "text" | "evidence" | "state";
const descriptions: Record<InspectorPanel, { title: string; purpose: string }> =
  {
    text: {
      title: "How the response becomes searchable text",
      purpose:
        "Compare the original response with its normalized view, inspect token and sentence boundaries, and see the lexical cues used for retrieval. These rules do not judge whether an answer is correct.",
    },
    evidence: {
      title: "From learner words to validated evidence",
      purpose:
        "The model proposes a claim and rubric judgments. The server checks the exact quote, assigned concept, criterion IDs and source citations before accepting the evidence. Validation checks provenance; it does not guarantee the interpretation is correct.",
    },
    state: {
      title: "How accepted evidence affects topic memory",
      purpose:
        "Compare the saved estimates before and after this response. Confidence, rubric coverage, response length and duplicate damping determine the weight. These are heuristic evidence estimates, not calibrated probabilities of understanding.",
    },
  };

function Json({ value }: { value: unknown }) {
  return (
    <pre className="nlp-json">
      {JSON.stringify(value, null, 2) ?? "No output recorded."}
    </pre>
  );
}
function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

export function ExecutionInspectors({
  execution,
  panel,
  learner,
  onResponse,
}: {
  execution: api.ExecutionDetail;
  panel: InspectorPanel;
  learner: string;
  onResponse: () => void;
}) {
  const [saved, setSaved] = useState<{
    id: string;
    record: AssessmentTurn;
  } | null>(null);
  const [recordError, setRecordError] = useState("");
  const [retry, setRetry] = useState(0);
  const canLoadSaved =
    execution.status === "completed" ||
    execution.metadata.cache_hit === true ||
    completedOutput(execution.stages, "persistence")?.committed === true;
  const turnId =
    canLoadSaved && typeof execution.metadata.turn_id === "string"
      ? execution.metadata.turn_id
      : "";
  useEffect(() => {
    setRecordError("");
    if (!turnId) return;
    const controller = new AbortController();
    api
      .getRecord(learner, turnId, controller.signal)
      .then((record) => {
        if (!controller.signal.aborted) setSaved({ id: turnId, record });
      })
      .catch((err) => {
        if (!controller.signal.aborted) setRecordError(err.message);
      });
    return () => controller.abort();
  }, [learner, turnId, retry]);
  const record = saved?.id === turnId ? saved.record : undefined;
  const last = (key: string) => completedOutput(execution.stages, key);
  const nlp =
    record?.nlp ?? (last("preprocessing") as AssessmentTurn["nlp"] | undefined);
  const evidence =
    record?.evidence ??
    (last("evidence") as unknown as ExtractedEvidence | undefined);
  const update = record
    ? {
        state_before: record.state_before,
        state_after: record.state_after,
        changes: record.state_changes,
      }
    : last("state_update");
  const available =
    panel === "text"
      ? !!nlp?.token_spans
      : panel === "evidence"
        ? !!evidence?.evidence_id
        : !!update;
  const stageKey =
    panel === "text"
      ? "preprocessing"
      : panel === "evidence"
        ? "evidence"
        : "state_update";
  const relevant = [...execution.stages]
    .reverse()
    .find((stage) => stage.key === stageKey);
  const description = descriptions[panel];
  return (
    <div className="nlp-inspector">
      <details className="nlp-panel-explainer">
        <summary>{description.title}</summary>
        <p>{description.purpose}</p>
      </details>
      {execution.metadata.cache_hit === true && (
        <p className="nlp-notice">
          This request reused an already saved response.{" "}
          {execution.metadata.discarded_computation === true
            ? "The concurrent computation in this timeline was discarded; the inspectors show the saved response. No additional evidence was committed."
            : "No new NLP stages ran and no new evidence was added."}
        </p>
      )}
      {recordError && (
        <p className="nlp-alert" role="alert">
          The complete saved response could not be loaded: {recordError}.
          Available trace data is shown below.{" "}
          <button
            className="text-button"
            onClick={() => setRetry((n) => n + 1)}
          >
            Retry saved response
          </button>
        </p>
      )}
      {!available ? (
        <div className="nlp-empty-panel" role="status">
          <h3>
            {execution.operation !== "answer"
              ? "Choose a learner response to inspect this step"
              : relevant?.status === "running" || execution.status === "running"
                ? "Waiting for this processing step"
                : "No accepted output for this step"}
          </h3>
          <p>
            {execution.operation !== "answer"
              ? "This execution prepares a topic or generates a question. Text transformations, Cognitive Evidence and state updates run when you submit your own explanation."
              : relevant?.status === "failed" || execution.status === "failed"
                ? "Processing failed before this output was accepted. The stage timeline records the failed attempts. A rejected interpretation is never shown as validated evidence."
                : turnId && !recordError
                  ? "Loading the complete saved response…"
                  : "This request did not record this output. Check the stage timeline or open a response in Saved responses."}
          </p>
          {execution.operation !== "answer" && (
            <button className="secondary-button" onClick={onResponse}>
              Inspect latest learner response
            </button>
          )}
        </div>
      ) : panel === "text" && nlp ? (
        <TextPanel
          nlp={nlp}
          retrieval={record?.extraction_trace.retrieval ?? last("retrieval")}
          complete={!!record}
        />
      ) : panel === "evidence" && evidence ? (
        <EvidencePanel
          evidence={evidence}
          record={record}
          grounding={last("grounding")}
          validation={last("validation")}
        />
      ) : update ? (
        <StatePanel
          update={
            update as {
              state_before: TopicState;
              state_after: TopicState;
              changes: AssessmentTurn["state_changes"];
            }
          }
          committed={!!record || last("persistence")?.committed === true}
          persistence={last("persistence")}
        />
      ) : null}
      {record && (
        <div className="nlp-record-source">
          <span>Complete saved response · {record.turn_id}</span>
          <button
            className="text-button"
            onClick={() => exportJSON(record, `${record.turn_id}.json`)}
          >
            Export complete record
          </button>
        </div>
      )}
    </div>
  );
}

function TextPanel({
  nlp,
  retrieval,
  complete,
}: {
  nlp: AssessmentTurn["nlp"];
  retrieval: unknown;
  complete: boolean;
}) {
  const tokens = inspectTokens(nlp.token_spans);
  return (
    <>
      <div className="nlp-io">
        <div>
          <h3>Original learner text</h3>
          <blockquote>{nlp.raw_text}</blockquote>
        </div>
        <div>
          <h3>Normalized view</h3>
          <blockquote>{nlp.normalized_text}</blockquote>
          <p className="nlp-caption">
            Unicode NFKC and whitespace normalization. Original text and
            quotation offsets are preserved.
          </p>
        </div>
      </div>
      <h3>Tokens and original character offsets</h3>
      <p className="nlp-caption">
        {nlp.token_count} tokens · Unicode code-point offsets [start, end) ·
        punctuation and mathematical symbols retained. These are regex tokens,
        not model subword tokens.
      </p>
      <div className="nlp-token-list">
        {tokens.items.map((token, i) => (
          <span key={i}>
            <strong>{token.token}</strong>
            <small>
              {token.start}:{token.end}
            </small>
          </span>
        ))}
      </div>
      {tokens.omitted > 0 && (
        <p className="nlp-notice">
          {tokens.omitted} tokens omitted from this bounded trace.{" "}
          {complete
            ? "See the exported record."
            : "The complete saved response loads automatically when available; failed requests have no saved response."}
        </p>
      )}
      <div className="nlp-io">
        <FeatureList
          title="Keywords"
          items={strings(nlp.keywords)}
          empty="No keywords extracted."
        />
        <div>
          <FeatureList
            title="Negation cues"
            items={strings(nlp.negation_cues)}
            empty="No matching negation cues."
          />
          <FeatureList
            title="Uncertainty cues"
            items={strings(nlp.uncertainty_cues)}
            empty="No matching uncertainty cues."
          />
        </div>
      </div>
      <p className="nlp-caption">
        English lexical cues are hints, not a sentiment analysis or a diagnosis.
        Keywords are normalized frequency counts after a small stop-word list.
      </p>
      <details>
        <summary>Sentence boundaries and response fingerprint</summary>
        <Json
          value={{
            sentences: nlp.sentences,
            response_fingerprint: nlp.response_hash,
            version: nlp.version,
          }}
        />
      </details>
      <details>
        <summary>Retrieved source passages and prior evidence</summary>
        <p>
          Local SQLite FTS5 uses Unicode tokenization, English Porter stemming
          and BM25 ranking. Reference material and learner evidence are
          retrieved separately. No dense learner embedding is computed.
        </p>
        <Json value={retrieval} />
      </details>
    </>
  );
}
function FeatureList({
  title,
  items,
  empty,
}: {
  title: string;
  items: string[];
  empty: string;
}) {
  return (
    <div>
      <h3>{title}</h3>
      {items.length ? (
        <ul className="nlp-feature-list">
          {items.map((item, i) => (
            <li key={i}>{item}</li>
          ))}
        </ul>
      ) : (
        <p className="nlp-caption">{empty}</p>
      )}
    </div>
  );
}
function EvidencePanel({
  evidence,
  record,
  grounding,
  validation,
}: {
  evidence: ExtractedEvidence;
  record?: AssessmentTurn;
  grounding: unknown;
  validation: unknown;
}) {
  const criteria = new Map(
    record?.question.criteria?.map((criterion) => [
      criterion.criterion_id,
      criterion.description,
    ]),
  );
  return (
    <>
      <h3>What the learner said</h3>
      <blockquote>{evidence.source_quote}</blockquote>
      <p className="nlp-caption">
        Verified original-text span [{evidence.source_start},{" "}
        {evidence.source_end}) · concept {evidence.concept_id}
      </p>
      <h3>What the model extracted</h3>
      <p>{evidence.claim}</p>
      <h3>Candidate interpretation</h3>
      <p>
        {evidence.category.replace(/_/g, " ")}
        {evidence.misconception ? ` · ${evidence.misconception}` : ""}
      </p>
      <p className="nlp-notice">
        Model confidence: {evidence.confidence} · uncalibrated. A candidate
        misconception is a revisable hypothesis.
      </p>
      <div className="nlp-io">
        <FeatureList
          title="Criteria met"
          items={strings(evidence.met_criteria).map(
            (id) => criteria.get(id) ?? id,
          )}
          empty="No criteria confirmed by this response."
        />
        <FeatureList
          title="Criteria still missing"
          items={strings(evidence.missing_criteria).map(
            (id) => criteria.get(id) ?? id,
          )}
          empty="No missing criteria reported for this question."
        />
      </div>
      <details>
        <summary>Concept grounding and source citations</summary>
        <Json
          value={{
            grounding,
            source_citations: evidence.source_citations,
            extractor_model: evidence.extractor_model,
          }}
        />
      </details>
      <details>
        <summary>Validation results</summary>
        <Json
          value={
            validation ?? {
              quote_verified: record?.extraction_trace.quote_validated,
              validation_attempts: record?.extraction_trace.validation_attempts,
            }
          }
        />
      </details>
      <details>
        <summary>Complete structured evidence</summary>
        <Json value={evidence} />
      </details>
    </>
  );
}
function StatePanel({
  update,
  committed,
  persistence,
}: {
  update: {
    state_before: TopicState;
    state_after: TopicState;
    changes: AssessmentTurn["state_changes"];
  };
  committed: boolean;
  persistence: unknown;
}) {
  const previous = update.state_before;
  const next = update.state_after;
  const change = update.changes;
  return (
    <>
      <p className="nlp-notice">
        {committed
          ? "The transaction committed this state."
          : "Computed only: persistence has not completed successfully. This state is not confirmed saved."}
      </p>
      {change && (
        <div className="nlp-change-summary">
          <h3>
            {change.weight === 0 ? "Estimate held" : "Evidence-weighted update"}
          </h3>
          <p>
            {change.weight === 0
              ? "The response was ambiguous, too short or below the confidence threshold. Numeric estimates stay unchanged; the observation is still recorded."
              : `Evidence weight ${change.weight}; rubric coverage ${Math.round(change.coverage * 100)}%; estimate change ${change.delta_estimate > 0 ? "+" : ""}${change.delta_estimate}.`}
          </p>
          {change.duplicate_response && (
            <p>
              Repeated response detected: evidence weight is reduced to 15%.
            </p>
          )}
        </div>
      )}
      {previous?.concepts && next?.concepts ? (
        <div className="nlp-table-scroll">
          <table className="nlp-state-table">
            <caption>
              Concept estimates and uncertainty before and after this response
            </caption>
            <thead>
              <tr>
                <th scope="col">Concept</th>
                <th scope="col">Before estimate</th>
                <th scope="col">After estimate</th>
                <th scope="col">Before uncertainty</th>
                <th scope="col">After uncertainty</th>
              </tr>
            </thead>
            <tbody>
              {Object.values(next.concepts)
                .filter(
                  (concept) => concept && typeof concept.name === "string",
                )
                .map((concept) => (
                  <tr key={concept.concept_id}>
                    <th scope="row">{concept.name}</th>
                    <td>
                      {previous.concepts[concept.concept_id]?.estimate ?? "—"}
                    </td>
                    <td>{concept.estimate}</td>
                    <td>
                      {previous.concepts[concept.concept_id]?.uncertainty ??
                        "—"}
                    </td>
                    <td>{concept.uncertainty}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="nlp-notice">
          The state was omitted from bounded telemetry. Open the complete saved
          response when available.
        </p>
      )}
      <details>
        <summary>Evidence-weighted calculation and belief changes</summary>
        <Json value={change} />
      </details>
      <div className="nlp-io">
        <details>
          <summary>Full state before</summary>
          <Json value={previous} />
        </details>
        <details>
          <summary>Full state after</summary>
          <Json value={next} />
        </details>
      </div>
      <details>
        <summary>Persistence outcome</summary>
        <Json value={persistence ?? { saved_record_loaded: committed }} />
      </details>
    </>
  );
}
