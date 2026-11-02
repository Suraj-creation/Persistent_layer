export interface ExecutionSummary {
  ordinal: number;
  trace_id: string;
  learner_id: string;
  conversation_id: string | null;
  question_id: string | null;
  operation: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  metadata: Record<string, unknown>;
}
export interface Stage {
  stage_id: string;
  sequence: number;
  key: string;
  name: string;
  technique: string;
  implementation: string;
  status: string;
  attempt: number;
  duration_ms: number | null;
  started_at: string;
  completed_at: string | null;
  parent_stage_id: string | null;
  previous_stage_id: string | null;
  input: Record<string, unknown> | null;
  output: Record<string, unknown> | null;
  parameters: Record<string, unknown>;
  error: Record<string, unknown> | null;
}
export interface ExecutionDetail extends ExecutionSummary {
  stages: Stage[];
  incomplete_warning: string | null;
}
export interface ExecutionPage {
  items: ExecutionSummary[];
  has_more: boolean;
  next_cursor: number | null;
  summary: {
    total: number;
    completed: number;
    failed: number;
    running: number;
    average_ms: number | null;
  };
}
export interface Technique {
  key: string;
  name: string;
  technique: string;
  implementation: string;
  purpose: string;
  justification: string;
}
export interface Research {
  generated_at: string;
  dataset: string;
  model: string;
  sample_size: number;
  notice: string;
  scorecard: {
    dimension: string;
    metric: string;
    baseline: number | null;
    text_baseline: number | null;
    proposed: number | null;
    improvement: number | null;
    status: string;
    evidence: string;
  }[];
  hypotheses: {
    id: string;
    hypothesis: string;
    capability: string;
    baseline: string;
    required_data: string;
    evaluation: string;
    status: string;
    tested: boolean;
  }[];
  legacy_benchmark: {
    status: string;
    comparable: boolean;
    artifact?: string;
    sha256?: string;
    results?: unknown;
    limitations?: string[];
  };
  evaluation_runs: {
    run_id: string;
    dataset: string;
    created_at: string;
    models: string[];
    sample_size: number;
  }[];
  selected_evaluation: unknown;
}

async function read<T>(
  path: string,
  learner: string,
  filters: Record<string, string | number | undefined> = {},
  signal?: AbortSignal,
): Promise<T> {
  const query = new URLSearchParams({ learner_id: learner });
  for (const [key, value] of Object.entries(filters))
    if (value !== undefined && value !== "") query.set(key, String(value));
  const response = await fetch(`/api/v2/observability/${path}?${query}`, {
    signal,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : `Observability request failed (${response.status}).`,
    );
  return body;
}
export const listExecutions = (
  learner: string,
  filters: Record<string, string | number | undefined>,
  signal?: AbortSignal,
) => read<ExecutionPage>("traces", learner, filters, signal);
export const getExecution = (
  learner: string,
  id: string,
  signal?: AbortSignal,
) =>
  read<ExecutionDetail>(
    `traces/${encodeURIComponent(id)}`,
    learner,
    {},
    signal,
  );
export const getTechniques = (signal?: AbortSignal) =>
  read<{ items: Technique[] }>("techniques", "", {}, signal);
export const getResearch = (
  learner: string,
  signal?: AbortSignal,
  runId?: string,
) => read<Research>("research", learner, { run_id: runId }, signal);
import type { AssessmentTurn } from "../types/assessment";

export interface SavedResponse {
  ordinal: number;
  turn_id: string;
  conversation_id: string;
  timestamp: string;
  question_id: string;
  question: string;
  trace_id: string | null;
  evidence_id: string;
  snapshot_id: string;
}
export const getRecords = (
  learner: string,
  filters: Record<string, string | number | undefined>,
  signal?: AbortSignal,
) =>
  read<{
    items: SavedResponse[];
    has_more: boolean;
    next_cursor: number | null;
  }>("records", learner, filters, signal);
export const getRecord = (learner: string, id: string, signal?: AbortSignal) =>
  read<AssessmentTurn>(
    `records/${encodeURIComponent(id)}`,
    learner,
    {},
    signal,
  );
