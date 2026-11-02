export interface ConceptDefinition {
  concept_id: string;
  name: string;
  description: string;
  prerequisites: string[];
}
export interface Hypothesis {
  hypothesis_id: string;
  description: string;
  score: number;
  status: string;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
}
export interface ConceptMemory {
  concept_id: string;
  name: string;
  alpha: number;
  beta: number;
  estimate: number;
  uncertainty: number;
  observation_count: number;
  informative_count: number;
  evidence_ids: string[];
  hypotheses: Record<string, Hypothesis>;
  missing_criteria: string[];
  last_category: string | null;
}
export interface TopicState {
  snapshot_id: string;
  timestamp: string;
  policy_version: string;
  concepts: Record<string, ConceptMemory>;
}
export interface SourceChunk {
  chunk_id: string;
  source_id: string;
  kind: string;
  content: string;
  metadata: Record<string, string | number>;
  rank: number | null;
  retrieval_method: string;
}
export interface Retrieval {
  query: string;
  scope: { learner_id: string; topic_key: string };
  method: string;
  materials: SourceChunk[];
  evidence: SourceChunk[];
}
export interface Criterion {
  criterion_id: string;
  description: string;
}
export interface AdaptivePlan {
  concept_id: string;
  kind: string;
  intent: string;
  reason: string;
  difficulty: string;
  missing_criteria: string[];
  required_criteria: Criterion[];
  evidence_ids: string[];
  state_snapshot_id: string;
  policy_version: string;
}
export interface GenerationTrace {
  trace_id: string;
  state_snapshot_id: string;
  context_sha256: string;
  model: string;
  duration_ms: number;
  attempts: number;
  plan: AdaptivePlan;
  context: {
    retrieval: Retrieval;
    state: TopicState;
    previous_questions: { prompt: string }[];
  };
  source_citations: string[];
  prompt_version: string;
}
export interface AssessmentQuestion {
  question_id: string;
  concept_id: string;
  kind: string;
  prompt: string;
  expected_focus: string;
  learning_support: string;
  source_citations: string[];
  generation_trace: GenerationTrace;
  criteria?: Criterion[];
  reference_answer?: string;
}
export interface ExtractedEvidence {
  evidence_id: string;
  concept_id: string;
  question_id: string;
  source_quote: string;
  source_start: number;
  source_end: number;
  claim: string;
  category: string;
  confidence: number;
  met_criteria: string[];
  missing_criteria: string[];
  misconception: string | null;
  contradicts_hypothesis_ids: string[];
  feedback: string;
  coaching: string;
  extractor_model: string;
  source_citations: string[];
}
export interface StateChange {
  concept_id: string;
  rule: string;
  confidence: number;
  coverage: number;
  novelty_factor: number;
  informativeness: number;
  duplicate_response: boolean;
  weight: number;
  delta_alpha: number;
  delta_beta: number;
  before_alpha: number;
  before_beta: number;
  after_alpha: number;
  after_beta: number;
  before_estimate: number;
  after_estimate: number;
  delta_estimate: number;
  before_uncertainty: number;
  after_uncertainty: number;
  formula: string;
  limitation: string;
}
export interface AssessmentTurn {
  turn_id: string;
  timestamp: string;
  question: AssessmentQuestion;
  response_text: string;
  nlp: {
    raw_text: string;
    normalized_text: string;
    token_count: number;
    token_spans: { token: string; start: number; end: number }[];
    sentences: { text: string; start: number; end: number }[];
    keywords: string[];
    negation_cues: string[];
    uncertainty_cues: string[];
    response_hash: string;
    version: string;
  };
  evidence: ExtractedEvidence;
  state_before: TopicState;
  state_after: TopicState;
  state_changes: StateChange;
  extraction_trace: {
    trace_id?: string;
    context_sha256: string;
    state_snapshot_id: string;
    model: string;
    retrieval: Retrieval;
    quote_validated: boolean;
    duration_ms: number;
    validation_attempts: string[];
  };
}
export interface Conversation {
  current_question_id: string | null;
  conversation_id: string;
  learner_id: string;
  topic_key: string;
  title: string;
  status: "active" | "needs_question" | "completed";
  version: number;
  created_at: string;
  updated_at: string;
  answered_count: number;
  last_error: string | null;
  generation_failure_code?: string | null;
  source_mode: string;
}
export interface SourceDocument {
  document_id: string;
  kind: string;
  title: string;
  character_count: number;
}
export interface AssessmentSession {
  pending_submission?: { trace_id: string; question_id: string; response_text: string; status: string; created_at: string } | null;
  conversation: Conversation;
  definition: {
    title: string;
    overview: string;
    concepts: ConceptDefinition[];
  };
  state: TopicState;
  turns: AssessmentTurn[];
  active_question: AssessmentQuestion | null;
  sources: SourceDocument[];
}
export interface ConceptProgress {
  concept_id: string;
  name: string;
  before_estimate: number;
  after_estimate: number;
  conversation_delta: number;
  uncertainty: number;
  observations: number;
  informative_observations: number;
  evidence_of_progress: boolean;
  missing_criteria: string[];
  hypotheses: Hypothesis[];
  supporting_quotes: { quote: string; evidence_id: string; category: string }[];
}
export interface ConversationReportData {
  report_id: string;
  generated_at: string;
  conversation: Conversation;
  source_documents: SourceDocument[];
  answered_count: number;
  concept_progress: ConceptProgress[];
  remaining_gaps: ConceptProgress[];
  unassessed_concepts: { concept_id: string; name: string; reason: string }[];
  evidence_of_progress: boolean;
  timeline: (AssessmentTurn & {
    next_question: AssessmentQuestion | null;
    next_question_reason: string;
  })[];
  interpretation: string;
  state_method: Record<string, string>;
}
