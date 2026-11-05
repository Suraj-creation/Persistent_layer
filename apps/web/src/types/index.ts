export interface ConceptLink {
  concept_id: string;
  concept_name: string;
  confidence: number;
  grounding_rationale: string;
}

export interface CandidateInterpretation {
  hypothesis_id: string;
  category: 'mastery' | 'misconception' | 'prerequisite_gap' | 'calculation_slip' | 'guessing' | 'ambiguous';
  description: string;
  hypothesis_confidence: number;
}

export interface CognitiveEvidence {
  evidence_id: string;
  interaction_id: string;
  learner_id: string;
  timestamp: string;
  source_type: string;
  source_quote: string;
  is_verbatim_quote: boolean;
  extracted_claim: string;
  polarity: 'supports' | 'contradicts' | 'neutral' | 'ambiguous';
  expressed_uncertainty: 'none' | 'low' | 'medium' | 'high' | 'explicitly_stated_doubt';
  concept_links: ConceptLink[];
  candidate_interpretation: CandidateInterpretation;
  extractor_model: string;
  prompt_version: string;
  schema_version: string;
  review_status: string;
}

export interface MisconceptionBelief {
  misconception_id: string;
  misconception_name: string;
  posterior_probability: number;
  supporting_evidence_ids: string[];
  contradicting_evidence_ids: string[];
  status: 'suspected' | 'confirmed' | 'resolved' | 'dismissed';
  last_updated: string;
}

export interface ConceptState {
  concept_id: string;
  concept_name: string;
  alpha?: number;
  beta?: number;
  mastery_estimate: number;
  uncertainty: number;
  evidence_sufficiency: 'insufficient' | 'sparse' | 'moderate' | 'sufficient';
  candidate_misconceptions: Record<string, MisconceptionBelief>;
  last_interaction_timestamp?: string;
  interaction_count: number;
}

export interface LearnerStateSnapshot {
  snapshot_id: string;
  learner_id: string;
  as_of_interaction_id: string;
  timestamp: string;
  concept_states: Record<string, ConceptState>;
  active_prerequisite_gaps: string[];
  policy_version: string;
}

export interface StateChangeEvent {
  event_id: string;
  learner_id: string;
  concept_id: string;
  prior_snapshot_id?: string;
  new_snapshot_id: string;
  triggering_evidence_id: string;
  delta_mastery: number;
  prior_uncertainty?: number;
  new_uncertainty?: number;
  prior_misconception_prob?: number;
  new_misconception_prob?: number;
  change_summary: string;
  applied_rule: string;
  math_trace?: Record<string, any>;
  timestamp: string;
}

export interface DiagnosticOption {
  option_key: string;
  option_text: string;
  is_correct: boolean;
  targeted_misconception_id?: string;
  expected_probability_given_misconception: number;
  expected_probability_given_mastery: number;
}

export interface DiagnosticItem {
  item_id: string;
  concept_id: string;
  question_text: string;
  options: DiagnosticOption[];
  diagnostic_rationale: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  evidence?: CognitiveEvidence;
  events?: StateChangeEvent[];
  diagnostic_probe?: DiagnosticItem;
  diagnostic_rationale?: string;
  selected_option?: string;
}

export interface ChatResponse {
  reply: string;
  evidence: CognitiveEvidence;
  new_state: LearnerStateSnapshot;
  events: StateChangeEvent[];
  recommended_diagnostic_probe?: DiagnosticItem;
  diagnostic_rationale?: string;
}

export interface ProcessInteractionResponse {
  interaction: any;
  evidence: CognitiveEvidence;
  new_state: LearnerStateSnapshot;
  events: StateChangeEvent[];
  recommended_diagnostic_probe?: DiagnosticItem;
  diagnostic_rationale?: string;
}

export interface DatasetAuditReport {
  dataset_name: string;
  total_records: number;
  total_students: number;
  total_items: number;
  total_concepts: number;
  text_availability_rate: number;
  misconception_label_rate: number;
  missingness_by_column: Record<string, number>;
  concept_distribution: Record<string, number>;
  leakage_risk_flags: string[];
  suitability_verdict: string;
  notes: string[];
}

export interface BenchmarkResults {
  dataset: string;
  sample_size: number;
  data_provenance?: string;
  research_validity?: string;
  caveats?: string[];
  tiers: {
    tier_1_correctness_only_pybkt: {
      name: string;
      description: string;
      metrics: {
        auc: number;
        accuracy: number;
        brier_score: number;
        ece: number;
        misconception_precision: number;
        misconception_recall: number;
        misconception_f1: number;
      };
    };
    tier_2_language_aware_baseline: {
      name: string;
      description: string;
      metrics: {
        auc: number;
        accuracy: number;
        brier_score: number;
        ece: number;
        misconception_precision: number;
        misconception_recall: number;
        misconception_f1: number;
      };
    };
    tier_3_proposed_persistent_state: {
      name: string;
      description: string;
      metrics: {
        auc: number;
        accuracy: number;
        brier_score: number;
        ece: number;
        misconception_precision: number;
        misconception_recall: number;
        misconception_f1: number;
      };
    };
  };
  comparison_delta_tier3_vs_tier1: {
    auc_delta: number;
    accuracy_delta: number;
    ece_reduction: number;
    misconception_f1_gain: number;
  };
}

export interface TrainingArtifactMetadata {
  name: string;
  path: string;
  type: string;
  available: boolean;
  expected_sha256?: string | null;
  actual_sha256?: string | null;
  checksum_verified: boolean;
}

export interface TrainingMetricSummary {
  mean_across_splits: number;
  sd_across_splits: number;
}

export interface TrainingResearchSummary {
  status: 'available' | 'partial';
  purpose: string;
  runtime_boundary: {
    bkt_used_for_live_learner_state: boolean;
    map_classifier_trained_or_used: boolean;
    eedi_graph_used_for_runtime_retrieval: boolean;
    live_state_method: string;
    notice: string;
  };
  assistments_bkt: {
    status: string;
    dataset?: string;
    source_url?: string;
    task: string;
    model: string;
    target: string;
    skill_filter?: string;
    split_design?: string;
    test_ratio?: number;
    data_audit: Record<string, number | string>;
    primary_split: Record<string, number>;
    primary_split_metrics: Record<string, number>;
    summary_across_splits: Array<{
      name: string;
      metrics: Record<string, TrainingMetricSummary>;
    }>;
    paired_bkt_minus_baselines: Record<string, Record<string, number>>;
    seeds: Array<{
      seed: number;
      train_rows: number;
      test_rows: number;
      train_students: number;
      test_students: number;
      models: Record<string, Record<string, number>>;
      artifacts: TrainingArtifactMetadata[];
    }>;
    skill_metrics: Record<string, Record<string, number>>;
    skill_parameters: Array<{
      skill: string;
      by_seed: Record<string, Record<string, number>>;
    }>;
    artifacts: TrainingArtifactMetadata[];
    data_sha256?: string;
    trained_at_utc?: string;
    evaluated_at_utc?: string;
    versions: Record<string, string>;
    limitations: string[];
    calibration_note?: string;
  };
  eedi_graph: {
    status: string;
    dataset?: string;
    revision?: string;
    license?: string;
    attribution?: string;
    source_url?: string;
    row_counts: Record<string, number>;
    unique_counts: Record<string, number>;
    weight_validation: Record<string, boolean | number>;
    file_sha256: Record<string, string>;
    data_role?: string;
    runtime_integrated: boolean;
  };
  map_classifier: {
    status: string;
    model_available: boolean;
    reason: string;
    method: string;
    planned_validation: string;
    planned_metrics: string[];
    runtime_integrated: boolean;
  };
  other_dataset_status: Array<{
    dataset: string;
    status: string;
    role: string;
    limitation: string;
  }>;
  related_reports: string[];
  warnings: string[];
}

export interface LearnerAuditReport {
  learner_id: string;
  report_markdown: string;
  total_interactions: number;
  total_evidence: number;
  total_audit_events: number;
  snapshot: LearnerStateSnapshot;
}
