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

export interface LearnerAuditReport {
  learner_id: string;
  report_markdown: string;
  total_interactions: number;
  total_evidence: number;
  total_audit_events: number;
  snapshot: LearnerStateSnapshot;
}
