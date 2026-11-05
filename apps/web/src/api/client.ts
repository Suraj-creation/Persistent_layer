import {
  CognitiveEvidence,
  LearnerStateSnapshot,
  StateChangeEvent,
  ProcessInteractionResponse,
  DatasetAuditReport,
  BenchmarkResults,
  TrainingResearchSummary,
  ChatResponse,
  ConversationTurn,
  RetrievalResult,
  LearnerAuditReport,
} from '../types';

const API_BASE = '/api/v1';

async function postEvent<T>(path: string, payload: unknown): Promise<T> {
  const requestKey = crypto.randomUUID();
  for (let attempt = 0; attempt < 2; attempt++) {
    let response: Response;
    try {
      response = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Idempotency-Key': requestKey },
        body: JSON.stringify(payload),
      });
    } catch (error) {
      if (attempt === 0) continue;
      throw error;
    }
    if (response.ok) {
      try { return await response.json(); }
      catch (error) { if (attempt === 0) continue; throw error; }
    }
    if (response.status >= 500 && attempt === 0) continue;
    const detail = await response.json().catch(() => ({}));
    throw new Error(typeof detail.detail === 'string' ? detail.detail : `Request failed (${response.status})`);
  }
  throw new Error('Service unavailable after retry');
}

export async function sendChatMessage(payload: {
  learner_id: string;
  message: string;
  item_id?: string;
  item_text?: string;
  selected_option?: string;
  chat_history?: { role: string; content: string }[];
}): Promise<ChatResponse> {
  return postEvent<ChatResponse>('/chat', payload);
}

export async function processInteraction(payload: {
  learner_id: string;
  item_id: string;
  item_text: string;
  item_concepts: string[];
  response_text: string;
  selected_option?: string;
  correctness: number;
  latency_seconds?: number;
}): Promise<ProcessInteractionResponse> {
  return postEvent<ProcessInteractionResponse>('/interactions/process', payload);
}

export async function getLearnerState(learnerId: string): Promise<LearnerStateSnapshot> {
  const res = await fetch(`${API_BASE}/learners/${encodeURIComponent(learnerId)}/state`);
  if (!res.ok) throw new Error(`Failed to load learner state: ${res.statusText}`);
  return res.json();
}

export async function getLearnerEvidence(learnerId: string): Promise<CognitiveEvidence[]> {
  const res = await fetch(`${API_BASE}/learners/${encodeURIComponent(learnerId)}/evidence`);
  if (!res.ok) throw new Error(`Failed to load evidence ledger: ${res.statusText}`);
  return res.json();
}

export async function getLearnerEvents(learnerId: string): Promise<StateChangeEvent[]> {
  const res = await fetch(`${API_BASE}/learners/${encodeURIComponent(learnerId)}/events`);
  if (!res.ok) throw new Error(`Failed to load state events: ${res.statusText}`);
  return res.json();
}

export async function replayLearnerTrajectory(learnerId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/learners/${encodeURIComponent(learnerId)}/replay`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to replay trajectory: ${res.statusText}`);
  return res.json();
}

export async function getDatasetAudits(): Promise<{
  eedi_diagnostic_questions: DatasetAuditReport;
  foundational_assistments: DatasetAuditReport;
}> {
  const res = await fetch(`${API_BASE}/audit`);
  if (!res.ok) throw new Error(`Failed to load audits: ${res.statusText}`);
  return res.json();
}

export async function getBenchmarkResults(): Promise<BenchmarkResults> {
  const res = await fetch(`${API_BASE}/benchmarks`);
  if (!res.ok) throw new Error(`Failed to load benchmarks: ${res.statusText}`);
  return res.json();
}

export async function getTrainingResearchSummary(): Promise<TrainingResearchSummary> {
  const res = await fetch(`${API_BASE}/training-report`);
  if (!res.ok) throw new Error(`Failed to load training report: ${res.statusText}`);
  return res.json();
}

export async function listLearners(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/learners`);
  if (!res.ok) throw new Error(`Failed to load learners: ${res.statusText}`);
  return res.json();
}

export async function getLearnerAuditReport(learnerId: string): Promise<LearnerAuditReport> {
  const res = await fetch(`${API_BASE}/learners/${encodeURIComponent(learnerId)}/audit_report`);
  if (!res.ok) throw new Error(`Failed to load audit report: ${res.statusText}`);
  return res.json();
}

export async function answerDiagnostic(payload: { learner_id: string; parent_interaction_id: string; item_id: string; option_key: string; explanation?: string }): Promise<ProcessInteractionResponse> {
  return postEvent<ProcessInteractionResponse>('/diagnostics/answer', payload);
}

export async function getConversation(learnerId: string): Promise<ConversationTurn[]> {
  const res = await fetch(`${API_BASE}/learners/${encodeURIComponent(learnerId)}/conversation`);
  if (!res.ok) throw new Error('Could not restore conversation');
  return res.json();
}

export async function queryKnowledge(query: string, mode: RetrievalResult['mode'] = 'mix', namespace = 'curriculum'): Promise<RetrievalResult> {
  const res = await fetch(`${API_BASE}/retrieval/query`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, mode, namespace }),
  });
  if (!res.ok) throw new Error(`Could not retrieve curriculum sources (${res.status})`);
  return res.json();
}
