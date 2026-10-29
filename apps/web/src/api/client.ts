import {
  CognitiveEvidence,
  LearnerStateSnapshot,
  StateChangeEvent,
  ProcessInteractionResponse,
  DatasetAuditReport,
  BenchmarkResults,
  ChatResponse,
  LearnerAuditReport,
} from '../types';

const API_BASE = '/api/v1';

export async function sendChatMessage(payload: {
  learner_id: string;
  message: string;
  item_id?: string;
  item_text?: string;
  selected_option?: string;
  chat_history?: { role: string; content: string }[];
}): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`Chat API error: ${res.statusText}`);
  }
  return res.json();
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
  const res = await fetch(`${API_BASE}/interactions/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.statusText}`);
  }
  return res.json();
}

export async function getLearnerState(learnerId: string): Promise<LearnerStateSnapshot> {
  const res = await fetch(`${API_BASE}/learners/${learnerId}/state`);
  if (!res.ok) throw new Error(`Failed to load learner state: ${res.statusText}`);
  return res.json();
}

export async function getLearnerEvidence(learnerId: string): Promise<CognitiveEvidence[]> {
  const res = await fetch(`${API_BASE}/learners/${learnerId}/evidence`);
  if (!res.ok) throw new Error(`Failed to load evidence ledger: ${res.statusText}`);
  return res.json();
}

export async function getLearnerEvents(learnerId: string): Promise<StateChangeEvent[]> {
  const res = await fetch(`${API_BASE}/learners/${learnerId}/events`);
  if (!res.ok) throw new Error(`Failed to load state events: ${res.statusText}`);
  return res.json();
}

export async function replayLearnerTrajectory(learnerId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/learners/${learnerId}/replay`, { method: 'POST' });
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

export async function listLearners(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/learners`);
  if (!res.ok) throw new Error(`Failed to load learners: ${res.statusText}`);
  return res.json();
}

export async function getLearnerAuditReport(learnerId: string): Promise<LearnerAuditReport> {
  const res = await fetch(`${API_BASE}/learners/${learnerId}/audit_report`);
  if (!res.ok) throw new Error(`Failed to load audit report: ${res.statusText}`);
  return res.json();
}
