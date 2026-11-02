import {
  AssessmentSession,
  Conversation,
  ConversationReportData,
} from "../types/assessment";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/v2${path}`, options);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      "The local API is unavailable. Start the backend on port 8000, then retry.",
    );
  }
  const data = await response.json().catch(() => null);
  if (!response.ok)
    throw new Error(
      typeof data?.detail === "string"
        ? data.detail
        : `The request could not be completed (${response.status}). Check your input and retry.`,
    );
  return data;
}
const post = <T>(path: string, payload: unknown) =>
  request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
export const getStatus = () =>
  request<{
    model_configured: boolean;
    model: string | null;
    retrieval: string;
  }>("/status");
export const conversations = (learner: string, signal?: AbortSignal) =>
  request<Conversation[]>(
    `/conversations?learner_id=${encodeURIComponent(learner)}`,
    { signal },
  );
export const loadConversation = (
  id: string,
  learner: string,
  signal?: AbortSignal,
) =>
  request<AssessmentSession>(
    `/conversations/${encodeURIComponent(id)}?learner_id=${encodeURIComponent(learner)}`,
    { signal },
  );
export const startConversation = (payload: {
  learner_id: string;
  topic: string;
  notes: string;
  request_id: string;
}) => post<AssessmentSession>("/conversations", payload);
export const submitAnswer = (
  id: string,
  payload: {
    learner_id: string;
    question_id: string;
    response_text: string;
    request_id: string;
    finish_after_answer?: boolean;
  },
) =>
  post<AssessmentSession>(
    `/conversations/${encodeURIComponent(id)}/answers`,
    payload,
  );
export const nextQuestion = (id: string, learner: string) =>
  post<AssessmentSession>(`/conversations/${encodeURIComponent(id)}/next`, {
    learner_id: learner,
  });
export const updateSources = (
  id: string,
  learner: string,
  mode: "study_notes" | "model_reference",
  notes = "",
) => post<AssessmentSession>(`/conversations/${encodeURIComponent(id)}/sources`, {
  learner_id: learner, mode, notes,
});
export const finishConversation = (id: string, learner: string) =>
  post<AssessmentSession>(`/conversations/${encodeURIComponent(id)}/finish`, {
    learner_id: learner,
  });
export const resumeConversation = (id: string, learner: string) =>
  post<AssessmentSession>(`/conversations/${encodeURIComponent(id)}/resume`, {
    learner_id: learner,
  });
export const loadReport = (id: string, learner: string, signal?: AbortSignal) =>
  request<ConversationReportData>(
    `/conversations/${encodeURIComponent(id)}/report?learner_id=${encodeURIComponent(learner)}`,
    { signal },
  );
