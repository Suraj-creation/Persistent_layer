export interface SavedDraft {
  questionId: string;
  questionPrompt: string;
  text: string;
}
type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function readDrafts(
  conversationId: string,
  storage: DraftStorage = localStorage,
): SavedDraft[] {
  const drafts = new Map<string, SavedDraft>();
  try {
    const records: SavedDraft[] = JSON.parse(
      storage.getItem(`learner-state:drafts:${conversationId}`) || "[]",
    );
    for (const record of records)
      if (
        record.questionId &&
        typeof record.text === "string" &&
        record.text.trim()
      )
        drafts.set(record.questionId, record);
  } catch {
    /* An invalid legacy cache cannot erase intact question-scoped records. */
  }
  try {
    const legacy = JSON.parse(
      storage.getItem(`learner-state:draft:${conversationId}`) || "null",
    );
    if (
      legacy?.questionId &&
      typeof legacy.text === "string" &&
      legacy.text.trim() &&
      !drafts.has(legacy.questionId)
    )
      drafts.set(legacy.questionId, {
        questionId: legacy.questionId,
        questionPrompt: legacy.questionPrompt || "Earlier question",
        text: legacy.text,
      });
  } catch {
    /* Keep valid records even if the old single-draft slot is corrupt. */
  }
  return [...drafts.values()];
}

export function saveDraft(
  conversationId: string,
  draft: SavedDraft,
  storage: DraftStorage = localStorage,
): SavedDraft[] {
  const records = readDrafts(conversationId, storage).filter(
    (item) => item.questionId !== draft.questionId,
  );
  if (draft.text.trim()) {
    records.push(draft);
    storage.setItem(
      `learner-state:draft:${conversationId}`,
      JSON.stringify(draft),
    );
  } else {
    try {
      if (
        JSON.parse(
          storage.getItem(`learner-state:draft:${conversationId}`) || "null",
        )?.questionId === draft.questionId
      )
        storage.removeItem(`learner-state:draft:${conversationId}`);
    } catch {
      /* The question-scoped list remains authoritative. */
    }
  }
  storage.setItem(
    `learner-state:drafts:${conversationId}`,
    JSON.stringify(records),
  );
  return records;
}
