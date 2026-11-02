// Telemetry is bounded and may contain omission markers instead of full values.
export function completedOutput(
  stages: {
    key: string;
    status: string;
    output: Record<string, unknown> | null;
  }[],
  key: string,
) {
  return (
    [...stages]
      .reverse()
      .find((stage) => stage.key === key && stage.status === "completed")
      ?.output ?? undefined
  );
}

export function initialExecution(
  items: { trace_id: string; operation: string }[],
) {
  return (
    (items.find((item) => item.operation === "answer") ?? items[0])?.trace_id ??
    null
  );
}

export function inspectTokens(values: unknown[]) {
  const items: { token: string; start: number; end: number }[] = [];
  let omitted = 0;
  for (const value of values) {
    if (!value || typeof value !== "object") continue;
    const token = value as Record<string, unknown>;
    if (typeof token.omitted_items === "number") omitted += token.omitted_items;
    if (
      typeof token.token === "string" &&
      typeof token.start === "number" &&
      typeof token.end === "number"
    )
      items.push({ token: token.token, start: token.start, end: token.end });
  }
  return { items, omitted };
}
