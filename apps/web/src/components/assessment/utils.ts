export const readable = (text: string) => text.replace(/_/g, " ");
export const displayProse = (text: string) =>
  text.replace(/\b(?:chunk|source)_[a-f0-9]{32}\b/gi, "the cited source");
export const percent = (number: number) => `${Math.round(number * 100)}%`;
export const displayDate = (value: string) =>
  new Date(value).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
export function exportJSON(value: unknown, filename: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
