import { useState } from "react";
import { Copy } from "lucide-react";
import { SavedDraft } from "./drafts";
import { MathText } from "./MathText";

export function SavedDrafts({ drafts }: { drafts: SavedDraft[] }) {
  const [copied, setCopied] = useState("");
  if (!drafts.length) return null;
  return (
    <section className="earlier-drafts">
      <details>
        <summary>
          {drafts.length === 1
            ? "An earlier unsent draft is still saved"
            : `${drafts.length} earlier unsent drafts are still saved`}
        </summary>
        <p>
          These drafts belong to their original questions and are not included
          in your graded evidence. You can recover their text without attaching
          an old answer to a different question.
        </p>
        {drafts.map((draft) => (
          <article key={draft.questionId}>
            <h3>
              <MathText text={draft.questionPrompt} />
            </h3>
            <blockquote>{draft.text}</blockquote>
            <button
              className="text-button"
              onClick={async () => {
                try {
                  await navigator.clipboard.writeText(draft.text);
                  setCopied(draft.questionId);
                } catch {
                  setCopied("unavailable");
                }
              }}
            >
              <Copy size={14} />
              {copied === draft.questionId ? "Copied" : "Copy draft text"}
            </button>
          </article>
        ))}
        {copied === "unavailable" && (
          <p role="status">Select the draft text above to copy it.</p>
        )}
      </details>
    </section>
  );
}
