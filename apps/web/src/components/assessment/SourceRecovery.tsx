import { useState } from "react";
import { BookOpen, ArrowRight } from "lucide-react";

export type SourceMode = "study_notes" | "model_reference";

export function SourceRecovery({ busy, onUpdate }: {
  busy: boolean;
  onUpdate: (mode: SourceMode, notes?: string) => void;
}) {
  const [notes, setNotes] = useState("");
  return (
    <details className="source-recovery">
      <summary><BookOpen size={16} /> Update study material</summary>
      <p>Supply subject material for this conversation, or use a model-authored reference. Your saved explanations and estimates stay intact.</p>
      <form onSubmit={(event) => { event.preventDefault(); onUpdate("study_notes", notes); }}>
        <label htmlFor="replacement-notes">Replacement study notes</label>
        <textarea id="replacement-notes" value={notes} onChange={(event) => setNotes(event.target.value)} rows={4} maxLength={60000} disabled={busy} placeholder="Paste definitions, explanations, or textbook passages about your topic." required />
        <small>Include at least 8 words of subject material. A grade or class label alone is not enough.</small>
        <div className="button-row">
          <button className="primary-button" disabled={busy || !notes.trim()}>Save notes & generate <ArrowRight size={15} /></button>
          <button type="button" className="secondary-button" disabled={busy} onClick={() => onUpdate("model_reference")}>Use model reference instead</button>
        </div>
      </form>
      <small>The model reference is explicitly unverified. It may contain errors; use your course material when available.</small>
    </details>
  );
}
