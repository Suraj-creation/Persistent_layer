import { useEffect, useState } from "react";
import { ArrowLeft, ArrowRight, X } from "lucide-react";

export function GuidedTour({
  active,
  hasSession,
  onClose,
  onTarget,
}: {
  active: boolean;
  hasSession: boolean;
  onClose: () => void;
  onTarget: (target: string) => void;
}) {
  const [step, setStep] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const steps = hasSession
    ? [
        {
          target: "question",
          title: "The guide asks; you explain.",
          text: "Each question is generated from your saved state and retrieved material. Explain your reasoning, including uncertainty.",
        },
        {
          target: "expectation",
          title: "Know what the question is checking.",
          text: "Open this to see the expected focus before answering. It describes the task without revealing the answer.",
        },
        {
          target: "inspector",
          title: "See how your state is built.",
          text: "Inspect concept estimates, exact quotes, missing criteria, and the saved-state decision behind the next question.",
        },
        {
          target: "reports",
          title: "Review the whole learning loop.",
          text: "Your report links questions, explanations, evidence, state calculations, and later checks of earlier gaps.",
        },
      ]
    : [
        {
          target: "topic",
          title: "Choose what you have already studied.",
          text: "Start with a topic you want to check. The platform builds a concept map and asks one focused question at a time.",
        },
        {
          target: "notes",
          title: "Ground the assessment in your material.",
          text: "Add notes if you have them. Without notes, the reference is model-authored and clearly labeled unverified.",
        },
        {
          target: "conversations",
          title: "Your real conversations live here.",
          text: "New conversations appear only after you start them. Return at any time; state and explanations are saved locally.",
        },
        {
          target: "reports",
          title: "Every conversation gets a report.",
          text: "Reports show evidence of progress and what still needs checking, with the original explanations behind each assessment.",
        },
      ];
  useEffect(() => {
    if (active) setStep(0);
  }, [active, hasSession]);
  useEffect(() => {
    if (!active) return;
    const target = steps[step].target;
    onTarget(target);
    setRect(null);
    const update = () => {
      const bounds = document
        .querySelector(`[data-guide="${target}"]`)
        ?.getBoundingClientRect();
      setRect(bounds?.width && bounds?.height ? bounds : null);
    };
    // The parent reveals the target pane before the next browser layout.
    const frame = requestAnimationFrame(() => {
      document
        .querySelector(`[data-guide="${target}"]`)
        ?.scrollIntoView({ block: "nearest" });
      update();
    });
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    const timer = window.setTimeout(update, 100);
    return () => {
      cancelAnimationFrame(frame);
      clearTimeout(timer);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [active, step, hasSession, onTarget]);
  if (!active || !rect) return null;
  const width = Math.min(326, window.innerWidth - 32);
  const left = Math.max(
    16,
    Math.min(rect.left, window.innerWidth - width - 16),
  );
  const top =
    rect.bottom + 220 < window.innerHeight
      ? rect.bottom + 12
      : Math.max(16, rect.top - 210);
  return (
    <>
      <div
        className="guide-highlight"
        style={{
          left: rect.left - 4,
          top: rect.top - 4,
          width: rect.width + 8,
          height: rect.height + 8,
        }}
      />
      <aside
        className="guide-tip"
        style={{ top, left, width, maxHeight: window.innerHeight - top - 16 }}
        aria-label="Workspace guide"
      >
        <div className="guide-tip-top">
          <span>
            Workspace guide · {step + 1} of {steps.length}
          </span>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close guide"
          >
            <X size={16} />
          </button>
        </div>
        <h3>{steps[step].title}</h3>
        <p>{steps[step].text}</p>
        <div className="guide-actions">
          <button
            className="text-button"
            disabled={step === 0}
            onClick={() => setStep((s) => s - 1)}
          >
            <ArrowLeft size={14} />
            Back
          </button>
          <button
            className="primary-button"
            onClick={() =>
              step === steps.length - 1 ? onClose() : setStep((s) => s + 1)
            }
          >
            {step === steps.length - 1 ? "Got it" : "Next"}
            <ArrowRight size={14} />
          </button>
        </div>
      </aside>
    </>
  );
}
