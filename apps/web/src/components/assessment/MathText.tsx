import { lazy, Suspense } from "react";
import { RenderBoundary } from "../RenderBoundary";

const MathRenderer = lazy(() => import("./MathRenderer").then(m => ({ default: m.MathText })));

/** Plain prose never downloads the math engine. Keep readable text while it loads. */
export function MathText({ text }: { text: string }) {
  if (!/\\[([]|\$/.test(text)) return <>{text}</>;
  return <RenderBoundary fallback={<>{text}</>}><Suspense fallback={<>{text}</>}><MathRenderer text={text} /></Suspense></RenderBoundary>;
}
