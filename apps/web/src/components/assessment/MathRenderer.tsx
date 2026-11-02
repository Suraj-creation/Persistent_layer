import { Fragment, useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/** Model-authored prose is escaped by React; only KaTeX's untrusted-mode output is HTML. */
export function MathText({ text }: { text: string }) {
  const segments = useMemo(() => {
    const result: { text: string; html?: string; display?: boolean }[] = [];
    // Explicit TeX delimiters are preferred. Single-dollar prices remain plain text.
    const pattern =
      /\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]|\$\$([\s\S]*?)\$\$|(?<!\\)\$([^$\n]+)\$/g;
    let start = 0;
    for (const match of text.matchAll(pattern)) {
      if (match.index! > start)
        result.push({ text: text.slice(start, match.index) });
      const expression = match[1] ?? match[2] ?? match[3] ?? match[4];
      const isPrice =
        match[4] !== undefined && !/[\\^_{}=+*/<>]|^[A-Za-z]$/.test(expression);
      if (isPrice) result.push({ text: match[0] });
      else {
        try {
          result.push({
            text: expression,
            display: match[2] !== undefined || match[3] !== undefined,
            html: katex.renderToString(expression, {
              trust: false,
              throwOnError: true,
              strict: "ignore",
              maxExpand: 1000,
              maxSize: 12,
              output: "htmlAndMathml",
              displayMode: match[2] !== undefined || match[3] !== undefined,
              macros: {},
            }),
          });
        } catch {
          result.push({ text: match[0] });
        }
      }
      start = match.index! + match[0].length;
    }
    if (start < text.length) result.push({ text: text.slice(start) });
    return result;
  }, [text]);
  return (
    <>
      {segments.map((segment, index) =>
        segment.html ? (
          <span
            key={index}
            className={
              segment.display ? "rendered-math display-math" : "rendered-math"
            }
            dangerouslySetInnerHTML={{ __html: segment.html }}
          />
        ) : (
          <Fragment key={index}>{segment.text}</Fragment>
        ),
      )}
    </>
  );
}
