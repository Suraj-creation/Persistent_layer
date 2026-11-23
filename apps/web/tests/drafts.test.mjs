import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import ts from "typescript";

const source = readFileSync(
  new URL("../src/components/assessment/drafts.ts", import.meta.url),
  "utf8",
);
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { readDrafts, saveDraft } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);
function memory() {
  const values = new Map();
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

test("editing a new question cannot overwrite a preserved earlier draft", () => {
  const storage = memory();
  saveDraft(
    "conversation",
    {
      questionId: "old",
      questionPrompt: "Original question?",
      text: "My unfinished reasoning.",
    },
    storage,
  );
  saveDraft(
    "conversation",
    {
      questionId: "new",
      questionPrompt: "A different question?",
      text: "New reasoning.",
    },
    storage,
  );
  assert.deepEqual(
    readDrafts("conversation", storage).map((d) => [d.questionId, d.text]),
    [
      ["old", "My unfinished reasoning."],
      ["new", "New reasoning."],
    ],
  );
});

test("submitting or clearing one question removes only that question draft", () => {
  const storage = memory();
  saveDraft(
    "conversation",
    { questionId: "old", questionPrompt: "Original?", text: "Keep this." },
    storage,
  );
  saveDraft(
    "conversation",
    { questionId: "new", questionPrompt: "New?", text: "Submit this." },
    storage,
  );
  saveDraft(
    "conversation",
    { questionId: "new", questionPrompt: "New?", text: "" },
    storage,
  );
  assert.deepEqual(
    readDrafts("conversation", storage).map((d) => d.questionId),
    ["old"],
  );
});

test("legacy draft survives migration, a new question, and a browser reload", () => {
  const storage = memory();
  storage.setItem(
    "learner-state:draft:conversation",
    JSON.stringify({ questionId: "old", text: "Legacy draft." }),
  );
  assert.equal(readDrafts("conversation", storage)[0].text, "Legacy draft.");
  saveDraft(
    "conversation",
    { questionId: "new", questionPrompt: "New?", text: "New draft." },
    storage,
  );
  assert.equal(
    readDrafts("conversation", storage).find((d) => d.questionId === "old")
      .text,
    "Legacy draft.",
  );
  assert.deepEqual(readDrafts("another-conversation", storage), []);
});
