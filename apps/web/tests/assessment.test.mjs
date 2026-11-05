import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

test('source recovery sends only the selected conversation, learner and material', async () => {
  const source = readFileSync(new URL('../src/api/assessment.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext}}).outputText;
  const api = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
  const previous = globalThis.fetch;
  const requests = [];
  globalThis.fetch = async (url, options) => {
    requests.push({url, options});
    return {ok: true, json: async () => ({active_question: {question_id: 'new-question'}})};
  };
  try {
    const result = await api.updateSources('conversation/a', 'profile', 'study_notes', 'Original notes\nwith their formatting.');
    assert.equal(result.active_question.question_id, 'new-question');
    assert.equal(requests[0].url, '/api/v2/conversations/conversation%2Fa/sources');
    assert.equal(requests[0].options.method, 'POST');
    assert.deepEqual(JSON.parse(requests[0].options.body), {learner_id: 'profile', mode: 'study_notes', notes: 'Original notes\nwith their formatting.'});
    await api.updateSources('conversation/a', 'profile', 'model_reference');
    assert.deepEqual(JSON.parse(requests[1].options.body), {learner_id: 'profile', mode: 'model_reference', notes: ''});
    globalThis.fetch = async () => ({ok: false, status: 422, json: async () => ({detail: 'Add at least 8 words of study material.'})});
    await assert.rejects(api.updateSources('conversation/a', 'profile', 'study_notes', 'Class 6th'), /8 words of study material/);
  } finally { globalThis.fetch = previous; }
});
