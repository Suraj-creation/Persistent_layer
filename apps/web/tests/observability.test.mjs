import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

test('trace filters are encoded and cancellation reaches the HTTP request', async () => {
  const source = readFileSync(new URL('../src/api/observability.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext}}).outputText;
  const api = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
  let received;
  const previous = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    received = {url, options};
    return {ok: true, json: async () => ({items: []})};
  };
  try {
    const controller = new AbortController();
    await api.listExecutions('profile', {conversation_id: 'a/b & c', operation: 'answer', before: 42}, controller.signal);
    const url = new URL(received.url, 'http://localhost');
    assert.equal(url.searchParams.get('conversation_id'), 'a/b & c');
    assert.equal(url.searchParams.get('before'), '42');
    assert.equal(url.searchParams.get('learner_id'), 'profile');
    assert.equal(received.options.signal, controller.signal);
    globalThis.fetch = async () => ({ok: false, status: 404, json: async () => ({detail: 'Execution not found for this learner.'})});
    await assert.rejects(api.getExecution('profile', 'other'), /Execution not found/);
  } finally { globalThis.fetch = previous; }
});
