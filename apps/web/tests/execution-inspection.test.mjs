import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';

async function helpers() {
  const source = readFileSync(new URL('../src/components/assessment/executionInspection.ts', import.meta.url), 'utf8');
  const compiled = ts.transpileModule(source, {compilerOptions: {module: ts.ModuleKind.ESNext}}).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(compiled).toString('base64')}`);
}

test('initial inspection prefers learner evidence over a newer generated question', async () => {
  const { initialExecution } = await helpers();
  assert.equal(initialExecution([{trace_id:'generation',operation:'generation'}, {trace_id:'answer',operation:'answer'}]), 'answer');
  assert.equal(initialExecution([{trace_id:'setup',operation:'create'}]), 'setup');
  assert.equal(initialExecution([]), null);
});

test('bounded telemetry markers never become tokens or invented complete output', async () => {
  const { inspectTokens, completedOutput } = await helpers();
  const tokens = inspectTokens([{token:'😀',start:0,end:1}, {omitted_items:200}, null]);
  assert.deepEqual(tokens, {items:[{token:'😀',start:0,end:1}], omitted:200});
  assert.equal(completedOutput([{key:'evidence',status:'failed',output:{claim:'rejected'}}], 'evidence'), undefined);
  assert.deepEqual(completedOutput([{key:'evidence',status:'completed',output:{claim:'valid'}}, {key:'evidence',status:'failed',output:{}}], 'evidence'), {claim:'valid'});
});
