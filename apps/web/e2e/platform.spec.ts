import { expect, test, type Page } from '@playwright/test';

test('improved workspace stays the default and keeps its styling after visiting training', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => localStorage.setItem('learner-state:guide:v2', 'seen'));
  await page.goto('/');
  await expect(page.getByLabel('Topic to assess')).toBeVisible();
  const workspaceStyles = () => page.locator('.sidebar').evaluate(element => {
    const style = getComputedStyle(element);
    return { background: style.backgroundColor, font: style.fontFamily, width: style.width };
  });
  const original = await workspaceStyles();
  await page.getByRole('button', { name: 'Training & retrieval', exact: true }).click();
  await page.getByRole('button', { name: 'Researcher', exact: true }).click();
  await expect(page.getByText('Training results, methods, and artifacts')).toBeVisible();
  await page.getByRole('button', { name: 'Learning workspace', exact: true }).click();
  await expect(page.getByLabel('Topic to assess')).toBeVisible();
  expect(await workspaceStyles()).toEqual(original);
  expect(new URL(page.url()).hash).toBe('');
  await page.reload();
  await expect(page.getByLabel('Topic to assess')).toBeVisible();
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  }
  expect(errors).toEqual([]);
});

async function openTraining(page: Page) {
  await page.goto('/#training');
  await page.getByRole('button', { name: 'AI Tutor', exact: true }).click();
}

test('lab submission, evidence, diagnostic answer, and replay form one persistent flow', async ({ page, request }) => {
  await openTraining(page);
  await page.getByRole('button', { name: 'Lab', exact: true }).click();
  await page.getByRole('button', { name: 'Submit & Run Cognitive Intelligence Pipeline' }).click();
  await expect(page.getByText('Extracted Cognitive Evidence', { exact: true })).toBeVisible();
  await expect(page.getByText('Active Diagnostic Probe (EIG Selection)')).toBeVisible();
  const diagnosticResponse = page.waitForResponse(response => response.url().endsWith('/diagnostics/answer') && response.request().method() === 'POST');
  await page.getByRole('button', { name: /^B:/ }).click();
  expect((await diagnosticResponse).status()).toBe(200);
  await expect(page.getByRole('status')).toContainText('Diagnostic answer and updated evidence saved');
  const ledger = await (await request.get('/api/v1/learners/student_A/evidence')).json();
  expect(ledger).toHaveLength(2);
  const replay = await (await request.post('/api/v1/learners/student_A/replay')).json();
  expect(replay.matches_persisted_state).toBe(true);
});

test('chat restores from backend after reload and switching learner isolates the conversation', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await openTraining(page);
  await page.locator('header select').selectOption('student_B');
  await page.getByRole('textbox').fill('I added the top and bottom numbers.');
  await page.getByRole('textbox').press('Enter');
  await expect(page.getByText('What common unit could you use for these fractions?', { exact: false }).last()).toBeVisible();
  await expect(page.getByText(/^Curriculum sources/)).toBeVisible();
  await page.reload();
  await page.locator('header select').selectOption('student_B');
  await expect(page.getByText('I added the top and bottom numbers.', { exact: true })).toBeVisible();
  await page.locator('header select').selectOption('student_C');
  await expect(page.getByText('I added the top and bottom numbers.', { exact: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('researcher training, retrieval modes, no-match fallback, audit, and replay views work', async ({ page, request }) => {
  await request.post('/api/v1/interactions/process', { data: { learner_id: 'researcher_e2e', item_id: 'ITEM_FRAC_01', item_text: '1/2 + 1/3', response_text: 'I added the top and bottom numbers.', selected_option: '2/5', correctness: 0 } });
  await openTraining(page);
  await page.getByRole('button', { name: 'Researcher', exact: true }).click();
  await expect(page.getByText('Training results, methods, and artifacts')).toBeVisible();
  await expect(page.getByText('0.6639 ± 0.0048', { exact: true }).first()).toBeVisible();
  await page.screenshot({ path: '../../docs/verification/final-merge/training-desktop.png' });
  await page.getByRole('button', { name: 'Retrieval Explorer', exact: true }).click();
  await page.getByRole('button', { name: 'Retrieve sources' }).click();
  await expect(page.getByRole('status')).toContainText('sources retrieved');
  await expect(page.getByText('Retrieved relationships')).toBeVisible();
  await page.getByLabel('Retrieval mode', { exact: true }).selectOption('naive');
  await page.getByRole('button', { name: 'Retrieve sources' }).click();
  await expect(page.getByRole('status')).toContainText('naive mode');
  await page.getByLabel('Question or reasoning to retrieve').fill('quantum quasar astrophysics');
  await page.getByRole('button', { name: 'Retrieve sources' }).click();
  await expect(page.getByRole('status')).toContainText('The system abstains');
  await page.getByRole('button', { name: 'Evidence Ledger', exact: true }).click();
  await expect(page.getByText('Auditable Cognitive Evidence Ledger')).toBeVisible();
  await page.getByRole('button', { name: 'Learner Trajectories', exact: true }).click();
  await page.locator('main select').selectOption('researcher_e2e');
  await page.getByRole('button', { name: 'Deterministic Replay' }).click();
  await expect(page.getByText(/Replayed 1 interactions/)).toBeVisible();
  await page.getByRole('button', { name: 'Dataset Audit', exact: true }).click();
  await expect(page.getByText('Audit Notes:').first()).toBeVisible();
  await page.getByRole('button', { name: '3-Tier Benchmarks', exact: true }).click();
  await expect(page.getByText('Demonstration data only.')).toBeVisible();
});

test('retrieval failure is visible and research view fits a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openTraining(page);
  await page.getByRole('button', { name: 'Researcher', exact: true }).click();
  await page.getByRole('button', { name: 'Retrieval Explorer', exact: true }).click();
  await page.route('**/api/v1/retrieval/query', route => route.fulfill({ status: 503, body: '{}' }));
  await page.getByRole('button', { name: 'Retrieve sources' }).click();
  await expect(page.getByRole('alert')).toContainText('Could not retrieve curriculum sources');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});

for (const failure of ['transport', 'truncated-json']) {
test(`a lost ${failure} response retries the same event key and commits one learner update`, async ({ page, request }) => {
  await openTraining(page);
  await page.locator('header select').selectOption('student_C');
  const initialEvidence = await (await request.get('/api/v1/learners/student_C/evidence')).json();
  let firstKey: string | undefined;
  let calls = 0;
  await page.route('**/api/v1/chat', async route => {
    calls++;
    const key = route.request().headers()['idempotency-key'];
    if (calls === 1) {
      firstKey = key;
      await route.fetch(); // Commit on the backend, then simulate a lost response.
      if (failure === 'transport') await route.abort('failed');
      else await route.fulfill({ status: 200, contentType: 'application/json', body: '{' });
    } else {
      expect(key).toBe(firstKey);
      await route.continue();
    }
  });
  await page.getByRole('textbox').fill('I added the top and bottom numbers.');
  await page.getByRole('textbox').press('Enter');
  await expect(page.getByText('What common unit could you use for these fractions?', { exact: false }).last()).toBeVisible();
  await expect.poll(() => calls).toBe(2);
  const evidence = await (await request.get('/api/v1/learners/student_C/evidence')).json();
  expect(evidence).toHaveLength(initialEvidence.length + 1);
});
}
