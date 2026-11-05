import { defineConfig } from '@playwright/test';
import path from 'node:path';

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  retries: 0,
  timeout: 30000,
  use: { baseURL: 'http://127.0.0.1:5193', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  reporter: [['list'], ['json', { outputFile: 'test-results/results.json' }]],
  webServer: [
    {
      command: '.venv/bin/python -m uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8013',
      cwd: path.resolve(import.meta.dirname, '../..'),
      url: 'http://127.0.0.1:8013/api/v1/health',
      env: { LEARNER_STATE_DB_PATH: `/tmp/persistent-layer-e2e-${process.pid}.db`, EVIDENCE_EXTRACTOR_MODE: 'rules' },
      reuseExistingServer: false,
    },
    {
      command: 'npm run dev -- --host 127.0.0.1 --port 5193 --strictPort',
      url: 'http://127.0.0.1:5193',
      env: { API_PROXY_TARGET: 'http://127.0.0.1:8013' },
      reuseExistingServer: false,
    },
  ],
});
