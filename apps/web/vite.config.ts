import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
declare const process: { env: Record<string, string | undefined> };

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.API_PROXY_TARGET || 'http://127.0.0.1:8002',
        changeOrigin: true,
      },
    },
  },
});
