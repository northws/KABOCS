import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies `/api` to the FastAPI backend.
// Both `/api/**` JSON endpoints and the SSE stream route work over the
// proxy without extra configuration.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: false,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
