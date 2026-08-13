import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/workshop/",
  plugins: [react()],
  build: {
    assetsInlineLimit: 0,
    emptyOutDir: true,
    outDir: "../src/kai/workshop/static",
    rollupOptions: {
      output: {
        assetFileNames: "app[extname]",
        chunkFileNames: "chunks/[name]-[hash].js",
        entryFileNames: "app.js",
      },
    },
    target: "es2022",
  },
  test: {
    clearMocks: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
  },
});
