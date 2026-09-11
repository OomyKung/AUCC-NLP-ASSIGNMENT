import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The FastAPI backend. Override with VITE_API_TARGET if you run it elsewhere.
const API_TARGET = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    // Listen on every interface. Without this Vite binds only to IPv6 [::1],
    // so http://127.0.0.1:5173 is refused -- and it also makes the dev server
    // reachable from a phone for real responsive testing.
    host: true,
    proxy: {
      // Captured video frames are served by FastAPI from the data directory.
      // Without this they resolve against the dev server, which answers every
      // unknown path with index.html for SPA routing -- so an <img> receives
      // HTML and renders as a broken image rather than a 404 anyone notices.
      '/media': {
        target: API_TARGET,
        changeOrigin: true,
      },
      // Proxy API calls to FastAPI so the browser sees a single origin in dev.
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        // Without this, a stopped backend surfaces as an opaque socket error.
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            const hint =
              `Cannot reach the backend at ${API_TARGET}. ` +
              'Start it with:  cd backend && uvicorn app.main:app --reload'
            console.error(`\n[api proxy] ${err.message}\n[api proxy] ${hint}\n`)
            // res is a ServerResponse for HTTP requests, a Socket for upgrades.
            if ('writeHead' in res && !res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'application/json' })
              res.end(JSON.stringify({ detail: hint }))
            }
          })
        },
      },
    },
  },
})
