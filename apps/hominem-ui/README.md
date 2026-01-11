# hominem-ui

Vue + Tailwind static SPA client for `hominem-agent`.

## Dev

1. Start `hominem-agent` (exposes `POST /api/chat`).
2. In this folder:
   - `npm install`
   - `npm run dev`

Defaults:
- UI dev server: `http://127.0.0.1:5173`
- Agent proxy target: `http://127.0.0.1:8020`

Override the dev proxy target with:
- `VITE_AGENT_URL=http://127.0.0.1:8020 npm run dev`

## Production build

- `npm run build` (outputs `dist/`)
- Serve `dist/` with any static server (or behind a reverse proxy).

If you serve the SPA from a different origin than `hominem-agent`, enable CORS in the agent via:
- `HOMINEM_UI_CORS_ORIGINS=https://your-ui.example` (see `python/hominem_agent/app.py`)

You can also bake in an agent base URL at build time:
- `VITE_AGENT_BASE_URL=https://your-agent.example npm run build`
