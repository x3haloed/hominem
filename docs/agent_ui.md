# Agent UI Quickstart

Goal: run a minimal loop `Open WebUI → hominem_agent → hominem_infer → hominem_agent → Open WebUI`.

## 1) Start `hominem_infer`

```bash
PYTHONPATH=python \
INFER_BACKEND=mlx_vlm \
INFER_HOST=127.0.0.1 \
INFER_PORT=8000 \
python3 -m hominem_infer
```

Optional:
- `INFER_MODEL_ID`: model to load (defaults in `python/hominem_infer/app.py`).

## 2) Start `hominem_agent` (OpenAI-compatible API)

```bash
PYTHONPATH=python \
HOMINEM_INFER_BASE_URL=http://127.0.0.1:8000/v1 \
HOMINEM_AGENT_HOST=127.0.0.1 \
HOMINEM_AGENT_PORT=8020 \
python3 -m hominem_agent
```

Open `http://127.0.0.1:8020/` for a quick health check.

## 3) Start Open WebUI

Set the OpenAI base URL to point at `hominem_agent`:
- Base URL: `http://127.0.0.1:8020/v1`
- API key: optional (can be blank)

## Notes

- Tool calling uses native OpenAI `tools`/`tool_calls` via Qwen-Agent `use_raw_api=True`.
- The agent includes minimal local file tools: `describe_file`, `extract_section`, `replace_section`.
