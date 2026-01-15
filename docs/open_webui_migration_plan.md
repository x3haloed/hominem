# Open WebUI Migration Plan (Replace hominem-ui)

Goal: delete `apps/hominem-ui` and run Open WebUI as the primary UI, with `hominem_agent`
as the OpenAI-compatible backend and `hominem_infer` as the inference engine. Use
OpenAI-style tool calling (no MCP).

## Scope

- UI replacement: Open WebUI handles chat UX, persistence, and admin tooling.
- Backend compatibility: `hominem_agent` becomes an OpenAI-compatible provider.
- Inference: `hominem_infer` remains the model runtime.
- Tool calling: OpenAI-style `tools` + `tool_calls` end-to-end.

Non-goals:
- Re-implement Open WebUI UI features inside `hominem-ui`.
- MCP tool server integration (explicitly out of scope).

## Plan Overview

1) Remove `apps/hominem-ui` and update docs/launcher scripts.
2) Decide how Open WebUI is included/pinned in the repo.
3) Add OpenAI-compatible API surface to `hominem_agent`.
4) Ensure tool execution and streaming match Open WebUI expectations.
5) Configure Open WebUI to point to `hominem_agent`.
6) Validate tool calling + artifacts + streaming in Open WebUI.

## Removal of Current UI

- Delete `apps/hominem-ui`.
- Remove any build/deploy references in `README.md`, scripts, or config that mention
  `hominem-ui` or Vite.
- Update `docs/agent_ui.md` to point to Open WebUI instead of local SPA.
- If `apps/serve` shells the UI in any way, simplify it to just run the API servers.

## `hominem_agent` Changes (OpenAI-Compatible Provider)

Implement OpenAI-compatible endpoints in `hominem_agent`:

- `GET /v1/models`
  - Return a list with the active model id(s) that route to `hominem_infer`.
- `POST /v1/chat/completions`
  - Accept OpenAI Chat Completions payloads (including `tools`, `tool_choice`,
    `stream`, and `stream_options`).
  - Return OpenAI-compliant responses.
  - If `stream=true`, return SSE (`text/event-stream`) with OpenAI delta chunks.

Key mapping and behavior:

- Tool calling:
  - Accept OpenAI `tools` in requests.
  - Ensure Qwen-Agent is configured with `use_raw_api=True` (already in place).
  - Execute tools within `hominem_agent`, return `tool_calls` in assistant messages.
- Reasoning content:
  - If `reasoning_content` is needed, embed in `assistant` content or a structured
    `metadata` field, but stay compatible with OpenAI response schema.
- Session and memory:
  - Open WebUI tracks chat history; `hominem_agent` should accept full `messages`
    and remain stateless per request, or persist in-memory only if needed.

Implementation approach:

- Remove any and all `/api/chat` and `/api/chat/stream` surface area that is not conformant to the new system.
- Do not keep legacy compatibility routes; the OpenAI-compatible API is the only supported surface.

## Open WebUI Source + Pinning Strategy

Pick one and document it clearly in repo tooling so upgrades are intentional:

- Python dependency (simplest):
  - Add a pinned version to `setup.py` (or `pyproject.toml` if we move) under extras or
    runtime deps: `open-webui==<version>`.
  - Use `open-webui serve` as an external process in deployment docs/scripts.

[removed bad options]

## Logic Relocation / Short-Circuiting

We should minimize logic duplication and align with Open WebUI conventions:

- Move UI-specific logic out of `hominem_agent` (if any) and keep it purely API.
- Keep tool orchestration inside `hominem_agent`; Open WebUI will call a single
  OpenAI-style backend.
- If any of the `apps/serve` logic wraps the UI, simplify it to run
  `hominem_infer` and `hominem_agent` only.
- Prefer Open WebUI for:
  - chat persistence
  - message rendering
  - artifact display (files/markdown)
  - settings UI

## Open WebUI Configuration

- Add `hominem_agent` as an OpenAI provider (base URL + optional API key).
- Ensure CORS and auth headers are compatible with Open WebUI's outgoing requests.
- Confirm SSE streaming and payload shapes align with Open WebUI's `/chat/completions`
  expectations.

## Validation Checklist

- Chat works with multi-turn context from Open WebUI.
- Tool calling flows: Open WebUI -> `hominem_agent` -> tools -> assistant/tool messages.
- Streaming responses appear correctly in Open WebUI.
- File/artifact attachments render as expected in Open WebUI (markdown + files).

## Work That Would Require a Fork

These are Open WebUI changes that are not supported via pipelines/tools today:

- Custom UI panels (e.g., a dedicated "computer use" control surface).
- Non-standard message renderers beyond markdown/attachments/tool calls.
- Pipeline-specific bespoke UI pages (beyond valves/config forms).
- Deep changes to chat layout or message timeline semantics.

If any of the above are required, plan on a fork or upstream PR.
