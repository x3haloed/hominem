# Unified Theory Chat (Canonical Server)

Minimal FastAPI server that runs the unified-theory runtime (`apps/serve/agent_runtime.py`) and persists
conversation state + messages to SQLite.

This replaces the older WebSocket/hot-swap serving system (removed).

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up database:**
   ```bash
   cp env.example .env
   # Edit .env to set DATABASE_PATH
   ```

3. **Start the server:**
   ```bash
   python main.py
   ```

   Open `http://localhost:8000` for the UI.

## API

- `GET /health` – liveness check
- `POST /chat` – run one turn (returns assistant + metrics)
- `POST /sleep` – drain `sleep_queue` to JSONL and optionally run online update
- `GET /messages` – fetch last messages for a conversation

## Configuration

### Environment Variables (.env)
```bash
# Database
DATABASE_PATH=storage/conversations.db

# Base model (HF id or local path)
BASE_MODEL_PATH=Qwen/Qwen3-1.7B

# Optional LoRA adapter (absolute path or name under artifacts/lora/)
AUTO_LOAD_LORA=

# Server
PORT=8000

# Frozen head checkpoints
MANIFOLD_CKPT=
REGIME_CKPT=

# Sleep logging / optional online update
SLEEP_UPDATE_ENABLED=false
SLEEP_UPDATE_CONFIG=
SLEEP_LOG_DIR=
```

## Project Structure
```
apps/serve/
├── main.py           # FastAPI server + endpoints
├── agent_runtime.py  # Unified-theory runtime pipeline
├── database.py       # SQLite persistence (messages + state)
├── static/
│   └── index.html    # Minimal UI
└── requirements.txt  # Python dependencies
```
