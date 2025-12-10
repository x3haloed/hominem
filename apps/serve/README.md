# Hominem LoRA Serving System

Real-time chat interface for LoRA models with emotion labeling and hot-swapping capabilities.

## Features

- 🗣️ **Real-time Chat**: WebSocket-powered conversation interface
- 🎭 **Emotion Labeling**: UI for labeling conversations with 6-axis emotion manifold
- 🔄 **Hot-Swapping**: Load and switch between LoRA models without downtime
- 📊 **Data Collection**: Automatic logging of conversations and labels to SQLite
- 🎨 **Modern UI**: ChatGPT-style interface with conversation management

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

3. **Load a model (optional):**
   ```bash
   python load_initial_model.py
   ```

4. **Start the server:**
   ```bash
   python main.py
   ```

   This will automatically open your browser to `http://localhost:8000`

## Architecture

### Database Schema
- **conversations**: Conversation threads
- **messages**: Individual messages within conversations
- **emotion_labels**: 6-axis emotion manifold labels
- **reward_labels**: Behavioral dimension labels
- **synthetic_data**: Training data from other sources

### Model Management
- **Hot-swapping**: Load multiple model versions simultaneously
- **Background loading**: Models load asynchronously without blocking
- **Automatic switching**: Switch between models between requests
- **Memory management**: GPU memory cleanup when unloading models

### Data Flow
1. **User Input** → WebSocket → Model Inference → Token Streaming
2. **Emotion Labeling** → UI Reactions → Database Persistence
3. **Model Training** → Background Loading → Hot-Swap Activation

## Configuration

### Environment Variables (.env)
```bash
# Database
DATABASE_PATH=storage/conversations.db

# Auto-load LoRA model on startup
AUTO_LOAD_LORA=default  # or "qwen3.1-7b" or full path

# Base model path (auto-detected if empty)
BASE_MODEL_PATH=

# Server
PORT=8000
DEBUG=true
```

### Model Loading

#### Auto-Loading on Startup
Set `AUTO_LOAD_LORA` in your `.env` file to automatically load a LoRA model when the server starts:

```bash
# Load specific version
AUTO_LOAD_LORA=qwen3.1-7b-v2

# Load latest version automatically
AUTO_LOAD_LORA=qwen3.1-7b  # Finds qwen3.1-7b-v2 if qwen3.1-7b doesn't exist

# Load from absolute path
AUTO_LOAD_LORA=/path/to/my/lora
```

**Versioning Strategy:**
- Use descriptive names: `qwen3.1-7b-week1`, `qwen3.1-7b-week2`
- Or semantic versions: `qwen3.1-7b-v1`, `qwen3.1-7b-v2`
- The system automatically finds the latest version when you specify a base name

The system will:
1. Find the LoRA adapter in `artifacts/lora/{name}/`
2. Auto-detect the base model from the LoRA's `adapter_config.json`
3. Load the model in the background
4. Activate it automatically

#### Manual Loading
Models can also be loaded via:
- **Web UI**: Click the ⚙️ button in the sidebar
- **API**: `POST /api/models/load` with base_model_path and lora_path
- **Script**: `python load_initial_model.py`

## API Endpoints

### Conversations
- `GET /api/conversations` - List all conversations
- `GET /api/conversations/{id}` - Get conversation with messages
- `POST /api/conversations` - Create new conversation

### Messages
- `POST /api/messages` - Send message (adds to DB, triggers inference)
- `POST /api/messages/{conv_id}/{msg_idx}/emotion` - Add emotion label

### Models
- `GET /api/models` - Get loaded model status
- `POST /api/models/load` - Load model in background
- `POST /api/models/switch/{version_id}` - Switch active model
- `DELETE /api/models/{version_id}` - Unload model

### WebSocket
- `ws://localhost:8000/ws/chat/{conversation_id}` - Real-time chat

## Emotion Labeling System

The UI provides reaction buttons for labeling emotions:
- 😊 +2/+1 (valence)
- 🚀 (arousal)
- 💔 (predictive discrepancy)
- ⏳/🪞 (temporal directionality)
- 🤗/🎭 (social broadcast)

Labels are stored with the message and used for training the emotion manifold.

## Hot-Swapping Workflow

1. **Background Training**: Train new LoRA in separate process
2. **Load New Model**: `POST /api/models/load` loads new version in background
3. **Wait for Load**: UI polls until model is ready
4. **Hot-Swap**: `POST /api/models/switch/{version_id}` switches active model
5. **Cleanup**: Unload old versions when memory is needed

## Development

### Project Structure
```
apps/serve/
├── main.py              # FastAPI server
├── database.py          # SQLite database manager
├── model_interface.py   # LoRA model management
├── schema.sql          # Database schema
├── static/             # Frontend assets
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt    # Python dependencies
├── env.example        # Environment template
└── load_initial_model.py  # Model loading helper
```

### Adding New Features
- **Database changes**: Update `schema.sql` and `database.py`
- **API endpoints**: Add to `main.py`
- **UI features**: Update `static/` files
- **Model features**: Update `model_interface.py`

## Troubleshooting

### Model Loading Issues
- Ensure `transformers`, `torch`, and `peft` are installed
- Check model paths exist and are accessible
- Verify GPU memory is available (if using CUDA)

### Database Issues
- Check `DATABASE_PATH` in `.env` points to writable location
- Run `python -c "import sqlite3; print('SQLite OK')"` to test SQLite

### WebSocket Issues
- Check browser console for connection errors
- Verify server is running on correct port
- Check CORS settings if accessing from different domain

## Integration with Training Pipeline

This serving system feeds directly into the dual-channel training pipeline:

1. **Conversation Data** → Emotion labels → Training batches
2. **New LoRA Models** → Background loading → Hot-swapping
3. **Weekly Retraining** → Automatic model updates

The system is designed to run continuously, collecting data for the weekly retraining cycles described in the main plan.
