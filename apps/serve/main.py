#!/usr/bin/env python3
"""
Hominem LoRA Serving System with Emotion Labeling UI

Launches a web interface for chatting with LoRA models and labeling emotions.
Provides real-time conversation and emotion labeling capabilities.
"""

import os
import asyncio
import threading
import webbrowser
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import sqlite3
import json
import time

from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

DATABASE_PATH = os.getenv("DATABASE_PATH", "storage/conversations.db")
AUTO_LOAD_LORA = os.getenv("AUTO_LOAD_LORA")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Resolve project root (apps/serve/ -> project root)
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LORA_DIR = ARTIFACTS_DIR / "lora"

# Ensure database directory exists
os.makedirs(Path(DATABASE_PATH).parent, exist_ok=True)

# Import our modules (these will be created)
try:
    from .database import DatabaseManager
    from .model_interface import ModelInterface
except ImportError:
    # Fallback for direct execution
    from database import DatabaseManager
    from model_interface import ModelInterface

# Global instances
db: Optional[DatabaseManager] = None
model: Optional[ModelInterface] = None
background_tasks: set = set()  # Track background tasks for cleanup

async def auto_load_lora_model(model_interface, lora_spec: str, base_model_path: str = None):
    """Automatically load a LoRA model on startup"""
    try:
        print(f"🔍 Processing LoRA spec: '{lora_spec}'")

        # Determine LoRA path
        if os.path.isabs(lora_spec):
            # Absolute path specified
            lora_path = Path(lora_spec)
            version_id = lora_path.name  # Use directory name as version
            print(f"📁 Using absolute path: {lora_path}")
        else:
            # Relative path from artifacts/lora/ (project root)
            lora_path = LORA_DIR / lora_spec
            version_id = lora_spec
            print(f"📁 Checking relative path: {lora_path}")

            # If the exact path doesn't exist, try to find the latest version
            if not lora_path.exists():
                print(f"⚠️  Path {lora_path} doesn't exist, looking for latest version...")
                lora_path = find_latest_lora_version(lora_spec)
                if lora_path:
                    version_id = lora_path.name
                    print(f"✅ Found latest version: {lora_path}")
                else:
                    print(f"❌ No versions found for '{lora_spec}'")
                    lora_path = None  # Ensure it's None if not found

        if not lora_path or not lora_path.exists():
            exists = lora_path.exists() if isinstance(lora_path, Path) else 'N/A'
            print(f"❌ Final check failed: lora_path={lora_path}, exists={exists}")
            print(f"⚠️  Auto-load LoRA not found: {lora_path}")
            return

        # Auto-detect base model if not specified
        if not base_model_path:
            base_model_path = auto_detect_base_model(lora_path)

        if not base_model_path:
            print(f"⚠️  Could not determine base model path for {lora_spec}")
            print("   Set BASE_MODEL_PATH in .env or ensure it's in the LoRA directory")
            return

        print(f"🔄 Auto-loading LoRA model: {version_id}")
        print(f"   Base: {base_model_path}")
        print(f"   LoRA: {lora_path}")

        # Load the model
        base_model_path_str = str(base_model_path) if base_model_path else None
        success = await model_interface.load_model_async(version_id, base_model_path_str, str(lora_path))

        if success:
            # Switch to the loaded model
            if model_interface.switch_to_version(version_id):
                print(f"✅ Auto-loaded and activated LoRA model: {version_id}")
            else:
                print(f"❌ Failed to activate auto-loaded model: {version_id}")
        else:
            print(f"❌ Failed to auto-load LoRA model: {version_id}")

    except Exception as e:
        print(f"❌ Error during auto-loading: {e}")

def find_latest_lora_version(base_name: str) -> Optional[Path]:
    """Find the latest version of a LoRA model (e.g., qwen3.1-7b-v2)"""
    lora_base_dir = LORA_DIR

    if not lora_base_dir.exists():
        print(f"❌ LoRA base directory not found: {lora_base_dir}")
        return None

    # Look for directories that start with the base name
    candidates = []
    for item in lora_base_dir.iterdir():
        if item.is_dir() and item.name.startswith(base_name):
            # Check if it has the required LoRA files
            if (item / "adapter_config.json").exists():
                candidates.append(item)

    if not candidates:
        return None

    # Sort by modification time (newest first)
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print(f"🔍 Found {len(candidates)} versions for '{base_name}', using latest: {candidates[0].name}")
    return candidates[0]

def auto_detect_base_model(lora_path: Path) -> Optional[str]:
    """Try to auto-detect the base model path from LoRA metadata"""
    try:
        # Check for adapter_config.json in LoRA directory
        config_path = lora_path / "adapter_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
                if base_model_name:
                    # First, check if it's an absolute/local path that exists
                    candidate = Path(base_model_name)
                    if candidate.is_absolute() and candidate.exists():
                        return str(candidate)

                    # Try resolving relative to known locations
                    potential_paths = [
                        lora_path / base_model_name,
                        BASE_DIR / base_model_name,
                        ARTIFACTS_DIR / "models" / base_model_name,
                        BASE_DIR / "models" / base_model_name,
                    ]
                    for path in potential_paths:
                        if path.exists():
                            return str(path)

                    # If nothing exists locally, assume it's a HuggingFace model ID
                    print(f"ℹ️  Using remote base model identifier: {base_model_name}")
                    return base_model_name
    except Exception as e:
        print(f"Warning: Could not read LoRA config in {lora_path}: {e}")

    return None


def list_available_loras() -> List[Dict[str, Any]]:
    """Discover available LoRA adapters from artifacts/lora"""
    options: List[Dict[str, Any]] = []

    if not LORA_DIR.exists():
        return options

    for item in LORA_DIR.iterdir():
        if not item.is_dir():
            continue

        adapter_file = item / "adapter_config.json"
        if not adapter_file.exists():
            continue

        base_model_name = None
        try:
            with open(adapter_file, "r") as f:
                config = json.load(f)
                base_model_name = config.get("base_model_name_or_path")
        except Exception as e:
            print(f"⚠️  Failed to read adapter config at {adapter_file}: {e}")

        detected_base = auto_detect_base_model(item)
        options.append({
            "id": item.name,
            "path": str(item),
            "base_model_name_or_path": base_model_name,
            "detected_base_model_path": detected_base,
            "updated_at": item.stat().st_mtime,
        })

    # Newest first
    options.sort(key=lambda x: x["updated_at"], reverse=True)
    return options


def list_available_base_models(lora_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collect unique base model identifiers/paths from env and LoRA configs"""
    candidates: Dict[str, Dict[str, Any]] = {}

    def add_candidate(identifier: str, source: str):
        if not identifier:
            return
        if identifier not in candidates:
            candidates[identifier] = {
                "id": identifier,
                "path": identifier,
                "source": source,
            }

    if BASE_MODEL_PATH:
        add_candidate(BASE_MODEL_PATH, "env")

    for lora in lora_options:
        add_candidate(lora.get("detected_base_model_path"), f"lora:{lora['id']}:detected")
        add_candidate(lora.get("base_model_name_or_path"), f"lora:{lora['id']}:config")

    # Sort for stable dropdown ordering
    return sorted(candidates.values(), key=lambda x: x["id"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db, model

    # Initialize database
    db = DatabaseManager(DATABASE_PATH)

    # Initialize model interface (placeholder for now)
    model = ModelInterface()

    # Auto-load LoRA model if specified
    print(f"🔍 Checking for auto-load: AUTO_LOAD_LORA='{AUTO_LOAD_LORA}', BASE_MODEL_PATH='{BASE_MODEL_PATH}'")
    if AUTO_LOAD_LORA:
        print(f"🚀 Starting auto-load for: {AUTO_LOAD_LORA}")
        await auto_load_lora_model(model, AUTO_LOAD_LORA, BASE_MODEL_PATH)
    else:
        print("⚠️  No AUTO_LOAD_LORA specified in .env")

    yield

    # Cleanup - comprehensive shutdown
    print("🧹 Starting application cleanup...")

    # Cancel any running background tasks
    if background_tasks:
        print(f"🛑 Cancelling {len(background_tasks)} background tasks...")
        for task in background_tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to cancel
        await asyncio.gather(*background_tasks, return_exceptions=True)
        background_tasks.clear()

    # Close database
    if db:
        print("💾 Closing database connection...")
        db.close()

    # Cleanup model resources
    if model:
        print("🧠 Cleaning up model resources...")
        try:
            # Unload all models to free GPU memory
            for version_id in model.get_loaded_versions():
                print(f"🗑️ Unloading model: {version_id}")
                model.unload_version(version_id)
        except Exception as e:
            print(f"⚠️ Error during model cleanup: {e}")

    print("✅ Application cleanup complete")

app = FastAPI(
    title="Hominem LoRA Serving System",
    description="Real-time chat and emotion labeling interface",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
# Get the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(script_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class MessageRequest(BaseModel):
    conversation_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class EmotionLabel(BaseModel):
    valence: Optional[float] = None  # -2 to +2
    arousal: Optional[float] = None  # 0 to 1
    dominance: Optional[float] = None  # -1 to 1
    predictive_discrepancy: Optional[float] = None  # -1 to 1
    temporal_directionality: Optional[float] = None  # -1 to 1
    social_broadcast: Optional[float] = None  # 0 to 1
    raw_indicators: Optional[Dict[str, Any]] = None  # UI reaction data
    notes: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface"""
    html_path = os.path.join(script_dir, "static", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/conversations")
async def get_conversations():
    """Get all conversations"""
    try:
        conversations = db.get_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a specific conversation with messages"""
    try:
        conversation = db.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conversations")
async def create_conversation():
    """Create a new conversation"""
    try:
        conversation_id = str(uuid.uuid4())
        db.create_conversation(conversation_id)
        return {"conversation_id": conversation_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/messages")
async def send_message(request: MessageRequest):
    """Send a message and get AI response"""
    try:
        # Add user message to database
        message_index = db.add_message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.content,
            metadata=request.metadata
        )

        # Get AI response (placeholder - will stream via WebSocket)
        # For now, return message info
        return {
            "message_index": message_index,
            "status": "processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/messages/{conversation_id}/{message_index}/emotion")
async def label_emotion(conversation_id: str, message_index: int, label: EmotionLabel):
    """Add emotion label to a message"""
    try:
        db.add_emotion_label(
            conversation_id=conversation_id,
            message_index=message_index,
            labeler="user",
            **label.dict(exclude_unset=True)
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model management endpoints
@app.get("/api/models")
async def get_models():
    """Get information about loaded models"""
    try:
        versions = model.get_loaded_versions()
        active_info = model.get_active_version_info()

        return {
            "loaded_versions": versions,
            "active_version": active_info,
            "device": model.device
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/available")
async def get_available_models():
    """List discoverable base models and LoRA adapters for UI selection"""
    try:
        lora_options = list_available_loras()
        base_models = list_available_base_models(lora_options)

        return {
            "base_models": base_models,
            "lora_adapters": lora_options
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/load")
async def load_model(request: Dict[str, Any]):
    """Load a model version in background"""
    try:
        version_id = request.get("version_id", f"v{int(time.time())}")
        base_model_path = request["base_model_path"]
        lora_path = request.get("lora_path")

        # Start background loading and track the task
        task = asyncio.create_task(
            model.load_model_async(version_id, base_model_path, lora_path)
        )
        background_tasks.add(task)

        # Clean up completed tasks
        background_tasks.difference_update(
            {t for t in background_tasks if t.done()}
        )

        return {
            "status": "loading",
            "version_id": version_id,
            "message": f"Loading model {version_id} in background"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/models/switch/{version_id}")
async def switch_model(version_id: str):
    """Switch to a different model version"""
    try:
        success = model.switch_to_version(version_id)
        if success:
            return {
                "status": "success",
                "active_version": version_id
            }
        else:
            raise HTTPException(status_code=404, detail="Model version not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/models/{version_id}")
async def unload_model(version_id: str):
    """Unload a model version"""
    try:
        model.unload_version(version_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat/{conversation_id}")
async def chat_websocket(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat with token streaming"""
    print(f"🔌 WebSocket connection established for conversation {conversation_id}")
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data["type"] == "send_message":
                content = data["content"]
                enable_thinking = data.get("enable_thinking", True)  # Default to thinking enabled
                metadata = data.get("metadata", {})

                print(f"📨 Received message in conversation {conversation_id}: {content[:100]}{'...' if len(content) > 100 else ''}")
                print(f"🧠 Thinking mode: {enable_thinking}")

                # Add user message to database
                message_index = db.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=content,
                    metadata={**metadata, "enable_thinking": enable_thinking}
                )

                print(f"💾 Saved user message {conversation_id}:{message_index}")

                # Send user message confirmation
                await websocket.send_json({
                    "type": "message_added",
                    "message_index": message_index,
                    "role": "user",
                    "content": content
                })

                print(f"🤖 Starting AI response generation for {conversation_id}:{message_index + 1} (thinking: {enable_thinking})")

                # Get conversation history for chat formatting
                conversation_data = db.get_conversation(conversation_id)
                conversation_history = []
                if conversation_data and "messages" in conversation_data:
                    # Convert to format expected by chat template
                    conversation_history = [
                        {"role": msg["role"], "content": msg["content"], "enable_thinking": enable_thinking}
                        for msg in conversation_data["messages"]
                    ]

                print(f"📚 Using conversation history: {len(conversation_history)} messages")

                # Generate AI response with streaming
                await model.generate_streaming_response(
                    websocket=websocket,
                    conversation_id=conversation_id,
                    message_index=message_index,
                    conversation_history=conversation_history,
                    enable_thinking=enable_thinking,
                    db=db
                )

            elif data["type"] == "label_emotion":
                # Handle emotion labeling
                label_data = data["label"]
                message_index = data["message_index"]

                print(f"🏷️  Saving emotion labels for {conversation_id}:{message_index}")

                db.add_emotion_label(
                    conversation_id=conversation_id,
                    message_index=message_index,
                    labeler="user",
                    **label_data
                )

                print(f"✅ Emotion labels saved for {conversation_id}:{message_index}")

                await websocket.send_json({
                    "type": "label_saved",
                    "message_index": message_index
                })

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

def launch_browser(port: int = 8000):
    """Launch browser after a short delay"""
    def _launch():
        import time
        time.sleep(1)  # Wait for server to start
        webbrowser.open(f"http://localhost:{port}")

    thread = threading.Thread(target=_launch)
    thread.daemon = True
    thread.start()

def main():
    """Main entry point"""
    port = int(os.getenv("PORT", 8000))

    print(f"🚀 Starting Hominem Serving System on http://localhost:{port}")
    print(f"📊 Database: {DATABASE_PATH}")

    # Launch browser in background thread (skip on errors)
    try:
        launch_browser(port)
    except Exception as e:
        print(f"Warning: Could not launch browser: {e}")

    # Start server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
