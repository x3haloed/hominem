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

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import sqlite3
import json

from dotenv import load_dotenv

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

DATABASE_PATH = os.getenv("DATABASE_PATH", "storage/conversations.db")

# Ensure database directory exists
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db, model

    # Initialize database
    db = DatabaseManager(DATABASE_PATH)

    # Initialize model interface (placeholder for now)
    model = ModelInterface()

    yield

    # Cleanup
    if db:
        db.close()

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

@app.websocket("/ws/chat/{conversation_id}")
async def chat_websocket(websocket: WebSocket, conversation_id: str):
    """WebSocket endpoint for real-time chat with token streaming"""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            if data["type"] == "send_message":
                content = data["content"]
                metadata = data.get("metadata", {})

                # Add user message to database
                message_index = db.add_message(
                    conversation_id=conversation_id,
                    role="user",
                    content=content,
                    metadata=metadata
                )

                # Send user message confirmation
                await websocket.send_json({
                    "type": "message_added",
                    "message_index": message_index,
                    "role": "user",
                    "content": content
                })

                # Generate AI response with streaming
                await model.generate_streaming_response(
                    websocket=websocket,
                    conversation_id=conversation_id,
                    message_index=message_index,
                    prompt=content
                )

            elif data["type"] == "label_emotion":
                # Handle emotion labeling
                label_data = data["label"]
                message_index = data["message_index"]

                db.add_emotion_label(
                    conversation_id=conversation_id,
                    message_index=message_index,
                    labeler="user",
                    **label_data
                )

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
