"""
Model interface for hominem serving system
Handles LoRA model loading and inference with token streaming and hot-swapping
"""

from typing import Any, Dict, Optional, List
import asyncio
import threading
import json
import time
from dataclasses import dataclass
from contextlib import asynccontextmanager
import os

# Model loading dependencies (loaded conditionally)
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Warning: transformers/peft not available. Install with: pip install transformers peft torch")

@dataclass
class ModelVersion:
    """Represents a loaded model version"""
    version_id: str
    base_model_path: str
    lora_path: Optional[str]
    tokenizer: Any = None
    model: Any = None
    pipeline: Any = None
    loaded_at: float = 0
    is_active: bool = False
    metadata: Dict[str, Any] = None

class ModelRegistry:
    """Manages multiple model versions for hot-swapping"""

    def __init__(self):
        self.models: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        self._lock = threading.RLock()

    def get_active_model(self) -> Optional[ModelVersion]:
        """Get the currently active model"""
        with self._lock:
            if self.active_version and self.active_version in self.models:
                return self.models[self.active_version]
        return None

    def set_active_version(self, version_id: str):
        """Atomically switch the active model version"""
        with self._lock:
            if version_id in self.models:
                # Mark old active as inactive
                if self.active_version and self.active_version in self.models:
                    self.models[self.active_version].is_active = False

                # Mark new one as active
                self.models[version_id].is_active = True
                self.active_version = version_id
                print(f"🔄 Switched to model version: {version_id}")

    def register_model(self, version: ModelVersion):
        """Register a model version"""
        with self._lock:
            self.models[version.version_id] = version
            print(f"📝 Registered model version: {version.version_id}")

    def unload_version(self, version_id: str):
        """Unload a model version from memory"""
        with self._lock:
            if version_id in self.models:
                version = self.models[version_id]
                # Clear GPU memory if available
                if hasattr(version.model, 'cpu'):
                    version.model.cpu()
                if torch and hasattr(torch, 'cuda'):
                    torch.cuda.empty_cache()

                del self.models[version_id]
                print(f"🗑️ Unloaded model version: {version_id}")

class ModelInterface:
    """Interface for LoRA model inference with streaming support and hot-swapping"""

    def __init__(self):
        self.registry = ModelRegistry()
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.max_memory = {}  # For model loading constraints

    async def generate_streaming_response(self, websocket, conversation_id: str,
                                        message_index: int, conversation_history: List[Dict[str, str]],
                                        enable_thinking: bool = True, db=None):
        """Generate streaming response and send via WebSocket"""
        print(f"🎯 Starting completion for {conversation_id}:{message_index + 1}")
        start_time = time.time()

        try:
            model_version = self.registry.get_active_model()
            if not model_version:
                print("❌ No active model loaded - using placeholder response")
                await websocket.send_json({
                    "type": "error",
                    "message": "No active model loaded"
                })
                return

            print(f"✅ Using model: {model_version.version_id}")

            assistant_index = message_index + 1

            # Send processing start
            await websocket.send_json({
                "type": "response_start",
                "message_index": assistant_index
            })

            # Format conversation using chat template
            formatted_prompt = self._format_chat_conversation(model_version.tokenizer, conversation_history, enable_thinking)

            # Use pipeline for generation
            if model_version.pipeline:
                print(f"🚀 Starting model inference with {model_version.version_id}")
                # Streaming generation
                response_text = ""
                start_time = time.time()

                # Generate with streaming - use proper EOS tokens
                eos_tokens = self._get_eos_tokens(model_version.tokenizer)

                outputs = model_version.pipeline(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=model_version.tokenizer.eos_token_id,
                    eos_token_id=eos_tokens,  # Use proper EOS tokens for chat
                    return_full_text=False
                )

                # For now, simulate streaming since pipeline doesn't support it directly
                # In production, you'd use a custom streaming implementation
                full_response = outputs[0]['generated_text']
                print(f"📄 Generated response ({len(full_response)} chars)")

                # Clean response (remove any EOS tokens that might have been generated)
                full_response = self._clean_generated_response(full_response, model_version.tokenizer)

                # Send in chunks to simulate streaming
                chunk_count = 0
                for chunk in self._chunk_text(full_response):
                    response_text += chunk
                    chunk_count += 1
                    await websocket.send_json({
                        "type": "token_chunk",
                        "message_index": assistant_index,
                        "chunk": chunk,
                        "is_complete": False
                    })
                    await asyncio.sleep(0.01)  # Simulate token timing

                processing_time = int((time.time() - start_time) * 1000)
                print(f"📤 Sent {chunk_count} chunks, {len(response_text)} total chars, {processing_time}ms")

                await websocket.send_json({
                    "type": "response_complete",
                    "message_index": assistant_index,
                    "full_response": response_text,
                    "token_count": len(model_version.tokenizer.encode(response_text)),
                    "processing_time_ms": processing_time
                })

                # Save assistant response to database
                if db:
                    try:
                        token_count = len(model_version.tokenizer.encode(response_text))
                        db.add_message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content=response_text,
                            token_count=token_count,
                            processing_time_ms=processing_time,
                            metadata={"enable_thinking": enable_thinking}
                        )
                        print(f"💾 Saved assistant response for {conversation_id}:{assistant_index}")
                    except Exception as db_error:
                        print(f"⚠️ Failed to save assistant message to database: {db_error}")

        except Exception as e:
            print(f"❌ Model generation error: {e}")
            await self._placeholder_response(websocket, message_index)

            elapsed = time.time() - start_time
            print(f"🏁 Completion finished for {conversation_id}:{message_index + 1} in {elapsed:.2f}s")

        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Model generation failed: {str(e)}"
            })

    async def _generate_with_model(self, websocket, model_version: ModelVersion,
                                 conversation_history: List[Dict[str, str]], message_index: int,
                                 enable_thinking: bool = True):
        """Generate response using the specified model with proper chat formatting"""
        if not MODELS_AVAILABLE:
            # Fallback placeholder response
            print("📝 Using placeholder response (transformers not available)")
            await self._placeholder_response(websocket, message_index)
            return

        try:
            print(f"🔄 Formatting conversation with chat template (thinking: {enable_thinking})")
            # Format conversation using chat template
            formatted_prompt = self._format_chat_conversation(model_version.tokenizer, conversation_history, enable_thinking)

            # Use pipeline for generation
            if model_version.pipeline:
                print(f"🚀 Starting model inference with {model_version.version_id}")
                # Streaming generation
                response_text = ""
                start_time = time.time()

                # Generate with streaming - use proper EOS tokens
                eos_tokens = self._get_eos_tokens(model_version.tokenizer)

                outputs = model_version.pipeline(
                    formatted_prompt,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=model_version.tokenizer.eos_token_id,
                    eos_token_id=eos_tokens,  # Use proper EOS tokens for chat
                    return_full_text=False
                )

                # For now, simulate streaming since pipeline doesn't support it directly
                # In production, you'd use a custom streaming implementation
                full_response = outputs[0]['generated_text']
                print(f"📄 Generated response ({len(full_response)} chars)")

                # Clean response (remove any EOS tokens that might have been generated)
                full_response = self._clean_generated_response(full_response, model_version.tokenizer)

                # Send in chunks to simulate streaming
                chunk_count = 0
                for chunk in self._chunk_text(full_response):
                    response_text += chunk
                    chunk_count += 1
                    await websocket.send_json({
                        "type": "token_chunk",
                        "message_index": message_index,
                        "chunk": chunk,
                        "is_complete": False
                    })
                    await asyncio.sleep(0.01)  # Simulate token timing

                processing_time = int((time.time() - start_time) * 1000)
                print(f"📤 Sent {chunk_count} chunks, {len(response_text)} total chars, {processing_time}ms")

                await websocket.send_json({
                    "type": "response_complete",
                    "message_index": message_index,
                    "full_response": response_text,
                    "token_count": len(model_version.tokenizer.encode(response_text)),
                    "processing_time_ms": processing_time
                })

        except Exception as e:
            print(f"❌ Model generation error: {e}")
            await self._placeholder_response(websocket, message_index)

    async def _placeholder_response(self, websocket, message_index: int):
        """Placeholder response when model is not available"""
        assistant_index = message_index + 1
        response_text = "Model loading... This is a placeholder response while the LoRA model is being integrated."
        print(f"📝 Sending placeholder response for message {message_index}")

        for chunk in self._chunk_text(response_text):
            await websocket.send_json({
                "type": "token_chunk",
                "message_index": assistant_index,
                "chunk": chunk,
                "is_complete": False
            })
            await asyncio.sleep(0.05)

        await websocket.send_json({
            "type": "response_complete",
            "message_index": assistant_index,
            "full_response": response_text,
            "token_count": len(response_text.split()),
            "processing_time_ms": 1000
        })

    def _format_chat_conversation(self, tokenizer, conversation_history: List[Dict[str, str]], enable_thinking: bool = True) -> str:
        """Format conversation history using chat template"""
        try:
            # Use tokenizer's apply_chat_template if available (transformers >= 4.34)
            if hasattr(tokenizer, 'apply_chat_template'):
                print(f"📝 Using chat template with {len(conversation_history)} messages (thinking: {enable_thinking})")
                return tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking  # Pass thinking parameter to template
                )
            else:
                # Fallback to manual formatting for older transformers
                print("⚠️ Chat template not available, using fallback formatting")
                return self._format_chat_fallback(conversation_history)
        except Exception as e:
            print(f"❌ Chat template formatting failed: {e}, using fallback")
            return self._format_chat_fallback(conversation_history)

    def _format_chat_fallback(self, conversation_history: List[Dict[str, str]]) -> str:
        """Fallback chat formatting when template is not available"""
        formatted = ""
        for message in conversation_history:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # Add generation prompt
        formatted += "<|im_start|>assistant\n"
        return formatted

    def _get_eos_tokens(self, tokenizer) -> List[int]:
        """Get EOS token IDs for stopping generation"""
        eos_tokens = []

        # Primary EOS token
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_tokens.extend(tokenizer.eos_token_id)
            else:
                eos_tokens.append(tokenizer.eos_token_id)

        # Chat-specific EOS tokens (like <|im_end|> for Qwen)
        eos_token_strings = ["<|im_end|>", "<|endoftext|>"]
        for token_str in eos_token_strings:
            try:
                token_id = tokenizer.encode(token_str, add_special_tokens=False)
                if token_id:
                    eos_tokens.extend(token_id)
            except:
                continue

        # Remove duplicates and return
        return list(set(eos_tokens))

    def _clean_generated_response(self, response: str, tokenizer) -> str:
        """Clean generated response by removing EOS tokens"""
        # Remove common EOS tokens
        eos_tokens = ["<|im_end|>", "<|endoftext|>", tokenizer.eos_token or "</s>"]

        cleaned = response
        for token in eos_tokens:
            cleaned = cleaned.replace(token, "")

        return cleaned.strip()

    def _format_chat_prompt(self, user_message: str) -> str:
        """Legacy method - kept for compatibility"""
        return f"Human: {user_message}\n\nAssistant:"

    def _chunk_text(self, text: str, chunk_size: int = 5):
        """Split text into chunks for streaming"""
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                yield chunk + " "

    async def load_model_async(self, version_id: str, base_model_path: str,
                              lora_path: Optional[str] = None) -> bool:
        """Load a model version asynchronously in background"""
        try:
            print(f"🔄 Loading model version {version_id} in background...")

            # Run model loading in thread pool to not block
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, self._load_model_sync, version_id, base_model_path, lora_path
            )

            if success:
                print(f"✅ Model version {version_id} loaded successfully")
                return True
            else:
                print(f"❌ Failed to load model version {version_id}")
                return False

        except Exception as e:
            print(f"❌ Error loading model {version_id}: {e}")
            return False

    def _load_model_sync(self, version_id: str, base_model_path: str,
                        lora_path: Optional[str] = None) -> bool:
        """Synchronous model loading (runs in thread pool)"""
        if not MODELS_AVAILABLE:
            print("⚠️ Model loading skipped - transformers/peft not available")
            return False

        try:
            print(f"Loading tokenizer from {base_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)

            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print(f"Loading base model from {base_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                max_memory=self.max_memory if self.device == "cuda" else None
            )

            # Load LoRA adapter if provided
            if lora_path and os.path.exists(lora_path):
                print(f"Loading LoRA adapter from {lora_path}")
                model = PeftModel.from_pretrained(model, lora_path)

            # Move to device
            if self.device == "cuda":
                model = model.to(self.device)

            # Create pipeline
            pipeline_instance = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            # Create model version
            version = ModelVersion(
                version_id=version_id,
                base_model_path=base_model_path,
                lora_path=lora_path,
                tokenizer=tokenizer,
                model=model,
                pipeline=pipeline_instance,
                loaded_at=time.time(),
                metadata={
                    "device": self.device,
                    "has_lora": lora_path is not None,
                    "model_size": self._estimate_model_size(model)
                }
            )

            self.registry.register_model(version)
            return True

        except Exception as e:
            print(f"Model loading failed: {e}")
            return False

    def _estimate_model_size(self, model) -> str:
        """Estimate model size for metadata"""
        if hasattr(model, 'num_parameters'):
            params = model.num_parameters()
            if params > 1e9:
                return f"{params/1e9:.1f}B"
            elif params > 1e6:
                return f"{params/1e6:.1f}M"
            else:
                return f"{params/1e3:.1f}K"
        return "unknown"

    def switch_to_version(self, version_id: str) -> bool:
        """Switch to a different model version"""
        if version_id in self.registry.models:
            self.registry.set_active_version(version_id)
            return True
        return False

    def get_loaded_versions(self) -> List[str]:
        """Get list of loaded model versions"""
        return list(self.registry.models.keys())

    def get_active_version_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the active model version"""
        active = self.registry.get_active_model()
        if active:
            return {
                "version_id": active.version_id,
                "base_model": active.base_model_path,
                "lora_path": active.lora_path,
                "loaded_at": active.loaded_at,
                "is_active": active.is_active,
                "metadata": active.metadata
            }
        return None

    def unload_version(self, version_id: str):
        """Unload a model version"""
        self.registry.unload_version(version_id)
