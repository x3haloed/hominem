#!/usr/bin/env python3
"""
Helper script to load an initial LoRA model for the serving system
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add the parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model_interface import ModelInterface

def main():
    """Load an initial model"""
    print("🤖 Hominem Model Loader")
    print("=" * 50)

    # Get model paths from environment or command line
    base_model_path = os.getenv("BASE_MODEL_PATH") or input("Base model path: ").strip()
    lora_path = os.getenv("LORA_PATH") or input("LoRA path (optional): ").strip() or None
    version_id = os.getenv("MODEL_VERSION") or f"v{int(time.time())}"

    if not base_model_path:
        print("❌ Base model path is required")
        return

    print(f"📂 Base model: {base_model_path}")
    print(f"🎯 LoRA path: {lora_path or 'None'}")
    print(f"🏷️  Version ID: {version_id}")
    print()

    # Initialize model interface
    model = ModelInterface()

    try:
        print("🔄 Loading model... (this may take several minutes)")
        import asyncio

        # Run async loading
        success = asyncio.run(model.load_model_async(version_id, base_model_path, lora_path))

        if success:
            print("✅ Model loaded successfully!")
            print(f"🔄 Switching to active model...")

            # Switch to the loaded model
            if model.switch_to_version(version_id):
                print(f"🎉 Model {version_id} is now active!")
                print()
                print("💡 You can now run the serving system:")
                print("   cd apps/serve && python main.py")
            else:
                print("❌ Failed to activate model")
        else:
            print("❌ Failed to load model")

    except KeyboardInterrupt:
        print("\n⏹️  Loading cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
