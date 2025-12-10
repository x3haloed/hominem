#!/usr/bin/env python3
"""
Debug script to test LoRA model loading
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    """Debug model loading"""
    print("🔍 LoRA Model Loading Debug")
    print("=" * 50)

    # Load environment
    load_dotenv()
    auto_load_lora = os.getenv("AUTO_LOAD_LORA")
    print(f"AUTO_LOAD_LORA from .env: '{auto_load_lora}'")

    # Check available LoRA models
    lora_base = "artifacts/lora"
    if os.path.exists(lora_base):
        print(f"📁 LoRA base directory exists: {lora_base}")
        models = [d for d in os.listdir(lora_base) if os.path.isdir(os.path.join(lora_base, d))]
        print(f"📂 Available LoRA models: {models}")

        # Check each model
        for model in models:
            model_path = os.path.join(lora_base, model)
            config_path = os.path.join(model_path, "adapter_config.json")
            model_path_safetensors = os.path.join(model_path, "adapter_model.safetensors")

            print(f"  🔍 Checking {model}:")
            print(f"    Path: {model_path}")
            print(f"    Config exists: {os.path.exists(config_path)}")
            print(f"    Model exists: {os.path.exists(model_path_safetensors)}")

            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        import json
                        config = json.load(f)
                        base_model = config.get('base_model_name_or_path', 'unknown')
                        print(f"    Base model: {base_model}")
                except Exception as e:
                    print(f"    Config read error: {e}")
    else:
        print(f"❌ LoRA base directory not found: {lora_base}")

    # Test the path resolution logic
    if auto_load_lora:
        print(f"\n🧪 Testing path resolution for '{auto_load_lora}':")

        if os.path.isabs(auto_load_lora):
            print("  📁 Absolute path detected")
            final_path = auto_load_lora
        else:
            print("  📁 Relative path, checking artifacts/lora/")
            final_path = os.path.join("artifacts", "lora", auto_load_lora)
            print(f"  📁 Full path: {final_path}")

            if not os.path.exists(final_path):
                print("  ⚠️  Exact path doesn't exist, would try find_latest_lora_version()")
                # Simulate find_latest_lora_version logic
                candidates = []
                if os.path.exists(lora_base):
                    for item in os.listdir(lora_base):
                        item_path = os.path.join(lora_base, item)
                        if os.path.isdir(item_path) and item.startswith(auto_load_lora):
                            config_path = os.path.join(item_path, "adapter_config.json")
                            if os.path.exists(config_path):
                                candidates.append(item_path)

                if candidates:
                    # Sort by modification time (newest first)
                    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    final_path = candidates[0]
                    print(f"  ✅ Found latest version: {final_path}")
                else:
                    print("  ❌ No candidates found"
        print(f"  🎯 Final path to load: {final_path}")
        print(f"  📁 Path exists: {os.path.exists(final_path) if final_path else False}")

if __name__ == "__main__":
    main()
