#!/usr/bin/env python3
"""
Quick and dirty script to tokenize messages from conversations.db using tiktoken
"""

import sqlite3
import tiktoken

def main():
    # Database path
    db_path = '/Users/chad/Documents/hominem/conversations.db'

    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all messages ordered by creation time
    cursor.execute('''
        SELECT role, content, think, created_at
        FROM messages
        WHERE conversation_id = 'canonical'
        ORDER BY created_at ASC
    ''')

    messages = []
    for role, content, think, created_at in cursor.fetchall():
        # Format message for chat template
        message = {
            'role': role,
            'content': content
        }

        # Add thinking content if present (for assistant messages)
        if think and role == 'assistant':
            message['content'] = f"{think}\n\n{content}"

        messages.append(message)

    conn.close()

    print(f"Loaded {len(messages)} messages from database")

    # Load Qwen tokenizer using tiktoken
    # Qwen models typically use a variant of the GPT-style tokenization
    # We'll try a few common encodings that work with Qwen models
    print("Loading tokenizer...")

    # Try different encodings that might work for Qwen
    encodings_to_try = [
        'cl100k_base',  # GPT-4 style (common for many models)
        'p50k_base',    # GPT-3 style
        'r50k_base',    # GPT-2 style
    ]

    tokenizer = None
    encoding_name = None

    for enc in encodings_to_try:
        try:
            tokenizer = tiktoken.get_encoding(enc)
            encoding_name = enc
            print(f"Using encoding: {enc}")
            break
        except Exception as e:
            print(f"Failed to load {enc}: {e}")
            continue

    if tokenizer is None:
        print("Could not load any tokenizer. Installing tiktoken...")
        import subprocess
        subprocess.run(['pip', 'install', 'tiktoken'], check=True)
        tokenizer = tiktoken.get_encoding('cl100k_base')
        encoding_name = 'cl100k_base'

    # Create Qwen-style chat template manually
    # Qwen typically uses <|im_start|>role\ncontent<|im_end|> format
    print("Applying Qwen chat template...")
    prompt_parts = []

    for msg in messages:
        if msg['role'] == 'user':
            prompt_parts.append(f"<|im_start|>user\n{msg['content']}<|im_end|>")
        elif msg['role'] == 'assistant':
            prompt_parts.append(f"<|im_start|>assistant\n{msg['content']}<|im_end|>")

    # Add generation prompt for next assistant response
    prompt_parts.append("<|im_start|>assistant\n")

    prompt = "\n".join(prompt_parts)

    # Count tokens
    print("Counting tokens...")
    tokens = tokenizer.encode(prompt)
    token_count = len(tokens)

    print(f"\nToken count: {token_count}")
    print(f"Character count: {len(prompt)}")
    print(f"Encoding used: {encoding_name}")
    print(".2f")

    # Show first 500 chars of the prompt
    print("\nFirst 500 characters of prompt:")
    print("-" * 50)
    print(prompt[:500])
    if len(prompt) > 500:
        print("...")
    print("-" * 50)

    # Show token breakdown for first few messages
    print("\nToken breakdown for first 3 messages:")
    print("-" * 30)
    for i, msg in enumerate(messages[:3]):
        msg_text = f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>"
        msg_tokens = tokenizer.encode(msg_text)
        print(f"Message {i+1} ({msg['role']}): {len(msg_tokens)} tokens")
        print(f"  Preview: {msg_text[:100]}{'...' if len(msg_text) > 100 else ''}")

if __name__ == "__main__":
    main()
