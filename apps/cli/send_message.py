#!/usr/bin/env python3
"""
Send a message to a running Hominem server and print the assistant reply.

Requires:
  - a running server (start with apps/cli/boot_model.py)
  - an existing conversation_id (create via POST /api/conversations, or UI)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import websocket
import threading
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Send message and return final reply (waits for emotion labeling).")
    parser.add_argument("conversation_id", help="Existing conversation id")
    parser.add_argument("message", help="User message to send")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--enable-self-awareness", action="store_true")
    parser.add_argument("--disable-self-awareness", action="store_true")
    parser.add_argument("--streaming", action="store_true", help="Use WebSocket streaming to capture partial responses")
    args = parser.parse_args()

    enable_thinking = True
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    enable_self_awareness = None
    if args.enable_self_awareness:
        enable_self_awareness = True
    if args.disable_self_awareness:
        enable_self_awareness = False

    if args.streaming:
        return stream_message(args)
    else:
        return complete_message(args)


def complete_message(args) -> int:
    enable_thinking = True
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    enable_self_awareness = None
    if args.enable_self_awareness:
        enable_self_awareness = True
    if args.disable_self_awareness:
        enable_self_awareness = False
    """Original synchronous completion method"""
    url = f"http://{args.host}:{args.port}/api/complete"
    payload = {
        "conversation_id": args.conversation_id,
        "content": args.message,
        "enable_thinking": enable_thinking,
        "metadata": {},
        "enable_self_awareness": enable_self_awareness,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = str(e)
        print(body, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    data = json.loads(body)
    print(data.get("assistant_response", ""))
    return 0


def stream_message(args) -> int:
    """WebSocket streaming method to capture partial responses"""
    ws_url = f"ws://{args.host}:{args.port}/ws/chat/{args.conversation_id}"

    enable_thinking = True
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    enable_self_awareness = None
    if args.enable_self_awareness:
        enable_self_awareness = True
    if args.disable_self_awareness:
        enable_self_awareness = False

    # Collected response data
    full_response = []
    response_started = False
    message_index = None

    def on_message(ws, message):
        nonlocal full_response, response_started, message_index
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "response_start":
                response_started = True
                message_index = data.get("message_index")
                print(f"[Response started for message {message_index}]", file=sys.stderr)

            elif msg_type == "token_chunk":
                if response_started:
                    chunk = data.get("chunk", "")
                    if chunk:
                        full_response.append(chunk)
                        print(chunk, end="", flush=True)

            elif msg_type == "response_complete":
                full_response_text = "".join(full_response)
                print(f"\n[Response complete - {len(full_response_text)} chars]", file=sys.stderr)
                ws.close()

            elif msg_type == "error":
                print(f"[Error: {data.get('message', 'Unknown error')}]", file=sys.stderr)
                ws.close()

        except json.JSONDecodeError:
            print(f"[Received non-JSON message: {message}]", file=sys.stderr)

    def on_error(ws, error):
        print(f"[WebSocket error: {error}]", file=sys.stderr)

    def on_close(ws, close_status_code, close_msg):
        print(f"\n[Connection closed: {close_status_code} - {close_msg}]", file=sys.stderr)

    def on_open(ws):
        # Send the initial message
        payload = {
            "type": "send_message",
            "content": args.message,
            "enable_thinking": enable_thinking,
            "enable_self_awareness": enable_self_awareness,
        }
        ws.send(json.dumps(payload))

    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )

    # Run with timeout handling
    def run_with_timeout():
        start_time = time.time()
        timeout_seconds = 120  # 2 minutes

        # Start WebSocket in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for completion or timeout
        while ws_thread.is_alive():
            if time.time() - start_time > timeout_seconds:
                print(f"\n[Timeout reached after {timeout_seconds} seconds]", file=sys.stderr)
                ws.close()
                break
            time.sleep(0.1)

    run_with_timeout()

    # Print final collected response
    final_response = "".join(full_response)
    if final_response.strip():
        print(f"\n[FINAL RESPONSE ({len(final_response)} chars)]:")
        print(final_response)
    else:
        print("[No response collected]", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



