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


if __name__ == "__main__":
    raise SystemExit(main())

