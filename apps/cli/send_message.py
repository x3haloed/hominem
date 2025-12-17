#!/usr/bin/env python3
"""
Send a message to the canonical Unified Theory Chat server and print the assistant reply.

Requires:
  - a running server (e.g. `python apps/serve/main.py` or `apps/cli/boot_model.py`)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Send message and return assistant reply.")
    parser.add_argument("conversation_id", help="Conversation id (default in UI is 'canonical')")
    parser.add_argument("message", help="User message to send")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    parser.add_argument("--print-metrics", action="store_true", help="Also print JSON metrics to stderr")
    args = parser.parse_args()

    enable_thinking = True
    if args.no_thinking:
        enable_thinking = False
    elif args.thinking:
        enable_thinking = True

    return complete_message(args, enable_thinking=enable_thinking)


def complete_message(args, *, enable_thinking: bool) -> int:
    url = f"http://{args.host}:{args.port}/chat"
    payload = {
        "conversation_id": args.conversation_id,
        "user_message": args.message,
        "enable_thinking": enable_thinking,
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
    assistant = data.get("assistant", "")
    print(assistant)
    if args.print_metrics:
        print(json.dumps(data.get("metrics", {}), indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
