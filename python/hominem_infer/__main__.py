"""CLI entry point for hominem_infer."""

from __future__ import annotations

import os

import uvicorn

from hominem_infer.app import app


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def main() -> None:
    _load_dotenv()
    host = os.getenv("INFER_HOST", "0.0.0.0")
    port = int(os.getenv("INFER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
