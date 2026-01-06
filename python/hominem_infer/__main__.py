"""CLI entry point for hominem_infer."""

from __future__ import annotations

import os

import uvicorn

from hominem_infer.app import app


def main() -> None:
    host = os.getenv("INFER_HOST", "0.0.0.0")
    port = int(os.getenv("INFER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
