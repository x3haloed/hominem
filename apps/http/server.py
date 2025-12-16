#!/usr/bin/env python3
"""
Thin wrapper to run the Unified Theory chat API.
Delegates to apps.serve.main (FastAPI app).
"""

from __future__ import annotations

import os
import uvicorn

from apps.serve.main import app


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

