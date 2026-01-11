from __future__ import annotations

import os

import uvicorn

from hominem_agent.app import app


def main() -> None:
    host = os.getenv("HOMINEM_AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("HOMINEM_AGENT_PORT", "8020"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

