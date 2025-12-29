from __future__ import annotations

import os

import uvicorn

from waterfall_web.app import app  # noqa: F401


def main() -> None:
    host = os.getenv("WF_HOST", "127.0.0.1")
    port = int(os.getenv("WF_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
