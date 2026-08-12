from __future__ import annotations

import asyncio
import sys

import uvicorn


def windows_selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Use the stable selector loop for LitSync's single-process Windows server."""
    return asyncio.SelectorEventLoop()


def main() -> None:
    print("LitSync is available at http://localhost:8000")
    loop = "run_litsync:windows_selector_loop_factory" if sys.platform == "win32" else "auto"
    uvicorn.run(
        "litsync_app.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop=loop,
    )


if __name__ == "__main__":
    main()
