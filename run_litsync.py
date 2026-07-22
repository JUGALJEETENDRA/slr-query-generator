from __future__ import annotations

import asyncio
import sys
import threading
import time
import urllib.request
import webbrowser

import uvicorn


def windows_safe_loop_factory() -> asyncio.AbstractEventLoop:
    """Avoid a Proactor accept/shutdown race on Windows.

    Uvicorn otherwise selects ProactorEventLoop for a single-process Windows
    server.  Python 3.13 can run a pending accept callback after the server has
    closed, which produces an AssertionError during an otherwise clean stop.
    The selector loop supports LitSync's socket server without that race.
    """
    return asyncio.SelectorEventLoop()


def open_when_ready() -> None:
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/status", timeout=1):
                webbrowser.open("http://localhost:8000")
                return
        except OSError:
            time.sleep(0.5)
    print("LitSync started, but the browser could not be opened automatically.")
    print("Open http://localhost:8000 in your browser.")


if __name__ == "__main__":
    threading.Thread(target=open_when_ready, daemon=True).start()
    loop = "run_litsync:windows_safe_loop_factory" if sys.platform == "win32" else "auto"
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop=loop,
    )
