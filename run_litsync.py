from __future__ import annotations

import threading
import time
import urllib.request
import webbrowser

import uvicorn


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
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
