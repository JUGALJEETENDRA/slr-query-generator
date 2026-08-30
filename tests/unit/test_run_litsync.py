from __future__ import annotations

import asyncio
from unittest.mock import patch

from run_litsync import main, windows_selector_loop_factory


def test_windows_loop_factory_returns_selector_loop() -> None:
    loop = windows_selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_launcher_uses_stable_windows_loop() -> None:
    with patch("run_litsync.uvicorn.run") as run:
        main()

    run.assert_called_once_with(
        "litsync_app.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="run_litsync:windows_selector_loop_factory",
    )
