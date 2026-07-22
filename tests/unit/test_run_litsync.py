from __future__ import annotations

import asyncio

from run_litsync import windows_safe_loop_factory


def test_windows_safe_loop_factory_uses_selector_event_loop() -> None:
    loop = windows_safe_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()
