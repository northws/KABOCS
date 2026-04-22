"""
EventHub — thread-safe bridge fan-out to multiple SSE subscribers.

A single background task drains events from ``bridge.events`` (a
``queue.Queue`` written by the optimizer worker thread) and broadcasts
each event to every currently-subscribed ``asyncio.Queue``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webui.backend.ui_bridge import WebUIBridge


class EventHub:
    """Fan-out hub: one producer (worker thread) → many consumers (SSE)."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._dispatch_task: asyncio.Task | None = None
        self._bridge_ref: "WebUIBridge | None" = None
        self._replay: list[dict] = []

    # ------------------------------------------------------------------
    def set_bridge(self, bridge: "WebUIBridge") -> None:
        """(Re)bind the hub to the currently active bridge and clear replay."""
        self._bridge_ref = bridge
        self._replay = []

    def record(self, event: dict) -> None:
        # keep last N events for new subscribers joining mid-run
        self._replay.append(event)
        if len(self._replay) > 500:
            self._replay = self._replay[-500:]

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self) -> None:
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except Exception:
                pass
            self._dispatch_task = None

    # ------------------------------------------------------------------
    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        # replay buffered history so late subscribers see prior context
        for ev in list(self._replay):
            await q.put(ev)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    # ------------------------------------------------------------------
    async def _dispatch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            # Pull one event from the worker-thread queue without blocking
            # the event loop.
            bridge = self._bridge_ref
            if bridge is None:
                await asyncio.sleep(0.2)
                continue
            try:
                event = await loop.run_in_executor(
                    None, lambda: bridge.events.get(timeout=0.5)
                )
            except Exception:
                continue
            if event is None:
                continue
            self.record(event)
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass
