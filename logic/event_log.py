"""append-only ordered log of basketball events."""

from __future__ import annotations
from typing import Any


class EventLog:
    """simple wrapper that collects events from the rule engine."""

    def __init__(self):
        self._events: list[dict[str, Any]] = []

    def add_events(self, events: list[dict[str, Any]]) -> None:
        self._events.extend(events)
