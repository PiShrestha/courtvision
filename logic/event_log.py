"""append-only ordered log of basketball events."""

from __future__ import annotations
from typing import Any


class EventLog:
    """simple wrapper that collects events from the rule engine."""

    def __init__(self):
        self._events: list[dict[str, Any]] = []

    def add_events(self, events: list[dict[str, Any]]) -> None:
        self._events.extend(events)

    def to_list(self) -> list[dict[str, Any]]:
        return list(self._events)

    def to_dataframe(self):
        """return events as a pandas dataframe, or none if pandas is missing."""
        try:
            import pandas as pd
        except Exception:
            return None
        return pd.DataFrame(self._events)
