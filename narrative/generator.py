"""event log -> scouting summary. uses gemini when the api key is set,
otherwise falls back to a deterministic local summary."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional at import
    genai = None


class NarrativeGenerator:
    """two-mode summary generator: gemini api or local fallback."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash"):
        self.model = model
        key = api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_enabled = False
        if genai is not None and key:
            genai.configure(api_key=key)
            self.gemini_enabled = True

    def generate(self, events: list[dict[str, Any]]) -> str:
        if self.gemini_enabled and genai is not None:
            prompt = self._build_prompt(events)
            response = genai.GenerativeModel(self.model).generate_content(prompt)
            return response.text or ""
        return self._local_summary(events)

    # ---- helpers ------------------------------------------------------------

    def _build_prompt(self, events: list[dict[str, Any]]) -> str:
        """format the event log into a scouting prompt for gemini."""
        lines = [
            "You are a basketball scout. Summarize this 1v1 event log.",
            "Keep it concise, factual, and mention who was most efficient.",
            "Event log:",
        ]
        for e in events:
            lines.append(
                f"frame={e.get('frame_id')} event={e.get('event')} player={e.get('player')}"
            )
        return "\n".join(lines)

    def _local_summary(self, events: list[dict[str, Any]]) -> str:
        """deterministic fallback when no api key is configured."""
