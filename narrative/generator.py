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
