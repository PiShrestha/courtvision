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
        if not events:
            return "No events were detected. Check camera angle, model weights, and ball visibility."

        per_player = _group_by_player(events)
        lines = ["CourtVision Summary", "", f"Total events: {len(events)}"]
        for player in sorted(per_player):
            c = per_player[player]
            attempts = c.get("shot_attempt", 0)
            made = c.get("shot_made", 0)
            pct = (100.0 * made / attempts) if attempts else 0.0
            lines.append(
                f"Player {player}: possessions={c.get('possession', 0)}, "
                f"shot_attempts={attempts}, shot_made={made}, fg%={pct:.1f}"
            )

        leader = _leader_by_fg(per_player)
        if leader is not None:
            lines += ["", f"Scouting note: Player {leader} was most efficient in this clip."]
        return "\n".join(lines)


def _group_by_player(events: list[dict[str, Any]]) -> dict[int, Counter[str]]:
    grouped: dict[int, Counter[str]] = {}
    for e in events:
        player = e.get("player")
        if not isinstance(player, int):
            continue
        grouped.setdefault(player, Counter())[str(e.get("event"))] += 1
    return grouped


def _leader_by_fg(per_player: dict[int, Counter[str]]) -> int | None:
    best_player, best_score = None, -1.0
    for player, c in per_player.items():
        attempts = c.get("shot_attempt", 0)
        made = c.get("shot_made", 0)
        score = (made / attempts) if attempts else 0.0
        if score > best_score:
            best_score, best_player = score, player
    return best_player
