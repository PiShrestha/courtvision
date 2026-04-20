"""per-player + overall field-goal stats from an event stream.

keeps the shape used by both run_demo_v2.py and offline_sweep_v2.py so
the aggregator can read either pathway without branching.
"""

from __future__ import annotations

from typing import Iterable


def field_goal_stats(events: Iterable[dict]) -> dict:
    """summarise shot events into per-player and overall FG stats.

    shape:
        {"per_player": {player_id: {attempts, made, miss, fg_pct}},
         "overall":   {attempts, made, fg_pct}}
    """
    per_player: dict[int, dict[str, int]] = {}
    for e in events:
        pid = int(e.get("player", -1))
        d = per_player.setdefault(pid, {"attempts": 0, "made": 0, "miss": 0})
        label = e.get("event")
        if label == "shot_attempt":
            d["attempts"] += 1
        elif label == "shot_made":
            d["made"] += 1
        elif label == "shot_miss":
            d["miss"] += 1

    out: dict[int, dict] = {}
    total_att = total_made = 0
    for pid, d in per_player.items():
        total_att += d["attempts"]
        total_made += d["made"]
        pct = d["made"] / d["attempts"] if d["attempts"] else None
        out[pid] = {**d, "fg_pct": round(pct, 3) if pct is not None else None}

    overall = {
        "attempts": total_att,
        "made": total_made,
        "fg_pct": round(total_made / total_att, 3) if total_att else None,
    }
    return {"per_player": out, "overall": overall}
