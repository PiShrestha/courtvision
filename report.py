"""builds the text stats report from pipeline outputs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2


def video_metadata(video_path: str) -> dict[str, Any]:
    """read basic metadata (fps, frame count, duration) from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "duration_s": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = (frame_count / fps) if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frame_count, "duration_s": duration}


def summarize_stats(tracks_by_frame: list[dict], events: list[dict]) -> dict[str, Any]:
    """roll tracks + events into counts for the final report."""
    class_counts: Counter[str] = Counter()
    for frame in tracks_by_frame:
        for track in frame.get("tracks", []):
            class_counts[str(track.get("class_name"))] += 1

    event_counts = Counter(str(e.get("event")) for e in events)

    per_player: dict[int, Counter[str]] = {}
    for event in events:
        player = event.get("player")
        if not isinstance(player, int):
            continue
        per_player.setdefault(player, Counter())[str(event.get("event"))] += 1

    return {
        "frames_processed": len(tracks_by_frame),
        "detection_counts": dict(class_counts),
        "event_counts": dict(event_counts),
        "per_player_events": {k: dict(v) for k, v in per_player.items()},
    }


def write_text_report(
    out_path: str,
    video_path: str,
    metadata: dict[str, Any],
    stats: dict[str, Any],
    summary: str,
) -> None:
    """write a human-readable report mixing stats and the narrative summary."""
    lines: list[str] = []
    lines += [
        "CourtVision Stats Report",
        "=" * 60,
        "",
        f"Video: {video_path}",
        f"FPS: {metadata.get('fps', 0.0):.2f}",
        f"Frames in file: {metadata.get('frame_count', 0)}",
        f"Duration (s): {metadata.get('duration_s', 0.0):.2f}",
        f"Frames processed: {stats.get('frames_processed', 0)}",
        "",
    ]

    lines += ["Detections", "-" * 60]
    for class_name, count in sorted(stats.get("detection_counts", {}).items()):
        lines.append(f"{class_name}: {count}")
    lines.append("")

    lines += ["Events", "-" * 60]
    for event_name, count in sorted(stats.get("event_counts", {}).items()):
        lines.append(f"{event_name}: {count}")
    lines.append("")

    lines += ["Per-Player Event Counts", "-" * 60]
    per_player = stats.get("per_player_events", {})
    if not per_player:
        lines.append("No player-attributed events detected.")
    else:
        for player in sorted(per_player):
            lines.append(f"Player {player}: {per_player[player]}")
    lines.append("")

    lines += ["Summary", "-" * 60, summary.strip(), ""]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
