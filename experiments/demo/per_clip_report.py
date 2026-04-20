"""turn a set of v2 runs into a structured per-clip report.

metrics (per user request, 2026-04-20):
  - shot_attempts per player (P1 / P2)
  - shot_made per player
  - shooting percentage = made / attempts per player (and overall)
  - possession time per player (seconds), plus who-has-it-when timeline

expects a directory of v2 runs produced by run_demo_v2.py — each run
has `*.meta.json`, `*.events.json`, and optionally `*.tracks.jsonl`.

usage:
    .venv/bin/python experiments/demo/per_clip_report.py \\
        --runs 'experiments/demo/outputs_v2_new_videos/*.meta.json' \\
        --out  experiments/demo/analysis_v2_new_videos/per_clip_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs",
                    default="experiments/demo/outputs_v2_new_videos/*.meta.json")
    ap.add_argument("--out",
                    default="experiments/demo/analysis_v2_new_videos/per_clip_report.md")
    return ap.parse_args()


def possession_seconds(events: list[dict], fps: float,
                        window_end_frame: int) -> dict[int, float]:
    """sum seconds each player holds possession between consecutive
    possession events (treating window_end_frame as the final boundary).
    """
    segments: list[tuple[int, int, int]] = []   # (start_frame, end_frame, player)
    last_player: int | None = None
    last_frame: int | None = None
    for e in events:
        if e.get("event") != "possession":
            continue
        f = int(e["frame_id"])
        p = int(e.get("player", -1))
        if last_player is not None and last_frame is not None:
            segments.append((last_frame, f, last_player))
        last_player, last_frame = p, f
    if last_player is not None and last_frame is not None:
        segments.append((last_frame, window_end_frame, last_player))

    seconds: dict[int, float] = defaultdict(float)
    for s, e, p in segments:
        seconds[p] += max(0, e - s) / max(1.0, fps)
    return dict(seconds)


def fg_per_player(events: list[dict]) -> dict[int, dict[str, int]]:
    per: dict[int, dict[str, int]] = defaultdict(
        lambda: {"attempts": 0, "made": 0, "miss": 0})
    for e in events:
        pid = int(e.get("player", -1))
        label = e.get("event")
        if label == "shot_attempt":
            per[pid]["attempts"] += 1
        elif label == "shot_made":
            per[pid]["made"] += 1
        elif label == "shot_miss":
            per[pid]["miss"] += 1
    return dict(per)


def fmt_pct(made: int, attempts: int) -> str:
    if attempts == 0:
        return "—"
    return f"{100 * made / attempts:.0f}%"


def fmt_secs(s: float) -> str:
    mins, secs = divmod(s, 60)
    if mins >= 1:
        return f"{int(mins)}m{int(secs):02d}s"
    return f"{secs:.1f}s"


def clip_report(meta_path: Path) -> dict:
    """load a single run's artefacts and compute its metrics."""
    meta = json.loads(meta_path.read_text())
    events_path = Path(str(meta_path).replace(".meta.json", ".events.json"))
    events = json.loads(events_path.read_text()) if events_path.exists() else []

    fps = float(meta.get("fps") or 30.0)
    start_f = int(round(float(meta.get("start", 0)) * fps))
    end_f = int(round(float(meta.get("end", 0)) * fps))
    duration_s = (end_f - start_f) / max(1.0, fps)

    fg = fg_per_player(events)
    poss = possession_seconds(events, fps, end_f)

    # focus on P1 + P2 only (per user requirement — background people ignored).
    primary = {}
    for pid in (1, 2):
        a = fg.get(pid, {}).get("attempts", 0)
        m = fg.get(pid, {}).get("made", 0)
        miss = fg.get(pid, {}).get("miss", 0)
        primary[pid] = {
            "attempts": a, "made": m, "miss": miss,
            "fg_pct": fmt_pct(m, a),
            "possession_s": round(poss.get(pid, 0.0), 1),
        }

    # anything that leaked past P1/P2 (escalated ids under old DuoTracker)
    # gets bucketed as "other" so the user can see noise at a glance.
    other_players = [p for p in fg if p not in (1, 2)]
    other = {
        "player_ids": other_players,
        "attempts": sum(fg[p]["attempts"] for p in other_players),
        "made": sum(fg[p]["made"] for p in other_players),
        "possession_s": round(sum(poss.get(p, 0.0) for p in other_players), 1),
    }

    # overall P1+P2 totals.
    total_att = sum(primary[p]["attempts"] for p in (1, 2))
    total_made = sum(primary[p]["made"] for p in (1, 2))
    total_poss = sum(primary[p]["possession_s"] for p in (1, 2))

    return {
        "run": meta_path.stem.replace(".meta", ""),
        "video": meta.get("video"),
        "start": meta.get("start"),
        "end": meta.get("end"),
        "duration_s": round(duration_s, 1),
        "fps": round(fps, 3),
        "model": meta.get("model"),
        "imgsz": meta.get("imgsz"),
        "hoop": meta.get("hoop"),
        "frames_processed": meta.get("frames_processed", 0),
        "total_events": meta.get("events_total", 0),
        "primary": primary,
        "other": other,
        "overall": {
            "attempts": total_att,
            "made": total_made,
            "fg_pct": fmt_pct(total_made, total_att),
            "possession_s": round(total_poss, 1),
            "possession_coverage": (round(total_poss / duration_s, 2)
                                    if duration_s > 0 else None),
        },
    }


def render_md(reports: list[dict]) -> str:
    lines: list[str] = [
        "# Per-clip report (v2)",
        "",
        "Focus is strictly the two primary slots (P1, P2). Any tracks "
        "that escalated past those ids are bucketed as 'other' — if that "
        "bucket is non-empty the DuoTracker lost identity during the clip.",
        "",
        f"total clips: **{len(reports)}**",
        "",
        "## headline table",
        "",
        "| clip | window | dur | fps | P1 att/made/FG% | P2 att/made/FG% | overall FG% | P1 poss | P2 poss | other |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        p1 = r["primary"][1]; p2 = r["primary"][2]
        clip = Path(r["video"] or "?").stem
        window = f"{r['start']:.0f}-{r['end']:.0f}s"
        lines.append(
            f"| {clip} | {window} | {fmt_secs(r['duration_s'])} | "
            f"{r['fps']} | {p1['attempts']}/{p1['made']}/{p1['fg_pct']} | "
            f"{p2['attempts']}/{p2['made']}/{p2['fg_pct']} | "
            f"{r['overall']['fg_pct']} | "
            f"{fmt_secs(p1['possession_s'])} | {fmt_secs(p2['possession_s'])} | "
            f"{r['other']['attempts'] + r['other']['made']} events "
            f"from ids {r['other']['player_ids'] or 'none'} |"
        )

    # per-clip detail.
    for r in reports:
        clip = Path(r["video"] or "?").stem
        window = f"{r['start']:.0f}–{r['end']:.0f}s"
        lines.extend([
            "",
            f"---",
            "",
            f"## {clip}  ({window})",
            "",
            f"- **duration:** {fmt_secs(r['duration_s'])} at {r['fps']} fps "
            f"({r['frames_processed']} frames processed)",
            f"- **model:** {r['model']} @ imgsz={r['imgsz']}",
            f"- **hoop:** {r['hoop']}",
            f"- **total events logged:** {r['total_events']}",
            "",
            "### per-player metrics",
            "",
            "| player | attempts | made | miss | FG% | possession |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for pid in (1, 2):
            d = r["primary"][pid]
            lines.append(
                f"| P{pid} | {d['attempts']} | {d['made']} | {d['miss']} | "
                f"{d['fg_pct']} | {fmt_secs(d['possession_s'])} |"
            )
        lines.append(
            f"| **total** | **{r['overall']['attempts']}** | "
            f"**{r['overall']['made']}** | "
            f"{r['primary'][1]['miss'] + r['primary'][2]['miss']} | "
            f"**{r['overall']['fg_pct']}** | "
            f"{fmt_secs(r['overall']['possession_s'])} "
            f"({r['overall']['possession_coverage']} coverage) |"
        )
        if r["other"]["player_ids"]:
            lines.extend([
                "",
                f"_DuoTracker identity warning:_ saw additional player ids "
                f"{r['other']['player_ids']} carrying "
                f"{r['other']['attempts']} attempts, "
                f"{r['other']['made']} makes, "
                f"{fmt_secs(r['other']['possession_s'])} possession. "
                "Treat this clip's per-player split cautiously.",
            ])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.runs))]
    if not paths:
        print(f"no runs matched {args.runs}")
        return 1
    reports = [clip_report(p) for p in paths]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(reports))
    (out.parent / "per_clip_report.json").write_text(
        json.dumps(reports, indent=2, default=str))
    print(f"wrote {out} ({len(reports)} clips)")
    print(f"and   {out.parent / 'per_clip_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
