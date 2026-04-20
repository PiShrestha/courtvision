"""offline parameter sweep for shot_attempt_v2 + shot_made_v2.

replays existing tracks.jsonl files (produced by previous run_demo.py runs)
through the v2 shot pipeline with varying tunables. no GPU / no perception
required. produces a new .events_v2.json and .meta_v2.json per (tracks x
param-set) combination.

usage:
    .venv/bin/python experiments/demo/offline_sweep_v2.py \
        --tracks-glob 'experiments/demo/outputs/run*.tracks.jsonl' \
        --matrix experiments/demo/matrix_v2_offline.csv \
        --outdir experiments/demo/outputs_v2

the matrix csv columns (header required):
    sweep_id, upward_trigger, history_frames, release_dist_px,
    approach_dist_px, cooldown_frames, occlusion_gap_frames,
    attempt_to_made_window, enter_zone_radius_factor,
    horizontal_pad_factor, min_downward_velocity

each tracks file combines with each sweep row → one output file per pair.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
import time
from pathlib import Path

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))

from _fg_stats import field_goal_stats       # noqa: E402
from hoop import Hoop                        # noqa: E402
from shot_pipeline_v2 import ShotPipelineV2  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracks-glob",
                    default="experiments/demo/outputs/run*.tracks.jsonl")
    ap.add_argument("--matrix",
                    default="experiments/demo/matrix_v2_offline.csv")
    ap.add_argument("--outdir", default="experiments/demo/outputs_v2")
    ap.add_argument("--hoop-config-dir",
                    default="experiments/demo/hoop_configs")
    ap.add_argument("--limit-tracks", type=int, default=0,
                    help="process only the first N tracks files (0 = all)")
    ap.add_argument("--only-video", default=None,
                    help="filter tracks files by video stem substring")
    return ap.parse_args()


def load_sweep_rows(matrix_path: Path) -> list[dict[str, str]]:
    with matrix_path.open() as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def resolve_hoop(tracks_path: Path, hoop_dir: Path) -> Hoop | None:
    """infer hoop config path from the tracks filename, which embeds the
    video stem: run5_1v1-mk_t268-338_... → hoop_configs/1v1-mk.json.
    """
    stem = tracks_path.name
    # find the first 1v1-* token.
    for token in stem.split("_"):
        if token.startswith("1v1-"):
            hp = hoop_dir / f"{token}.json"
            if hp.exists():
                return Hoop.from_json(hp)
    return None


def replay_tracks_file(
    tracks_path: Path,
    hoop: Hoop,
    row: dict[str, str],
) -> tuple[list[dict], dict]:
    pipe = ShotPipelineV2(
        anchor=hoop,
        tracker_kind="static",   # offline: no frames, static anchor only
        upward_trigger=float(row["upward_trigger"]),
        history_frames=int(row["history_frames"]),
        release_dist_px=float(row["release_dist_px"]),
        approach_dist_px=float(row["approach_dist_px"]),
        cooldown_frames=int(row["cooldown_frames"]),
        occlusion_gap_frames=int(row["occlusion_gap_frames"]),
        attempt_to_made_window=int(row["attempt_to_made_window"]),
        enter_zone_radius_factor=float(row["enter_zone_radius_factor"]),
        horizontal_pad_factor=float(row["horizontal_pad_factor"]),
        min_downward_velocity=float(row["min_downward_velocity"]),
    )
    events: list[dict] = []
    frames_processed = 0
    start = time.time()
    with tracks_path.open() as f:
        for line in f:
            frame_data = json.loads(line)
            frame_id = int(frame_data["frame_id"])
            tracks = frame_data.get("tracks", [])
            players = [t for t in tracks if t.get("class_name") == "player"]
            balls = [t for t in tracks if t.get("class_name") == "ball"]
            ball_bbox = balls[0]["bbox"] if balls else None
            events.extend(pipe.update(None, frame_id, ball_bbox, players))
            frames_processed += 1
    elapsed = time.time() - start

    fg = field_goal_stats(events)
    summary = {
        "frames_processed": frames_processed,
        "events_total": len(events),
        "possession_events": sum(1 for e in events if e["event"] == "possession"),
        "shot_attempt_events": sum(1 for e in events if e["event"] == "shot_attempt"),
        "shot_made_events": sum(1 for e in events if e["event"] == "shot_made"),
        "shot_miss_events": sum(1 for e in events if e["event"] == "shot_miss"),
        "replay_seconds": round(elapsed, 3),
        "field_goal_stats": fg,
    }
    return events, summary


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        print(f"matrix csv not found: {matrix_path}", file=sys.stderr); return 1
    rows = load_sweep_rows(matrix_path)
    if not rows:
        print("matrix is empty", file=sys.stderr); return 1

    tracks_files = [Path(p) for p in sorted(glob.glob(args.tracks_glob))]
    if args.only_video:
        tracks_files = [p for p in tracks_files if args.only_video in p.name]
    if args.limit_tracks > 0:
        tracks_files = tracks_files[: args.limit_tracks]
    if not tracks_files:
        print("no tracks files matched", file=sys.stderr); return 1

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    hoop_dir = Path(args.hoop_config_dir)

    total = len(tracks_files) * len(rows)
    done = 0
    skipped_no_hoop = 0
    for tracks_path in tracks_files:
        hoop = resolve_hoop(tracks_path, hoop_dir)
        if hoop is None:
            skipped_no_hoop += 1
            continue
        # also pull the matching meta.json so we carry perception context.
        source_meta: dict = {}
        # filenames contain "." in values like "c0.25" — can't use Path.with_suffix.
        meta_path = Path(str(tracks_path).replace(".tracks.jsonl", ".meta.json"))
        if meta_path.exists():
            try:
                source_meta = json.loads(meta_path.read_text())
            except Exception:
                source_meta = {}
        for row in rows:
            events, summary = replay_tracks_file(tracks_path, hoop, row)
            sweep_id = row["sweep_id"]
            base = tracks_path.name.replace(".tracks.jsonl", "")
            out_prefix = outdir / f"{base}__sw{sweep_id}"
            out_events = Path(str(out_prefix) + ".events_v2.json")
            out_meta = Path(str(out_prefix) + ".meta_v2.json")
            out_events.write_text(json.dumps(events, indent=2))
            meta: dict = {
                "source_tracks": str(tracks_path),
                "source_meta": str(meta_path) if meta_path.exists() else None,
                "sweep_id": sweep_id,
                "hoop": hoop.to_dict(),
                **{k: _coerce(v) for k, v in row.items() if k != "sweep_id"},
                **summary,
            }
            # inherit perception config from the source run so downstream
            # aggregation can slice v2 results by detector config too.
            for key in ("video", "start", "end", "model", "imgsz", "confidence",
                        "tracker_config"):
                if key in source_meta:
                    meta[f"source_{key}"] = source_meta[key]
            out_meta.write_text(json.dumps(meta, indent=2))
            done += 1
            if done % 25 == 0 or done == total:
                print(f"progress: {done}/{total}")
    print(f"done. wrote {done} pairs. skipped {skipped_no_hoop} tracks "
          f"files with no resolvable hoop config.")
    return 0


def _coerce(s: str) -> float | int | str:
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


if __name__ == "__main__":
    raise SystemExit(main())
