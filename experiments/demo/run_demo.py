"""end-to-end demo: yolov8 detection + duo tracker + shot detector + viz.

usage:
    cd experiments/demo
    .venv/bin/python run_demo.py \
        --video ../../1v1-mk.mov --start 120 --end 180 \
        --hoop experiments/demo/hoop_configs/1v1-mk.json \
        --out experiments/demo/outputs/demo_1v1-mk_t120-180

produces under <out>:
    <out>.mp4           annotated video
    <out>.events.json   event log
    <out>.meta.json     run parameters + timings
    <out>.tracks.jsonl  per-frame tracks (one json object per line)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

# make sibling modules importable regardless of cwd.
SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))

# stage-1 tracker from the main project.
REPO_ROOT = SELF_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perception.tracker import Tracker             # noqa: E402  (after sys.path insert)

from duo_tracker import DuoTracker                 # noqa: E402
from hoop import Hoop, detect_hoop                 # noqa: E402
from shot_detector import ShotDetector             # noqa: E402
from viz import annotate_video                     # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--hoop", default=None, help="json hoop config path (optional).")
    ap.add_argument("--upward-trigger", type=float, default=-10.0)
    ap.add_argument("--made-window", type=int, default=45)
    ap.add_argument("--hoop-pad", type=int, default=12)
    ap.add_argument("--possession-dist-px", type=float, default=140.0)
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--out", required=True, help="output prefix (no extension)")
    return ap.parse_args()


def load_hoop(video_path: str, hoop_path: str | None,
              start: float, end: float | None) -> Hoop:
    if hoop_path and Path(hoop_path).exists():
        return Hoop.from_json(hoop_path)
    h = detect_hoop(video_path, start_seconds=start, end_seconds=end or start + 60.0)
    if h is None:
        raise RuntimeError("auto hoop detection failed; supply --hoop config")
    return h


def main() -> int:
    args = parse_args()
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    meta: dict = {
        "video": args.video,
        "start": args.start,
        "end": args.end,
        "model": args.model,
        "imgsz": args.imgsz,
        "confidence": args.confidence,
        "tracker_config": args.tracker,
        "upward_trigger": args.upward_trigger,
        "made_window": args.made_window,
        "hoop_pad": args.hoop_pad,
        "possession_dist_px": args.possession_dist_px,
    }

    hoop = load_hoop(args.video, args.hoop, args.start, args.end)
    meta["hoop"] = hoop.to_dict()

    tracker = Tracker(weights_path=args.model, confidence_threshold=args.confidence,
                       tracker_config=args.tracker, imgsz=args.imgsz)
    duo = DuoTracker()
    shot = ShotDetector(hoop=hoop, upward_velocity_trigger=args.upward_trigger,
                          made_window_frames=args.made_window, hoop_pad=args.hoop_pad,
                          possession_dist_px=args.possession_dist_px)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr); return 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    start_frame = int(round(args.start * fps))
    end_frame = int(round(args.end * fps)) if args.end is not None else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracks_by_frame: list[dict] = []
    events: list[dict] = []
    tracks_jsonl = out_prefix.with_suffix(".tracks.jsonl").open("w")

    frame_id = start_frame
    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if end_frame is not None and frame_id >= end_frame: break

            raw = tracker.update(frame, frame_id=frame_id)
            players_only = [t for t in raw if t.get("class_name") == "player"]
            balls_only = [t for t in raw if t.get("class_name") == "ball"]

            # duo-track the top players; keep ball as-is (single best box).
            duo_players = duo.update(frame, players_only, frame_id=frame_id)
            ball_box = balls_only[0]["bbox"] if balls_only else None
            frame_events = shot.update(frame_id, ball_box, duo_players)
            events.extend(frame_events)

            tracks = list(duo_players) + (balls_only[:1] if balls_only else [])
            frame_dict = {
                "frame_id": frame_id,
                "frame_height": frame.shape[0],
                "frame_width": frame.shape[1],
                "tracks": tracks,
            }
            tracks_by_frame.append(frame_dict)
            tracks_jsonl.write(json.dumps(frame_dict) + "\n")
            frame_id += 1
            processed += 1
    finally:
        tracks_jsonl.close(); cap.release()

    perception_s = time.time() - start_wall

    # write event log.
    events_path = out_prefix.with_suffix(".events.json")
    events_path.write_text(json.dumps(events, indent=2))

    # annotated video (slow; second pass over source).
    if args.save_video:
        video_out = out_prefix.with_suffix(".mp4")
        annotate_video(args.video, tracks_by_frame, events, hoop,
                        str(video_out), start_seconds=args.start, end_seconds=args.end)

    total_s = time.time() - start_wall
    poss = sum(1 for e in events if e["event"] == "possession")
    att = sum(1 for e in events if e["event"] == "shot_attempt")
    made = sum(1 for e in events if e["event"] == "shot_made")
    meta.update({
        "frames_processed": processed,
        "events_total": len(events),
        "possession_events": poss,
        "shot_attempt_events": att,
        "shot_made_events": made,
        "perception_seconds": round(perception_s, 2),
        "total_seconds": round(total_s, 2),
    })
    out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    print(f"done. processed {processed} frames, {len(events)} events "
           f"(possession={poss} attempts={att} made={made}) in {total_s:.1f}s")
    print(f"  events: {events_path}")
    print(f"  meta:   {out_prefix.with_suffix('.meta.json')}")
    if args.save_video:
        print(f"  video:  {out_prefix.with_suffix('.mp4')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
