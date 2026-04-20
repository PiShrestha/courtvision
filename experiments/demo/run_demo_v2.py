"""end-to-end v2 demo: yolov8 + duo tracker + rim tracker + shot pipeline v2.

mirrors run_demo.py but swaps the v1 ShotDetector for ShotPipelineV2 (which
includes RimTracker for per-frame rim localisation and the stricter
attempt/made rules).

usage:
    .venv/bin/python experiments/demo/run_demo_v2.py \
        --video 1v1-mk.mov --start 268.37 --end 338.09 \
        --hoop experiments/demo/hoop_configs/1v1-mk.json \
        --model yolov8m.pt --imgsz 1280 --confidence 0.30 \
        --out experiments/demo/outputs_v2/livemk_268-338

produces four artefacts with the same schema as v1 plus a field_goal_stats
block in meta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))
REPO_ROOT = SELF_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perception.tracker import Tracker                  # noqa: E402

from _fg_stats import field_goal_stats                   # noqa: E402
from custom_tracker import CustomTracker                  # noqa: E402
from duo_tracker import DuoTracker                       # noqa: E402
from hoop import Hoop, detect_hoop                       # noqa: E402
from shot_pipeline_v2 import ShotPipelineV2              # noqa: E402
from viz import annotate_video                           # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--hoop", default=None)
    # rim tracker
    ap.add_argument("--rim-tracker-kind", default="static",
                    choices=["csrt", "kcf", "mil", "static"])
    ap.add_argument("--rim-reseed-every", type=int, default=90)
    # attempt v2
    ap.add_argument("--upward-trigger", type=float, default=-10.0)
    ap.add_argument("--history-frames", type=int, default=6)
    ap.add_argument("--release-dist-px", type=float, default=60.0)
    ap.add_argument("--approach-dist-px", type=float, default=260.0)
    ap.add_argument("--cooldown-frames", type=int, default=20)
    ap.add_argument("--possession-dist-px", type=float, default=140.0)
    # made v2
    ap.add_argument("--occlusion-gap-frames", type=int, default=10)
    ap.add_argument("--attempt-to-made-window", type=int, default=90)
    ap.add_argument("--enter-zone-radius-factor", type=float, default=2.0)
    ap.add_argument("--horizontal-pad-factor", type=float, default=1.5)
    ap.add_argument("--min-downward-velocity", type=float, default=1.5)
    # custom basketball model (optional — routes `hoop` class into the rim
    # tracker per-frame, bypassing CSRT/MIL drift).
    ap.add_argument("--custom-model", default=None,
                    help="path to a YOLOv8 weights file with a hoop/rim class")
    ap.add_argument("--hoop-conf", type=float, default=0.4,
                    help="min confidence for hoop detections (default 0.4)")
    # io
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def _best_hoop(hoops: list[dict], anchor_center: tuple[int, int]) -> list[float] | None:
    """pick the hoop detection closest to the JSON anchor (resolves ties
    when a full-court clip sees both rims in one frame)."""
    if not hoops:
        return None
    ax, ay = anchor_center

    def _dist(h):
        x1, y1, x2, y2 = h["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (cx - ax) ** 2 + (cy - ay) ** 2

    return min(hoops, key=_dist)["bbox"]


def load_hoop(video_path: str, hoop_path: str | None,
              start: float, end: float | None) -> Hoop:
    if hoop_path and Path(hoop_path).exists():
        return Hoop.from_json(hoop_path)
    h = detect_hoop(video_path, start_seconds=start,
                    end_seconds=end or start + 60.0)
    if h is None:
        raise RuntimeError("auto hoop detection failed; supply --hoop config")
    return h


def main() -> int:
    args = parse_args()
    out_prefix = Path(args.out); out_prefix.parent.mkdir(parents=True, exist_ok=True)

    start_wall = time.time()
    meta: dict = {k: getattr(args, k) for k in (
        "video", "start", "end", "model", "imgsz", "confidence", "tracker",
        "rim_tracker_kind", "rim_reseed_every",
        "upward_trigger", "history_frames", "release_dist_px",
        "approach_dist_px", "cooldown_frames", "possession_dist_px",
        "occlusion_gap_frames", "attempt_to_made_window",
        "enter_zone_radius_factor", "horizontal_pad_factor",
        "min_downward_velocity",
    )}
    meta["pipeline"] = "v2"

    hoop = load_hoop(args.video, args.hoop, args.start, args.end)
    meta["hoop"] = hoop.to_dict()

    # read the source fps early so we can scale rule windows at pipeline init.
    _cap_probe = cv2.VideoCapture(args.video)
    fps = float(_cap_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    _cap_probe.release()
    meta["fps"] = round(fps, 3)

    if args.custom_model:
        # custom basketball checkpoint with a hoop class routes per-frame
        # rim detections into the rim tracker; the JSON hoop is kept only
        # as a fallback for frames where no hoop is detected.
        tracker = CustomTracker(
            weights_path=args.custom_model,
            confidence_threshold=args.confidence,
            tracker_config=args.tracker, imgsz=args.imgsz,
            confidence_per_class={"hoop": args.hoop_conf},
        )
        meta["custom_model"] = args.custom_model
        meta["custom_class_map"] = tracker.describe_class_map()
        print(f"* custom model class map: {meta['custom_class_map']}")
    else:
        tracker = Tracker(weights_path=args.model,
                          confidence_threshold=args.confidence,
                          tracker_config=args.tracker, imgsz=args.imgsz)
    duo = DuoTracker()
    pipe = ShotPipelineV2(
        anchor=hoop,
        tracker_kind=args.rim_tracker_kind,
        reseed_every=args.rim_reseed_every,
        upward_trigger=args.upward_trigger,
        history_frames=args.history_frames,
        release_dist_px=args.release_dist_px,
        approach_dist_px=args.approach_dist_px,
        cooldown_frames=args.cooldown_frames,
        possession_dist_px=args.possession_dist_px,
        occlusion_gap_frames=args.occlusion_gap_frames,
        attempt_to_made_window=args.attempt_to_made_window,
        enter_zone_radius_factor=args.enter_zone_radius_factor,
        horizontal_pad_factor=args.horizontal_pad_factor,
        min_downward_velocity=args.min_downward_velocity,
        fps=fps,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr); return 1
    start_frame = int(round(args.start * fps))
    end_frame = int(round(args.end * fps)) if args.end is not None else None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracks_by_frame: list[dict] = []
    events: list[dict] = []
    hoop_trace: list[dict] = []
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
            hoops_only = [t for t in raw if t.get("class_name") == "hoop"]
            duo_players = duo.update(frame, players_only, frame_id=frame_id)
            ball_box = balls_only[0]["bbox"] if balls_only else None
            # highest-confidence hoop wins. typical custom YOLO returns 0-2
            # hoop detections per frame; we take the one nearest the JSON
            # anchor to break ties when a full-court clip sees both rims.
            hoop_box = _best_hoop(hoops_only, anchor_center=hoop.center) if hoops_only else None

            frame_events = pipe.update(frame, frame_id, ball_box, duo_players,
                                          hoop_bbox=hoop_box)
            events.extend(frame_events)
            th = pipe.current_hoop()
            hoop_trace.append({
                "frame_id": frame_id, "center": list(th.center),
                "radius": int(th.radius), "source": th.source,
            })

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
    events_path = out_prefix.with_suffix(".events.json")
    events_path.write_text(json.dumps(events, indent=2))

    if args.save_video:
        video_out = out_prefix.with_suffix(".mp4")
        annotate_video(args.video, tracks_by_frame, events, hoop,
                       str(video_out), start_seconds=args.start, end_seconds=args.end)

    total_s = time.time() - start_wall
    fg = field_goal_stats(events)
    meta.update({
        "frames_processed": processed,
        "events_total": len(events),
        "possession_events": sum(1 for e in events if e["event"] == "possession"),
        "shot_attempt_events": sum(1 for e in events if e["event"] == "shot_attempt"),
        "shot_made_events": sum(1 for e in events if e["event"] == "shot_made"),
        "shot_miss_events": sum(1 for e in events if e["event"] == "shot_miss"),
        "perception_seconds": round(perception_s, 2),
        "total_seconds": round(total_s, 2),
        "field_goal_stats": fg,
    })
    out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    # rim trace is verbose; dump to a sibling file for debugging.
    out_prefix.with_suffix(".rim_trace.jsonl").write_text(
        "\n".join(json.dumps(h) for h in hoop_trace))

    att = meta["shot_attempt_events"]; made = meta["shot_made_events"]
    miss = meta["shot_miss_events"]
    print(f"v2 done. {processed} frames. attempts={att} made={made} miss={miss} "
          f"fg_pct={fg['overall']['fg_pct']}  in {total_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
