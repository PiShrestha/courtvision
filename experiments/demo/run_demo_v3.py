"""v3 cli runner: v2 rules + optional yoloe / rtmpose / sam 3 fusion."""

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
from duo_tracker import DuoTracker                       # noqa: E402
from hoop import Hoop, detect_hoop                       # noqa: E402
from shot_pipeline_v3 import ShotPipelineV3, V3Config    # noqa: E402
from viz import annotate_video                           # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--model", default="yolo26l.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--confidence", type=float, default=0.30)
    ap.add_argument("--tracker", default="botsort.yaml")
    ap.add_argument("--hoop", default=None)
    ap.add_argument("--rim-tracker-kind", default="flow",
                    choices=["csrt", "kcf", "mil", "flow", "static"])
    ap.add_argument("--rim-reseed-every", type=int, default=90)
    ap.add_argument("--upward-trigger", type=float, default=-10.0)
    ap.add_argument("--history-frames", type=int, default=6)
    ap.add_argument("--release-dist-px", type=float, default=60.0)
    ap.add_argument("--approach-dist-px", type=float, default=260.0)
    ap.add_argument("--cooldown-frames", type=int, default=20)
    ap.add_argument("--possession-dist-px", type=float, default=140.0)
    ap.add_argument("--possession-switch-evidence", type=int, default=4)
    ap.add_argument("--possession-switch-ratio", type=float, default=1.25)
    ap.add_argument("--occlusion-gap-frames", type=int, default=15)
    ap.add_argument("--attempt-to-made-window", type=int, default=120)
    ap.add_argument("--enter-zone-radius-factor", type=float, default=2.5)
    ap.add_argument("--horizontal-pad-factor", type=float, default=2.0)
    ap.add_argument("--min-downward-velocity", type=float, default=1.0)
    # v3 backends
    ap.add_argument("--use-yoloe", action="store_true")
    ap.add_argument("--yoloe-weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--yoloe-confidence", type=float, default=0.15)
    ap.add_argument("--use-pose", action="store_true")
    ap.add_argument("--pose-device", default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--pose-mode", default="balanced",
                    choices=["lightweight", "balanced", "performance"])
    ap.add_argument("--pose-score-threshold", type=float, default=0.5)
    ap.add_argument("--sam3-cache", default=None,
                    help="path to a pre-built sam3 jsonl cache for this clip")
    ap.add_argument("--strict-consensus", action="store_true",
                    help="require >=2 sources and conf>=strict_min for fused rim")
    ap.add_argument("--strict-min-confidence", type=float, default=0.8)
    ap.add_argument("--strict-min-sources", type=int, default=2)
    ap.add_argument("--save-video", action="store_true", default=True)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def load_hoop(video_path: str, hoop_path: str | None,
              start: float, end: float | None) -> Hoop:
    if hoop_path and Path(hoop_path).exists():
        return Hoop.from_json(hoop_path)
    h = detect_hoop(video_path, start_seconds=start,
                    end_seconds=end or start + 60.0)
    if h is None:
        raise RuntimeError("auto hoop detection failed; supply --hoop config")
    return h


def _best_hoop_bbox(yoloe_rim_signals, anchor_center) -> list[float] | None:
    # yoloe rim detection nearest the json anchor.
    if not yoloe_rim_signals:
        return None
    ax, ay = anchor_center

    def _d(s):
        cx, cy = (s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2
        return (cx - ax) ** 2 + (cy - ay) ** 2
    return min(yoloe_rim_signals, key=_d).bbox


def main() -> int:
    args = parse_args()
    out_prefix = Path(args.out); out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # probe source fps to scale frame-based rule windows.
    _probe = cv2.VideoCapture(args.video)
    fps = float(_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    _probe.release()

    start_wall = time.time()
    meta: dict = {k: getattr(args, k) for k in (
        "video", "start", "end", "model", "imgsz", "confidence", "tracker",
        "rim_tracker_kind", "rim_reseed_every",
        "upward_trigger", "history_frames", "release_dist_px",
        "approach_dist_px", "cooldown_frames", "possession_dist_px",
        "possession_switch_evidence", "possession_switch_ratio",
        "occlusion_gap_frames", "attempt_to_made_window",
        "enter_zone_radius_factor", "horizontal_pad_factor",
        "min_downward_velocity",
    )}
    meta["pipeline"] = "v3"
    meta["fps"] = round(fps, 3)
    meta["backends"] = {
        "yoloe": bool(args.use_yoloe),
        "pose": bool(args.use_pose),
        "sam3_cache": args.sam3_cache,
    }

    hoop = load_hoop(args.video, args.hoop, args.start, args.end)
    meta["hoop"] = hoop.to_dict()

    tracker = Tracker(weights_path=args.model,
                      confidence_threshold=args.confidence,
                      tracker_config=args.tracker, imgsz=args.imgsz)
    duo = DuoTracker()

    cfg = V3Config(
        tracker_kind=args.rim_tracker_kind,
        reseed_every=args.rim_reseed_every,
        upward_trigger=args.upward_trigger,
        history_frames=args.history_frames,
        release_dist_px=args.release_dist_px,
        approach_dist_px=args.approach_dist_px,
        cooldown_frames=args.cooldown_frames,
        possession_dist_px=args.possession_dist_px,
        possession_switch_evidence=args.possession_switch_evidence,
        possession_switch_ratio=args.possession_switch_ratio,
        occlusion_gap_frames=args.occlusion_gap_frames,
        attempt_to_made_window=args.attempt_to_made_window,
        enter_zone_radius_factor=args.enter_zone_radius_factor,
        horizontal_pad_factor=args.horizontal_pad_factor,
        min_downward_velocity=args.min_downward_velocity,
        fps=fps,
        use_yoloe=args.use_yoloe,
        use_pose=args.use_pose,
        sam3_cache_path=args.sam3_cache,
        pose_score_threshold=args.pose_score_threshold,
        strict_consensus=args.strict_consensus,
        strict_min_confidence=args.strict_min_confidence,
        strict_min_sources=args.strict_min_sources,
    )
    pipe = ShotPipelineV3(anchor=hoop, config=cfg)

    # optional backends
    ovt = None
    if args.use_yoloe:
        from open_vocab_tracker import OpenVocabTracker
        ovt = OpenVocabTracker(weights_path=args.yoloe_weights, backend="yoloe",
                                imgsz=args.imgsz,
                                confidence_threshold=args.yoloe_confidence,
                                source_tag="yoloe")
    pose_est = None
    if args.use_pose:
        from pose_estimator import PoseEstimator
        pose_est = PoseEstimator(device=args.pose_device, mode=args.pose_mode)

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

    yoloe_total = 0.0
    pose_total = 0.0

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
            duo_players = duo.update(frame, players_only, frame_id=frame_id)
            ball_box = balls_only[0]["bbox"] if balls_only else None

            yoloe_signals = None
            if ovt is not None:
                t0 = time.time()
                yoloe_signals = ovt.update(frame, frame_id)
                yoloe_total += time.time() - t0

            pose_signals = None
            if pose_est is not None and duo_players:
                t0 = time.time()
                pose_signals = pose_est.emit_signals(frame, duo_players, frame_id)
                pose_total += time.time() - t0

            frame_events = pipe.update(
                frame, frame_id, ball_box, duo_players,
                yoloe_signals=yoloe_signals,
                pose_signals=pose_signals,
            )
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
                       str(video_out), start_seconds=args.start,
                       end_seconds=args.end, rim_trace=hoop_trace)

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
        "backend_timings": {
            "yoloe_seconds": round(yoloe_total, 2),
            "pose_seconds": round(pose_total, 2),
        },
        "field_goal_stats": fg,
    })
    out_prefix.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    out_prefix.with_suffix(".rim_trace.jsonl").write_text(
        "\n".join(json.dumps(h) for h in hoop_trace))

    att = meta["shot_attempt_events"]; made = meta["shot_made_events"]
    miss = meta["shot_miss_events"]
    print(f"v3 done. {processed} frames. attempts={att} made={made} miss={miss} "
           f"fg_pct={fg['overall']['fg_pct']}  total={total_s:.1f}s "
           f"yoloe={yoloe_total:.1f}s pose={pose_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
