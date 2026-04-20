#!/usr/bin/env python3
"""courtvision entry point. runs perception -> symbolic -> narrative on a video."""

from __future__ import annotations

import argparse

import config as app_config
from logic.event_log import EventLog
from logic.homography import Homography
from logic.rules import RuleEngine
from narrative.generator import NarrativeGenerator
from perception.pipeline import PerceptionPipeline
from report import summarize_stats, video_metadata, write_text_report


def run_perception(video_path: str, cfg: dict) -> list[dict]:
    """stage 1: detect + track every frame in the configured window."""
    p = cfg["perception"]
    v = cfg["video"]
    homography = (
        Homography.from_config(cfg["homography_config"])
        if cfg.get("homography_config")
        else None
    )
    pipeline = PerceptionPipeline(
        video_path,
        weights_path=p["model"],
        homography=homography,
        start_seconds=v["start_seconds"],
        end_seconds=v.get("end_seconds"),
        stride=v["stride"],
        confidence_threshold=p["confidence"],
        imgsz=p["imgsz"],
        tracker_config=p["tracker"],
    )
    return list(pipeline.run())


def run_logic(tracks_by_frame: list[dict], cfg: dict) -> list[dict]:
    """stage 2: deterministic rule engine over per-frame tracks."""
    s = cfg["symbolic"]
    engine = RuleEngine(
        court_possession_dist=s["court_possession_dist"],
        pixel_possession_dist=s["pixel_possession_dist"],
    )
    log = EventLog()
    for frame in tracks_by_frame:
        log.add_events(engine.process_frame(frame))
    return log.to_list()


def run_narrative(events: list[dict], cfg: dict) -> str:
    """stage 3: event log -> scouting summary."""
    gen = NarrativeGenerator(model=cfg["narrative"]["gemini_model"])
    return gen.generate(events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CourtVision - neuro-symbolic basketball analytics."
    )
    parser.add_argument("--video", required=True, help="input video file.")
    parser.add_argument("--config", default=None, help="yaml config file (default: ./config.yaml).")
    parser.add_argument("--out-txt", dest="out_txt", default=None, help="output report path.")

    # overrides; any unset flag falls back to config.yaml.
    parser.add_argument("--start", type=float, default=None, help="start time in seconds.")
    parser.add_argument("--end", type=float, default=None, help="end time in seconds.")
    parser.add_argument("--stride", type=int, default=None, help="process every nth frame.")
    parser.add_argument("--model", default=None, help="yolo weights or shorthand.")
    parser.add_argument("--confidence", type=float, default=None, help="detector threshold.")
    parser.add_argument("--imgsz", type=int, default=None, help="yolo inference resolution.")
    parser.add_argument("--tracker", default=None, help="bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--homography-config", dest="homography_config", default=None,
                         help="json file with pixel<->court correspondences.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = app_config.apply_cli_overrides(app_config.load(args.config), args)

    print("=" * 50)
    print("  CourtVision - neuro-symbolic basketball analytics")
    print("=" * 50, "\n")

    v, p = cfg["video"], cfg["perception"]
    print("* stage 1: perception")
    end_label = "EOF" if v.get("end_seconds") is None else f"{v['end_seconds']}s"
    print(f"  window: start={v['start_seconds']}s end={end_label} stride={v['stride']}")
    print(f"  model={p['model']} conf={p['confidence']} imgsz={p['imgsz']} tracker={p['tracker']}")
    tracks = run_perception(args.video, cfg)
    print(f"  got {len(tracks)} frames of tracking data.\n")

    print("* stage 2: symbolic reasoning")
    events = run_logic(tracks, cfg)
    print(f"  detected {len(events)} game events.\n")

    print("* stage 3: narrative report")
    summary = run_narrative(events, cfg)
    print(summary)

    out_path = cfg.get("out_txt") or "outputs/stats_report.txt"
    write_text_report(
        out_path=out_path,
        video_path=args.video,
        metadata=video_metadata(args.video),
        stats=summarize_stats(tracks, events),
        summary=summary,
    )
    print(f"\nSaved report: {out_path}")


if __name__ == "__main__":
    main()
