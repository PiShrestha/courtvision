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
