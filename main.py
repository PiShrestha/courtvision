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
