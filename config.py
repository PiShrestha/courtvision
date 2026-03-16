"""loads config.yaml and merges cli overrides on top."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load(path: str | Path | None = None) -> dict[str, Any]:
    """read the yaml config from disk."""
    target = Path(path) if path else DEFAULT_CONFIG_PATH
    with target.open() as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict[str, Any], args: Namespace) -> dict[str, Any]:
    """overlay any non-none cli argument on top of the config dict."""
    mapping = {
        # cli flag -> (section, key)
