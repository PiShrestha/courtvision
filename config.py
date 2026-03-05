"""loads config.yaml and merges cli overrides on top."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent
