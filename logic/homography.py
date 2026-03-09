"""planar homography for mapping pixels to court coordinates."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class Homography:
    """transforms a pixel (px, py) to a court (x, y) on the floor plane.

    typical setup:
        h = Homography()
        h.set_correspondences(pixel_pts, court_pts)   # at least 4 pairs
        h.compute_homography()
