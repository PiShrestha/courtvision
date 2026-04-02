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
        court_xy = h.pixel_to_court(px, py)
    """

    def __init__(self):
        self.pixel_points: np.ndarray | None = None
        self.court_points: np.ndarray | None = None
        self.H: np.ndarray | None = None

    def set_correspondences(self, pixel_pts, court_pts) -> None:
        """store pixel<->court pairs. needs >=4 correspondences."""
        pixel_arr = np.asarray(pixel_pts, dtype=np.float32)
        court_arr = np.asarray(court_pts, dtype=np.float32)

        if pixel_arr.shape != court_arr.shape:
            raise ValueError(
                f"pixel and court shapes must match, got {pixel_arr.shape} vs {court_arr.shape}"
            )
        if pixel_arr.ndim != 2 or pixel_arr.shape[1] != 2:
            raise ValueError(f"correspondences must be shape (N, 2), got {pixel_arr.shape}")
        if pixel_arr.shape[0] < 4:
            raise ValueError(f"need at least 4 correspondences, got {pixel_arr.shape[0]}")

        self.pixel_points = pixel_arr
        self.court_points = court_arr

    def compute_homography(self) -> None:
        """solve for the 3x3 matrix using ransac."""
        if self.pixel_points is None or self.court_points is None:
            raise RuntimeError("call set_correspondences() first")
        H, _ = cv2.findHomography(self.pixel_points, self.court_points, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("cv2.findHomography failed")
        self.H = H

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """project a single pixel to court coordinates."""
        if self.H is None:
            raise RuntimeError("call compute_homography() first")
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
