"""per-frame rim localisation via multi-scale template matching.

loads a gallery of tight rim crops (produced by extract_rim_templates.py)
and, for each frame, picks the best (template, scale) that matches. returns
a TrackedHoop-compatible dict so rim_tracker can consume it.

design notes:
- tm_ccoeff_normed is scale-invariant enough to distinguish rim vs not-rim
  but not truly scale-invariant — we loop over a small set of scales per
  template to cover zoom variation.
- the first-pass match can be expensive; we support a "search window"
  restricted to a box around the previous match to keep per-frame cost low.
- on cold start (no previous match) we search the whole frame at all scales
  and all templates to find a reliable anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _orange_fraction(bgr_crop: np.ndarray) -> float:
    """fraction of pixels in a bgr crop that fall in the rim-orange hsv range."""
    if bgr_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _ORANGE_LO1, _ORANGE_HI1)
    m2 = cv2.inRange(hsv, _ORANGE_LO2, _ORANGE_HI2)
    mask = cv2.bitwise_or(m1, m2)
    return float(mask.sum() / 255) / mask.size


SCALES = (0.45, 0.60, 0.80, 1.00, 1.25, 1.60)
DEFAULT_MIN_SCORE = 0.55   # tm_ccoeff_normed threshold for a confident match
DEFAULT_MIN_ORANGE_FRAC = 0.03   # matched region must have >=3% orange pixels

# hsv bounds for a basketball-rim orange; tuned once from rims_mk samples.
# two ranges because OpenCV hsv wraps at 180 and pure red is close to 0 and 180.
_ORANGE_LO1 = (0,   100, 100)
_ORANGE_HI1 = (15,  255, 255)
_ORANGE_LO2 = (160, 100, 100)
_ORANGE_HI2 = (180, 255, 255)


@dataclass
class TemplateMatch:
    center: tuple[int, int]
    radius: int
    score: float
    template_idx: int
    scale: float


class TemplateRimDetector:
    """detect the rim in a frame using a gallery of scaled templates."""

    def __init__(self, template_dir: str | Path,
                 min_score: float = DEFAULT_MIN_SCORE,
                 scales: tuple[float, ...] = SCALES,
                 upper_frac: float = 0.65,
                 min_orange_frac: float = DEFAULT_MIN_ORANGE_FRAC) -> None:
        """upper_frac restricts the cold-start search to the top fraction
        of the frame (outdoor rims live high). min_orange_frac rejects
        matches whose area has too few orange pixels (eliminates
        scoreboard/UI overlays that happen to template-match well)."""
        self.template_dir = Path(template_dir)
        self.min_score = float(min_score)
        self.scales = tuple(scales)
        self.upper_frac = float(upper_frac)
        self.min_orange_frac = float(min_orange_frac)
        self.templates = self._load_templates(self.template_dir)
        self.templates_bgr = self._load_templates_bgr(self.template_dir)
        if not self.templates:
            raise RuntimeError(f"no templates under {template_dir}")

    @staticmethod
    def _load_templates(d: Path) -> list[np.ndarray]:
        if not d.exists():
            return []
        out: list[np.ndarray] = []
        for p in sorted(d.glob("rim_*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None or min(img.shape) < 8:
                continue
            out.append(img)
        return out

    @staticmethod
    def _load_templates_bgr(d: Path) -> list[np.ndarray]:
        if not d.exists():
            return []
        return [cv2.imread(str(p)) for p in sorted(d.glob("rim_*.png"))
                if cv2.imread(str(p)) is not None]

    def detect(self, frame: np.ndarray,
               search_center: tuple[int, int] | None = None,
               search_half: int = 240) -> TemplateMatch | None:
        """return best match in frame, optionally constrained to a window."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        h, w = gray.shape

        # optional search window around last known center to reduce cost.
        # cold-start: restrict to upper portion of frame (outdoor rims live high).
        if search_center is not None:
            cx, cy = search_center
            x0 = max(0, cx - search_half); x1 = min(w, cx + search_half)
            y0 = max(0, cy - search_half); y1 = min(h, cy + search_half)
            roi = gray[y0:y1, x0:x1]
            offset = (x0, y0)
        else:
            y_cut = int(h * self.upper_frac)
            roi = gray[:y_cut, :]
            offset = (0, 0)

        # full-color frame used for the orange-gate post-check.
        frame_bgr = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        candidates: list[TemplateMatch] = []
        for ti, tmpl in enumerate(self.templates):
            th0, tw0 = tmpl.shape
            for s in self.scales:
                tw, th = max(8, int(round(tw0 * s))), max(8, int(round(th0 * s)))
                if tw >= roi.shape[1] or th >= roi.shape[0]:
                    continue
                scaled = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(roi, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val < self.min_score:
                    continue
                cx_abs = max_loc[0] + offset[0]
                cy_abs = max_loc[1] + offset[1]
                # orange-gate: reject matches whose patch has too few orange pixels
                patch = frame_bgr[cy_abs:cy_abs + th, cx_abs:cx_abs + tw]
                if _orange_fraction(patch) < self.min_orange_frac:
                    continue
                candidates.append(TemplateMatch(
                    center=(cx_abs + tw // 2, cy_abs + th // 2),
                    radius=max(tw, th) // 2,
                    score=float(max_val),
                    template_idx=ti, scale=float(s)))
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.score)
