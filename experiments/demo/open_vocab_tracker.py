"""YOLOE / YOLO-World open-vocabulary detector wrapper.

emits PerceptionSignal objects (one per detection) that the
ConsensusFuser can combine with other detectors. unlike CustomTracker
(which needs a basketball-specific weight file), open-vocab detectors
accept TEXT PROMPTS at inference and match them against an internal
image-text alignment — no custom weights required.

available backends (detected at construction time from ultralytics):

  yoloe       THU-MIG YOLOE, ICCV 2025. +3.5 AP over YOLO-World v2,
              matches YOLO11 speed. handles 1200+ categories via an
              internal vocabulary; text prompts optional.
  yoloworld   CVPR 2024 YOLO-World v2. older but stable.

both ship with ultralytics >= 8.3. we pick whichever the installed
version exposes; yoloe wins when both are present.

security posture: same as custom_tracker.py — YOLO(path.pt)
deserializes pickle. the weight files we use here are the official
open-weight releases from Ultralytics (yoloe-11s-seg.pt,
yolov8s-worldv2.pt, etc.) shipped directly from huggingface; not
arbitrary third-party checkpoints. if you want a pinned sha, record
the one ultralytics downloads on first use.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from ultralytics import YOLOE, YOLOWorld
except Exception:  # pragma: no cover
    YOLOE = None
    YOLOWorld = None

from fusion import PerceptionSignal


# canonical vocabulary → prompt phrases we use when the caller doesn't
# supply their own. biased toward what basketball footage actually contains.
DEFAULT_PROMPTS: dict[str, list[str]] = {
    "rim": ["basketball rim", "basketball hoop", "rim", "hoop"],
    "ball": ["basketball", "basketball ball", "orange basketball", "ball"],
    "player": ["basketball player", "person", "player"],
    "backboard": ["basketball backboard", "backboard"],
}


class OpenVocabTracker:
    """open-vocabulary detector wrapping YOLOE or YOLO-World.

    args:
        weights_path:  path to the ultralytics weight file. common choices:
                         yoloe-11s-seg.pt   smallest, fastest
                         yoloe-11m-seg.pt   medium (recommended default)
                         yoloe-11l-seg.pt   largest, highest accuracy
                       omit to let ultralytics download the default weight.
        backend:       "yoloe" (preferred) or "yoloworld". resolved at
                       construction; falls back to whichever is available.
        prompts:       optional {canonical_target: [phrase, ...]}. overrides
                       DEFAULT_PROMPTS. the prompts are set_classes()-ed on
                       the model once at init so inference is fast.
        imgsz:         inference resolution. 640 or 1280.
        confidence_threshold: per-class min confidence.
        source_tag:    how produced signals self-identify. default "yoloe".
    """

    def __init__(
        self,
        weights_path: str = "yoloe-11m-seg.pt",
        *,
        backend: str = "yoloe",
        prompts: dict[str, list[str]] | None = None,
        imgsz: int = 640,
        confidence_threshold: float = 0.15,
        source_tag: str | None = None,
    ) -> None:
        self.imgsz = int(imgsz)
        self.conf = float(confidence_threshold)
        self.prompts = dict(DEFAULT_PROMPTS)
        if prompts:
            self.prompts.update(prompts)

        # flatten to one list of phrases; remember phrase → canonical mapping.
        self._phrase_to_canonical: dict[str, str] = {}
        flat_phrases: list[str] = []
        for canonical, phrases in self.prompts.items():
            for p in phrases:
                if p not in self._phrase_to_canonical:
                    self._phrase_to_canonical[p] = canonical
                    flat_phrases.append(p)

        self.backend, self.model = self._build(backend, weights_path)
        self.source = source_tag or self.backend

        # wire the text prompts into the model (does the one-off text encoding).
        if hasattr(self.model, "set_classes"):
            self.model.set_classes(flat_phrases,
                                   self.model.get_text_pe(flat_phrases)
                                   if hasattr(self.model, "get_text_pe") else flat_phrases)
        elif hasattr(self.model, "set_classes_with_prompts"):
            self.model.set_classes_with_prompts(flat_phrases)

    def describe_prompts(self) -> dict:
        return {
            "backend": self.backend,
            "canonical_to_phrases": self.prompts,
            "phrase_to_canonical": self._phrase_to_canonical,
        }

    def update(self, frame: np.ndarray, frame_id: int
                ) -> list[PerceptionSignal]:
        """detect on one frame, emit PerceptionSignals bucketed by canonical."""
        if self.model is None:
            return []
        results = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
            save=False, device=None,
        )
        if not results:
            return []
        out: list[PerceptionSignal] = []
        boxes = results[0].boxes
        if boxes is None:
            return out
        names = self.model.names     # {int: phrase} after set_classes()
        for conf, cls, xyxy in zip(
            boxes.conf.tolist(),
            boxes.cls.int().tolist(),
            boxes.xyxy.tolist(),
        ):
            phrase = names.get(int(cls)) if isinstance(names, dict) else names[int(cls)]
            canonical = self._phrase_to_canonical.get(phrase)
            if canonical is None:
                continue
            out.append(PerceptionSignal(
                source=self.source, target=canonical,    # type: ignore[arg-type]
                frame_id=int(frame_id),
                confidence=float(conf),
                bbox=[float(v) for v in xyxy],
                extras={"phrase": phrase},
            ))
        return out

    # ---- internals ----------------------------------------------------------

    def _build(self, preferred: str, weights_path: str):
        """resolve the backend and load the weight file."""
        preferred = preferred.lower()
        if preferred == "yoloe" and YOLOE is not None:
            try:
                return "yoloe", YOLOE(weights_path)
            except Exception:
                pass   # fall through to the fallback
        if preferred in ("yoloworld", "yolo-world") and YOLOWorld is not None:
            return "yoloworld", YOLOWorld(weights_path)
        # fallback chain: yoloe → yoloworld.
        if YOLOE is not None:
            try:
                return "yoloe", YOLOE(weights_path)
            except Exception:
                pass
        if YOLOWorld is not None:
            return "yoloworld", YOLOWorld(weights_path)
        raise RuntimeError(
            "neither YOLOE nor YOLOWorld is available in this "
            "ultralytics install (need ultralytics >= 8.3)")


def iter_all_signals(trackers: Iterable[OpenVocabTracker],
                      frame: np.ndarray, frame_id: int
                      ) -> list[PerceptionSignal]:
    """run multiple open-vocab trackers on one frame; flatten the signals."""
    out: list[PerceptionSignal] = []
    for t in trackers:
        out.extend(t.update(frame, frame_id))
    return out
