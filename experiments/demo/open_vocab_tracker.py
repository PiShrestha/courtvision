"""open-vocab detector (yoloe / yolo-world) emitting PerceptionSignals."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    from ultralytics import YOLOE, YOLOWorld
except Exception:  # pragma: no cover
    YOLOE = None
    YOLOWorld = None

from fusion import PerceptionSignal


# canonical target -> default prompt phrases for basketball footage.
DEFAULT_PROMPTS: dict[str, list[str]] = {
    "rim": ["basketball rim", "basketball hoop", "rim", "hoop"],
    "ball": ["basketball", "basketball ball", "orange basketball", "ball"],
    "player": ["basketball player", "person", "player"],
    "backboard": ["basketball backboard", "backboard"],
}


class OpenVocabTracker:
    """yoloe / yolo-world detector; prompts text phrases at init."""

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

        # flatten phrases; keep a phrase -> canonical map for dispatch.
        self._phrase_to_canonical: dict[str, str] = {}
        flat_phrases: list[str] = []
        for canonical, phrases in self.prompts.items():
            for p in phrases:
                if p not in self._phrase_to_canonical:
                    self._phrase_to_canonical[p] = canonical
                    flat_phrases.append(p)

        self.backend, self.model = self._build(backend, weights_path)
        self.source = source_tag or self.backend

        # one-off text encoding of the prompt phrases into the model.
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
        # detect on one frame; emit PerceptionSignals per canonical target.
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
        # resolve backend (yoloe preferred, yoloworld fallback) and load weights.
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
    # run multiple trackers on a frame; flatten signals.
    out: list[PerceptionSignal] = []
    for t in trackers:
        out.extend(t.update(frame, frame_id))
    return out
