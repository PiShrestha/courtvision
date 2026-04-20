"""YOLOv8 tracker that consumes a custom basketball checkpoint with a
`hoop` class (and keeps working on COCO weights when no custom model
is provided).

Motivation: perception/tracker.py (main pipeline) is hard-coded to
COCO class ids {0: player, 32: ball} — it drops every other class.
When the user provides a basketball-specific checkpoint that emits
`hoop`, `rim`, or `backboard`, we want those detections to reach the
shot pipeline as well.

Design:
- Reads the loaded YOLO model's `names` dict at construction time and
  builds a case-insensitive alias map to our canonical vocabulary
  {"player", "ball", "hoop", "backboard"}.
- Emits the same {track_id, bbox, confidence, class_name} dict shape
  as perception/tracker.Tracker. Callers filtering by
  class_name == "ball" keep working; new code filtering by "hoop"
  picks up the new class automatically.
- Passes through Ultralytics tracking state identically to the main
  Tracker.

Security note: `YOLO(path)` deserializes pickle. Only point this at a
weights file you trust. See download_basketball_model.py for the
curated download recipes.
"""

from __future__ import annotations

from typing import Any

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional at import
    YOLO = None


# canonical domain vocabulary the rest of the demo assumes.
CANONICAL = ("player", "ball", "hoop", "backboard")

# default aliases map model-emitted class names (lowercased, hyphen-or-underscore-
# normalized) to our canonical names. extend via the `class_aliases` constructor arg.
DEFAULT_ALIASES: dict[str, str] = {
    # player variants
    "person": "player",
    "player": "player",
    "players": "player",
    # ball variants
    "ball": "ball",
    "basketball": "ball",
    "sports-ball": "ball",
    "sports_ball": "ball",
    # hoop variants (the whole reason this module exists)
    "hoop": "hoop",
    "rim": "hoop",
    "basketball-hoop": "hoop",
    "basketball_hoop": "hoop",
    "net": "hoop",
    # backboard (useful for bank-shot classification later)
    "backboard": "backboard",
    "board": "backboard",
}


def _canonicalise(name: str) -> str:
    """normalise a class name for alias lookup."""
    return name.lower().replace(" ", "-").replace("_", "-").strip()


class CustomTracker:
    """drop-in for perception.tracker.Tracker that supports hoop class.

    args:
        weights_path: path to a .pt file. COCO weights work too (they
            yield only player + ball, as before).
        confidence_threshold: min detection confidence.
        tracker_config: bytetrack.yaml or botsort.yaml.
        imgsz: inference resolution.
        class_aliases: extra {emitted_name_lowercase: canonical} pairs.
            lets callers map a checkpoint's idiosyncratic labels
            (e.g. "HOOP_L", "rim_area") to the canonical vocab.
        confidence_per_class: optional {canonical_name: min_conf} to
            apply tighter thresholds per class (e.g. hoop=0.5).
    """

    def __init__(
        self,
        weights_path: str,
        confidence_threshold: float = 0.35,
        tracker_config: str = "bytetrack.yaml",
        imgsz: int = 640,
        class_aliases: dict[str, str] | None = None,
        confidence_per_class: dict[str, float] | None = None,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.tracker_config = tracker_config
        self.imgsz = int(imgsz)
        self.per_class_conf = {k: float(v) for k, v in (confidence_per_class or {}).items()}
        self.model = YOLO(weights_path) if YOLO is not None else None
        self._class_id_to_canonical = self._build_class_map(class_aliases or {})

    # ---- public api ---------------------------------------------------------

    def update(self, frame, frame_id: int) -> list[dict[str, Any]]:
        """detect + track one bgr frame, return canonical detections."""
        if self.model is None:
            return []

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        # some detection pipelines return boxes without track ids (first frame,
        # or hoops when no appearance to match). we keep hoops without track
        # ids as "instance-only" detections with track_id=-1 so RimTracker can
        # still consume them.
        has_ids = boxes.id is not None
        track_ids = boxes.id.int().tolist() if has_ids else [-1] * len(boxes)

        tracks: list[dict[str, Any]] = []
        for tid, conf, cls_id, xyxy in zip(
            track_ids,
            boxes.conf.tolist(),
            boxes.cls.int().tolist(),
            boxes.xyxy.tolist(),
        ):
            canon = self._class_id_to_canonical.get(int(cls_id))
            if canon is None:
                continue
            gate = self.per_class_conf.get(canon, self.confidence_threshold)
            if conf < gate:
                continue
            tracks.append({
                "track_id": int(tid),
                "bbox": [float(v) for v in xyxy],
                "confidence": float(conf),
                "class_name": canon,
            })
        return tracks

    def reset(self) -> None:
        """clear tracker state between independent videos."""
        if self.model is None:
            return
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        for t in getattr(predictor, "trackers", []) or []:
            if hasattr(t, "reset"):
                t.reset()

    def describe_class_map(self) -> dict[int, str]:
        """return the resolved class-id -> canonical name map (for logging)."""
        return dict(self._class_id_to_canonical)

    # ---- internals ----------------------------------------------------------

    def _build_class_map(self, extra_aliases: dict[str, str]) -> dict[int, str]:
        """resolve the loaded model's class names against the alias table.

        returns a dict mapping the model's integer class id to our canonical
        name, or {} if the model doesn't expose a `names` attribute.
        """
        if self.model is None:
            return {}
        names = getattr(self.model, "names", None) or {}
        aliases = {_canonicalise(k): v for k, v in DEFAULT_ALIASES.items()}
        aliases.update({_canonicalise(k): v for k, v in extra_aliases.items()})

        resolved: dict[int, str] = {}
        for cls_id, raw_name in names.items():
            key = _canonicalise(str(raw_name))
            target = aliases.get(key)
            if target in CANONICAL:
                resolved[int(cls_id)] = target
        return resolved
