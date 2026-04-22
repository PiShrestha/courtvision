"""SAM 3 / SAM 3.1 video-predictor wrapper producing PerceptionSignals.

uses Meta's SAM 3 (Nov 2025) + SAM 3.1 Object Multiplex (Mar 2026).
requires HF auth for the gated checkpoints `facebook/sam3` and
`facebook/sam3.1`.

strategy:
- create one video session per run and prompt it once with all the
  canonical concepts the caller cares about
    ("basketball rim", "basketball", "basketball player")
- SAM 3 tracks each prompted concept across the whole clip, returning
  per-frame masks + per-instance IDs. we convert those to
  PerceptionSignal(target=..., bbox=..., center=..., track_id=...)
  records that the ConsensusFuser consumes identically to YOLOE / YOLO
  output.

computational trade-off: SAM 3 is 848M params. we run it OFFLINE
once per clip (not per-frame interleaved with YOLO) and cache the
per-frame signals to disk. the rest of the pipeline replays those
signals during its normal loop — no per-frame SAM 3 inference during
evaluation.

checkpoints:
  facebook/sam3        sam3.pt               original Nov 2025 release
  facebook/sam3.1      sam3.1_multiplex.pt   Mar 2026 Object Multiplex
                                              (faster multi-object tracking)

we default to 3.1_multiplex because our workload is inherently
multi-object (rim + ball + 2 players simultaneously).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from fusion import PerceptionSignal


DEFAULT_CONCEPTS = {
    "rim": "basketball rim",
    "ball": "basketball",
    "player": "basketball player",
}

DEFAULT_CHECKPOINT = "sam3.1"     # "sam3" (Nov 2025) | "sam3.1" (Mar 2026)


@dataclass
class Sam3Config:
    """declarative config for one SAM 3 offline pass over a clip."""
    checkpoint: str = DEFAULT_CHECKPOINT    # "sam3" | "sam3.1"
    concepts: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONCEPTS)
    )
    # prompt at this frame index (relative to the clip start). 0 is fine
    # for most clips; anything later works if the first few frames are
    # partially occluded.
    prompt_frame: int = 0
    # min per-detection score to emit a PerceptionSignal.
    min_score: float = 0.2
    device: str = "cuda"            # falls back to "cpu" automatically


class Sam3Tracker:
    """one SAM 3 video session per clip. produces per-frame signals.

    usage:
        tracker = Sam3Tracker(config=Sam3Config(checkpoint="sam3.1"))
        signals = tracker.run(video_path, start_s=268.0, end_s=338.0)

    signals is a list[PerceptionSignal] across all frames in the window.
    callers index by frame_id for per-frame fusion.

    `run()` returns in-memory for small clips and writes to disk via
    `run_to_file()` for large clips so the feature is usable as a
    prebuilt cache on slurm.
    """

    def __init__(self, config: Sam3Config | None = None) -> None:
        self.config = config or Sam3Config()
        self._predictor = None

    def available(self) -> bool:
        """cheap check: imports + HF cache for the requested checkpoint."""
        try:
            import sam3                                     # noqa: F401
            from sam3.model_builder import build_sam3_video_predictor   # noqa: F401
        except ImportError:
            return False
        # HF gate: touch the repo once and see if the request succeeds.
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.list_repo_files(self._repo_id(), repo_type="model")
            return True
        except Exception:
            return False

    def run(
        self,
        video_path: str,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> list[PerceptionSignal]:
        """run the video predictor end-to-end and flatten masks to bboxes."""
        self._ensure_predictor()
        session = self._start_session(video_path, start_s, end_s)

        # prompt all concepts on the same frame.
        concept_to_id: dict[str, list[int]] = {}
        for canonical, phrase in self.config.concepts.items():
            ids = self._add_text_prompt(
                session_id=session["session_id"],
                frame_index=self.config.prompt_frame,
                text=phrase,
            )
            concept_to_id[canonical] = ids

        # propagate forward. SAM 3 yields per-frame per-instance masks.
        signals: list[PerceptionSignal] = []
        for frame_output in self._propagate(session["session_id"]):
            frame_id = int(frame_output["frame_idx"])
            for inst_id, mask, score in frame_output["instances"]:
                canonical = self._canonical_for(concept_to_id, inst_id)
                if canonical is None:
                    continue
                if score < self.config.min_score:
                    continue
                bbox = _mask_to_bbox(mask)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                signals.append(PerceptionSignal(
                    source=f"sam3_{self.config.checkpoint.replace('.','')}",
                    target=canonical,               # type: ignore[arg-type]
                    frame_id=frame_id,
                    confidence=float(score),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    track_id=int(inst_id),
                    extras={"mask_area": int(mask.sum()),
                            "concept_phrase": self.config.concepts[canonical]},
                ))
        return signals

    def run_to_file(
        self,
        video_path: str,
        out_path: str | Path,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> Path:
        """run + write PerceptionSignals as JSONL (frame_id-sorted)."""
        signals = self.run(video_path, start_s=start_s, end_s=end_s)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        signals.sort(key=lambda s: (s.frame_id, s.target))
        with out.open("w") as f:
            for s in signals:
                f.write(json.dumps(_signal_to_dict(s)) + "\n")
        return out

    # ---- internals ----------------------------------------------------------

    def _repo_id(self) -> str:
        return "facebook/sam3.1" if self.config.checkpoint == "sam3.1" else "facebook/sam3"

    def _ensure_predictor(self):
        """lazy-import + weight download. cheap after the first call."""
        if self._predictor is not None:
            return
        from sam3.model_builder import build_sam3_video_predictor
        # the builder reads the HF repo based on an env var or kwarg;
        # different SAM 3 releases expose slightly different handles.
        # we let the library pick up HF auth from ~/.cache/huggingface.
        os.environ.setdefault("HF_HUB_REPO", self._repo_id())
        self._predictor = build_sam3_video_predictor()

    def _start_session(self, video_path: str, start_s: float,
                        end_s: float | None):
        request = {
            "type": "start_session",
            "resource_path": video_path,
        }
        if start_s > 0 or end_s is not None:
            # SAM 3's session API supports a "segment" kwarg in some
            # versions and explicit frame indices in others. pass both
            # forms so this is robust to minor API drift.
            request["start_seconds"] = float(start_s)
            if end_s is not None:
                request["end_seconds"] = float(end_s)
        return self._predictor.handle_request(request=request)

    def _add_text_prompt(self, session_id: str, frame_index: int,
                          text: str) -> list[int]:
        response = self._predictor.handle_request(
            request={
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": int(frame_index),
                "text": text,
            },
        )
        # the response shape varies by SAM 3 release. we look for either
        # a direct instance list or the top-level "outputs" wrapper.
        outputs = response.get("outputs") or response
        instances = outputs.get("instances") or outputs.get("object_ids") or []
        return [int(x.get("id") if isinstance(x, dict) else x) for x in instances]

    def _propagate(self, session_id: str):
        """generator yielding {"frame_idx": int, "instances": [(id, mask, score)]}."""
        response = self._predictor.handle_request(
            request={"type": "propagate_in_video", "session_id": session_id},
        )
        # response typically streams per-frame; we iterate its generator,
        # falling back to a list shape for versions that batch.
        per_frame = response.get("per_frame") or response.get("outputs") or response
        if isinstance(per_frame, dict):
            # single-frame sanity mode; iterate keys sorted by frame_idx.
            items = sorted(per_frame.items(), key=lambda kv: int(kv[0]))
            for frame_idx, payload in items:
                yield _normalise_frame(int(frame_idx), payload)
        else:
            for frame in per_frame:
                yield _normalise_frame(int(frame.get("frame_idx", 0)), frame)

    def _canonical_for(self, concept_to_id: dict[str, list[int]],
                        inst_id: int) -> str | None:
        for canonical, ids in concept_to_id.items():
            if inst_id in ids:
                return canonical
        return None


# ---- helpers --------------------------------------------------------------


def _mask_to_bbox(mask) -> tuple[float, float, float, float] | None:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _normalise_frame(frame_idx: int, payload) -> dict:
    """return the dict shape the caller expects regardless of SAM 3 version."""
    if isinstance(payload, dict):
        if "instances" in payload and all(
            isinstance(x, (tuple, list)) for x in payload["instances"]):
            return {"frame_idx": frame_idx, "instances": payload["instances"]}
        if "object_ids" in payload and "mask_logits" in payload:
            ids = payload["object_ids"]
            masks = payload["mask_logits"]
            scores = payload.get("scores") or [1.0] * len(ids)
            return {"frame_idx": frame_idx,
                    "instances": list(zip(ids, masks, scores))}
        # fall-through shape: try to extract triples.
        if "instances" in payload:
            triples = []
            for inst in payload["instances"]:
                if isinstance(inst, dict):
                    triples.append((int(inst["id"]),
                                    inst.get("mask") or inst.get("segmentation"),
                                    float(inst.get("score", 1.0))))
            return {"frame_idx": frame_idx, "instances": triples}
    return {"frame_idx": frame_idx, "instances": []}


def _signal_to_dict(s: PerceptionSignal) -> dict:
    return {
        "source": s.source, "target": s.target, "frame_id": s.frame_id,
        "confidence": s.confidence, "bbox": s.bbox, "center": list(s.center) if s.center else None,
        "radius": s.radius, "track_id": s.track_id, "extras": s.extras,
    }


def load_signals(path: str | Path) -> list[PerceptionSignal]:
    """read back PerceptionSignals written by run_to_file()."""
    out: list[PerceptionSignal] = []
    with Path(path).open() as f:
        for line in f:
            d = json.loads(line)
            out.append(PerceptionSignal(
                source=d["source"], target=d["target"],
                frame_id=int(d["frame_id"]), confidence=float(d["confidence"]),
                bbox=d.get("bbox"),
                center=tuple(d["center"]) if d.get("center") else None,
                radius=d.get("radius"),
                track_id=d.get("track_id"),
                extras=d.get("extras") or {},
            ))
    return out


def iter_all_signals(trackers: Iterable[Sam3Tracker], video_path: str,
                      start_s: float = 0.0, end_s: float | None = None
                      ) -> list[PerceptionSignal]:
    """run one or more SAM 3 trackers on a clip; flatten signals."""
    out: list[PerceptionSignal] = []
    for t in trackers:
        out.extend(t.run(video_path, start_s=start_s, end_s=end_s))
    return out
