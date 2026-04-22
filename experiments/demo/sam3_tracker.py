"""sam 3 / sam 3.1 video predictor wrapper emitting PerceptionSignals."""

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

DEFAULT_CHECKPOINT = "sam3.1"


@dataclass
class Sam3Config:
    """config for one sam 3 offline pass over a clip."""
    checkpoint: str = DEFAULT_CHECKPOINT        # "sam3" | "sam3.1"
    concepts: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONCEPTS)
    )
    prompt_frame: int = 0
    min_score: float = 0.2
    device: str = "cuda"


class Sam3Tracker:
    """one sam 3 video session per clip; emits per-frame PerceptionSignals."""

    def __init__(self, config: Sam3Config | None = None) -> None:
        self.config = config or Sam3Config()
        self._predictor = None

    def available(self) -> bool:
        # imports + hf gate in one cheap check.
        try:
            import sam3                                     # noqa: F401
            from sam3.model_builder import build_sam3_video_predictor   # noqa: F401
        except ImportError:
            return False
        try:
            from huggingface_hub import HfApi
            HfApi().list_repo_files(self._repo_id(), repo_type="model")
            return True
        except Exception:
            return False

    def run(
        self,
        video_path: str,
        start_s: float = 0.0,
        end_s: float | None = None,
    ) -> list[PerceptionSignal]:
        self._ensure_predictor()
        session = self._start_session(video_path, start_s, end_s)

        # prompt all concepts at the seed frame.
        concept_to_id: dict[str, list[int]] = {}
        for canonical, phrase in self.config.concepts.items():
            ids = self._add_text_prompt(
                session_id=session["session_id"],
                frame_index=self.config.prompt_frame,
                text=phrase,
            )
            concept_to_id[canonical] = ids

        # propagate forward; one signal per (frame, instance) above threshold.
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
        # run + write jsonl sorted by (frame_id, target).
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
        # lazy import + weight download; cheap after first call.
        if self._predictor is not None:
            return
        from sam3.model_builder import build_sam3_video_predictor
        os.environ.setdefault("HF_HUB_REPO", self._repo_id())
        self._predictor = build_sam3_video_predictor()

    def _start_session(self, video_path: str, start_s: float,
                        end_s: float | None):
        # sam 3's start_session does not accept start/end kwargs; callers
        # must pre-cut the clip. offload_video_to_cpu keeps frames on host ram.
        return self._predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_path,
                "offload_video_to_cpu": True,
            },
        )

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
        outputs = response.get("outputs")
        if outputs is None:
            return []
        obj_ids = _safe_list(outputs.get("out_obj_ids"))
        return [int(x) for x in obj_ids]

    def _propagate(self, session_id: str):
        # propagate is a stream request (yields per frame).
        for response in self._predictor.handle_stream_request(
            request={
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
            },
        ):
            frame_idx = int(response.get("frame_index", 0))
            out = response.get("outputs")
            if out is None:
                yield {"frame_idx": frame_idx, "instances": []}
                continue
            obj_ids = _safe_list(out.get("out_obj_ids"))
            masks_raw = out.get("out_binary_masks")
            masks = list(masks_raw) if masks_raw is not None else []
            probs_raw = out.get("output_probs")
            probs = _safe_list(probs_raw) if probs_raw is not None \
                    else [1.0] * len(obj_ids)
            instances = []
            for oid, mask, prob in zip(obj_ids, masks, probs):
                instances.append((int(oid), mask, float(prob)))
            yield {"frame_idx": frame_idx, "instances": instances}

    def _canonical_for(self, concept_to_id: dict[str, list[int]],
                        inst_id: int) -> str | None:
        for canonical, ids in concept_to_id.items():
            if inst_id in ids:
                return canonical
        return None


# ---- helpers --------------------------------------------------------------


def _mask_to_bbox(mask) -> tuple[float, float, float, float] | None:
    # mask -> axis-aligned bbox. accepts torch tensor (cpu/cuda), ndarray, or list.
    if hasattr(mask, "detach") and hasattr(mask, "cpu"):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _safe_list(obj):
    # torch tensor / ndarray / list / None -> python list.
    if obj is None:
        return []
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return list(obj)


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
    # replay PerceptionSignals written by run_to_file().
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
    # run multiple trackers on a clip; flatten the output.
    out: list[PerceptionSignal] = []
    for t in trackers:
        out.extend(t.run(video_path, start_s=start_s, end_s=end_s))
    return out
