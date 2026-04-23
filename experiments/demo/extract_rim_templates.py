"""extract tight rim crops from rims_mk/*.png using sam 3.1 text grounding.

each input png (a screenshot or still showing the basketball hoop) is
converted to a single-frame mp4, fed through Sam3Tracker with the rim
concept, and cropped to a tight bbox around the returned mask. the
cropped rim PNGs are saved to <out_dir>/rim_<N>.png and a manifest.json
records the bboxes + chosen phrases so the template matcher can resume
without re-running sam3.

usage:
    python experiments/demo/extract_rim_templates.py \\
        --input-dir experiments/demo/rims_mk \\
        --out-dir   experiments/demo/rims_mk/templates \\
        --pad-frac  0.12
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sam3_tracker import Sam3Tracker, Sam3Config


def png_to_frame_mp4(png: Path, fps: int = 30, n_frames: int = 6) -> Path:
    """wrap a still png in a tiny mp4. sam3 start_session needs a video path.
    we use a short duplicated clip so the predictor has something to propagate."""
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    # ffmpeg: loop the still for n_frames / fps seconds at fps.
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(png),
        "-t", f"{n_frames / fps:.3f}", "-r", str(fps),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(tmp),
    ]
    subprocess.run(cmd, check=True)
    return tmp


def tight_bbox_from_signals(signals, min_conf: float = 0.25) -> tuple[int,int,int,int] | None:
    """union of rim bboxes across frames, returned as the enclosing rect."""
    bbs = [s.bbox for s in signals
           if s.target == "rim" and s.confidence >= min_conf and s.bbox]
    if not bbs:
        return None
    arr = np.array(bbs, dtype=np.float32)
    x1 = float(arr[:, 0].min()); y1 = float(arr[:, 1].min())
    x2 = float(arr[:, 2].max()); y2 = float(arr[:, 3].max())
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def pad_bbox(bbox: tuple[int,int,int,int], img_shape, frac: float) -> tuple[int,int,int,int]:
    h, w = img_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1; bh = y2 - y1
    px = int(round(bw * frac)); py = int(round(bh * frac))
    return (max(0, x1 - px), max(0, y1 - py),
            min(w, x2 + px), min(h, y2 + py))


def extract_one(png: Path, tracker: Sam3Tracker, out_dir: Path,
                pad_frac: float) -> dict:
    """run sam3 on one still, save the tight rim crop, return metadata."""
    mp4 = png_to_frame_mp4(png)
    try:
        signals = tracker.run(str(mp4), start_s=0.0)
    finally:
        mp4.unlink(missing_ok=True)

    bb = tight_bbox_from_signals(signals)
    img = cv2.imread(str(png))
    if bb is None:
        return {"source": png.name, "status": "no_rim_signal",
                "chosen_phrases": dict(tracker._chosen_phrases)}

    padded = pad_bbox(bb, img.shape, pad_frac)
    x1, y1, x2, y2 = padded
    crop = img[y1:y2, x1:x2]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rim_{png.stem}.png"
    cv2.imwrite(str(out_path), crop)
    return {
        "source":          png.name,
        "status":          "ok",
        "tight_bbox":      list(bb),
        "padded_bbox":     list(padded),
        "crop":            out_path.name,
        "crop_size":       [int(crop.shape[1]), int(crop.shape[0])],
        "chosen_phrases":  dict(tracker._chosen_phrases),
        "n_rim_signals":   sum(1 for s in signals if s.target == "rim"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir",   type=Path, required=True)
    ap.add_argument("--pad-frac",  type=float, default=0.12,
                    help="fraction of bbox to pad on each side")
    args = ap.parse_args()

    pngs = sorted(args.input_dir.glob("*.png")) + sorted(args.input_dir.glob("*.jpg"))
    pngs = [p for p in pngs if "templates" not in p.parts]   # skip outputs dir
    if not pngs:
        raise SystemExit(f"no pngs in {args.input_dir}")

    # sam3.1 with a rim-focused concept + multi-phrase fallback.
    cfg = Sam3Config(
        checkpoint="sam3.1",
        concepts={"rim": ["basketball hoop", "hoop",
                          "orange rim", "basketball rim"]},
        prompt_frames=[0],   # still image; single frame is fine
    )
    tracker = Sam3Tracker(cfg)
    tracker._ensure_predictor()

    results = []
    for p in pngs:
        print(f"extracting {p.name} ...", flush=True)
        try:
            meta = extract_one(p, tracker, args.out_dir, args.pad_frac)
        except Exception as e:
            meta = {"source": p.name, "status": f"error: {type(e).__name__}: {e}"}
        print(f"  -> {meta.get('status')}  "
              f"{'crop=' + meta.get('crop', '') if meta.get('crop') else ''}")
        results.append(meta)

    # write manifest
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\nextracted {n_ok}/{len(results)} templates")
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
