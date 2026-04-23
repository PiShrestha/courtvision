"""stage-3 vlm reasoning pass: gemini reviews rule-based events and agrees /
disagrees, enabling rule+vlm F1 vs rule-only F1 comparisons.

input: events.json (from run_demo_v3.py) + the source video file.
output: events_vlm.json — same events with `vlm_verdict` and `vlm_reason`
fields added per shot event. possession events are passed through
unreviewed (too frequent; verifying them would waste api calls).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
from google import genai
from google.genai import types
from PIL import Image


SHOT_EVENTS = {"shot_attempt", "shot_made", "shot_miss"}

PROMPT_TEMPLATE = (
    "You are reviewing a basketball 1v1 clip. The rule-based system labelled "
    "frame {frame} as event='{event}' for player={player}. I'm showing you "
    "{n_frames} frames sampled around that moment (~1s before to ~1s after).\n\n"
    "Respond with STRICT JSON only, no prose: "
    '{{"verdict": "agree" | "disagree", "reason": "<=15 words"}}.\n'
    "Agree if the frames clearly show the labelled event; disagree if the "
    "frames show a different action (e.g. pass, fake, dribble, rebound)."
)


def sample_frames(video: Path, frame_id: int, fps: float,
                  window_s: float = 1.0, n: int = 8) -> list[Image.Image]:
    """grab n evenly-spaced frames within ±window_s around frame_id."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    span = int(round(window_s * fps))
    lo = max(0, frame_id - span)
    hi = min(total - 1 if total > 0 else frame_id + span, frame_id + span)
    if hi <= lo:
        cap.release()
        return []
    idxs = [lo + round(i * (hi - lo) / max(1, n - 1)) for i in range(n)]
    frames: list[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # bgr -> rgb -> pil (gemini accepts pil images directly).
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def call_gemini(client, model_name: str, event: dict,
                frames: list[Image.Image]) -> dict:
    """one gemini call per shot event. returns {verdict, reason} or error dict."""
    if not frames:
        return {}
    prompt = PROMPT_TEMPLATE.format(
        frame=event.get("frame_id"),
        event=event.get("event"),
        player=event.get("player"),
        n_frames=len(frames),
    )
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=[prompt, *frames],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        text = (resp.text or "").strip()
        return json.loads(text)
    except Exception as e:
        return {"verdict": "error", "reason": str(e)[:80]}


def verify_events(events_path: Path, video_path: Path,
                  out_path: Path, model_name: str) -> dict:
    with events_path.open() as f:
        events = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    t0 = time.time()
    reviewed = agree = disagree = errors = 0
    for e in events:
        if e.get("event") not in SHOT_EVENTS:
            continue
        frames = sample_frames(video_path, int(e["frame_id"]), fps)
        verdict = call_gemini(client, model_name, e, frames)
        reviewed += 1
        e["vlm_verdict"] = verdict.get("verdict", "error")
        e["vlm_reason"] = verdict.get("reason", "")
        if verdict.get("verdict") == "agree":
            agree += 1
        elif verdict.get("verdict") == "disagree":
            disagree += 1
        else:
            errors += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(events, f, indent=2)

    stats = {
        "events_total": len(events),
        "shot_events_reviewed": reviewed,
        "vlm_agree": agree,
        "vlm_disagree": disagree,
        "vlm_error": errors,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(stats))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="path to events.json")
    ap.add_argument("--video",  required=True, help="source clip video")
    ap.add_argument("--out",    required=True, help="output events_vlm.json path")
    ap.add_argument("--model",  default="gemini-2.5-flash",
                    help="free-tier gemini model; 2.5-flash has high quota + multimodal")
    args = ap.parse_args()
    verify_events(Path(args.events), Path(args.video), Path(args.out), args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
