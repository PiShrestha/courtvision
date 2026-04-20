"""lightweight evaluation harness: precision / recall / F1 against a
human-annotated ground-truth csv.

GT format (csv, header required):
    frame_id, event, player
where event ∈ {possession, shot_attempt, shot_made, shot_miss}.

matching: each predicted event matches a GT event iff
    - same `event` label,
    - |frame_id_pred - frame_id_gt| <= tolerance_frames (default 15).
player IDs are NOT required to match by identity, because the DuoTracker's
{1,2} labels aren't meaningful across annotators — we record agreement as a
separate metric ("player_agreement_rate"), not as a hard match gate.

usage:
    .venv/bin/python experiments/demo/evaluate_v2.py \
        --pred experiments/demo/outputs_v2/xxx.events.json \
        --gt experiments/demo/gt/1v1-mk_t268-338.csv \
        --tolerance 15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


EVENTS_OF_INTEREST = ("shot_attempt", "shot_made")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="events.json path")
    ap.add_argument("--gt", required=True, help="ground-truth csv path")
    ap.add_argument("--tolerance", type=int, default=15,
                    help="max |frame| distance to count a match (default 15)")
    ap.add_argument("--out", default=None, help="optional json path to write metrics")
    return ap.parse_args()


def load_gt(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            fid = row.get("frame_id", "").strip()
            ev = row.get("event", "").strip()
            if not fid or not ev:
                continue
            try:
                fid_int = int(float(fid))
            except ValueError:
                continue
            pl = row.get("player", "").strip()
            out.append({
                "frame_id": fid_int,
                "event": ev,
                "player": int(float(pl)) if pl else None,
            })
    return out


def load_pred(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def match_events(pred: list[dict], gt: list[dict], tol: int,
                 event_type: str) -> dict:
    p = sorted([e for e in pred if e.get("event") == event_type],
               key=lambda e: e["frame_id"])
    g = sorted([e for e in gt if e.get("event") == event_type],
               key=lambda e: e["frame_id"])
    # greedy nearest-matching on frame_id.
    matched_p: set[int] = set()
    matched_g: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for gi, gev in enumerate(g):
        candidates = [(abs(pev["frame_id"] - gev["frame_id"]), pi)
                       for pi, pev in enumerate(p) if pi not in matched_p
                       and abs(pev["frame_id"] - gev["frame_id"]) <= tol]
        if not candidates:
            continue
        _, pi = min(candidates)
        matched_p.add(pi); matched_g.add(gi)
        pairs.append((gi, pi))
    tp = len(pairs)
    fp = len(p) - tp
    fn = len(g) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    # player agreement rate on matched pairs.
    agree = 0; eligible = 0
    for gi, pi in pairs:
        if g[gi]["player"] is None:
            continue
        eligible += 1
        if g[gi]["player"] == p[pi].get("player"):
            agree += 1
    agree_rate = (agree / eligible) if eligible else None
    return {
        "event": event_type,
        "pred_count": len(p),
        "gt_count": len(g),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "player_agreement_rate": round(agree_rate, 3) if agree_rate is not None else None,
    }


def main() -> int:
    args = parse_args()
    pred = load_pred(Path(args.pred))
    gt = load_gt(Path(args.gt))
    results: dict[str, dict] = {
        "pred_path": args.pred,
        "gt_path": args.gt,
        "tolerance_frames": args.tolerance,
    }
    for ev in EVENTS_OF_INTEREST:
        results[ev] = match_events(pred, gt, args.tolerance, ev)

    # pretty print.
    print(f"pred: {args.pred}")
    print(f"gt:   {args.gt}   tolerance=±{args.tolerance} frames")
    print(f"{'event':<14} {'GT':>4} {'pred':>4} {'TP':>3} {'FP':>3} {'FN':>3} "
          f"{'prec':>6} {'rec':>6} {'F1':>6} {'play-agree':>11}")
    for ev in EVENTS_OF_INTEREST:
        r = results[ev]
        pa = "n/a" if r["player_agreement_rate"] is None else f"{r['player_agreement_rate']:.2f}"
        print(f"{ev:<14} {r['gt_count']:>4} {r['pred_count']:>4} "
              f"{r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
              f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {pa:>11}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
