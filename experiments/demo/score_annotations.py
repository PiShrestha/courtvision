"""score filled-in annotation worksheets against the model's events.json.

computes per-event-type precision, recall, F1, plus confusion between
shot_attempt / shot_made / shot_miss when the annotator marked wrong_type.

usage:
    python experiments/demo/score_annotations.py \\
        --worksheet-dir experiments/demo/annotation_worksheets \\
        --variant-dir   experiments/demo/outputs_v3_ablation/full_ensemble_26 \\
        --out           experiments/demo/annotation_worksheets/f1_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SHOT_EVENTS = ["shot_attempt", "shot_made", "shot_miss"]


def read_worksheet(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if not row.get("section") or row["section"].startswith("#"):
                continue
            rows.append(row)
    return rows


def score_clip(worksheet: Path, events_path: Path) -> dict:
    rows = read_worksheet(worksheet)
    with events_path.open() as f:
        model_events = [e for e in json.load(f) if e.get("event") in SHOT_EVENTS]

    # a predicted event is TP if annotator marked `ok`, FP if `false_positive`,
    # and counts as a WrongType error if `wrong_type`.
    tp = Counter()
    fp = Counter()
    wrong = Counter()
    missed = Counter()
    corrections: list[tuple[str, str]] = []

    for r in rows:
        section = r.get("section", "").strip()
        verdict = r.get("annotator_verdict", "").strip().lower()
        mevent  = r.get("model_event", "").strip()
        cevent  = r.get("annotator_correct_event", "").strip()
        if section == "predicted":
            if verdict == "ok":
                tp[mevent] += 1
            elif verdict == "false_positive":
                fp[mevent] += 1
            elif verdict == "wrong_type":
                wrong[mevent] += 1
                if cevent:
                    corrections.append((mevent, cevent))
                    # also counts as TP for the corrected class for recall, since
                    # the underlying event happened; the predicted class was wrong.
                    tp[cevent] += 1
                    fp[mevent] += 1
            # blank verdict -> unscored; skip silently
        elif section == "missed":
            if cevent:
                missed[cevent] += 1

    return {
        "clip": worksheet.stem.replace(".worksheet", ""),
        "model_shot_events": len(model_events),
        "tp": dict(tp),
        "fp": dict(fp),
        "wrong_type": dict(wrong),
        "missed": dict(missed),
        "corrections": corrections,
    }


def aggregate(per_clip: list[dict]) -> dict:
    tp_all = Counter()
    fp_all = Counter()
    fn_all = Counter()   # missed (annotator-added)
    wrong_all = Counter()
    n_events = 0

    for c in per_clip:
        for k, v in c["tp"].items():      tp_all[k] += v
        for k, v in c["fp"].items():      fp_all[k] += v
        for k, v in c["missed"].items():  fn_all[k] += v
        for k, v in c["wrong_type"].items(): wrong_all[k] += v
        n_events += c["model_shot_events"]

    per_type = {}
    for t in SHOT_EVENTS:
        tp = tp_all.get(t, 0)
        fp = fp_all.get(t, 0)
        fn = fn_all.get(t, 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_type[t] = {"tp": tp, "fp": fp, "fn": fn,
                       "precision": round(precision, 3),
                       "recall":    round(recall, 3),
                       "f1":        round(f1, 3)}

    # overall micro-f1 across the three shot types
    tp_m = sum(per_type[t]["tp"] for t in SHOT_EVENTS)
    fp_m = sum(per_type[t]["fp"] for t in SHOT_EVENTS)
    fn_m = sum(per_type[t]["fn"] for t in SHOT_EVENTS)
    p_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0.0
    r_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
    f_m = 2 * p_m * r_m / (p_m + r_m) if (p_m + r_m) > 0 else 0.0
    micro = {"tp": tp_m, "fp": fp_m, "fn": fn_m,
             "precision": round(p_m, 3), "recall": round(r_m, 3),
             "f1": round(f_m, 3)}

    return {
        "n_clips": len(per_clip),
        "n_model_events": n_events,
        "per_event_type": per_type,
        "micro": micro,
        "wrong_type_errors": dict(wrong_all),
    }


def render_markdown(agg: dict, per_clip: list[dict]) -> str:
    lines = [
        f"# F1 on annotated subset ({agg['n_clips']} clips, "
        f"{agg['n_model_events']} model-predicted shot events)",
        "",
        "## Per event type",
        "",
        "| event | TP | FP | FN | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for t in SHOT_EVENTS:
        s = agg["per_event_type"][t]
        lines.append(f"| {t} | {s['tp']} | {s['fp']} | {s['fn']} | "
                     f"{s['precision']} | {s['recall']} | {s['f1']} |")
    m = agg["micro"]
    lines += [
        f"| **micro** | **{m['tp']}** | **{m['fp']}** | **{m['fn']}** | "
        f"**{m['precision']}** | **{m['recall']}** | **{m['f1']}** |",
        "",
        "## Per-clip breakdown",
        "",
    ]
    for c in per_clip:
        lines.append(f"- `{c['clip']}`  model_events={c['model_shot_events']}  "
                     f"tp={sum(c['tp'].values())}  "
                     f"fp={sum(c['fp'].values())}  "
                     f"wrong_type={sum(c['wrong_type'].values())}  "
                     f"missed={sum(c['missed'].values())}")
    if agg["wrong_type_errors"]:
        lines += ["", "## Wrong-type errors (counted as FP + TP-for-correct-class)",
                  f"{agg['wrong_type_errors']}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worksheet-dir", required=True, type=Path)
    ap.add_argument("--variant-dir",   required=True, type=Path)
    ap.add_argument("--out",           required=True, type=Path)
    args = ap.parse_args()

    per_clip = []
    for ws in sorted(args.worksheet_dir.glob("*.worksheet.csv")):
        stem = ws.name.replace(".worksheet.csv", "")
        events_path = args.variant_dir / f"{stem}.events.json"
        if not events_path.exists():
            print(f"warn: no events.json for {stem}, skipping")
            continue
        per_clip.append(score_clip(ws, events_path))

    if not per_clip:
        raise SystemExit("no worksheets scored; check --worksheet-dir and --variant-dir")

    agg = aggregate(per_clip)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_markdown(agg, per_clip))
    print(render_markdown(agg, per_clip))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
