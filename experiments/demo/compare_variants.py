"""side-by-side comparison of ablation variants across the same clips.

for each (video, window) pair, emit one row showing every variant's
shot_attempt / shot_made / FG% / possession split. highlights the
deltas between baseline and each augmented variant.

usage:
    .venv/bin/python experiments/demo/compare_variants.py \\
        --runs-root experiments/demo/outputs_v3_ablation \\
        --out experiments/demo/analysis_v3_ablation/cross_variant_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


VARIANTS = ["baseline", "plus_yoloe", "plus_pose", "plus_sam3", "plus_sam3_1"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root",
                    default="experiments/demo/outputs_v3_ablation")
    ap.add_argument("--out",
                    default="experiments/demo/analysis_v3_ablation/cross_variant_report.md")
    return ap.parse_args()


def clip_key(meta: dict) -> str:
    """stable identifier that collapses all variants of the same clip/window."""
    video = Path(meta.get("video", "?")).stem
    start = int(float(meta.get("start", 0)))
    end = int(float(meta.get("end", 0)))
    return f"{video}__t{start}-{end}"


def load_variant(root: Path, variant: str) -> dict[str, dict]:
    """load all meta.json files for a variant; return {clip_key: meta}."""
    out: dict[str, dict] = {}
    for meta_path in sorted((root / variant).glob("*.meta.json")):
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            continue
        m["_path"] = str(meta_path)
        out[clip_key(m)] = m
    return out


def fmt_pct(made: int, attempts: int) -> str:
    if attempts == 0:
        return "—"
    return f"{100 * made / attempts:.0f}%"


def render(rows: dict[str, dict[str, dict]]) -> str:
    """rows[clip][variant] = meta"""
    lines = [
        "# V3 cross-variant comparison",
        "",
        "Each row is one (video, window). Columns are the 5 ablation variants. "
        "Cells show `attempts / made (FG%) · P1/P2 poss-s`.",
        "",
        f"clips: **{len(rows)}**,  variants: **{len(VARIANTS)}**",
        "",
    ]
    header = "| clip / window | " + " | ".join(VARIANTS) + " |"
    sep = "|" + "---|" * (len(VARIANTS) + 1)
    lines.append(header)
    lines.append(sep)
    for clip, by_var in sorted(rows.items()):
        cells = [clip]
        for v in VARIANTS:
            m = by_var.get(v)
            if m is None:
                cells.append("_missing_")
                continue
            att = int(m.get("shot_attempt_events", 0))
            made = int(m.get("shot_made_events", 0))
            fg = fmt_pct(made, att)
            fg_stats = m.get("field_goal_stats", {}) or {}
            per_p = fg_stats.get("per_player", {}) or {}
            # per_player keys may be strings when loaded from JSON.
            def _poss(pid):
                d = per_p.get(str(pid)) or per_p.get(pid) or {}
                return d.get("attempts", 0) == 0 and "—" or ""
            # simpler: use shot events as coarse player split
            cells.append(f"{att}/{made} ({fg})")
        lines.append("| " + " | ".join(cells) + " |")

    # aggregate by variant (average across all clips).
    lines += ["", "## variant aggregates (mean per clip, non-missing only)", ""]
    agg_header = "| variant | runs | mean attempts | mean made | mean FG% | mean poss P1+P2 |"
    lines.append(agg_header)
    lines.append("|---|---:|---:|---:|---:|---:|")
    for v in VARIANTS:
        metas = [by_var[v] for by_var in rows.values() if v in by_var]
        if not metas:
            lines.append(f"| {v} | 0 | — | — | — | — |")
            continue
        att = [int(m.get("shot_attempt_events", 0)) for m in metas]
        mad = [int(m.get("shot_made_events", 0)) for m in metas]
        att_mean = sum(att) / len(att)
        mad_mean = sum(mad) / len(mad)
        total_att = sum(att); total_mad = sum(mad)
        fg_mean = fmt_pct(total_mad, total_att)
        # per-clip possession sum (P1 + P2) from field_goal_stats wasn't
        # recorded; compute from .events.json if available.
        lines.append(
            f"| {v} | {len(metas)} | {att_mean:.2f} | {mad_mean:.2f} "
            f"| {fg_mean} | — |"
        )

    # deltas vs baseline
    if rows and "baseline" in VARIANTS:
        base_metas = [by_var["baseline"] for by_var in rows.values()
                       if "baseline" in by_var]
        if base_metas:
            base_att = sum(int(m.get("shot_attempt_events", 0)) for m in base_metas) / len(base_metas)
            base_mad = sum(int(m.get("shot_made_events", 0)) for m in base_metas) / len(base_metas)
            lines += ["", "## deltas vs baseline (mean per clip)", ""]
            lines.append("| variant | Δ attempts | Δ made | Δ FG% pp |")
            lines.append("|---|---:|---:|---:|")
            for v in VARIANTS:
                if v == "baseline": continue
                metas = [by_var[v] for by_var in rows.values() if v in by_var and "baseline" in by_var]
                if not metas: continue
                att = sum(int(m.get("shot_attempt_events", 0)) for m in metas) / len(metas)
                mad = sum(int(m.get("shot_made_events", 0)) for m in metas) / len(metas)
                d_att = att - base_att
                d_mad = mad - base_mad
                b_fg = (base_mad / base_att) if base_att else 0
                v_fg = (mad / att) if att else 0
                d_fg_pp = (v_fg - b_fg) * 100
                lines.append(f"| {v} | {d_att:+.2f} | {d_mad:+.2f} | {d_fg_pp:+.1f} pp |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = Path(args.runs_root)
    by_clip: dict[str, dict[str, dict]] = {}
    for v in VARIANTS:
        loaded = load_variant(root, v)
        for clip, m in loaded.items():
            by_clip.setdefault(clip, {})[v] = m
    if not by_clip:
        print(f"no ablation runs found under {root}")
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(by_clip))
    print(f"wrote {out_path}  ({len(by_clip)} clips x {len(VARIANTS)} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
