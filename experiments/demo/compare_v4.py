"""cross-variant comparison across v3 ablation + v4 ensemble (9 variants total)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


V3_VARIANTS = ["baseline", "plus_yoloe", "plus_pose", "plus_sam3", "plus_sam3_1"]
V4_VARIANTS = ["full_ensemble", "full_strict", "yoloe_pose", "pose_sam3_1"]
ALL_VARIANTS = V3_VARIANTS + V4_VARIANTS


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-root", default="experiments/demo/outputs_v3_ablation")
    ap.add_argument("--v4-root", default="experiments/demo/outputs_v4_ensemble")
    ap.add_argument("--out",
                    default="experiments/demo/analysis_v4_ensemble/cross_all_variants.md")
    return ap.parse_args()


def clip_key(meta: dict) -> str:
    video = Path(meta.get("video", "?")).stem
    start = int(float(meta.get("start", 0)))
    end = int(float(meta.get("end", 0)))
    return f"{video}__t{start}-{end}"


def load(root: Path, variant: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for meta_path in sorted((root / variant).glob("*.meta.json")):
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            continue
        out[clip_key(m)] = m
    return out


def fmt_pct(made: int, att: int) -> str:
    if att == 0:
        return "—"
    return f"{100 * made / att:.0f}%"


def render(rows: dict[str, dict[str, dict]]) -> str:
    lines = [
        "# All-variants cross-comparison (v3 ablation + v4 ensemble)",
        "",
        f"clips: **{len(rows)}**,  variants: **{len(ALL_VARIANTS)}**",
        "",
        "## headline: mean metrics per variant",
        "",
        "| variant | runs | mean att | mean made | mean miss | overall FG% | Δ att vs baseline | Δ made vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_mean_att = base_mean_made = 0.0
    base_metas = [by_var["baseline"] for by_var in rows.values() if "baseline" in by_var]
    if base_metas:
        base_mean_att = sum(int(m.get("shot_attempt_events", 0)) for m in base_metas) / len(base_metas)
        base_mean_made = sum(int(m.get("shot_made_events", 0)) for m in base_metas) / len(base_metas)

    for v in ALL_VARIANTS:
        metas = [by_var[v] for by_var in rows.values() if v in by_var]
        if not metas:
            lines.append(f"| {v} | 0 | — | — | — | — | — | — |")
            continue
        att = [int(m.get("shot_attempt_events", 0)) for m in metas]
        mad = [int(m.get("shot_made_events", 0)) for m in metas]
        mis = [int(m.get("shot_miss_events", 0)) for m in metas]
        att_m = sum(att) / len(att)
        mad_m = sum(mad) / len(mad)
        mis_m = sum(mis) / len(mis)
        fg = fmt_pct(sum(mad), sum(att))
        d_att = att_m - base_mean_att
        d_mad = mad_m - base_mean_made
        lines.append(
            f"| {v} | {len(metas)} | {att_m:.2f} | {mad_m:.2f} | {mis_m:.2f} "
            f"| {fg} | {d_att:+.2f} | {d_mad:+.2f} |"
        )

    lines += ["", "## per-clip grid (attempts / made / FG%)", ""]
    header = "| clip / window | " + " | ".join(ALL_VARIANTS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ALL_VARIANTS) + 1))
    for clip, by_var in sorted(rows.items()):
        cells = [clip]
        for v in ALL_VARIANTS:
            m = by_var.get(v)
            if m is None:
                cells.append("_—_"); continue
            att = int(m.get("shot_attempt_events", 0))
            mad = int(m.get("shot_made_events", 0))
            cells.append(f"{att}/{mad}/{fmt_pct(mad, att)}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    v3 = Path(args.v3_root)
    v4 = Path(args.v4_root)
    by_clip: dict[str, dict[str, dict]] = {}
    for v in V3_VARIANTS:
        for clip, m in load(v3, v).items():
            by_clip.setdefault(clip, {})[v] = m
    for v in V4_VARIANTS:
        for clip, m in load(v4, v).items():
            by_clip.setdefault(clip, {})[v] = m
    if not by_clip:
        print(f"no runs found under {v3} or {v4}")
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(by_clip))
    print(f"wrote {out_path}  ({len(by_clip)} clips x {len(ALL_VARIANTS)} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
