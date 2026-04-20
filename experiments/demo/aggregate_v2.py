"""aggregate v2 sweep outputs (offline + live) into CSV + summary + charts.

reads every *.meta_v2.json (offline sweep) and every run_demo_v2 meta.json
(live), merges them on a shared schema, emits:
  - experiments/demo/analysis_v2/results_v2.csv
  - experiments/demo/analysis_v2/summary_v2.md
  - chart_attempts_vs_sweep.png, chart_made_vs_sweep.png,
    chart_fg_pct_per_player.png, chart_v1_vs_v2.png (if v1 files present)
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TUNABLE_KEYS = (
    "upward_trigger", "history_frames", "release_dist_px",
    "approach_dist_px", "cooldown_frames",
    "occlusion_gap_frames", "attempt_to_made_window",
    "enter_zone_radius_factor", "horizontal_pad_factor",
    "min_downward_velocity",
)


def flatten(meta: dict) -> dict:
    fg = meta.get("field_goal_stats", {}) or {}
    overall = fg.get("overall", {}) or {}
    row = {
        "path": meta.get("_path"),
        "sweep_id": meta.get("sweep_id"),
        "pipeline": meta.get("pipeline", "v2_offline"),
        "video": meta.get("video") or meta.get("source_video"),
        "start": meta.get("start") or meta.get("source_start"),
        "end": meta.get("end") or meta.get("source_end"),
        "source_model": meta.get("model") or meta.get("source_model"),
        "source_imgsz": meta.get("imgsz") or meta.get("source_imgsz"),
        "source_confidence": meta.get("confidence") or meta.get("source_confidence"),
        "frames_processed": meta.get("frames_processed", 0),
        "possession_events": meta.get("possession_events", 0),
        "shot_attempt_events": meta.get("shot_attempt_events", 0),
        "shot_made_events": meta.get("shot_made_events", 0),
        "shot_miss_events": meta.get("shot_miss_events", 0),
        "overall_attempts": overall.get("attempts", 0),
        "overall_made": overall.get("made", 0),
        "overall_fg_pct": overall.get("fg_pct"),
        "replay_seconds": meta.get("replay_seconds"),
        "perception_seconds": meta.get("perception_seconds"),
        "total_seconds": meta.get("total_seconds"),
    }
    for k in TUNABLE_KEYS:
        row[k] = meta.get(k)
    return row


def load_all(meta_globs: list[str]) -> pd.DataFrame:
    rows = []
    for g in meta_globs:
        for p in sorted(glob.glob(g)):
            try:
                m = json.loads(Path(p).read_text())
            except Exception:
                continue
            m["_path"] = p
            rows.append(flatten(m))
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, outdir: Path) -> None:
    out = [
        "# Shot-made v2 sweep summary",
        "",
        f"total runs: **{len(df)}**",
        f"videos: {df['video'].nunique()}",
        f"unique sweep_ids: {df['sweep_id'].nunique()}",
        "",
        "## event totals by sweep_id (mean across tracks files)",
    ]
    if df.empty or df["sweep_id"].isna().all():
        out.append("_no sweep data available_")
    else:
        pivot = df.groupby("sweep_id").agg(
            n=("shot_attempt_events", "size"),
            att_mean=("shot_attempt_events", "mean"),
            made_mean=("shot_made_events", "mean"),
            miss_mean=("shot_miss_events", "mean"),
            fg_pct_mean=("overall_fg_pct", "mean"),
        ).round(2)
        out.append(pivot.to_markdown())

    out.append("")
    out.append("## sensitivity to each tunable (mean shot_made across runs)")
    for k in TUNABLE_KEYS:
        if df[k].nunique() > 1:
            sub = df.groupby(k)[["shot_attempt_events", "shot_made_events",
                                   "shot_miss_events", "overall_fg_pct"]].mean().round(2)
            out.append(f"\n**{k}**\n")
            out.append(sub.to_markdown())
    out.append("")
    out.append("## per-video totals")
    if not df.empty:
        pv = df.groupby("video").agg(
            runs=("shot_attempt_events", "size"),
            att_sum=("shot_attempt_events", "sum"),
            made_sum=("shot_made_events", "sum"),
            miss_sum=("shot_miss_events", "sum"),
        )
        out.append(pv.to_markdown())
    (outdir / "summary_v2.md").write_text("\n".join(out))


def chart_attempts_vs_made(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty: return
    fig, ax = plt.subplots(figsize=(9, 5))
    for sw, sub in df.groupby("sweep_id"):
        ax.scatter(sub["shot_attempt_events"], sub["shot_made_events"],
                   alpha=0.7, label=str(sw))
    ax.set_xlabel("shot_attempts per run")
    ax.set_ylabel("shot_made per run")
    ax.set_title("attempts vs made across runs (v2)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncols=3)
    fig.tight_layout()
    fig.savefig(outdir / "chart_attempts_vs_made.png", dpi=140)
    plt.close(fig)


def chart_tunable_effect(df: pd.DataFrame, outdir: Path) -> None:
    varied = [k for k in TUNABLE_KEYS if df[k].nunique() > 1]
    if not varied: return
    fig, axes = plt.subplots(1, len(varied), figsize=(3 * len(varied), 4),
                              sharey=False)
    if len(varied) == 1:
        axes = [axes]
    for ax, k in zip(axes, varied):
        g = df.groupby(k).agg(att=("shot_attempt_events", "mean"),
                               made=("shot_made_events", "mean"),
                               miss=("shot_miss_events", "mean"))
        g.plot(kind="line", marker="o", ax=ax)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel(k, fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("mean events per run by tunable (all videos)")
    fig.tight_layout()
    fig.savefig(outdir / "chart_tunable_effects.png", dpi=140)
    plt.close(fig)


def chart_v1_vs_v2(df_v2: pd.DataFrame, v1_glob: str, outdir: Path) -> None:
    v1_rows = []
    for p in sorted(glob.glob(v1_glob)):
        try:
            m = json.loads(Path(p).read_text())
        except Exception:
            continue
        ev_path = p.replace(".meta.json", ".events.json")
        events: list[dict] = []
        if Path(ev_path).exists():
            try:
                events = json.loads(Path(ev_path).read_text())
            except Exception:
                events = []
        v1_rows.append({
            "path": p,
            "video": m.get("video"),
            "start": m.get("start"), "end": m.get("end"),
            "model": m.get("model"), "imgsz": m.get("imgsz"),
            "attempts": sum(1 for e in events if e.get("event") == "shot_attempt"),
            "made": sum(1 for e in events if e.get("event") == "shot_made"),
            "miss": 0,
        })
    v1 = pd.DataFrame(v1_rows)
    if v1.empty or df_v2.empty:
        return
    v1_summary = v1[["attempts", "made", "miss"]].mean()
    v2_summary = df_v2[["shot_attempt_events", "shot_made_events",
                          "shot_miss_events"]].mean()
    summary = pd.DataFrame({
        "v1": v1_summary.values,
        "v2": v2_summary.values,
    }, index=["attempt", "made", "miss"])
    fig, ax = plt.subplots(figsize=(7, 4))
    summary.plot(kind="bar", ax=ax)
    ax.set_title("v1 vs v2: mean event counts per run")
    ax.set_ylabel("events per run (mean)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(outdir / "chart_v1_vs_v2.png", dpi=140)
    plt.close(fig)
    summary.round(2).to_csv(outdir / "v1_vs_v2.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-glob",
                    default="experiments/demo/outputs_v2*/*.meta_v2.json")
    ap.add_argument("--live-glob",
                    default="experiments/demo/outputs_v2_live*/*.meta.json")
    ap.add_argument("--v1-glob",
                    default="experiments/demo/outputs/run*.meta.json",
                    help="for v1-vs-v2 chart")
    ap.add_argument("--outdir", default="experiments/demo/analysis_v2")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = load_all([args.v2_glob, args.live_glob])
    if df.empty:
        print("no v2 outputs found"); return 1
    df.to_csv(outdir / "results_v2.csv", index=False)
    write_summary(df, outdir)
    chart_attempts_vs_made(df, outdir)
    chart_tunable_effect(df, outdir)
    chart_v1_vs_v2(df, args.v1_glob, outdir)
    print(f"wrote {len(df)} rows to {outdir}/results_v2.csv")
    print(f"wrote summary + charts to {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
