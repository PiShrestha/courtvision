"""aggregate the demo-matrix outputs into raw csv + tables + multiple charts.

usage:
    .venv/bin/python experiments/demo/aggregate_matrix.py \
        --glob 'experiments/demo/outputs/run*.meta.json' \
        --outdir experiments/demo/analysis
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


def load_records(meta_paths: list[Path]) -> pd.DataFrame:
    rows = []
    for mp in meta_paths:
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            continue
        # events sidecar.
        events_path = mp.with_suffix("").with_suffix(".events.json")
        events: list[dict] = []
        if events_path.exists():
            try:
                events = json.loads(events_path.read_text())
            except Exception:
                events = []
        possession = sum(1 for e in events if e.get("event") == "possession")
        attempts = sum(1 for e in events if e.get("event") == "shot_attempt")
        made = sum(1 for e in events if e.get("event") == "shot_made")
        unique_players = {e.get("player") for e in events if isinstance(e.get("player"), int)}
        rows.append({
            "meta_path": str(mp),
            "video": meta.get("video"),
            "start": meta.get("start"),
            "end": meta.get("end"),
            "model": meta.get("model"),
            "imgsz": meta.get("imgsz"),
            "confidence": meta.get("confidence"),
            "tracker": meta.get("tracker_config"),
            "upward_trigger": meta.get("upward_trigger"),
            "made_window": meta.get("made_window"),
            "hoop_pad": meta.get("hoop_pad"),
            "possession_dist_px": meta.get("possession_dist_px"),
            "hoop_center_x": meta.get("hoop", {}).get("center", [0, 0])[0],
            "hoop_center_y": meta.get("hoop", {}).get("center", [0, 0])[1],
            "hoop_radius": meta.get("hoop", {}).get("radius"),
            "frames_processed": meta.get("frames_processed", 0),
            "possession_events": possession,
            "shot_attempt_events": attempts,
            "shot_made_events": made,
            "distinct_players_in_events": len(unique_players),
            "perception_seconds": meta.get("perception_seconds"),
            "total_seconds": meta.get("total_seconds"),
        })
    df = pd.DataFrame(rows)
    if df.empty: return df

    # dedupe by (video, window, full config) keeping the newest mtime.
    # this prevents mixing buggy pre-fix runs with the reruns that followed.
    df["mtime"] = df["meta_path"].map(lambda p: Path(p).stat().st_mtime)
    group_keys = ["video", "start", "end", "model", "imgsz", "confidence",
                   "upward_trigger", "made_window", "hoop_pad"]
    df = (df.sort_values("mtime")
             .drop_duplicates(subset=group_keys, keep="last")
             .drop(columns=["mtime"])
             .reset_index(drop=True))

    df["window_len_s"] = df["end"] - df["start"]
    df["events_per_minute"] = (df["possession_events"]
                                 + df["shot_attempt_events"]
                                 + df["shot_made_events"]) / (df["window_len_s"] / 60).clip(lower=1e-6)
    df["shot_success_rate"] = df["shot_made_events"] / df["shot_attempt_events"].replace(0, np.nan)
    df["config"] = (df["model"].str.replace(".pt", "", regex=False)
                     + "_i" + df["imgsz"].astype(str)
                     + "_c" + df["confidence"].astype(str)
                     + "_ut" + df["upward_trigger"].astype(str)
                     + "_mw" + df["made_window"].astype(str)
                     + "_hp" + df["hoop_pad"].astype(str))
    return df


def write_tables(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "results_raw.csv", index=False)

    summary = [
        "# Demo matrix summary",
        "",
        f"total runs: **{len(df)}**",
        f"videos: {df['video'].nunique()}",
        f"windows: {df.groupby(['video', 'start']).ngroups}",
        f"configs: {df['config'].nunique()}",
        "",
        "## event totals by detector config (collapsing shot parameters)",
        df.groupby(["model", "imgsz", "confidence"]).agg(
            runs=("possession_events", "size"),
            possessions=("possession_events", "mean"),
            attempts=("shot_attempt_events", "mean"),
            made=("shot_made_events", "mean"),
            events_per_min=("events_per_minute", "mean"),
            mean_seconds=("perception_seconds", "mean"),
        ).round(2).to_markdown(),
        "",
        "## shot-attempt sensitivity by upward_trigger (all detectors)",
        df.groupby(["upward_trigger"]).agg(
            runs=("shot_attempt_events", "size"),
            attempts_mean=("shot_attempt_events", "mean"),
            attempts_std=("shot_attempt_events", "std"),
            made_mean=("shot_made_events", "mean"),
        ).round(2).to_markdown(),
        "",
        "## shot_made sensitivity by hoop_pad and made_window",
        df.pivot_table(values="shot_made_events",
                         index="hoop_pad", columns="made_window",
                         aggfunc="mean").round(2).to_markdown(),
        "",
        "## per-video event totals",
        df.groupby("video").agg(
            runs=("possession_events", "size"),
            possessions=("possession_events", "sum"),
            attempts=("shot_attempt_events", "sum"),
            made=("shot_made_events", "sum"),
        ).to_markdown(),
        "",
        "## runtime (perception seconds) by model x imgsz",
        df.pivot_table(values="perception_seconds",
                         index="model", columns="imgsz",
                         aggfunc="mean").round(1).to_markdown(),
    ]
    (outdir / "summary.md").write_text("\n".join(summary))


def chart_events_per_config(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty: return
    pivot = df.groupby(["model", "imgsz", "confidence"]).agg(
        possession=("possession_events", "mean"),
        attempt=("shot_attempt_events", "mean"),
        made=("shot_made_events", "mean"),
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", stacked=False, ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("events per run (mean)")
    ax.set_title("event counts by detector config")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "chart_events_per_detector.png", dpi=140)
    plt.close(fig)


def chart_shot_param_sweep(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, param, ylabel in zip(
        axes,
        ["upward_trigger", "made_window", "hoop_pad"],
        ["shot_attempts", "shot_made", "shot_made"],
    ):
        grp = df.groupby(param).agg(
            att_mean=("shot_attempt_events", "mean"),
            made_mean=("shot_made_events", "mean"),
        )
        if param == "upward_trigger":
            y = grp["att_mean"]
        else:
            y = grp["made_mean"]
        y.plot(kind="line", marker="o", ax=ax)
        ax.set_title(f"{ylabel} vs {param}")
        ax.set_xlabel(param)
        ax.set_ylabel(ylabel + " (mean)")
        ax.grid(alpha=0.3)
    fig.suptitle("shot detector sensitivity")
    fig.tight_layout()
    fig.savefig(outdir / "chart_shot_sensitivity.png", dpi=140)
    plt.close(fig)


def chart_per_video_bars(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty: return
    agg = df.groupby("video").agg(
        possession=("possession_events", "sum"),
        attempt=("shot_attempt_events", "sum"),
        made=("shot_made_events", "sum"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    agg.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("event totals per video (summed across runs)")
    ax.set_ylabel("events")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "chart_per_video.png", dpi=140)
    plt.close(fig)


def chart_runtime_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty: return
    pv = df.pivot_table(values="perception_seconds", index="model",
                         columns="imgsz", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(pv.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
    ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
    for i in range(len(pv.index)):
        for j in range(len(pv.columns)):
            ax.text(j, i, f"{pv.values[i, j]:.0f}s", ha="center", va="center",
                     color="white", fontsize=10)
    ax.set_title("perception seconds per run (mean)")
    fig.colorbar(im, ax=ax, label="seconds")
    fig.tight_layout()
    fig.savefig(outdir / "chart_runtime_heatmap.png", dpi=140)
    plt.close(fig)


def chart_distribution_attempts(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty: return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, sub in df.groupby("model"):
        ax.hist(sub["shot_attempt_events"], bins=20, alpha=0.5, label=model)
    ax.set_title("distribution of shot_attempt events by model")
    ax.set_xlabel("shot_attempt count per run")
    ax.set_ylabel("frequency")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "chart_attempts_distribution.png", dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="experiments/demo/outputs/run*.meta.json")
    ap.add_argument("--outdir", default="experiments/demo/analysis")
    args = ap.parse_args()

    meta_paths = [Path(p) for p in sorted(glob.glob(args.glob))]
    if not meta_paths:
        print(f"no meta files matched {args.glob}"); return 1
    df = load_records(meta_paths)
    if df.empty:
        print("no records loaded"); return 1
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_tables(df, outdir)
    chart_events_per_config(df, outdir)
    chart_shot_param_sweep(df, outdir)
    chart_per_video_bars(df, outdir)
    chart_runtime_heatmap(df, outdir)
    chart_distribution_attempts(df, outdir)
    print(f"wrote {len(df)} rows to {outdir}/results_raw.csv")
    print(f"wrote summary: {outdir}/summary.md")
    print(f"wrote charts:  {outdir}/chart_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
