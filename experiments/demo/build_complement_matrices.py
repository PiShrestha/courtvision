"""build ablation matrices for the 6 variants NOT in jpj8rf's ASAP run.

jpj8rf is running baseline + plus_sam3_1 + full_ensemble (203 tasks).
auj7tx should run the complement: 4 v3 variants + 2 v4 variants = 300 tasks.
together, both accounts cover the full 9-variant ablation in parallel.
"""

from __future__ import annotations

import csv
from pathlib import Path

# the base 50-clip manifest both sides share.
BASE_CSV = Path("experiments/demo/matrix_v2_50clips_flow.csv")


def build_v3_complement(base: list[dict]) -> list[dict]:
    # variants NOT run by jpj8rf (who does baseline + plus_sam3_1).
    variants = [
        ("plus_yoloe", {"use_yoloe": 1, "use_pose": 0, "sam3_cache": ""}),
        ("plus_pose",  {"use_yoloe": 0, "use_pose": 1, "sam3_cache": ""}),
        ("plus_sam3",  {"use_yoloe": 0, "use_pose": 0, "sam3_cache": "sam3"}),
    ]
    return _expand(base, variants, v4=False)


def build_v4_complement(base: list[dict]) -> list[dict]:
    # v4 variants NOT run by jpj8rf (who does full_ensemble).
    variants = [
        ("full_strict",  {"use_yoloe": 1, "use_pose": 1, "sam3_cache": "sam3.1", "strict_consensus": 1}),
        ("yoloe_pose",   {"use_yoloe": 1, "use_pose": 1, "sam3_cache": "",        "strict_consensus": 0}),
        ("pose_sam3_1",  {"use_yoloe": 0, "use_pose": 1, "sam3_cache": "sam3.1", "strict_consensus": 0}),
    ]
    return _expand(base, variants, v4=True)


def _expand(base: list[dict], variants: list[tuple[str, dict]],
             v4: bool) -> list[dict]:
    rows = []
    task_id = 0
    for vname, flags in variants:
        for r in base:
            rr = dict(r)
            rr["variant"] = vname
            rr["task_id"] = str(task_id)
            for k, v in flags.items():
                rr[k] = str(v)
            if v4 and "strict_consensus" not in rr:
                rr["strict_consensus"] = "0"
            rows.append(rr)
            task_id += 1
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    if not rows:
        raise SystemExit(f"no rows to write for {out}")
    # sbatch scripts parse positionally via `IFS=, read -r _TASKID VARIANT RUN_ID ...`,
    # so task_id + variant must lead the header regardless of dict insertion order.
    all_cols = list(rows[0].keys())
    lead = [c for c in ("task_id", "variant") if c in all_cols]
    cols = lead + [c for c in all_cols if c not in lead]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    with BASE_CSV.open() as f:
        base = list(csv.DictReader(f))
    v3 = build_v3_complement(base)
    v4 = build_v4_complement(base)
    write_csv(v3, Path("experiments/demo/matrix_v3_ablation.csv"))
    write_csv(v4, Path("experiments/demo/matrix_v4_ensemble.csv"))
    print(f"v3 complement: {len(v3)} rows  (variants: plus_yoloe, plus_pose, plus_sam3)")
    print(f"v4 complement: {len(v4)} rows  (variants: full_strict, yoloe_pose, pose_sam3_1)")
    print(f"total: {len(v3) + len(v4)} tasks across both matrices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
