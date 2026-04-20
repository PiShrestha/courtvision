# Demo Run Log

Appended chronologically during overnight development.
Every step logs: what was tried, what happened, next action.

## Phase 1 — workspace setup
- Isolated workspace under `experiments/demo/` (git-untracked).
- Target video: `1v1-mk.mov` (only verified source at start).
- Other videos (`1v1-ddg.mp4`, `1v1-friga.mp4`, `1v1-jason.mp4`) still transferring.
- Overnight Slurm sweep (48 baseline-config jobs) is running in parallel on a separate flow.

## Phase 2 — hoop detection
- Attempted Hough circles on `1v1-mk.mov`. Top candidates were all corner blobs:
  - (84, 85), (1831, 84), (239, 102), ... — dominated by corner logos / vignettes.
- Switched to **color-persistence** (HSV orange mask across ~30 sampled frames) +
  connected components. Found a plausible hoop at **(855, 202), ~36 px radius**
  on `1v1-mk.mov`. Wrote to `hoop_configs/1v1-mk.json` with `source=auto`.
- Same approach failed on `1v1-ddg.mp4` and `1v1-jason.mp4` — no persistent
  orange blobs of the right size. Wrote **placeholder** hoop configs at
  (960, 200, 40) with `source=manual_guess`. Sample frames saved as
  `hoop_configs/<video>_sample.jpg` for review.
- Consequence: `shot_made` accuracy on ddg/jason will be low until hoop
  positions are corrected. `shot_attempt` still works.

## Phase 3 — duo tracker
- Initial test (5 s, `1v1-mk.mov` @ t=120 s):
  - Exactly 2 persistent IDs (P1, P2) held across all 150 frames. ✅
  - But P2 was a phantom at (1759, 10)–(1906, 154) — a tiny 147×145 corner box.
    The "top-2 by confidence" bootstrap picked a non-player.
- Fix: added a **minimum bbox size filter** at bootstrap and matching:
  `min_bbox_height_frac=0.12`, `min_bbox_width_frac=0.03` (both relative to
  frame size). Drops corner watermark / spectator boxes before they can seed
  a slot.

## Phase 4 — experiment matrix
- Axes: detector = {model × imgsz × confidence} = 12, shot detector = 3 presets.
- Videos = 3 working out of 4 (friga has a bad moov atom from the transfer).
- Windows = 3 random 60–90 s per video (seed=7), drawn from the middle 60 %.
- Total runs: **324** (1v1-mk: 108, 1v1-ddg: 108, 1v1-jason: 108).
- Saving annotated mp4 for 1 in 6 tasks → ~54 annotated videos total (~5–6 GB).

## Phase 5 — submitted
- Baseline old-pipeline sweep: Slurm job **11879707** (36 tasks, concurrency 6)
  on manifest `scripts/manifest.csv` via the existing `scripts/run_array.sbatch`.
  Aggregator **11879708** (afterany dependency).
- Demo matrix: Slurm job **11879709** (324 tasks, concurrency 8) on
  `experiments/demo/matrix.csv` via `experiments/demo/run_matrix.sbatch`.
  Aggregator **11879710** (afterany dependency).
- Artefacts in morning:
  - `outputs/analysis/summary.md` + `chart_*.png` (baseline old pipeline)
  - `experiments/demo/analysis/summary.md` + `chart_*.png` (demo matrix)
  - `experiments/demo/outputs/run*.{mp4,events.json,meta.json,tracks.jsonl}`

## Phase 6 — diagnosis of first batch failures
- **First baseline array (11879707)**: 0/36 succeeded. Error: `RuntimeError: GET was unable to find an engine to execute this computation` at `F.conv2d`. cuDNN dispatch failure — torch 2.11.0+cu130 wheel doesn't match the CUDA runtime on some nodes.
- **First demo array (11879709)**: 90/324 succeeded, 234 failed. Same error pattern. Cross-tab by node confirmed: all 234 failures landed on `udc-an33-37` (174) or `udc-an33-38` (60). All 90 successes landed on `udc-ba*` / `udc-an40-25` / `udc-an28-1` / `udc-an38-*`. Those are A6000 nodes; the an33 nodes are a different architecture.

## Phase 7 — first retry (cancelled)
- Set `--partition=gpu-a6000` in both sbatch files.
- Retry baseline 11921329 failed immediately with a different error:
  `SyntaxError: unterminated string literal` in the Python heredoc that
  serialises metadata. Root cause: Python's `csv.DictWriter` writes CRLF by
  default on Linux too (it's platform-agnostic); the bash `IFS=, read` leaves
  `\r` on the last field value; the `\r` makes the Python string literal
  look unterminated.
- Cancelled retry (4 tasks already failed). Fixed the bug two ways:
  - `build_manifest.py` / `build_matrix.py`: `lineterminator="\n"` on DictWriter.
  - `run_array.sbatch` / `run_matrix.sbatch`: defensive `| tr -d '\r'` on the
    row read, so any external CSV works.
- Also stripped `\r` from the already-written `scripts/manifest.csv` and
  `experiments/demo/matrix.csv` with `sed -i 's/\r$//'`.

## Phase 8 — smoke test on gpu-a6000
- Submitted `sbatch --array=0-1 scripts/run_array.sbatch` → job 11921400.
- Both tasks COMPLETED. Produced real `outputs/*_11921400_{0,1}.txt` with
  event counts > 0. Confirmed both fixes work.

## Phase 9 — full retry submitted
- baseline retry: **11921419** (36 tasks, concurrency 6, gpu-a6000)
- demo retry:     **11921420** (234 tasks, concurrency 8, gpu-a6000)
- baseline aggregate: **11921421** (afterany)
- demo aggregate:     **11921422** (afterany)

Expected wall time: ~45-60 min for demo retry at concurrency 8.
Combined with 90 runs already complete, final `experiments/demo/analysis/summary.md` will cover the full 324 runs.
