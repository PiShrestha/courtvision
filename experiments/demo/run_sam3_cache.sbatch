#!/bin/bash
# pre-compute SAM 3 PerceptionSignals for every clip in the 50-clip matrix.
# one task per clip. writes JSONL under experiments/demo/sam3_cache/.
#
# env knobs:
#   MATRIX       matrix csv to read clip rows from (default 50-clip flow)
#   CHECKPOINT   "sam3" | "sam3.1" — which facebook/* repo to load
#
# gpu-a6000 is safe (we're on cs6770_sp26 per the balance audit).
# SAM 3 is ~848M params; A100 would be faster but a6000 has the SU headroom.

#SBATCH --job-name=cv_sam3_cache
#SBATCH --output=logs/sam3_cache_%A_%a.out
#SBATCH --error=logs/sam3_cache_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --account=cs6770_sp26
# memory bumped 32G -> 48G after first attempt OOM'd: SAM 3 session API
# does not scope to time windows, so it preloads the entire 16 min video
# into RAM (~28k frames at 1080p60). we now pre-extract the 60-90s window
# with ffmpeg below so memory scales with the window, not the full source.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

MATRIX="${MATRIX:-experiments/demo/matrix_v2_50clips_flow.csv}"
CHECKPOINT="${CHECKPOINT:-sam3.1}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
row=$(sed -n "$((TASK_ID + 2))p" "${MATRIX}" | tr -d '\r')
[ -z "${row}" ] && { echo "no row for task ${TASK_ID}"; exit 1; }

IFS=, read -r RUN_ID VIDEO START END _HOOP _MODEL _IMGSZ _CONF \
              _REST <<< "${row}"

stem="$(basename "${VIDEO}" | cut -d. -f1)"
out_dir="experiments/demo/sam3_cache/${CHECKPOINT}"
mkdir -p "${out_dir}" logs "$HOME/.triton/cache"

# triton JIT-compiles tiny cuda_utils.c per process; the system's
# /usr/include/python3.11/ directory lacks Python.h, so the compile fails.
# prepend a known-good Python.h location (miniforge ships one) and pin
# the triton cache so the first task compiles once and all subsequent
# tasks in the array reuse the .so.
export CPATH="/apps/software/standard/core/miniforge/24.3.0-py3.11/include/python3.11:${CPATH:-}"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
out_path="${out_dir}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"

# pre-extract the window to a temp mp4 so SAM 3 doesn't load the full source.
clip_tmp="/tmp/cv_sam3_${SLURM_JOB_ID}_${TASK_ID}.mp4"
cleanup() { rm -f "${clip_tmp}" 2>/dev/null || true; }
trap cleanup EXIT
dur=$(awk "BEGIN {printf \"%.3f\", ${END} - ${START}}")
echo "=================================================================="
echo "SAM 3 cache task ${TASK_ID}  checkpoint=${CHECKPOINT}"
echo "  video  : ${VIDEO}  window ${START}-${END}s (${dur}s)"
echo "  clip   : ${clip_tmp}"
echo "  out    : ${out_path}"
echo "  node   : $(hostname)"
echo "=================================================================="

module load ffmpeg 2>/dev/null || true
ffmpeg -nostdin -y -ss "${START}" -i "${VIDEO}" -t "${dur}" \
    -c copy "${clip_tmp}" 2>&1 | tail -3

python - <<PY
import sys, json, time
sys.path.insert(0, "experiments/demo")
from sam3_tracker import Sam3Tracker, Sam3Config

cfg = Sam3Config(checkpoint="${CHECKPOINT}")
tracker = Sam3Tracker(cfg)
t0 = time.time()
# feed the pre-extracted clip; SAM 3 reads frames from index 0 which
# now correspond to original video time ${START}s. we remap frame ids
# back in the output so downstream consumers align with the original.
raw_signals = tracker.run("${clip_tmp}", start_s=0.0, end_s=None)

import cv2
cap = cv2.VideoCapture("${VIDEO}")
fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
cap.release()
offset = int(round(float("${START}") * fps))

# shift frame ids to absolute frame indices in the source video.
for s in raw_signals:
    s.frame_id = int(s.frame_id) + offset

# write the shifted signals.
from pathlib import Path
out_p = Path("${out_path}")
out_p.parent.mkdir(parents=True, exist_ok=True)
raw_signals.sort(key=lambda s: (s.frame_id, s.target))
with out_p.open("w") as f:
    for s in raw_signals:
        f.write(json.dumps({
            "source": s.source, "target": s.target,
            "frame_id": s.frame_id, "confidence": s.confidence,
            "bbox": s.bbox, "center": list(s.center) if s.center else None,
            "radius": s.radius, "track_id": s.track_id,
            "extras": s.extras,
        }) + "\n")
print(f"wrote {out_p} ({len(raw_signals)} signals) in {time.time()-t0:.1f}s")
PY
