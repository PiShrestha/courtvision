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
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --account=cs6770_sp26

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
mkdir -p "${out_dir}" logs
out_path="${out_dir}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"

echo "=================================================================="
echo "SAM 3 cache task ${TASK_ID}  checkpoint=${CHECKPOINT}"
echo "  video  : ${VIDEO}  window ${START}-${END}s"
echo "  out    : ${out_path}"
echo "  node   : $(hostname)"
echo "=================================================================="

python - <<PY
import sys, json, time
sys.path.insert(0, "experiments/demo")
from sam3_tracker import Sam3Tracker, Sam3Config

cfg = Sam3Config(checkpoint="${CHECKPOINT}")
tracker = Sam3Tracker(cfg)
t0 = time.time()
out = tracker.run_to_file("${VIDEO}", "${out_path}",
                            start_s=float("${START}"), end_s=float("${END}"))
print(f"wrote {out}  in {time.time()-t0:.1f}s")
PY
