#!/bin/bash
# slurm array task runner for the demo experiment matrix.
# reads a row of experiments/demo/matrix.csv and runs run_demo.py with its params.

#SBATCH --job-name=cv_demo
#SBATCH --output=logs/demo_%A_%a.out
#SBATCH --error=logs/demo_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --account=cs6770_sp26

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

MATRIX="${MATRIX:-experiments/demo/matrix.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
line_no=$((TASK_ID + 2))
row=$(sed -n "${line_no}p" "${MATRIX}" | tr -d '\r')
[ -z "${row}" ] && { echo "no row for task ${TASK_ID}"; exit 1; }

IFS=, read -r RUN_ID VIDEO START END HOOP_CONFIG MODEL IMGSZ CONFIDENCE TRACKER POSSESSION_DIST UPWARD MADE_WIN HOOP_PAD <<< "${row}"

run_tag="run${RUN_ID}_$(basename "${VIDEO}" | cut -d. -f1)_t${START%.*}-${END%.*}_$(basename "${MODEL}" .pt)_i${IMGSZ}_c${CONFIDENCE}_ut${UPWARD}_mw${MADE_WIN}_hp${HOOP_PAD}"
OUT_PREFIX="experiments/demo/outputs/${run_tag}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"

mkdir -p logs experiments/demo/outputs
source .venv/bin/activate

echo "=================================================================="
echo "array task ${TASK_ID}  run_id=${RUN_ID}"
echo "  video        : ${VIDEO}  window ${START}-${END}s"
echo "  perception   : ${MODEL} imgsz=${IMGSZ} conf=${CONFIDENCE}"
echo "  shot         : upward=${UPWARD} made_window=${MADE_WIN} hoop_pad=${HOOP_PAD}"
echo "  hoop_config  : ${HOOP_CONFIG}"
echo "  output       : ${OUT_PREFIX}"
echo "=================================================================="

# save annotated mp4 for ~1 in 6 tasks so disk stays under 6 GB total.
# enough to spot-check every video x detector bucket visually.
save_video="--no-save-video"
if [ "$((TASK_ID % 6))" = "0" ]; then
    save_video="--save-video"
fi

python experiments/demo/run_demo.py \
    --video "${VIDEO}" \
    --start "${START}" --end "${END}" \
    --model "${MODEL}" --imgsz "${IMGSZ}" --confidence "${CONFIDENCE}" \
    --tracker "${TRACKER}" \
    --possession-dist-px "${POSSESSION_DIST}" \
    --upward-trigger "${UPWARD}" \
    --made-window "${MADE_WIN}" \
    --hoop-pad "${HOOP_PAD}" \
    --hoop "${HOOP_CONFIG}" \
    ${save_video} \
    --out "${OUT_PREFIX}"
