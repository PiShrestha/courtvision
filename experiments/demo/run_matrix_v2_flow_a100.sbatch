#!/bin/bash
# 50-clip flow rerun on A100 via the biocomplexity account.
# a100 nodes (udc-an28/34/36/37) are physically separate from the broken
# udc-an33-* v100 nodes called out in CLAUDE.local.md, so no exclude is
# needed.

#SBATCH --job-name=cv_v2_flow_a100
#SBATCH --output=logs/v2_flow_a100_%A_%a.out
#SBATCH --error=logs/v2_flow_a100_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

MATRIX="${MATRIX:-experiments/demo/matrix_v2_50clips_flow.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
line_no=$((TASK_ID + 2))
row=$(sed -n "${line_no}p" "${MATRIX}" | tr -d '\r')
[ -z "${row}" ] && { echo "no row for task ${TASK_ID}"; exit 1; }

IFS=, read -r RUN_ID VIDEO START END HOOP_CONFIG MODEL IMGSZ CONFIDENCE \
              RIM_KIND RIM_RESEED UPWARD HIST RELEASE APPROACH COOLDOWN POSS_DIST \
              OCCLUSION MADE_WIN ENTER_FACTOR HORIZ_FACTOR MIN_DV <<< "${row}"

run_tag="v2flow_run${RUN_ID}_$(basename "${VIDEO}" | cut -d. -f1)_t${START%.*}-${END%.*}_$(basename "${MODEL}" .pt)_i${IMGSZ}_rim-${RIM_KIND}_a100"
OUT_PREFIX="experiments/demo/outputs_v2_flow/${run_tag}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"

mkdir -p logs experiments/demo/outputs_v2_flow
source .venv/bin/activate
echo "running on $(hostname) — $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python experiments/demo/run_demo_v2.py \
    --video "${VIDEO}" --start "${START}" --end "${END}" \
    --model "${MODEL}" --imgsz "${IMGSZ}" --confidence "${CONFIDENCE}" \
    --hoop "${HOOP_CONFIG}" \
    --rim-tracker-kind "${RIM_KIND}" --rim-reseed-every "${RIM_RESEED}" \
    --upward-trigger "${UPWARD}" --history-frames "${HIST}" \
    --release-dist-px "${RELEASE}" --approach-dist-px "${APPROACH}" \
    --cooldown-frames "${COOLDOWN}" --possession-dist-px "${POSS_DIST}" \
    --occlusion-gap-frames "${OCCLUSION}" \
    --attempt-to-made-window "${MADE_WIN}" \
    --enter-zone-radius-factor "${ENTER_FACTOR}" \
    --horizontal-pad-factor "${HORIZ_FACTOR}" \
    --min-downward-velocity "${MIN_DV}" \
    --save-video --out "${OUT_PREFIX}"
