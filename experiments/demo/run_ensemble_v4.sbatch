#!/bin/bash
# v4 ensemble ablation: 4 variants x 50 clips = 200 tasks.
# runs the same shot_pipeline_v3 with combinations of yoloe / rtmpose /
# sam 3.1 / strict-consensus toggles. reuses the sam 3 cache produced
# by run_sam3_cache.sbatch.

#SBATCH --job-name=cv_v4_ens
#SBATCH --output=logs/v4_ens_%A_%a.out
#SBATCH --error=logs/v4_ens_%A_%a.err
#SBATCH --time=01:30:00   # full_strict/yoloe_pose/pose_sam3_1 need pose headroom
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --exclude=udc-an33-[37-38]   # v100 nodes; torch 2.11+cu130 lacks sm_70 kernels
#SBATCH --account=cs6770_sp26

set -euo pipefail
umask 0022         # keep outputs group+other readable
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

MATRIX="${MATRIX:-experiments/demo/matrix_v4_ensemble.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
row=$(sed -n "$((TASK_ID + 2))p" "${MATRIX}" | tr -d '\r')
[ -z "${row}" ] && { echo "no row for task ${TASK_ID}"; exit 1; }

IFS=, read -r _TASKID VARIANT RUN_ID VIDEO START END HOOP_CONFIG MODEL IMGSZ CONFIDENCE \
              RIM_KIND RIM_RESEED UPWARD HIST RELEASE APPROACH COOLDOWN POSS_DIST \
              OCCLUSION MADE_WIN ENTER_FACTOR HORIZ_FACTOR MIN_DV \
              USE_YOLOE USE_POSE SAM3_CACHE STRICT <<< "${row}"

stem="$(basename "${VIDEO}" | cut -d. -f1)"
run_tag="v4ens_${VARIANT}_run${RUN_ID}_${stem}_t${START%.*}-${END%.*}"
OUT_PREFIX="experiments/demo/outputs_v4_ensemble/${VARIANT}/${run_tag}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
mkdir -p logs "$(dirname "${OUT_PREFIX}")"

mkdir -p "$HOME/.triton/cache"
export CPATH="/apps/software/standard/core/miniforge/24.3.0-py3.11/include/python3.11:${CPATH:-}"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .venv/bin/activate

echo "v4 ensemble task ${TASK_ID}  variant=${VARIANT}  clip=${stem} t${START}-${END}"
echo "  yoloe=${USE_YOLOE} pose=${USE_POSE} sam3=${SAM3_CACHE} strict=${STRICT}"

flags=()
[ "${USE_YOLOE}" = "1" ] && flags+=("--use-yoloe")
[ "${USE_POSE}" = "1" ] && flags+=("--use-pose")
[ "${STRICT}" = "1" ] && flags+=("--strict-consensus")
if [ -n "${SAM3_CACHE}" ]; then
    cache_path="experiments/demo/sam3_cache/${SAM3_CACHE}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"
    [ -f "${cache_path}" ] && flags+=("--sam3-cache" "${cache_path}") \
        || echo "warn: sam3 cache missing: ${cache_path}"
fi

save="--no-save-video"
[ "$((TASK_ID % 5))" = "0" ] && save="--save-video"

python experiments/demo/run_demo_v3.py \
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
    ${flags[@]+"${flags[@]}"} ${save} \
    --out "${OUT_PREFIX}"
