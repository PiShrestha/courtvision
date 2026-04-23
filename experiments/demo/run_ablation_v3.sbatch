#!/bin/bash
# 5-variant ablation array runner.
# reads one row of matrix_v3_ablation.csv and runs run_demo_v3.py with the
# appropriate backend flags for that row's variant.
#
# the matrix CSV column order is:
#   task_id, variant, run_id, video, start, end, hoop_config, model, imgsz,
#   confidence, rim_tracker_kind, rim_reseed_every, upward_trigger,
#   history_frames, release_dist_px, approach_dist_px, cooldown_frames,
#   possession_dist_px, occlusion_gap_frames, attempt_to_made_window,
#   enter_zone_radius_factor, horizontal_pad_factor, min_downward_velocity,
#   use_yoloe, use_pose, sam3_cache

#SBATCH --job-name=cv_v3_abl
#SBATCH --output=logs/v3_abl_%A_%a.out
#SBATCH --error=logs/v3_abl_%A_%a.err
#SBATCH --time=01:30:00   # pose variants hit timeout at 40m; bumped with margin
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

MATRIX="${MATRIX:-experiments/demo/matrix_v3_ablation.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
row=$(sed -n "$((TASK_ID + 2))p" "${MATRIX}" | tr -d '\r')
[ -z "${row}" ] && { echo "no row for task ${TASK_ID}"; exit 1; }

IFS=, read -r _TASKID VARIANT RUN_ID VIDEO START END HOOP_CONFIG MODEL IMGSZ CONFIDENCE \
              RIM_KIND RIM_RESEED UPWARD HIST RELEASE APPROACH COOLDOWN POSS_DIST \
              OCCLUSION MADE_WIN ENTER_FACTOR HORIZ_FACTOR MIN_DV \
              USE_YOLOE USE_POSE SAM3_CACHE <<< "${row}"

stem="$(basename "${VIDEO}" | cut -d. -f1)"
run_tag="v3abl_${VARIANT}_run${RUN_ID}_${stem}_t${START%.*}-${END%.*}"
OUT_PREFIX="experiments/demo/outputs_v3_ablation/${VARIANT}/${run_tag}_${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
mkdir -p logs "$(dirname "${OUT_PREFIX}")"

mkdir -p "$HOME/.triton/cache"
# make sure triton can find Python.h during its JIT compiles.
export CPATH="/apps/software/standard/core/miniforge/24.3.0-py3.11/include/python3.11:${CPATH:-}"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
source .venv/bin/activate
echo "=================================================================="
echo "v3 ablation task ${TASK_ID}  variant=${VARIANT}  clip=${stem} t${START}-${END}"
echo "  flags: use_yoloe=${USE_YOLOE} use_pose=${USE_POSE} sam3_cache=${SAM3_CACHE}"
echo "  node : $(hostname)"
echo "=================================================================="

# build backend flag list from the variant's bit flags.
flags=()
if [ "${USE_YOLOE}" = "1" ]; then flags+=("--use-yoloe"); fi
if [ "${USE_POSE}" = "1" ]; then flags+=("--use-pose"); fi
if [ -n "${SAM3_CACHE}" ]; then
    cache_path="experiments/demo/sam3_cache/${SAM3_CACHE}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"
    if [ -f "${cache_path}" ]; then
        flags+=("--sam3-cache" "${cache_path}")
    else
        echo "WARNING: SAM 3 cache missing for variant ${VARIANT}: ${cache_path}"
        echo "falling back to non-SAM3 run."
    fi
fi

# keep annotated mp4 for every 5th task so we can spot-check visually.
if [ "$((TASK_ID % 5))" = "0" ]; then
    save="--save-video"
else
    save="--no-save-video"
fi

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
    ${flags[@]+"${flags[@]}"} \
    ${save} \
    --out "${OUT_PREFIX}"
