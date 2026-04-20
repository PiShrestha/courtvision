#!/bin/bash
# =============================================================================
# CourtVision pipeline runner.
#
# Works two ways:
#   1. Slurm submission:   sbatch scripts/run_courtvision.sh
#   2. Local / interactive: bash scripts/run_courtvision.sh
#
# All settings are environment variables so you don't have to edit the file
# for each run. Example:
#
#   VIDEO=videos/game1.mov START=120 END=210 STRIDE=2 \
#       sbatch scripts/run_courtvision.sh
#
#   VIDEO=videos/clip.mov bash scripts/run_courtvision.sh
#
# Outputs:
#   logs/courtvision_<jobid>.out  (stdout / stderr when run via sbatch)
#   outputs/<videoname>_<jobid>.txt   (stats report)
# =============================================================================

# ----- Slurm directives -------------------------------------------------------
# Edit partition / account / time as needed for your cluster.
#SBATCH --job-name=courtvision
#SBATCH --output=logs/courtvision_%j.out
#SBATCH --error=logs/courtvision_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=cs6770_sp26

set -euo pipefail

# ----- Configuration (override via env vars) ----------------------------------
VIDEO="${VIDEO:?VIDEO is required, e.g. VIDEO=videos/game.mov}"
START="${START:-0}"
END="${END:-}"                       # empty = end of video
STRIDE="${STRIDE:-1}"
HOMOGRAPHY="${HOMOGRAPHY:-}"         # path to JSON correspondences (optional)
MODEL="${MODEL:-yolov8m.pt}"
CONFIDENCE="${CONFIDENCE:-0.35}"
IMGSZ="${IMGSZ:-640}"
TRACKER="${TRACKER:-bytetrack.yaml}"

# Encode the perception config in the output filename so a sweep produces
# distinguishable artefacts.
JOB_TAG="${SLURM_JOB_ID:-local$$}"
VIDEO_BASE="$(basename "${VIDEO}")"
VIDEO_STEM="${VIDEO_BASE%.*}"
MODEL_TAG="$(basename "${MODEL}" .pt)"
CONF_TAG="conf$(printf "%.0f" "$(echo "${CONFIDENCE} * 100" | bc 2>/dev/null || echo "${CONFIDENCE}")")"
TRACKER_TAG="$(basename "${TRACKER}" .yaml)"
RUN_TAG="${MODEL_TAG}_${CONF_TAG}_imgsz${IMGSZ}_${TRACKER_TAG}"
END_TAG="${END:-EOF}"
WINDOW_TAG="t${START}-${END_TAG}"
OUT_TXT="${OUT_TXT:-outputs/${VIDEO_STEM}_${WINDOW_TAG}_${RUN_TAG}_${JOB_TAG}.txt}"

# ----- Move to repo root and activate venv -----------------------------------
# Under sbatch, $0 / BASH_SOURCE point at Slurm's spool copy of the script,
# not the original. Use SLURM_SUBMIT_DIR (which is the directory sbatch was
# run from) when present, fall back to BASH_SOURCE for `bash scripts/...` use.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

if [ ! -f .venv/bin/activate ]; then
    echo "ERROR: .venv not found at ${REPO_ROOT}/.venv" >&2
    echo "Create it with: /usr/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p logs outputs

# ----- Build CLI args ---------------------------------------------------------
ARGS=(
    --video "${VIDEO}"
    --out-txt "${OUT_TXT}"
    --start "${START}"
    --stride "${STRIDE}"
    --model "${MODEL}"
    --confidence "${CONFIDENCE}"
    --imgsz "${IMGSZ}"
    --tracker "${TRACKER}"
)
[ -n "${END}" ]        && ARGS+=(--end "${END}")
[ -n "${HOMOGRAPHY}" ] && ARGS+=(--homography-config "${HOMOGRAPHY}")

# ----- Provenance log ---------------------------------------------------------
echo "===================================================================="
echo "CourtVision run"
echo "  job tag      : ${JOB_TAG}"
echo "  host         : $(hostname)"
echo "  date         : $(date -Iseconds)"
echo "  python       : $(python --version 2>&1)"
echo "  video        : ${VIDEO}"
echo "  start..end   : ${START}s..${END:-EOF}"
echo "  stride       : ${STRIDE}"
echo "  model        : ${MODEL}"
echo "  confidence   : ${CONFIDENCE}"
echo "  imgsz        : ${IMGSZ}"
echo "  tracker      : ${TRACKER}"
echo "  homography   : ${HOMOGRAPHY:-<none>}"
echo "  output       : ${OUT_TXT}"
echo "  cuda visible : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  command      : python main.py ${ARGS[*]}"
echo "===================================================================="

python main.py "${ARGS[@]}"
echo "Done. Stats report at: ${OUT_TXT}"
