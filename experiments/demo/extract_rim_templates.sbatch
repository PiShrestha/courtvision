#!/bin/bash
# one-shot sam 3.1 pass to cut tight rim crops from rims_mk/*.png.

#SBATCH --job-name=cv_rim_templates
#SBATCH --output=logs/rim_templates_%j.out
#SBATCH --error=logs/rim_templates_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=udc-an33-[37-38]
#SBATCH --account=nssac_students

set -euo pipefail
umask 0022
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

mkdir -p "$HOME/.triton/cache" logs
export CPATH="/apps/software/standard/core/miniforge/24.3.0-py3.11/include/python3.11:${CPATH:-}"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .venv/bin/activate

module load ffmpeg 2>/dev/null || true

: "${RIM_INPUT_DIR:=experiments/demo/rims_mk}"
: "${RIM_OUT_DIR:=${RIM_INPUT_DIR}/templates}"
: "${RIM_PAD_FRAC:=0.12}"
echo "extracting from ${RIM_INPUT_DIR} -> ${RIM_OUT_DIR} (pad=${RIM_PAD_FRAC})"

python experiments/demo/extract_rim_templates.py \
    --input-dir "${RIM_INPUT_DIR}" \
    --out-dir   "${RIM_OUT_DIR}" \
    --pad-frac  "${RIM_PAD_FRAC}"
