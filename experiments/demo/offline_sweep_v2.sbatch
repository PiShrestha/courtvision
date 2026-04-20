#!/bin/bash
# offline v2 sweep: replay every existing tracks.jsonl through the v2
# pipeline with every row of matrix_v2_offline.csv. cpu-only, fast.
#
# submit once; produces N_tracks * N_sweep output files under outputs_v2/.

#SBATCH --job-name=cv_v2_offline
#SBATCH --output=logs/v2_offline_%j.out
#SBATCH --error=logs/v2_offline_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate
mkdir -p logs experiments/demo/outputs_v2

MATRIX="${MATRIX:-experiments/demo/matrix_v2_offline.csv}"
TRACKS_GLOB="${TRACKS_GLOB:-experiments/demo/outputs/run*.tracks.jsonl}"
OUTDIR="${OUTDIR:-experiments/demo/outputs_v2}"

echo "======================================================"
echo "v2 offline sweep"
echo "  matrix      : ${MATRIX}"
echo "  tracks glob : ${TRACKS_GLOB}"
echo "  outdir      : ${OUTDIR}"
echo "======================================================"

python experiments/demo/offline_sweep_v2.py \
    --tracks-glob "${TRACKS_GLOB}" \
    --matrix "${MATRIX}" \
    --outdir "${OUTDIR}"
