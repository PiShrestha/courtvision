#!/bin/bash
# offline v2 rule-parameter sweep over the 50 new tracks.jsonl files.
# CPU-only; fires on standard partition afterany the GPU matrices finish.
#
# replays each of the 50 tracks files through the 15-row rule-variant
# matrix (matrix_v2_offline.csv) -> 750 replays. runs in ~1-2 min.

#SBATCH --job-name=cv_v2_50c_sweep
#SBATCH --output=logs/v2_50c_sweep_%j.out
#SBATCH --error=logs/v2_50c_sweep_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=biocomplexity

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate
mkdir -p experiments/demo/outputs_v2_50c_sweep

python experiments/demo/offline_sweep_v2.py \
    --tracks-glob 'experiments/demo/outputs_v2_flow/*.tracks.jsonl' \
    --matrix experiments/demo/matrix_v2_offline.csv \
    --outdir experiments/demo/outputs_v2_50c_sweep

python experiments/demo/aggregate_v2.py \
    --v2-glob 'experiments/demo/outputs_v2_50c_sweep/*.meta_v2.json' \
    --live-glob '/nonexistent/*.meta.json' \
    --v1-glob '/nonexistent/*.meta.json' \
    --outdir experiments/demo/analysis_v2_50c_sweep
