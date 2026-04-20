#!/bin/bash
# aggregate v2 sweep outputs into csv + summary.md + charts.

#SBATCH --job-name=cv_v2_agg
#SBATCH --output=logs/v2_agg_%j.out
#SBATCH --error=logs/v2_agg_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

python experiments/demo/aggregate_v2.py \
    --v2-glob 'experiments/demo/outputs_v2*/*.meta_v2.json' \
    --live-glob 'experiments/demo/outputs_v2_live*/*.meta.json' \
    --v1-glob 'experiments/demo/outputs/run*.meta.json' \
    --outdir experiments/demo/analysis_v2
