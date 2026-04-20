#!/bin/bash
# runs once the matrix array finishes, via --dependency=afterany:<array_id>.

#SBATCH --job-name=cv_demo_agg
#SBATCH --output=logs/demo_agg_%j.out
#SBATCH --error=logs/demo_agg_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

python experiments/demo/aggregate_matrix.py \
    --glob 'experiments/demo/outputs/run*.meta.json' \
    --outdir experiments/demo/analysis

echo "done."
ls -lh experiments/demo/analysis/
