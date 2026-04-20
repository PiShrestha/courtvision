#!/bin/bash
# aggregate + per-clip report for the new-videos v2 matrix.
# runs afterany the array completes.

#SBATCH --job-name=cv_v2_newv_agg
#SBATCH --output=logs/v2_newv_agg_%j.out
#SBATCH --error=logs/v2_newv_agg_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

# standard aggregation (csv + summary + charts).
python experiments/demo/aggregate_v2.py \
    --v2-glob '/nonexistent/*.meta_v2.json' \
    --live-glob 'experiments/demo/outputs_v2_new_videos/*.meta.json' \
    --v1-glob 'experiments/demo/outputs/run*.meta.json' \
    --outdir experiments/demo/analysis_v2_new_videos

# per-clip report with the metrics the user asked for.
python experiments/demo/per_clip_report.py \
    --runs 'experiments/demo/outputs_v2_new_videos/*.meta.json' \
    --out experiments/demo/analysis_v2_new_videos/per_clip_report.md
