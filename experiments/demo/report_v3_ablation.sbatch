#!/bin/bash
# aggregator + per-clip + per-variant comparison report for the v3 ablation.

#SBATCH --job-name=cv_v3_abl_agg
#SBATCH --output=logs/v3_abl_agg_%j.out
#SBATCH --error=logs/v3_abl_agg_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

# per-variant standard aggregation
for variant in baseline plus_yoloe plus_pose plus_sam3 plus_sam3_1; do
    echo "=== aggregating ${variant} ==="
    python experiments/demo/per_clip_report.py \
        --runs "experiments/demo/outputs_v3_ablation/${variant}/*.meta.json" \
        --out "experiments/demo/analysis_v3_ablation/${variant}/per_clip_report.md" || true
done

# cross-variant comparison (per clip, all 5 variants side-by-side)
python experiments/demo/compare_variants.py \
    --runs-root experiments/demo/outputs_v3_ablation \
    --out experiments/demo/analysis_v3_ablation/cross_variant_report.md \
    || echo "compare_variants not yet built; skipping"
