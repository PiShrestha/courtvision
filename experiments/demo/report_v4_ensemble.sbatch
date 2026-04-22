#!/bin/bash
# v4 ensemble aggregator: per-variant reports + cross-variant table that
# includes BOTH the v3 ablation variants and the v4 ensemble variants.

#SBATCH --job-name=cv_v4_ens_agg
#SBATCH --output=logs/v4_ens_agg_%j.out
#SBATCH --error=logs/v4_ens_agg_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
umask 0022         # keep outputs group+other readable
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

for v in full_ensemble full_strict yoloe_pose pose_sam3_1; do
    python experiments/demo/per_clip_report.py \
        --runs "experiments/demo/outputs_v4_ensemble/${v}/*.meta.json" \
        --out "experiments/demo/analysis_v4_ensemble/${v}/per_clip_report.md" || true
done

# merged comparison across all 9 variants (v3 5 + v4 4).
python experiments/demo/compare_variants.py \
    --runs-root experiments/demo/outputs_v3_ablation \
    --out experiments/demo/analysis_v4_ensemble/v3_only_report.md || true

python experiments/demo/compare_v4.py \
    --v3-root experiments/demo/outputs_v3_ablation \
    --v4-root experiments/demo/outputs_v4_ensemble \
    --out experiments/demo/analysis_v4_ensemble/cross_all_variants.md || true
