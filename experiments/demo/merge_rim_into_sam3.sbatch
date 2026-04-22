#!/bin/bash
# append rim-only cache signals into the primary sam3.1 cache jsonls
# so the ablation pipeline picks up all three targets (ball, player, rim)
# from one file per clip.

#SBATCH --job-name=cv_rim_merge
#SBATCH --output=logs/rim_merge_%j.out
#SBATCH --error=logs/rim_merge_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

src_dir="experiments/demo/sam3_cache/sam3.1_rim"
dst_dir="experiments/demo/sam3_cache/sam3.1"
appended=0
skipped=0
for rim_f in "${src_dir}"/*.sam3.jsonl; do
    base=$(basename "${rim_f}")
    dst_f="${dst_dir}/${base}"
    if [ ! -f "${dst_f}" ]; then
        echo "skip ${base} (no base cache file)"
        skipped=$((skipped + 1))
        continue
    fi
    # idempotency: skip if rim already merged (look for a rim line in dst).
    if grep -q '"target": "rim"' "${dst_f}" 2>/dev/null; then
        echo "skip ${base} (rim already present)"
        skipped=$((skipped + 1))
        continue
    fi
    cat "${rim_f}" >> "${dst_f}"
    appended=$((appended + 1))
done
echo "merged ${appended} rim files, skipped ${skipped}"
