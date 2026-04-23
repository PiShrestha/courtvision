#!/bin/bash
# stage-3 vlm reasoning pass over the full_ensemble_26 events.
# iterates every events.json in outputs_v3_ablation/full_ensemble_26/, grabs
# the matching source clip path from the sidecar meta.json, calls gemini 2.0
# flash on each shot event, and writes events_vlm.json next to the input.
#
# runs on the standard partition (no gpu needed — gemini is api-bound).

#SBATCH --job-name=cv_vlm_verify
#SBATCH --output=logs/vlm_verify_%j.out
#SBATCH --error=logs/vlm_verify_%j.err
#SBATCH --time=00:45:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=standard
#SBATCH --account=cs6770_sp26

set -euo pipefail
umask 0022
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
source .venv/bin/activate

if [ -z "${GEMINI_API_KEY:-}" ] && [ -f .env ]; then
    set -a; source .env; set +a
fi
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be set (via env or .env)}"

variant_dir="experiments/demo/outputs_v3_ablation/full_ensemble_26"
shopt -s nullglob
reviewed=0
errors=0
for events_f in "${variant_dir}"/*.events.json; do
    meta_f="${events_f%.events.json}.meta.json"
    out_f="${events_f%.events.json}.events_vlm.json"
    if [ -f "${out_f}" ]; then
        echo "skip ${events_f} (already verified)"
        continue
    fi
    [ -f "${meta_f}" ] || { echo "no meta for ${events_f}"; errors=$((errors+1)); continue; }
    video=$(python -c "import json,sys; print(json.load(open('${meta_f}')).get('video',''))")
    [ -f "${video}" ] || { echo "missing video ${video}"; errors=$((errors+1)); continue; }
    echo "verify ${events_f}  video=${video}"
    python experiments/demo/vlm_verify.py \
        --events "${events_f}" --video "${video}" --out "${out_f}" \
        --model gemini-2.0-flash \
        || { echo "vlm_verify failed for ${events_f}"; errors=$((errors+1)); }
    reviewed=$((reviewed+1))
done
echo "vlm verify done: ${reviewed} clips processed, ${errors} errors"
