#!/bin/bash
# supplementary rim-only sam 3.1 cache pass.
# prompts only "basketball hoop" (empirically works; "basketball rim" returns 0)
# and writes rim signals to sam3_cache/sam3.1_rim/. a follow-up merge step
# appends these into the existing sam3.1/*.jsonl so the ablation sees the
# combined ball + player + rim data without re-running the heavier prompts.

#SBATCH --job-name=cv_sam3_rim
#SBATCH --output=logs/sam3_rim_%A_%a.out
#SBATCH --error=logs/sam3_rim_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=nssac_students

set -euo pipefail
umask 0022         # keep outputs group+other readable for collaborators
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

MATRIX="${MATRIX:-experiments/demo/matrix_v2_50clips_flow.csv}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"
row=$(sed -n "$((TASK_ID + 2))p" "${MATRIX}" | tr -d '\r')
[ -z "${row}" ] && { echo "no row for task ${TASK_ID}"; exit 1; }
IFS=, read -r RUN_ID VIDEO START END _REST <<< "${row}"

stem="$(basename "${VIDEO}" | cut -d. -f1)"
out_dir="experiments/demo/sam3_cache/sam3.1_rim"
out_path="${out_dir}/run${RUN_ID}_${stem}_t${START%.*}-${END%.*}.sam3.jsonl"
mkdir -p "${out_dir}" logs "$HOME/.triton/cache"

export CPATH="/apps/software/standard/core/miniforge/24.3.0-py3.11/include/python3.11:${CPATH:-}"
export TRITON_CACHE_DIR="$HOME/.triton/cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source .venv/bin/activate

clip_tmp="/tmp/cv_rim_${SLURM_JOB_ID}_${TASK_ID}.mp4"
trap "rm -f ${clip_tmp} 2>/dev/null || true" EXIT
dur=$(awk "BEGIN {printf \"%.3f\", ${END} - ${START}}")

echo "rim-only cache task ${TASK_ID}  ${stem} t${START}-${END}"
module load ffmpeg 2>/dev/null || true
ffmpeg -nostdin -y -ss "${START}" -i "${VIDEO}" -t "${dur}" \
    -r 30 -c:v libx264 -preset ultrafast -crf 23 -an "${clip_tmp}" 2>&1 | tail -3

python - <<PY
import sys, json, time, cv2
sys.path.insert(0, "experiments/demo")
from sam3_tracker import Sam3Tracker, Sam3Config
from pathlib import Path

# only prompt for rim; reuse the same multi-phrase fallback logic.
cfg = Sam3Config(checkpoint="sam3.1",
                   concepts={"rim": ["basketball hoop", "hoop",
                                     "orange rim", "basketball rim"]})
tracker = Sam3Tracker(cfg)
t0 = time.time()
raw = tracker.run("${clip_tmp}", start_s=0.0)

cap = cv2.VideoCapture("${VIDEO}")
fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
cap.release()
offset = int(round(float("${START}") * fps))
ratio = fps / 30.0
for s in raw:
    s.frame_id = offset + int(round(int(s.frame_id) * ratio))

out = Path("${out_path}")
out.parent.mkdir(parents=True, exist_ok=True)
raw.sort(key=lambda s: (s.frame_id, s.target))
with out.open("w") as f:
    for s in raw:
        f.write(json.dumps({
            "source": s.source, "target": s.target, "frame_id": s.frame_id,
            "confidence": s.confidence, "bbox": s.bbox,
            "center": list(s.center) if s.center else None,
            "radius": s.radius, "track_id": s.track_id, "extras": s.extras,
        }) + "\n")
print(f"wrote {out} ({len(raw)} rim signals) in {time.time()-t0:.1f}s")
print(f"chosen phrase: {tracker._chosen_phrases}")
PY
