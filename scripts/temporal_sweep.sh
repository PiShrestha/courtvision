#!/bin/bash
# =============================================================================
# Temporal sweep: same perception config, six different 90s windows spread
# across the full 16-minute video. Used to find out whether failure modes
# (over-detection, ID churn, missed shots) are uniform across the clip
# (= structural bug in our code) or local to specific scenes (= the camera
# state at that moment is the problem).
#
# Config used: the winner of the parameter sweep -- yolov8x + imgsz=1280 +
# bytetrack -- so any failures we observe are NOT due to a weak detector.
#
# Usage:
#   bash scripts/temporal_sweep.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VIDEO="1v1-mk.mov"
MODEL="yolov8x.pt"
CONFIDENCE="0.35"
IMGSZ="1280"
TRACKER="bytetrack.yaml"

# Six non-overlapping 90s windows across the 960s video.
# Each row: START END
windows=(
    "0   90"
    "180 270"
    "360 450"
    "540 630"
    "720 810"
    "870 960"
)

echo "Temporal sweep over ${VIDEO} (${#windows[@]} windows, 90s each)"
echo "Config: ${MODEL} conf=${CONFIDENCE} imgsz=${IMGSZ} ${TRACKER}"
echo

for i in "${!windows[@]}"; do
    read -r START END <<< "${windows[$i]}"
    n=$((i + 1))
    JOB_NAME="cv_temporal${n}_t${START}-${END}"

    out=$(VIDEO="${VIDEO}" START="${START}" END="${END}" \
          MODEL="${MODEL}" CONFIDENCE="${CONFIDENCE}" \
          IMGSZ="${IMGSZ}" TRACKER="${TRACKER}" \
          sbatch --job-name="${JOB_NAME}" scripts/run_courtvision.sh)

    jobid=$(echo "${out}" | awk '{print $NF}')
    printf "  [%d] window %4ss..%4ss -> job %s\n" "${n}" "${START}" "${END}" "${jobid}"
done

echo
echo "Submitted. Check status with:  squeue -u \$USER"
echo "When done, compare with:       bash scripts/compare_temporal.sh"
