#!/bin/bash
# =============================================================================
# Submit a parameter sweep over the first 90 seconds of 1v1-mk.mov.
#
# Six independent jobs targeting the three failure modes from the baseline run:
#   1. player over-detection
#   2. sparse ball detection
#   3. ID fragmentation
#
# Each job is submitted to the GPU partition under cs6770_sp26 and writes
# its report to outputs/<videoname>_<config>_<jobid>.txt so all six are
# distinguishable when you compare them tomorrow.
#
# Usage:
#   bash scripts/sweep_90s.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VIDEO="1v1-mk.mov"
START=0
END=90

# Each row: MODEL CONFIDENCE IMGSZ TRACKER          (whitespace-separated)
configs=(
    "yolov8m.pt 0.35 640  bytetrack.yaml"   # 1: baseline
    "yolov8m.pt 0.45 640  bytetrack.yaml"   # 2: tighter conf vs over-detection
    "yolov8m.pt 0.25 1280 bytetrack.yaml"   # 3: low conf + 2x input for ball recall
    "yolov8l.pt 0.35 640  bytetrack.yaml"   # 4: bigger model, default conf/imgsz
    "yolov8x.pt 0.35 1280 bytetrack.yaml"   # 5: kitchen sink
    "yolov8m.pt 0.35 640  botsort.yaml"     # 6: alternative tracker
)

echo "Submitting ${#configs[@]} jobs to cs6770_sp26 / gpu partition..."
echo "Video: ${VIDEO}   window: ${START}s..${END}s"
echo

for i in "${!configs[@]}"; do
    read -r MODEL CONFIDENCE IMGSZ TRACKER <<< "${configs[$i]}"
    n=$((i + 1))
    JOB_NAME="cv_sweep${n}_$(basename "${MODEL}" .pt)_c${CONFIDENCE}_i${IMGSZ}"

    out=$(VIDEO="${VIDEO}" START="${START}" END="${END}" \
          MODEL="${MODEL}" CONFIDENCE="${CONFIDENCE}" \
          IMGSZ="${IMGSZ}" TRACKER="${TRACKER}" \
          sbatch --job-name="${JOB_NAME}" scripts/run_courtvision.sh)

    jobid=$(echo "${out}" | awk '{print $NF}')
    printf "  [%d] %-40s -> job %s\n" "${n}" "${MODEL} conf=${CONFIDENCE} imgsz=${IMGSZ} ${TRACKER}" "${jobid}"
done

echo
echo "Submitted. Check status with:  squeue -u \$USER"
echo "Cancel all sweep jobs with:    squeue -u \$USER --name=cv_sweep1,cv_sweep2,cv_sweep3,cv_sweep4,cv_sweep5,cv_sweep6 -h -o %i | xargs -r scancel"
echo "When done, compare with:       bash scripts/compare_sweep.sh"
