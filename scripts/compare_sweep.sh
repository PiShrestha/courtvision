#!/bin/bash
# =============================================================================
# Print a side-by-side summary of every CourtVision report under outputs/
# whose filename matches the sweep pattern. Designed for the morning after
# you submit scripts/sweep_90s.sh.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

shopt -s nullglob
reports=(outputs/*_yolov8*_conf*_imgsz*.txt)

if [ ${#reports[@]} -eq 0 ]; then
    echo "No sweep reports found under outputs/."
    echo "Expected pattern: outputs/<video>_yolov8?_confNN_imgsz###_<tracker>_<jobid>.txt"
    exit 1
fi

printf "%-60s | %7s | %7s | %7s | %5s | %6s | %5s\n" \
    "config" "frames" "players" "ball" "poss" "shotAt" "ids"
printf "%.s-" $(seq 1 110); echo

for report in "${reports[@]}"; do
    base=$(basename "${report}" .txt)
    config=$(echo "${base}" | sed -E 's/_[0-9]+$//')   # strip trailing _<jobid>

    frames=$(grep -E "^Frames processed:" "${report}" | awk '{print $3}')
    players=$(grep -E "^player:" "${report}" | awk '{print $2}')
    ball=$(grep -E "^ball:" "${report}" | awk '{print $2}')
    poss=$(grep -E "^possession:" "${report}" | awk '{print $2}')
    shot=$(grep -E "^shot_attempt:" "${report}" | awk '{print $2}')
    # Count distinct player IDs as a proxy for ID-fragmentation severity.
    ids=$(grep -E "^Player [0-9]+:" "${report}" | wc -l)

    printf "%-60s | %7s | %7s | %7s | %5s | %6s | %5s\n" \
        "${config}" "${frames:-?}" "${players:-?}" "${ball:-0}" \
        "${poss:-0}" "${shot:-0}" "${ids:-0}"
done

echo
echo "Reading guide:"
echo "  players  - total player detections across all frames; lower (closer to frames*2) = less over-detection"
echo "  ball     - total ball detections; higher = better recall on the ball"
echo "  poss     - emitted possession events; should be in single digits if tracking is stable"
echo "  shotAt   - emitted shot_attempt events"
echo "  ids      - distinct player track IDs in the report; lower = less ID fragmentation"
echo "  Ground truth for a clean 1v1 90s clip: players~=frames*2, ball~=frames, poss~=4-12, ids~=2-4"
