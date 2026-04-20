#!/bin/bash
# =============================================================================
# Print a side-by-side summary of every CourtVision report under outputs/
# whose filename matches the temporal-sweep pattern (t<start>-<end>).
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

shopt -s nullglob
reports=(outputs/*_t[0-9]*-[0-9]*_yolov8*.txt)

if [ ${#reports[@]} -eq 0 ]; then
    echo "No temporal-sweep reports found under outputs/."
    echo "Expected pattern: outputs/<video>_t<start>-<end>_<model>_..._<jobid>.txt"
    exit 1
fi

# Sort by start time so the table reads chronologically.
mapfile -t reports < <(
    for r in "${reports[@]}"; do
        # Extract start seconds from "_t<start>-<end>_" segment for sort key.
        s=$(basename "${r}" | sed -E 's/.*_t([0-9]+)-[0-9]+_.*/\1/')
        printf "%010d\t%s\n" "${s}" "${r}"
    done | sort -n | cut -f2
)

printf "%-12s | %7s | %7s | %7s | %5s | %6s | %5s\n" \
    "window" "frames" "players" "ball" "poss" "shotAt" "ids"
printf "%.s-" $(seq 1 70); echo

for report in "${reports[@]}"; do
    base=$(basename "${report}" .txt)
    window=$(echo "${base}" | sed -E 's/.*_t([0-9]+-[0-9]+)_.*/\1/')

    frames=$(grep -E "^Frames processed:" "${report}" | awk '{print $3}')
    players=$(grep -E "^player:" "${report}" | awk '{print $2}')
    ball=$(grep -E "^ball:" "${report}" | awk '{print $2}')
    poss=$(grep -E "^possession:" "${report}" | awk '{print $2}')
    shot=$(grep -E "^shot_attempt:" "${report}" | awk '{print $2}')
    ids=$(grep -E "^Player [0-9]+:" "${report}" | wc -l)

    printf "%-12s | %7s | %7s | %7s | %5s | %6s | %5s\n" \
        "${window}" "${frames:-?}" "${players:-?}" "${ball:-0}" \
        "${poss:-0}" "${shot:-0}" "${ids:-0}"
done

echo
echo "What to look for:"
echo "  - If players/frame is roughly constant across windows, over-detection"
echo "    is uniform - it's a structural detection problem (spectators, blur)."
echo "  - If it spikes in some windows and not others, those windows have"
echo "    something unusual (camera cut to crowd, zoom in, scene change)."
echo "  - If ball recall is much lower in some windows, those parts are"
echo "    where the ball is hidden (in net, behind player) more often."
echo "  - If ID count is high in EVERY window, fragmentation is structural."
echo "  - If shot attempts cluster in some windows and are zero in others,"
echo "    those windows have actual play vs warmup/dead time."
