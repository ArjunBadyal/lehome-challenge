#!/bin/bash
# Merge teleop demos into the existing per-category dataset for retraining.
# Usage: bash scripts/teleop_helper/merge_teleop.sh <category>
#   category ∈ {top_short, top_long, pant_long, pant_short}
#
# Logic:
#   - Discover all numbered teleop datasets under Datasets/teleop_recovery/<category>_*/NNN/
#   - Use scripts.dataset merge to combine them with the existing
#     Datasets/example/<category>_merged/ dataset
#   - Output: Datasets/teleop_merged/<category>_merged/
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
if [[ -z "$CATEGORY" ]]; then
    echo "Usage: bash $0 <category>"
    echo "  category ∈ {top_short, top_long, pant_long, pant_short}"
    exit 1
fi
case "$CATEGORY" in
    top_short|top_long|pant_long|pant_short) ;;
    *) echo "Invalid category: $CATEGORY"; exit 1 ;;
esac

EXISTING="Datasets/example/${CATEGORY}_merged"
OUT="Datasets/teleop_merged/${CATEGORY}_merged"

# Find all teleop datasets for this category.
# dataset_record saves to the first numbered child under dataset_root, e.g.
# Datasets/teleop_recovery/top_short_seen_9/001/.
TELEOP_DIRS=()
declare -A SEEN_DIRS=()
for root in Datasets/teleop_recovery/${CATEGORY}_*; do
    [[ -d "$root" ]] || continue
    if [[ -d "$root/meta" ]] && find "$root/data" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
        SEEN_DIRS["${root%/}"]=1
    fi
    for d in "$root"/[0-9][0-9][0-9]; do
        if [[ -d "$d/meta" ]] && find "$d/data" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
            SEEN_DIRS["${d%/}"]=1
        fi
    done
done
for d in "${!SEEN_DIRS[@]}"; do
    if [[ -d "$d" ]]; then
        TELEOP_DIRS+=("$d")
    fi
done
IFS=$'\n' TELEOP_DIRS=($(sort <<<"${TELEOP_DIRS[*]}"))
unset IFS

if [[ ${#TELEOP_DIRS[@]} -eq 0 ]]; then
    echo "No teleop demos found under Datasets/teleop_recovery/${CATEGORY}_*"
    echo "Record some first using: bash scripts/teleop_helper/launch_${CATEGORY}_*.sh"
    exit 1
fi

echo "Merging into $OUT:"
echo "  baseline: $EXISTING"
for d in "${TELEOP_DIRS[@]}"; do
    n_files=$(find "$d/data" -name '*.parquet' 2>/dev/null | wc -l || echo 0)
    echo "  + $d  (~$n_files parquet file(s))"
done

mkdir -p "$(dirname "$OUT")"
SOURCES=("$EXISTING" "${TELEOP_DIRS[@]}")
SOURCE_LIST=$(python - "${SOURCES[@]}" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)

./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset merge \
    --source_roots "$SOURCE_LIST" \
    --output_root "$OUT" \
    --output_repo_id "${CATEGORY}_with_teleop"

echo ""
echo "Merged dataset ready at: $OUT"
echo "Next: bash scripts/teleop_helper/retrain.sh $CATEGORY"
