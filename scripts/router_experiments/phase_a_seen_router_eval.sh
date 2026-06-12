#!/bin/bash
# Fast sanity eval for the seen-garment router on three garments in one category.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
THRESH="${2:-0.95}"
if [[ -z "$CATEGORY" ]]; then
    echo "Usage: bash $0 <top_short|top_long|pant_long|pant_short> [conf_threshold]"
    exit 1
fi

case "$CATEGORY" in
    top_short)
        GARMENTS=(Top_Short_Seen_9 Top_Short_Seen_0 Top_Short_Unseen_0)
        DATASET="Datasets/example/top_short_merged"
        ;;
    top_long)
        GARMENTS=(Top_Long_Seen_7 Top_Long_Seen_0 Top_Long_Unseen_0)
        DATASET="Datasets/example/top_long_merged"
        ;;
    pant_long)
        GARMENTS=(Pant_Long_Seen_3 Pant_Long_Seen_0 Pant_Long_Unseen_1)
        DATASET="Datasets/example/pant_long_merged"
        ;;
    pant_short)
        GARMENTS=(Pant_Short_Seen_0 Pant_Short_Seen_6 Pant_Short_Unseen_0)
        DATASET="Datasets/example/pant_short_merged"
        ;;
    *) echo "Unknown category: $CATEGORY"; exit 1 ;;
esac

LOG_DIR="/tmp/phase_a_seen_router_${CATEGORY}_t${THRESH}"
mkdir -p "$LOG_DIR"
EVAL_LIST="$LOG_DIR/eval_list.txt"
printf '%s\n' "${GARMENTS[@]}" > "$EVAL_LIST"

echo "=== Seen-router Phase A: $CATEGORY threshold=$THRESH ==="
cat "$EVAL_LIST"

SEEN_ROUTER_CONF_THRESHOLD="$THRESH" \
SEEN_ROUTER_MAP_PATH=outputs/seen_garment_router/conservative_best_checkpoints.json \
PORTFOLIO_DEFAULT_TOP_SHORT=aug_55k \
PORTFOLIO_DEFAULT_TOP_LONG=aug_90k \
PORTFOLIO_DEFAULT_PANT_LONG=aug_90k \
PORTFOLIO_DEFAULT_PANT_SHORT=aug_60k \
    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type seen_garment_router \
        --policy_path outputs/seen_garment_router/seen_garment_classifier.pt \
        --dataset_root "$DATASET" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type "$CATEGORY" \
        --eval_list_override "$EVAL_LIST" \
        --max_steps 600 --num_episodes 5 \
        --enable_cameras --device cpu --headless \
        2>&1 | tee "$LOG_DIR/full.log"

echo "=== Summary ==="
grep -E "Success Rate:|  (Top|Pant)_|SeenRouter]" "$LOG_DIR/full.log" | tail -80
