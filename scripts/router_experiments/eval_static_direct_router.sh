#!/bin/bash
# Evaluate a fixed per-category router:
#   classifier -> static direct-best specialist per predicted category
#
# This disables the kNN portfolio path entirely. It is useful when the
# per-garment portfolio is noisier than a single strong checkpoint.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CAT="${1:-}"
if [[ -z "$CAT" ]]; then
    echo "Usage: bash $0 <top_short|top_long|pant_long|pant_short>"
    exit 1
fi

case "$CAT" in
    top_short) DATASET="Datasets/example/top_short_merged" ;;
    top_long) DATASET="Datasets/example/top_long_merged" ;;
    pant_long) DATASET="Datasets/example/pant_long_merged" ;;
    pant_short) DATASET="Datasets/example/pant_short_merged" ;;
    *) echo "Unknown category: $CAT"; exit 1 ;;
esac

LOG="/tmp/static_direct_router_${CAT}.log"
echo "=== Static-direct router eval: ${CAT} ==="
echo "Map: top_short=aug_55k top_long=aug_90k pant_long=aug_90k pant_short=aug_60k"
echo "Log: $LOG"

PORTFOLIO_HYBRID_CATEGORIES="__none__" \
PORTFOLIO_USE_MULTICAM=1 \
PORTFOLIO_DEFAULT_TOP_SHORT=aug_55k \
PORTFOLIO_DEFAULT_TOP_LONG=aug_90k \
PORTFOLIO_DEFAULT_PANT_LONG=aug_90k \
PORTFOLIO_DEFAULT_PANT_SHORT=aug_60k \
ROUTER_GT_CATEGORY="$CAT" \
    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type portfolio_router \
        --policy_path outputs/classifier/garment_classifier.pt \
        --dataset_root "$DATASET" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type "$CAT" \
        --max_steps 600 \
        --num_episodes 5 \
        --enable_cameras \
        --device cpu \
        --headless \
        2>&1 | tee "$LOG"

echo ""
echo "=== Summary: ${CAT} ==="
grep -aE "Success Rate:|  (Top|Pant)_|fallback route|Classified:" "$LOG" | tail -120
