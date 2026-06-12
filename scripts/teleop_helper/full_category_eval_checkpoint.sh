#!/bin/bash
# Evaluate one LeRobot checkpoint on the full category eval list.
# Usage:
#   bash scripts/teleop_helper/full_category_eval_checkpoint.sh \
#     <top_short|top_long|pant_long|pant_short> <checkpoint_dir> <tag>
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
CKPT_DIR="${2:-}"
TAG="${3:-manual}"

case "$CATEGORY" in
    top_short|top_long|pant_long|pant_short) ;;
    *)
        echo "Usage: bash $0 <top_short|top_long|pant_long|pant_short> <checkpoint_dir> <tag>"
        exit 1
        ;;
esac

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint not found: $CKPT_DIR"
    exit 1
fi

DATASET="Datasets/example/${CATEGORY}_merged"
LOG="/tmp/full_eval_${CATEGORY}_${TAG}.log"

echo "=== Full category eval: ${CATEGORY} / ${TAG} ==="
echo "Checkpoint: $CKPT_DIR"
echo "Dataset:    $DATASET"
echo "Log:        $LOG"

./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
    --policy_type lerobot \
    --policy_path "$CKPT_DIR" \
    --dataset_root "$DATASET" \
    --task_description "Fold a garment with bimanual robot arms" \
    --garment_type "$CATEGORY" \
    --max_steps 600 --num_episodes 5 \
    --enable_cameras --device cpu --headless \
    2>&1 | tee "$LOG"

echo ""
echo "=== Summary (${CATEGORY} / ${TAG}) ==="
grep -E "Success Rate:|  (Top|Pant)_" "$LOG" | grep -v "Saved video\|Switching" | tail -30
