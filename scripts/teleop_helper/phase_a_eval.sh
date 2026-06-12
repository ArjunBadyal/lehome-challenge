#!/bin/bash
# Phase A: fast 3-garment sanity check for fine-tuned checkpoints.
# For each (category, checkpoint), run 5-ep eval on:
#   - the training target (Seen_9 / Seen_3)
#   - one holdout Seen (Seen_0 / Seen_0)
#   - one Unseen (Unseen_0 / Unseen_1)
# Total: 3 garments × 5 ep × 1 ckpt × 2 categories = 30 episodes ≈ 25-30 min CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
CKPT="${2:-}"
if [[ -z "$CATEGORY" || -z "$CKPT" ]]; then
    echo "Usage: bash $0 <top_short|pant_long> <ckpt_step>"
    exit 1
fi

case "$CATEGORY" in
    top_short)
        GARMENTS=(Top_Short_Seen_9 Top_Short_Seen_0 Top_Short_Unseen_0)
        DATASET="Datasets/example/top_short_merged"
        DEFAULT_RUN_DIR="outputs/train/act_top_short_mixed10p_seen9"
        ;;
    pant_long)
        GARMENTS=(Pant_Long_Seen_3 Pant_Long_Seen_0 Pant_Long_Unseen_1)
        DATASET="Datasets/example/pant_long_merged"
        DEFAULT_RUN_DIR="outputs/train/act_pant_long_mixed10p_seen3"
        ;;
    *) echo "category must be top_short or pant_long"; exit 1 ;;
esac

# Allow override via $RUN_DIR env var, e.g. RUN_DIR=outputs/train/...other...
RUN_DIR="${RUN_DIR:-$DEFAULT_RUN_DIR}"
CKPT_DIR="${RUN_DIR}/checkpoints/${CKPT}/pretrained_model"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "Checkpoint not found: $CKPT_DIR"
    exit 1
fi

RUN_TAG="$(basename "$RUN_DIR")"
LOG_DIR="/tmp/phase_a_eval_${RUN_TAG}_${CKPT}"
mkdir -p "$LOG_DIR"

# Build eval-list override file with the 3 target garments
EVAL_LIST="${LOG_DIR}/eval_list.txt"
printf '%s\n' "${GARMENTS[@]}" > "$EVAL_LIST"

echo "=== Phase A: ${CATEGORY} ckpt ${CKPT} ==="
echo "Checkpoint: $CKPT_DIR"
echo "Eval list:  $EVAL_LIST"
cat "$EVAL_LIST"

./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
    --policy_type lerobot \
    --policy_path "$CKPT_DIR" \
    --dataset_root "$DATASET" \
    --task_description "Fold a garment with bimanual robot arms" \
    --garment_type "$CATEGORY" \
    --eval_list_override "$EVAL_LIST" \
    --max_steps 600 --num_episodes 5 \
    --enable_cameras --device cpu --headless \
    2>&1 | tee "${LOG_DIR}/full.log"

echo ""
echo "=== Per-garment results ==="
grep -E "Success Rate|Garment:|Final result" "${LOG_DIR}/full.log" | tail -30
