#!/bin/bash
# Collect successful Pant-Long Seen eval rollouts as LeRobot demos.
#
# This is self-imitation data: run the current policy on Seen garments only and
# let scripts.eval save successful episodes as a dataset. Do not include Unseen
# garments in this collection.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

POLICY_PATH="${POLICY_PATH:-outputs/train/act_pant_long_aug/checkpoints/090000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-Datasets/example/pant_long_merged}"
OUT_ROOT="${OUT_ROOT:-Datasets/eval_success/pant_long_aug90_seen_success}"
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
LOG="${LOG:-/tmp/collect_eval_success_pant_long_seen.log}"

if [[ ! -d "$POLICY_PATH" ]]; then
    echo "Missing policy: $POLICY_PATH"
    exit 1
fi

mkdir -p "$OUT_ROOT"
EVAL_LIST="/tmp/pant_long_seen_eval_success_list.txt"
cat > "$EVAL_LIST" <<'EOF'
Pant_Long_Seen_0
Pant_Long_Seen_1
Pant_Long_Seen_2
Pant_Long_Seen_3
Pant_Long_Seen_4
Pant_Long_Seen_5
Pant_Long_Seen_6
Pant_Long_Seen_7
Pant_Long_Seen_8
Pant_Long_Seen_9
EOF

echo "=========================================================================="
echo " PANT-LONG SEEN EVAL-SUCCESS DEMO COLLECTION"
echo "=========================================================================="
echo " Policy      : $POLICY_PATH"
echo " Dataset     : $DATASET_ROOT"
echo " Output root : $OUT_ROOT"
echo " Episodes    : $NUM_EPISODES per Seen garment"
echo " Max steps   : $MAX_STEPS"
echo " Device      : env cpu, policy $POLICY_DEVICE"
echo " Eval list   : $EVAL_LIST"
cat "$EVAL_LIST"
echo "=========================================================================="

POLICY_DEVICE="$POLICY_DEVICE" ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
    --policy_type lerobot \
    --policy_path "$POLICY_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --task_description "Fold a garment with bimanual robot arms" \
    --garment_type pant_long \
    --eval_list_override "$EVAL_LIST" \
    --max_steps "$MAX_STEPS" --num_episodes "$NUM_EPISODES" \
    --save_datasets \
    --eval_dataset_path "$OUT_ROOT" \
    --enable_cameras --device cpu --headless \
    2>&1 | tee "$LOG"

echo ""
echo "Eval-success collection complete. Datasets:"
find "$OUT_ROOT" -mindepth 1 -maxdepth 2 -type f -path '*/meta/info.json' -print | sort
