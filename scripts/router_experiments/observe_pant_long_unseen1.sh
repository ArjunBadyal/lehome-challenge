#!/bin/bash
# Launch a visible one-garment eval for Pant_Long_Unseen_1 using the packaged
# submission policy. Intended for manual observation/debugging after full evals.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

LOG="/tmp/observe_pant_long_unseen1.log"
EVAL_LIST="/tmp/observe_pant_long_unseen1.txt"
NUM_EPISODES="${NUM_EPISODES:-3}"

printf '%s\n' "Pant_Long_Unseen_1" > "$EVAL_LIST"

echo "=== Observe Pant_Long_Unseen_1 ==="
echo "Policy: submission/policy.py"
echo "Episodes: $NUM_EPISODES"
echo "Log: $LOG"

LEHOME_POLICY_DEVICE=cpu \
POLICIES_ROOT="$PWD/submission/policies" \
    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type submission_bundle \
        --policy_path "$PWD/submission/policies/classifier/garment_classifier.pt" \
        --dataset_root "Datasets/example/pant_long_merged" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type pant_long \
        --eval_list_override "$EVAL_LIST" \
        --max_steps 600 \
        --num_episodes "$NUM_EPISODES" \
        --enable_cameras \
        --device cpu \
        2>&1 | tee "$LOG"
