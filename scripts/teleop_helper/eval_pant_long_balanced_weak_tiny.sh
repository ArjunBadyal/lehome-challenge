#!/bin/bash
# Evaluate Pant-Long balanced-weak checkpoints on a targeted guard suite.
#
# Suite:
#   - weak targets: Seen_9, Seen_8, Seen_6
#   - regression guards: Seen_0, Seen_3, Unseen_1
#
# This is a mid-cost filter before spending time on a full category eval.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

RUN_DIR="${RUN_DIR:-outputs/train/act_pant_long_balanced_weak_tiny}"
DATASET="${DATASET:-Datasets/example/pant_long_merged}"
LOG_ROOT="${LOG_ROOT:-/tmp/eval_pant_long_balanced_weak_tiny}"
CKPTS="${CKPTS:-090000 090250 090500 090750 091000}"
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"

mkdir -p "$LOG_ROOT"
EVAL_LIST="${LOG_ROOT}/pant_long_guard_eval_list.txt"
cat > "$EVAL_LIST" <<'EOF'
Pant_Long_Seen_9
Pant_Long_Seen_8
Pant_Long_Seen_6
Pant_Long_Seen_0
Pant_Long_Seen_3
Pant_Long_Unseen_1
EOF

echo "=========================================================================="
echo " PANT-LONG BALANCED-WEAK CHECKPOINT EVAL"
echo "=========================================================================="
echo " Run dir     : $RUN_DIR"
echo " Dataset     : $DATASET"
echo " Checkpoints : $CKPTS"
echo " Episodes    : $NUM_EPISODES per garment"
echo " Eval list   : $EVAL_LIST"
cat "$EVAL_LIST"
echo "=========================================================================="

summary="${LOG_ROOT}/summary.tsv"
printf "checkpoint\tsuccess_rate_line\n" > "$summary"

for ckpt in $CKPTS; do
    ckpt_dir="${RUN_DIR}/checkpoints/${ckpt}/pretrained_model"
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "Skipping missing checkpoint: $ckpt_dir"
        continue
    fi

    log_dir="${LOG_ROOT}/${ckpt}"
    mkdir -p "$log_dir"
    log="${log_dir}/full.log"

    echo ""
    echo "=========================================================================="
    echo " Evaluating Pant-Long checkpoint ${ckpt}"
    echo "=========================================================================="
    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type lerobot \
        --policy_path "$ckpt_dir" \
        --dataset_root "$DATASET" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type pant_long \
        --eval_list_override "$EVAL_LIST" \
        --max_steps "$MAX_STEPS" --num_episodes "$NUM_EPISODES" \
        --enable_cameras --device cpu --headless \
        2>&1 | tee "$log"

    rate="$(grep -E "Success Rate:" "$log" | tail -1 || true)"
    printf "%s\t%s\n" "$ckpt" "$rate" | tee -a "$summary"
done

echo ""
echo "=========================================================================="
echo " Summary"
echo "=========================================================================="
cat "$summary"
