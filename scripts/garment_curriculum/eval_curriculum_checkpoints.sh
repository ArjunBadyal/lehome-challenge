#!/bin/bash
# Evaluate curriculum ACT checkpoints on guard suites before any full eval.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
case "$CATEGORY" in
    top_short)
        RUN_DIR="${RUN_DIR:-outputs/train/act_top_short_curriculum}"
        DATASET="Datasets/example/top_short_merged"
        GARMENTS=(Top_Short_Seen_0 Top_Short_Seen_5 Top_Short_Seen_9 Top_Short_Unseen_0 Top_Short_Unseen_1)
        ;;
    pant_long)
        RUN_DIR="${RUN_DIR:-outputs/train/act_pant_long_curriculum}"
        DATASET="Datasets/example/pant_long_merged"
        GARMENTS=(Pant_Long_Seen_0 Pant_Long_Seen_3 Pant_Long_Seen_8 Pant_Long_Seen_9 Pant_Long_Unseen_0 Pant_Long_Unseen_1)
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long>"
        exit 2
        ;;
esac

LIST="/tmp/curriculum_guard_${CATEGORY}.txt"
printf '%s\n' "${GARMENTS[@]}" > "$LIST"
SUMMARY="/tmp/curriculum_guard_${CATEGORY}_summary.tsv"
printf "checkpoint\tsuccess_rate\n" > "$SUMMARY"
NUM_EPISODES="${NUM_EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-600}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-*}"

for ckpt in $(find "$RUN_DIR/checkpoints" -maxdepth 1 -mindepth 1 -type d -name "$CHECKPOINT_GLOB" -printf '%f\n' | sort); do
    CKPT_DIR="$RUN_DIR/checkpoints/$ckpt/pretrained_model"
    [[ -f "$CKPT_DIR/model.safetensors" ]] || continue
    LOG="/tmp/curriculum_guard_${CATEGORY}_${ckpt}.log"
    echo "=== Guard eval $CATEGORY $ckpt ==="
    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type lerobot \
        --policy_path "$CKPT_DIR" \
        --dataset_root "$DATASET" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type "$CATEGORY" \
        --eval_list_override "$LIST" \
        --max_steps "$MAX_STEPS" \
        --num_episodes "$NUM_EPISODES" \
        --enable_cameras \
        --device cpu \
        --headless \
        2>&1 | tee "$LOG"
    rate=$(grep -a "Success Rate:" "$LOG" | tail -1 | awk '{print $NF}')
    printf "%s\t%s\n" "$ckpt" "${rate:-NA}" | tee -a "$SUMMARY"
done

echo "Summary: $SUMMARY"
