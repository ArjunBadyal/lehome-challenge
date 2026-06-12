#!/bin/bash
# Harvest successful ACT+residual trajectories on Seen-derived curriculum variants.
#
# Uses dataset_record.py rather than eval --save_datasets because the recorder
# finalizes LeRobot datasets reliably and supports release/clearance gating.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
RESIDUAL_CKPT="${2:-}"
VARIANT_LIST="${3:-}"

case "$CATEGORY" in
    top_short)
        DATASET_ROOT="${DATASET_ROOT:-Datasets/example/top_short_merged}"
        DEFAULT_LIST="outputs/garment_curriculum/eval_lists/top_short_variants.txt"
        EARLY_SCHEDULE="${EARLY_SCHEDULE:-160:2:3.0:,220:3:2.4:2}"
        AUTO_RESTART="${AUTO_RESTART:-540}"
        ;;
    pant_long)
        DATASET_ROOT="${DATASET_ROOT:-Datasets/example/pant_long_merged}"
        DEFAULT_LIST="outputs/garment_curriculum/eval_lists/pant_long_variants.txt"
        EARLY_SCHEDULE="${EARLY_SCHEDULE:-180:2:3.2:,280:3:2.6:2}"
        AUTO_RESTART="${AUTO_RESTART:-560}"
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long> <residual_checkpoint.pt> [variant_list]"
        exit 2
        ;;
esac

if [[ ! -f "$RESIDUAL_CKPT" ]]; then
    echo "Residual checkpoint not found: $RESIDUAL_CKPT"
    exit 1
fi

VARIANT_LIST="${VARIANT_LIST:-$DEFAULT_LIST}"
if [[ ! -f "$VARIANT_LIST" ]]; then
    echo "Variant list not found: $VARIANT_LIST"
    exit 1
fi

TARGET_PER_GARMENT="${TARGET_PER_GARMENT:-2}"
MAX_GARMENTS="${MAX_GARMENTS:-12}"
OUTPUT_BASE="${OUTPUT_BASE:-Datasets/garment_curriculum/residual_success/${CATEGORY}}"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"

mkdir -p "$OUTPUT_BASE"
mapfile -t GARMENTS < <(grep -v '^[[:space:]]*$' "$VARIANT_LIST" | head -n "$MAX_GARMENTS")

echo "=== Residual success harvest: $CATEGORY ==="
echo "Checkpoint: $RESIDUAL_CKPT"
echo "Output:     $OUTPUT_BASE"
echo "Target:     $TARGET_PER_GARMENT per garment, ${#GARMENTS[@]} garments"

for garment in "${GARMENTS[@]}"; do
    out_root="$OUTPUT_BASE/$garment"
    score_log="/tmp/curriculum_harvest_${CATEGORY}_${garment}.jsonl"
    run_log="/tmp/curriculum_harvest_${CATEGORY}_${garment}.log"
    mkdir -p "$out_root"
    rm -f "$score_log"

    echo ""
    echo "===== Harvest $garment ====="
    LEHOME_GLOBAL_KEYBOARD=0 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
        --task LeHome-BiSO101-Direct-Garment-v2 \
        --garment_name "$garment" \
        --garment_version Release \
        --teleop_device bi-keyboard \
        --num_envs 1 \
        --enable_record \
        --dataset_root "$out_root" \
        --num_episode "$TARGET_PER_GARMENT" \
        --disable_depth \
        --log_success \
        --auto_start_record \
        --auto_save_success \
        --auto_save_min_steps 140 \
        --auto_save_require_release \
        --auto_save_min_gripper_open 0.20 \
        --auto_save_min_gripper_cloth_distance "${MIN_CLOTH_CLEARANCE:-0.025}" \
        --early_restart_schedule "$EARLY_SCHEDULE" \
        --auto_restart_fail_steps "$AUTO_RESTART" \
        --score_probe_steps 120,160,220,280,340,420,520 \
        --score_probe_log "$score_log" \
        --safe_assist_hotkeys \
        --assist_policy_type residual \
        --assist_policy_path "$RESIDUAL_CKPT" \
        --assist_policy_device "$ASSIST_POLICY_DEVICE" \
        --device cpu \
        2>&1 | tee "$run_log"
done

echo "Harvest complete. Validate with:"
echo "  find $OUTPUT_BASE -mindepth 2 -maxdepth 2 -type d -name '[0-9][0-9][0-9]' -print0 | xargs -0 python scripts/garment_curriculum/validate_lerobot_dataset.py"
