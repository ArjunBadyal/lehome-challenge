#!/bin/bash
# Harvest successful ACT-only self-imitation demos on Seen-derived variants.
#
# This uses dataset_record.py because it finalizes LeRobot datasets reliably.
# It never uses Unseen garments and never reads garment labels at inference; the
# garment name here is only a training-time data-collection target.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
VARIANT_LIST="${2:-}"

case "$CATEGORY" in
    top_short)
        DEFAULT_LIST="outputs/garment_curriculum/eval_lists/top_short_easy_variants.txt"
        POLICY_PATH="${POLICY_PATH:-outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model}"
        DATASET_ROOT="${DATASET_ROOT:-Datasets/example/top_short_merged}"
        OUTPUT_BASE="${OUTPUT_BASE:-Datasets/garment_curriculum/act_variant_success/top_short_easy}"
        # Top-Short often closes the sleeve/side condition before the upward
        # fold. Do not gate at step 220; let late successful folds complete.
        EARLY_RESTART_SCHEDULE="${EARLY_RESTART_SCHEDULE:-160:2:3.0:}"
        AUTO_SAVE_MIN_STEPS="${AUTO_SAVE_MIN_STEPS:-140}"
        AUTO_RESTART_FAIL_STEPS="${AUTO_RESTART_FAIL_STEPS:-540}"
        SCORE_PROBE_STEPS="${SCORE_PROBE_STEPS:-120,160,220,280,340,420,520}"
        ;;
    pant_long)
        DEFAULT_LIST="outputs/garment_curriculum/eval_lists/pant_long_easy_variants.txt"
        POLICY_PATH="${POLICY_PATH:-outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model}"
        DATASET_ROOT="${DATASET_ROOT:-Datasets/example/pant_long_merged}"
        OUTPUT_BASE="${OUTPUT_BASE:-Datasets/garment_curriculum/act_variant_success/pant_long_easy}"
        EARLY_RESTART_SCHEDULE="${EARLY_RESTART_SCHEDULE:-180:2:3.2:,280:3:2.6:2}"
        AUTO_SAVE_MIN_STEPS="${AUTO_SAVE_MIN_STEPS:-180}"
        AUTO_RESTART_FAIL_STEPS="${AUTO_RESTART_FAIL_STEPS:-560}"
        SCORE_PROBE_STEPS="${SCORE_PROBE_STEPS:-120,160,220,280,340,420,520,560}"
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long> [variant_list]"
        exit 2
        ;;
esac

VARIANT_LIST="${VARIANT_LIST:-$DEFAULT_LIST}"
if [[ ! -f "$VARIANT_LIST" ]]; then
    echo "Variant list not found: $VARIANT_LIST"
    exit 1
fi
if [[ ! -d "$POLICY_PATH" ]]; then
    echo "Policy path not found: $POLICY_PATH"
    exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root not found: $DATASET_ROOT"
    exit 1
fi

TARGET_PER_VARIANT="${TARGET_PER_VARIANT:-1}"
MAX_VARIANTS="${MAX_VARIANTS:-32}"
MAX_ATTEMPTS_PER_VARIANT="${MAX_ATTEMPTS_PER_VARIANT:-6}"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
MIN_CLOTH_CLEARANCE="${MIN_CLOTH_CLEARANCE:-0.025}"
AUTO_SAVE_MIN_GRIPPER_OPEN="${AUTO_SAVE_MIN_GRIPPER_OPEN:-0.20}"
AUTO_SAVE_SETTLE_STEPS="${AUTO_SAVE_SETTLE_STEPS:-90}"
SAVE_NEAR_MISS="${SAVE_NEAR_MISS:-0}"
NEAR_MISS_MIN_PASSED="${NEAR_MISS_MIN_PASSED:-4}"
NEAR_MISS_MAX_WORST_CLOSE_RATIO="${NEAR_MISS_MAX_WORST_CLOSE_RATIO:-1.10}"
AUTO_GRIP_HOLD_START_STEP="${AUTO_GRIP_HOLD_START_STEP:--1}"
AUTO_GRIP_HOLD_END_STEP="${AUTO_GRIP_HOLD_END_STEP:--1}"
AUTO_GRIP_RELEASE_UNTIL_STEP="${AUTO_GRIP_RELEASE_UNTIL_STEP:--1}"
DISABLE_DEPTH="${DISABLE_DEPTH:-1}"

mkdir -p "$OUTPUT_BASE"
mapfile -t VARIANTS < <(grep -v '^[[:space:]]*$' "$VARIANT_LIST" | head -n "$MAX_VARIANTS")

echo "=========================================================================="
echo " ACT VARIANT SUCCESS HARVEST: $CATEGORY"
echo "=========================================================================="
echo " Variants   : $VARIANT_LIST (${#VARIANTS[@]} selected)"
echo " Output     : $OUTPUT_BASE"
echo " Target     : $TARGET_PER_VARIANT success(es) per variant"
echo " Attempts   : max $MAX_ATTEMPTS_PER_VARIANT failed attempt(s) before skip"
echo " Policy     : $POLICY_PATH"
echo " Dataset    : $DATASET_ROOT"
echo " Release    : require gripper open >= $AUTO_SAVE_MIN_GRIPPER_OPEN, clearance >= ${MIN_CLOTH_CLEARANCE}m"
echo " Settle     : require success+release to hold for $AUTO_SAVE_SETTLE_STEPS extra step(s)"
echo " NearMiss   : $SAVE_NEAR_MISS (min_passed=$NEAR_MISS_MIN_PASSED, worst_close<=$NEAR_MISS_MAX_WORST_CLOSE_RATIO)"
echo " GripTail   : hold_start=$AUTO_GRIP_HOLD_START_STEP hold_end=$AUTO_GRIP_HOLD_END_STEP release_until=$AUTO_GRIP_RELEASE_UNTIL_STEP"
echo " Early gate : $EARLY_RESTART_SCHEDULE"
echo "=========================================================================="

for garment in "${VARIANTS[@]}"; do
    out_root="$OUTPUT_BASE/$garment"
    score_log="/tmp/act_variant_${CATEGORY}_${garment}_scores.jsonl"
    run_log="/tmp/act_variant_${CATEGORY}_${garment}.log"
    depth_args=()
    if [[ "$DISABLE_DEPTH" == "1" ]]; then
        depth_args+=(--disable_depth)
    fi
    near_miss_args=()
    if [[ "$SAVE_NEAR_MISS" == "1" ]]; then
        near_miss_args+=(
            --auto_save_near_miss
            --auto_save_near_miss_require_release
            --auto_save_near_miss_min_passed "$NEAR_MISS_MIN_PASSED"
            --auto_save_near_miss_max_worst_close_ratio "$NEAR_MISS_MAX_WORST_CLOSE_RATIO"
        )
    fi
    auto_grip_args=()
    if [[ "$AUTO_GRIP_HOLD_START_STEP" != "-1" || "$AUTO_GRIP_HOLD_END_STEP" != "-1" || "$AUTO_GRIP_RELEASE_UNTIL_STEP" != "-1" ]]; then
        auto_grip_args+=(
            --auto_grip_hold_start_step "$AUTO_GRIP_HOLD_START_STEP"
            --auto_grip_hold_end_step "$AUTO_GRIP_HOLD_END_STEP"
            --auto_grip_release_until_step "$AUTO_GRIP_RELEASE_UNTIL_STEP"
        )
    fi

    mkdir -p "$out_root"
    rm -f "$score_log"

    valid_count=0
    while IFS= read -r ds; do
        if python scripts/garment_curriculum/validate_lerobot_dataset.py "$ds" >/dev/null 2>&1; then
            valid_count=$((valid_count + 1))
        fi
    done < <(find "$out_root" -mindepth 1 -maxdepth 1 -type d -name '[0-9][0-9][0-9]' 2>/dev/null | sort)
    if (( valid_count >= TARGET_PER_VARIANT )); then
        echo ""
        echo "===== Skip $garment: already has $valid_count valid demo(s) ====="
        continue
    fi
    needed=$((TARGET_PER_VARIANT - valid_count))

    echo ""
    echo "===== Harvest $garment ($needed needed, $valid_count valid existing) ====="
    LEHOME_GLOBAL_KEYBOARD=0 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
        --task "LeHome-BiSO101-Direct-Garment-v2" \
        --garment_name "$garment" \
        --garment_version Release \
        --teleop_device bi-keyboard \
        --num_envs 1 \
        --enable_record \
        --dataset_root "$out_root" \
        --num_episode "$needed" \
        --log_success \
        --auto_start_record \
        --auto_save_success \
        --auto_save_min_steps "$AUTO_SAVE_MIN_STEPS" \
        --auto_save_success_settle_steps "$AUTO_SAVE_SETTLE_STEPS" \
        --auto_save_require_release \
        --auto_save_min_gripper_open "$AUTO_SAVE_MIN_GRIPPER_OPEN" \
        --auto_save_min_gripper_cloth_distance "$MIN_CLOTH_CLEARANCE" \
        --score_probe_steps "$SCORE_PROBE_STEPS" \
        --score_probe_log "$score_log" \
        --early_restart_schedule "$EARLY_RESTART_SCHEDULE" \
        --auto_restart_fail_steps "$AUTO_RESTART_FAIL_STEPS" \
        --max_attempts_per_episode "$MAX_ATTEMPTS_PER_VARIANT" \
        "${near_miss_args[@]}" \
        "${auto_grip_args[@]}" \
        --safe_assist_hotkeys \
        --assist_policy_type lerobot \
        --assist_policy_path "$POLICY_PATH" \
        --assist_dataset_root "$DATASET_ROOT" \
        --assist_policy_device "$ASSIST_POLICY_DEVICE" \
        "${depth_args[@]}" \
        --device cpu \
        2>&1 | tee "$run_log"
done

echo ""
echo "Harvest complete. Valid demos:"
while IFS= read -r ds; do
    if python scripts/garment_curriculum/validate_lerobot_dataset.py "$ds" >/dev/null 2>&1; then
        echo "$ds"
    fi
done < <(find "$OUTPUT_BASE" -mindepth 2 -maxdepth 2 -type d -name '[0-9][0-9][0-9]' 2>/dev/null | sort)
