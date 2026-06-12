#!/bin/bash
# Collect small successful ACT-generated demo sets for weak Pant-Long Seen garments.
#
# Purpose:
#   Base aug90 is the best general Pant-Long checkpoint, but its weak cells are
#   Pant_Long_Seen_9, Seen_8, and Seen_6. Harvesting only successful ACT rollouts
#   on these Seen garments gives us targeted coverage without training on Unseen.
#
# Usage:
#   NUM_SUCCESS_PER_GARMENT=5 bash scripts/teleop_helper/harvest_pant_long_weak_seen.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
NUM_SUCCESS_PER_GARMENT="${NUM_SUCCESS_PER_GARMENT:-5}"
OUTPUT_BASE="${OUTPUT_BASE:-Datasets/teleop_recovery/pant_long_weak_seen_retracted_act_harvest}"
POLICY_PATH="${POLICY_PATH:-outputs/train/act_pant_long_aug/checkpoints/090000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-Datasets/example/pant_long_merged}"
AUTO_SAVE_MIN_STEPS="${AUTO_SAVE_MIN_STEPS:-180}"
AUTO_RESTART_FAIL_STEPS="${AUTO_RESTART_FAIL_STEPS:-520}"
AUTO_SAVE_REQUIRE_RELEASE="${AUTO_SAVE_REQUIRE_RELEASE:-0}"
AUTO_SAVE_MIN_GRIPPER_OPEN="${AUTO_SAVE_MIN_GRIPPER_OPEN:-0.20}"
AUTO_SAVE_MIN_GRIPPER_CLOTH_DISTANCE="${AUTO_SAVE_MIN_GRIPPER_CLOTH_DISTANCE:-0.0}"
EARLY_RESTART_SCHEDULE="${EARLY_RESTART_SCHEDULE:-120::3.0:,160::2.6:,220::1.5:1,280:::2}"
GARMENTS="${GARMENTS:-Pant_Long_Seen_9 Pant_Long_Seen_8 Pant_Long_Seen_6}"
DISABLE_DEPTH="${DISABLE_DEPTH:-1}"

mkdir -p "$OUTPUT_BASE"

harvest_one() {
    local garment="$1"
    local out_root="${OUTPUT_BASE}/${garment}"
    local score_log="/tmp/harvest_weak_${garment}_scores.jsonl"
    local run_log="/tmp/harvest_weak_${garment}.log"

    mkdir -p "$out_root"
    rm -f "$score_log"
    local release_args=()
    local depth_args=()
    if [[ "$AUTO_SAVE_REQUIRE_RELEASE" == "1" ]]; then
        release_args+=(--auto_save_require_release)
        release_args+=(--auto_save_min_gripper_open "$AUTO_SAVE_MIN_GRIPPER_OPEN")
        release_args+=(--auto_save_min_gripper_cloth_distance "$AUTO_SAVE_MIN_GRIPPER_CLOTH_DISTANCE")
    fi
    if [[ "$DISABLE_DEPTH" == "1" ]]; then
        depth_args+=(--disable_depth)
    fi

    echo ""
    echo "=========================================================================="
    echo " PANT-LONG WEAK-SEEN ACT HARVEST: ${garment}"
    echo "=========================================================================="
    echo " Save dir   : ${out_root}"
    echo " Target     : ${NUM_SUCCESS_PER_GARMENT} successful demos"
    echo " Policy     : ${POLICY_PATH}"
    echo " Dataset    : ${DATASET_ROOT}"
    echo " Save gate  : success after step ${AUTO_SAVE_MIN_STEPS}"
    echo " Release    : require=${AUTO_SAVE_REQUIRE_RELEASE}, gripper>=${AUTO_SAVE_MIN_GRIPPER_OPEN}, clearance>=${AUTO_SAVE_MIN_GRIPPER_CLOTH_DISTANCE}m"
    echo " Depth      : disabled=${DISABLE_DEPTH}"
    echo " Early gate : ${EARLY_RESTART_SCHEDULE}"
    echo " Log        : ${run_log}"
    echo " Score log  : ${score_log}"
    echo "=========================================================================="

    LEHOME_GLOBAL_KEYBOARD=0 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
        --task "LeHome-BiSO101-Direct-Garment-v2" \
        --garment_name "$garment" \
        --garment_version Release \
        --teleop_device bi-keyboard \
        --num_envs 1 \
        --enable_record \
        --dataset_root "$out_root" \
        --num_episode "$NUM_SUCCESS_PER_GARMENT" \
        --log_success \
        --auto_start_record \
        --auto_save_success \
        --auto_save_min_steps "$AUTO_SAVE_MIN_STEPS" \
        "${release_args[@]}" \
        --score_probe_steps 120,160,220,280,340,390,500,560,620 \
        --score_probe_log "$score_log" \
        --early_restart_step 280 \
        --early_restart_close_ratio 3.2 \
        --early_restart_min_passed 3 \
        --early_restart_schedule "$EARLY_RESTART_SCHEDULE" \
        --auto_restart_fail_steps "$AUTO_RESTART_FAIL_STEPS" \
        --safe_assist_hotkeys \
        --enable_click_ik \
        --assist_policy_type lerobot \
        --assist_policy_path "$POLICY_PATH" \
        --assist_dataset_root "$DATASET_ROOT" \
        --assist_policy_device "$ASSIST_POLICY_DEVICE" \
        "${depth_args[@]}" \
        --device cpu \
        2>&1 | tee "$run_log"
}

echo "=========================================================================="
echo " PANT-LONG WEAK-SEEN ACT HARVEST QUEUE"
echo "=========================================================================="
echo " Output base : ${OUTPUT_BASE}"
echo " Target each : ${NUM_SUCCESS_PER_GARMENT}"
echo " Garments    : ${GARMENTS}"
echo " Device      : ${ASSIST_POLICY_DEVICE}"
echo " Save gate   : success after step ${AUTO_SAVE_MIN_STEPS}; fail at ${AUTO_RESTART_FAIL_STEPS}"
echo " Release     : require=${AUTO_SAVE_REQUIRE_RELEASE}, gripper>=${AUTO_SAVE_MIN_GRIPPER_OPEN}, clearance>=${AUTO_SAVE_MIN_GRIPPER_CLOTH_DISTANCE}m"
echo " Depth       : disabled=${DISABLE_DEPTH}"
echo " Early gate  : ${EARLY_RESTART_SCHEDULE}"
echo " Note        : Seen garments only; no Unseen demo collection."
echo "=========================================================================="

for garment in ${GARMENTS}; do
    harvest_one "$garment"
done

echo ""
echo "Pant-Long weak-seen harvest complete."
