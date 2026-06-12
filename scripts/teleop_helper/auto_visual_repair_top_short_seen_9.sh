#!/bin/bash
# Unattended visual-repair demo collector for Top_Short_Seen_9.
# ACT runs the prefix; at AUTO_REPAIR_STEP the E-key visual repair macro
# grasps top/shoulder landmarks after ACT's sleeve fold, applies one correction,
# and the recorder
# saves only if the official success checker passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/teleop_recovery/top_short_seen_9_auto_visual_v7"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
AUTO_REPAIR_STEP="${AUTO_REPAIR_STEP:-320}"
AUTO_SAVE_MIN_STEPS="${AUTO_SAVE_MIN_STEPS:-$((AUTO_REPAIR_STEP + 180))}"
NUM_SUCCESS="${NUM_SUCCESS:-10}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " AUTO VISUAL REPAIR: Top_Short_Seen_9"
echo "=========================================================================="
echo " Save dir    : $OUTPUT_ROOT"
echo " Target      : $NUM_SUCCESS successful repair demos"
echo " Trigger     : ACT prefix for $AUTO_REPAIR_STEP steps, then E-macro"
echo " Save rule   : auto-save only when env success checker passes"
echo " Abort       : Ctrl+C"
echo "=========================================================================="
echo ""

LEHOME_GLOBAL_KEYBOARD=0 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
    --task "LeHome-BiSO101-Direct-Garment-v2" \
    --garment_name Top_Short_Seen_9 \
    --garment_version Release \
    --teleop_device bi-keyboard \
    --num_envs 1 \
    --enable_record \
    --dataset_root "$OUTPUT_ROOT" \
    --num_episode "$NUM_SUCCESS" \
    --log_success \
    --assist_policy_type lerobot \
    --assist_policy_path outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model \
    --assist_dataset_root Datasets/example/top_short_merged \
    --assist_policy_device "$ASSIST_POLICY_DEVICE" \
    --enable_click_ik \
    --click_ik_steps 40 \
    --visual_repair_grasp top \
    --auto_visual_repair_step "$AUTO_REPAIR_STEP" \
    --auto_visual_repair_attempts 1 \
    --auto_visual_repair_settle_steps 40 \
    --visual_repair_debug_dir "/tmp/lehome_visual_repair_top_short_seen_9_v7" \
    --visual_repair_debug_every 20 \
    --auto_save_success \
    --auto_save_min_steps "$AUTO_SAVE_MIN_STEPS" \
    --enable_cameras \
    --device cpu \
    2>&1 | tee "/tmp/auto_visual_repair_top_short_seen_9.log"
