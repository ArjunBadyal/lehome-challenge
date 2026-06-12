#!/bin/bash
# Teleop recording on Top_Short_Seen_9 — teaches the SAME skill as Unseen_0
# (collar fold + general clean fold) WITHOUT leaking on the eval test set.
#
# Why this garment: Seen_9 currently scores 0–20% across all checkpoints.
# All 5 conditions fail by ~20cm. Demonstrate a complete clean fold:
#   sleeves brought together, body folded, collar (shoulder caps) brought to
#   center over body. ~5–10 successful demos targeting the missed steps.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/teleop_recovery/top_short_seen_9"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " TELEOP: Top_Short_Seen_9 — clean fold demos (proxy for collar skill)"
echo "=========================================================================="
echo " Save dir : $OUTPUT_ROOT"
echo " Goal     : let ACT start, then manually fix collar/shoulder fold"
echo " Why this : trains the missing collar/shoulder-fold skill via a Seen garment"
echo "            (NOT Unseen_0 — that is the eval test set; teleoping on it leaks)"
echo " Assist   : S=start ACT recording, M=manual takeover, R=resume, P=pause physics"
echo " Grip     : C=force closed+pause physics, V=open/release+pause physics, Z=ACT grippers"
echo " Repair   : E=visual landmark IK grasp→fold→release macro (slow, top landmarks)"
echo " AutoFold : G=after C, plan bimanual move toward garment centroid (no clicks)"
echo " ClickIK  : after C/V, click the 'Click-IK Top Camera' window to move nearest arm"
echo " LiveIK   : P/C/V physics-pause, then use scripts/teleop_helper/send_ik_command.py"
echo " SafeMode : after P/C/V, all Isaac hotkeys are ignored until LiveIK resume"
echo " Save     : auto-save on checker success; X=restart bad starts before pause; N/D/M/E/G/ESC disabled"
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
    --num_episode 30 \
    --log_success \
    --auto_save_success \
    --auto_save_min_steps 120 \
    --safe_assist_hotkeys \
    --assist_policy_type lerobot \
    --assist_policy_path outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model \
    --assist_dataset_root Datasets/example/top_short_merged \
    --assist_policy_device "$ASSIST_POLICY_DEVICE" \
    --enable_click_ik \
    --click_ik_steps 70 \
    --visual_repair_grasp top \
    --enable_cameras \
    --device cpu \
    2>&1 | tee "/tmp/teleop_top_short_seen_9.log"
