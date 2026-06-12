#!/bin/bash
# Teleop recording for Top_Short_Seen_5 (precision-fold near-miss case).
# Bi-keyboard teleop. Demos saved to Datasets/teleop_recovery/top_short_seen_5/.
#
# Failure mode: ACT gets ~20% success here. The fold is approximately right
# but corners don't fully meet (d(0,4), d(1,5) ~18cm vs target 6.3).
# Demonstrate symmetric, complete-fold versions where both sleeves meet.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/teleop_recovery/top_short_seen_5"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " TELEOP: Top_Short_Seen_5 — clean symmetric fold"
echo "=========================================================================="
echo " Save dir : $OUTPUT_ROOT"
echo " Goal     : let ACT start, then manually tighten symmetric sleeve/corner fold"
echo " Assist   : S=start ACT recording, M=manual takeover, R=resume, P=pause physics"
echo " Grip     : C=force closed+pause physics, V=open/release+pause physics, Z=ACT grippers"
echo " Repair   : E=visual landmark IK grasp→fold→release macro (slow, top landmarks)"
echo " AutoFold : G=after C, plan bimanual move toward garment centroid (no clicks)"
echo " ClickIK  : after C/V, click the 'Click-IK Top Camera' window to move nearest arm"
echo " Save     : N=success+save, X=restart, D=discard, ESC=abort"
echo "=========================================================================="
echo ""

LEHOME_GLOBAL_KEYBOARD=1 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
    --task "LeHome-BiSO101-Direct-Garment-v2" \
    --garment_name Top_Short_Seen_5 \
    --garment_version Release \
    --teleop_device bi-keyboard \
    --num_envs 1 \
    --enable_record \
    --dataset_root "$OUTPUT_ROOT" \
    --num_episode 30 \
    --log_success \
    --assist_policy_type lerobot \
    --assist_policy_path outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model \
    --assist_dataset_root Datasets/example/top_short_merged \
    --assist_policy_device "$ASSIST_POLICY_DEVICE" \
    --enable_click_ik \
    --click_ik_steps 70 \
    --visual_repair_grasp top \
    --enable_cameras \
    --device cpu \
    2>&1 | tee "/tmp/teleop_top_short_seen_5.log"
