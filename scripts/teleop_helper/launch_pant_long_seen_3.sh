#!/bin/bash
# Teleop recording on Pant_Long_Seen_3 — teaches second-leg recovery (the
# Pant_Long_Unseen_1 failure mode) WITHOUT leaking on the eval test set.
#
# Why this garment: Seen_3 has d(1,5) failure (right leg loose) on 80% of
# attempts across all checkpoints — same mode as Unseen_1 but on training data.
# Demonstrate folding both legs symmetrically into the body.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/teleop_recovery/pant_long_seen_3"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " TELEOP: Pant_Long_Seen_3 — second-leg fold (proxy for Unseen_1 mode)"
echo "=========================================================================="
echo " Save dir : $OUTPUT_ROOT"
echo " Goal     : let ACT start, then manually fix the loose second-leg fold"
echo " Why this : trains the missing second-leg-fold skill on a Seen garment"
echo "            (NOT Unseen_1 — that is the eval test set; teleoping on it leaks)"
echo " Assist   : S=start ACT recording, M=manual takeover, R=resume, P=pause physics"
echo " Grip     : C=force closed+pause physics, V=open/release+pause physics, Z=ACT grippers"
echo " Repair   : E=visual landmark IK grasp→fold→release macro (slow, bottom landmarks)"
echo " AutoFold : G=after C, plan bimanual move toward garment centroid (no clicks)"
echo " ClickIK  : after C/V, click the 'Click-IK Top Camera' window to move nearest arm"
echo " Save     : N=success+save, X=restart, D=discard, ESC=abort"
echo "=========================================================================="
echo ""

LEHOME_GLOBAL_KEYBOARD=1 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
    --task "LeHome-BiSO101-Direct-Garment-v2" \
    --garment_name Pant_Long_Seen_3 \
    --garment_version Release \
    --teleop_device bi-keyboard \
    --num_envs 1 \
    --enable_record \
    --dataset_root "$OUTPUT_ROOT" \
    --num_episode 30 \
    --log_success \
    --assist_policy_type lerobot \
    --assist_policy_path outputs/train/act_pant_long_aug/checkpoints/090000/pretrained_model \
    --assist_dataset_root Datasets/example/pant_long_merged \
    --assist_policy_device "$ASSIST_POLICY_DEVICE" \
    --enable_click_ik \
    --click_ik_steps 70 \
    --visual_repair_grasp bottom \
    --enable_cameras \
    --device cpu \
    2>&1 | tee "/tmp/teleop_pant_long_seen_3.log"
