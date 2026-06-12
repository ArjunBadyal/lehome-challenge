#!/bin/bash
# ACT-only success harvester for Top_Short_Seen_9.
#
# This does not use manual save or IK repair. It repeatedly runs the current
# Top-Short ACT policy, saves only episodes that pass the official success
# checker, and discards/retries failures after a fixed step budget.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/teleop_recovery/top_short_seen_9_act_harvest"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " ACT HARVEST: Top_Short_Seen_9"
echo "=========================================================================="
echo " Save dir : $OUTPUT_ROOT"
echo " Mode     : auto-start ACT, auto-save successes, auto-restart failures"
echo " Success  : look for '[AutoSave] ... saving episode' in the terminal"
echo " Abort    : Ctrl+C"
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
    --num_episode 10 \
    --log_success \
    --auto_start_record \
    --auto_save_success \
    --auto_save_min_steps 120 \
    --score_probe_steps 120,160,220,280,340,390,520 \
    --score_probe_log /tmp/harvest_act_top_short_seen_9_scores.jsonl \
    --early_restart_step 220 \
    --early_restart_close_ratio 3.2 \
    --early_restart_min_passed 3 \
    --auto_restart_fail_steps 520 \
    --safe_assist_hotkeys \
    --enable_click_ik \
    --assist_policy_type lerobot \
    --assist_policy_path outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model \
    --assist_dataset_root Datasets/example/top_short_merged \
    --assist_policy_device "$ASSIST_POLICY_DEVICE" \
    --device cpu \
    2>&1 | tee "/tmp/harvest_act_top_short_seen_9.log"
