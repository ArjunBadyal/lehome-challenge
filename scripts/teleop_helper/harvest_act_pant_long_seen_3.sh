#!/bin/bash
# ACT-only success harvester for Pant_Long_Seen_3.
#
# Mirrors the Top_Short harvester: auto-start ACT, auto-save successes,
# auto-restart failures. Pant_Long success criteria:
#   d(p[0],p[4]) <= ~3.5 cm  (left waist -> knee)
#   d(p[1],p[5]) <= ~3.5 cm  (right waist -> knee)
#   d(p[0],p[2]) >= ~3.9 cm  (anatomical, auto-pass)
#   d(p[1],p[3]) >= ~3.9 cm  (anatomical, auto-pass)
# So passed>=3 means at least one waist-knee pair has closed.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/teleop_recovery/pant_long_seen_3_act_harvest"
ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " ACT HARVEST: Pant_Long_Seen_3"
echo "=========================================================================="
echo " Save dir : $OUTPUT_ROOT"
echo " Mode     : auto-start ACT, auto-save successes, auto-restart failures"
echo " Success  : look for '[AutoSave] ... saving episode' in the terminal"
echo " Abort    : Ctrl+C"
echo "=========================================================================="
echo ""

LEHOME_GLOBAL_KEYBOARD=0 ./third_party/IsaacLab/isaaclab.sh -p -m scripts.dataset_sim record \
    --task "LeHome-BiSO101-Direct-Garment-v2" \
    --garment_name Pant_Long_Seen_3 \
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
    --score_probe_log /tmp/harvest_act_pant_long_seen_3_scores.jsonl \
    --early_restart_step 280 \
    --early_restart_close_ratio 3.2 \
    --early_restart_min_passed 3 \
    --auto_restart_fail_steps 540 \
    --safe_assist_hotkeys \
    --enable_click_ik \
    --assist_policy_type lerobot \
    --assist_policy_path golden_checkpoints/pant_long_80k/pretrained_model \
    --assist_dataset_root Datasets/example/pant_long_merged \
    --assist_policy_device "$ASSIST_POLICY_DEVICE" \
    --device cpu \
    2>&1 | tee "/tmp/harvest_act_pant_long_seen_3.log"
