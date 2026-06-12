#!/bin/bash
# Collect additional ACT-generated successful demos on Seen garments only.
#
# Rationale: the previous fine-tunes overfit because the added demos were
# concentrated on Top_Short_Seen_9 / Pant_Long_Seen_3. This queue collects a
# broader set of successful rollouts so a later base+demos merge has less drift.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

ASSIST_POLICY_DEVICE="${ASSIST_POLICY_DEVICE:-cuda}"
NUM_SUCCESS_PER_GARMENT="${NUM_SUCCESS_PER_GARMENT:-5}"
OUTPUT_BASE="${OUTPUT_BASE:-Datasets/teleop_recovery/diverse_seen_act_harvest}"

mkdir -p "$OUTPUT_BASE"

harvest_one() {
    local garment="$1"
    local policy_path="$2"
    local dataset_root="$3"
    local early_step="$4"
    local fail_steps="$5"
    local out_root="${OUTPUT_BASE}/${garment}"
    local score_log="/tmp/harvest_diverse_${garment}_scores.jsonl"
    local run_log="/tmp/harvest_diverse_${garment}.log"

    mkdir -p "$out_root"
    rm -f "$score_log"

    echo ""
    echo "=========================================================================="
    echo " DIVERSE ACT HARVEST: ${garment}"
    echo "=========================================================================="
    echo " Save dir   : ${out_root}"
    echo " Target     : ${NUM_SUCCESS_PER_GARMENT} successful demos"
    echo " Policy     : ${policy_path}"
    echo " Dataset    : ${dataset_root}"
    echo " Gate       : early_step=${early_step}, fail_steps=${fail_steps}"
    echo " Log        : ${run_log}"
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
        --auto_save_min_steps 120 \
        --score_probe_steps 120,160,220,280,340,390,520 \
        --score_probe_log "$score_log" \
        --early_restart_step "$early_step" \
        --early_restart_close_ratio 3.2 \
        --early_restart_min_passed 3 \
        --auto_restart_fail_steps "$fail_steps" \
        --safe_assist_hotkeys \
        --enable_click_ik \
        --assist_policy_type lerobot \
        --assist_policy_path "$policy_path" \
        --assist_dataset_root "$dataset_root" \
        --assist_policy_device "$ASSIST_POLICY_DEVICE" \
        --device cpu \
        2>&1 | tee "$run_log"
}

echo "=========================================================================="
echo " DIVERSE SEEN ACT HARVEST QUEUE"
echo "=========================================================================="
echo " Output base : ${OUTPUT_BASE}"
echo " Target each : ${NUM_SUCCESS_PER_GARMENT}"
echo " Device      : ${ASSIST_POLICY_DEVICE}"
echo " Note        : Seen garments only; no Unseen demo collection."
echo "=========================================================================="

# Top-short: include holdout-like Seen_0 and a non-9 garment to reduce the
# Seen_9-only bias that caused catastrophic forgetting.
harvest_one \
    Top_Short_Seen_0 \
    outputs/train/act_top_short_aug/checkpoints/045000/pretrained_model \
    Datasets/example/top_short_merged \
    220 520

harvest_one \
    Top_Short_Seen_2 \
    outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model \
    Datasets/example/top_short_merged \
    220 520

# Pant-long: collect both the prior target and a holdout-style seen garment.
harvest_one \
    Pant_Long_Seen_0 \
    outputs/train/act_pant_long_aug/checkpoints/085000/pretrained_model \
    Datasets/example/pant_long_merged \
    280 540

harvest_one \
    Pant_Long_Seen_3 \
    golden_checkpoints/pant_long_80k/pretrained_model \
    Datasets/example/pant_long_merged \
    280 540

echo ""
echo "Diverse Seen harvest complete."
