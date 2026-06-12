#!/bin/bash
# Unattended Pant-Long weak-garment pipeline:
#   1. Harvest strict post-release successful ACT demos on weak Seen garments.
#   2. Merge them with the existing balanced/diverse Pant-Long replay dataset.
#   3. Fine-tune from aug90 with low LR and small checkpoint increments.
#   4. Evaluate base + fine-tuned checkpoints on a Pant-Long guard suite.
#
# Intended to run overnight without Isaac UI interaction.
set -u -o pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

PIPELINE_LOG="${PIPELINE_LOG:-/tmp/overnight_pant_long_weak_pipeline.log}"
HARVEST_TARGET="${HARVEST_TARGET:-3}"
MIN_TOTAL_DEMOS="${MIN_TOTAL_DEMOS:-3}"
STRICT_TIMEOUT_PER_GARMENT="${STRICT_TIMEOUT_PER_GARMENT:-75m}"
RELAXED_TIMEOUT_PER_GARMENT="${RELAXED_TIMEOUT_PER_GARMENT:-45m}"
EXTRA_STEPS="${EXTRA_STEPS:-1000}"
LR="${LR:-1e-6}"
WEAK_BASES="${WEAK_BASES:-Datasets/teleop_recovery/pant_long_weak_seen_act_harvest Datasets/teleop_recovery/pant_long_weak_seen_retracted_act_harvest}"
HARVEST_OUTPUT_BASE="${HARVEST_OUTPUT_BASE:-Datasets/teleop_recovery/pant_long_weak_seen_retracted_act_harvest}"
RUN_DIR="${RUN_DIR:-outputs/train/act_pant_long_balanced_weak_tiny}"
EVAL_CKPTS="${EVAL_CKPTS:-090000 090250 090500 090750 091000}"
COLLECT_EVAL_SUCCESS="${COLLECT_EVAL_SUCCESS:-1}"

exec > >(tee -a "$PIPELINE_LOG") 2>&1

kill_sim() {
    local pids
    pids="$(
        ps -eo pid=,args= \
            | grep -E "python -m scripts\.dataset_sim record|python -m scripts\.eval|bash ./third_party/IsaacLab/isaaclab\.sh -p -m scripts\.(dataset_sim|eval)|/kit/kit|isaacsim" \
            | grep -v grep \
            | awk '{print $1}' \
            || true
    )"
    if [[ -n "$pids" ]]; then
        echo "[pipeline] Killing stray Isaac/eval processes:"
        echo "$pids"
        kill $pids 2>/dev/null || true
        sleep 3
        pids="$(
            ps -eo pid=,args= \
                | grep -E "python -m scripts\.dataset_sim record|python -m scripts\.eval|bash ./third_party/IsaacLab/isaaclab\.sh -p -m scripts\.(dataset_sim|eval)|/kit/kit|isaacsim" \
                | grep -v grep \
                | awk '{print $1}' \
                || true
        )"
        if [[ -n "$pids" ]]; then
            kill -9 $pids 2>/dev/null || true
            sleep 2
        fi
    fi
}

count_finalized_weak_datasets() {
    local total=0
    local count
    for weak_base in ${WEAK_BASES}; do
        count="$(find "$weak_base" -mindepth 3 -maxdepth 4 -type f -path "*/meta/info.json" 2>/dev/null | wc -l)"
        total=$((total + count))
    done
    echo "$total"
}

count_episodes_for_garment() {
    local garment="$1"
    python - "$garment" ${WEAK_BASES} <<'PY'
from pathlib import Path
import sys
import pyarrow.parquet as pq

garment = sys.argv[1]
total = 0
for base_arg in sys.argv[2:]:
    root = Path(base_arg) / garment
    if not root.exists():
        continue
    for ep_file in sorted(root.glob("*/meta/episodes/chunk-*/file-*.parquet")):
        try:
            total += pq.read_table(ep_file).num_rows
        except Exception:
            pass
print(total)
PY
}

total_weak_episodes() {
    local total=0
    local count
    for garment in Pant_Long_Seen_9 Pant_Long_Seen_8 Pant_Long_Seen_6; do
        count="$(count_episodes_for_garment "$garment")"
        total=$((total + count))
    done
    echo "$total"
}

run_harvest_round() {
    local round_name="$1"
    local timeout_per_garment="$2"
    local schedule="$3"
    local save_min_steps="$4"
    local clearance="$5"
    local fail_steps="$6"

    echo ""
    echo "[pipeline] Harvest round: $round_name"
    echo "[pipeline] schedule=$schedule save_min_steps=$save_min_steps clearance=$clearance fail_steps=$fail_steps timeout=$timeout_per_garment"
    for garment in Pant_Long_Seen_9 Pant_Long_Seen_8 Pant_Long_Seen_6; do
        local current
        local remaining
        current="$(count_episodes_for_garment "$garment")"
        remaining=$((HARVEST_TARGET - current))
        if [[ "$remaining" -le 0 ]]; then
            echo "[pipeline] $garment already has $current demos; skipping."
            continue
        fi
        echo "[pipeline] $garment has $current demos; collecting up to $remaining more."
        set +e
        NUM_SUCCESS_PER_GARMENT="$remaining" \
        GARMENTS="$garment" \
        OUTPUT_BASE="$HARVEST_OUTPUT_BASE" \
        EARLY_RESTART_SCHEDULE="$schedule" \
        AUTO_SAVE_MIN_STEPS="$save_min_steps" \
        AUTO_SAVE_REQUIRE_RELEASE=0 \
        AUTO_SAVE_MIN_GRIPPER_CLOTH_DISTANCE="$clearance" \
        AUTO_RESTART_FAIL_STEPS="$fail_steps" \
        DISABLE_DEPTH=1 \
        timeout "$timeout_per_garment" bash scripts/teleop_helper/harvest_pant_long_weak_seen.sh
        local rc=$?
        set -e
        echo "[pipeline] $round_name / $garment harvest exit code: $rc"
        kill_sim
        current="$(count_episodes_for_garment "$garment")"
        echo "[pipeline] $garment demos after $round_name: $current"
    done
}

echo "=========================================================================="
echo " OVERNIGHT PANT-LONG WEAK PIPELINE"
echo "=========================================================================="
echo " Started       : $(date -Is)"
echo " Harvest target: $HARVEST_TARGET per weak Seen garment"
echo " Strict timeout: $STRICT_TIMEOUT_PER_GARMENT per garment"
echo " Relax timeout : $RELAXED_TIMEOUT_PER_GARMENT per garment"
echo " Weak bases    : $WEAK_BASES"
echo " Harvest output: $HARVEST_OUTPUT_BASE"
echo " Fine-tune     : +${EXTRA_STEPS} steps, LR=${LR}"
echo " Eval ckpts    : $EVAL_CKPTS"
echo " Eval-success : $COLLECT_EVAL_SUCCESS"
echo " Log           : $PIPELINE_LOG"
echo "=========================================================================="

kill_sim

echo ""
echo "[pipeline] Stage 1/4: strict weak-seen harvest"
run_harvest_round \
    "strict" "$STRICT_TIMEOUT_PER_GARMENT" \
    "120::3.0:,160::2.6:,220::1.5:1,280:::2" \
    "180" "0.0" "520"

if [[ "$(total_weak_episodes)" -lt $((HARVEST_TARGET * 3)) ]]; then
    echo "[pipeline] Strict pass did not hit full quota; running relaxed fallback."
    run_harvest_round \
        "relaxed" "$RELAXED_TIMEOUT_PER_GARMENT" \
        "120::3.4:,180::2.8:,260::1.8:1,360:::1" \
        "180" "0.0" "560"
fi

weak_count="$(count_finalized_weak_datasets)"
weak_episodes="$(total_weak_episodes)"
echo "[pipeline] Finalized weak harvest datasets found: $weak_count"
echo "[pipeline] Finalized weak episodes found: $weak_episodes"
if [[ "$COLLECT_EVAL_SUCCESS" == "1" ]]; then
    echo ""
    echo "[pipeline] Stage 1b/4: collect Seen eval-success demos"
    kill_sim
    bash scripts/teleop_helper/collect_eval_success_pant_long_seen.sh
fi

eval_success_count="$(
    find Datasets/eval_success/pant_long_aug90_seen_success -mindepth 2 -maxdepth 3 -type f -path '*/meta/info.json' 2>/dev/null | wc -l
)"
echo "[pipeline] Eval-success datasets found: $eval_success_count"
if [[ "$weak_episodes" -lt "$MIN_TOTAL_DEMOS" && "$eval_success_count" -lt 1 ]]; then
    echo "[pipeline] Not enough weak/eval-success demos; skipping merge/train/eval."
    exit 0
fi

echo ""
echo "[pipeline] Stage 2/4: clean merge"
if ! bash scripts/teleop_helper/merge_pant_long_balanced_weak_seen.sh; then
    echo "[pipeline] Merge failed; aborting before training."
    exit 1
fi

echo ""
echo "[pipeline] Stage 3/4: tiny balanced fine-tune"
LR="$LR" bash scripts/teleop_helper/finetune_pant_long_balanced_weak_tiny.sh "$EXTRA_STEPS"

echo ""
echo "[pipeline] Stage 4/4: CPU guard-suite eval"
kill_sim
RUN_DIR="$RUN_DIR" CKPTS="$EVAL_CKPTS" bash scripts/teleop_helper/eval_pant_long_balanced_weak_tiny.sh

echo ""
echo "=========================================================================="
echo " PIPELINE COMPLETE"
echo "=========================================================================="
echo " Finished: $(date -Is)"
echo " Summary : /tmp/eval_pant_long_balanced_weak_tiny/summary.tsv"
cat /tmp/eval_pant_long_balanced_weak_tiny/summary.tsv 2>/dev/null || true
