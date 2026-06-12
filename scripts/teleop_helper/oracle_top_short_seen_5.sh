#!/bin/bash
# Scripted check_point oracle for Top_Short_Seen_5 — collar-fold proxy.
# Runs unattended: snapshots cloth check_points at episode start, drives
# bimanual IK to fold hem (p[4],p[5]) onto collar (p[0],p[1]), evaluates
# the standard challenge checker, saves successes / discards failures.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/scripted_oracle/top_short_seen_5"
NUM_EPISODE="${NUM_EPISODE:-5}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " ORACLE: Top_Short_Seen_5 — scripted bimanual fold (hem -> collar)"
echo "=========================================================================="
echo " Save dir   : $OUTPUT_ROOT"
echo " Target     : $NUM_EPISODE successful demos (failures auto-retried)"
echo " Compliance : check_points read at training time only; eval still hidden"
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
    --num_episode "$NUM_EPISODE" \
    --log_success \
    --scripted_oracle top_short \
    --enable_cameras \
    --device cpu \
    2>&1 | tee "/tmp/oracle_top_short_seen_5.log"
