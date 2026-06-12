#!/bin/bash
# Scripted check_point oracle for Pant_Long_Seen_3 — second-leg fold proxy.
# Runs unattended: snapshots cloth check_points at episode start, drives
# bimanual IK to fold waist (p[0],p[1]) onto knee (p[4],p[5]), evaluates
# the standard challenge checker, saves successes / discards failures.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

OUTPUT_ROOT="Datasets/scripted_oracle/pant_long_seen_3"
NUM_EPISODE="${NUM_EPISODE:-5}"
mkdir -p "$OUTPUT_ROOT"

echo "=========================================================================="
echo " ORACLE: Pant_Long_Seen_3 — scripted bimanual fold (waist -> knee)"
echo "=========================================================================="
echo " Save dir   : $OUTPUT_ROOT"
echo " Target     : $NUM_EPISODE successful demos (failures auto-retried)"
echo " Compliance : check_points read at training time only; eval still hidden"
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
    --num_episode "$NUM_EPISODE" \
    --log_success \
    --scripted_oracle pant_long \
    --enable_cameras \
    --device cpu \
    2>&1 | tee "/tmp/oracle_pant_long_seen_3.log"
