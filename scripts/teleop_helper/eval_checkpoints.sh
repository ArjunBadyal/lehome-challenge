#!/bin/bash
# Evaluate every checkpoint produced by the teleop retrain (5-ep CPU per checkpoint).
# Usage: bash scripts/teleop_helper/eval_checkpoints.sh <category>
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
case "$CATEGORY" in
    top_short|top_long|pant_long|pant_short) ;;
    *) echo "Usage: bash $0 <category>  ∈ {top_short, top_long, pant_long, pant_short}"; exit 1 ;;
esac

CKPT_DIR="outputs/train/act_${CATEGORY}_teleop/checkpoints"
DATASET="Datasets/example/${CATEGORY}_merged"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "No teleop retrain checkpoints found at $CKPT_DIR"
    exit 1
fi

PROGRESS="/tmp/teleop_eval_${CATEGORY}_progress.log"
: > "$PROGRESS"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS"; }

log "Evaluating teleop retrain checkpoints for $CATEGORY"

for step_dir in "$CKPT_DIR"/0*; do
    [[ -L "$step_dir" ]] && continue  # skip 'last' symlink
    step=$(basename "$step_dir")
    label="$(printf '%03dk' $((10#$step / 1000)))"
    log "==== ${CATEGORY} teleop $label ===="

    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type lerobot \
        --policy_path "${step_dir}/pretrained_model" \
        --dataset_root "$DATASET" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type "$CATEGORY" \
        --max_steps 600 --num_episodes 5 \
        --enable_cameras --device cpu --headless \
        > "/tmp/teleop_eval_${CATEGORY}_${label}.log" 2>&1
    rc=$?
    log "$label done (exit=$rc)"
    grep -E "Success Rate:|  ${CATEGORY^}" "/tmp/teleop_eval_${CATEGORY}_${label}.log" \
        | grep -v "Saved video\|Switching" | tail -14 | tee -a "$PROGRESS"
    echo "" | tee -a "$PROGRESS"
done

log "==== ${CATEGORY^^} TELEOP EVALS DONE ===="
echo ""
echo "Pick the best checkpoint by Success Rate. Then update submission/assemble_policies.sh"
echo "to point ${CATEGORY} at outputs/train/act_${CATEGORY}_teleop/checkpoints/<best_step>"
