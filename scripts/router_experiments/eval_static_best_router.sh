#!/bin/bash
set -euo pipefail

cd /home/arjun/lehome-challenge
source .venv/bin/activate

PROGRESS=/tmp/eval_static_best_router_progress.log
: > "$PROGRESS"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PROGRESS"
}

log "Static-best portfolio-router eval: classifier category -> most-common-best checkpoint"
log "Map: top_short=aug_50k, top_long=aug_90k, pant_long=aug_90k, pant_short=aug_50k"

declare -A DATASETS=(
    ["top_short"]="Datasets/example/top_short_merged"
    ["top_long"]="Datasets/example/top_long_merged"
    ["pant_long"]="Datasets/example/pant_long_merged"
    ["pant_short"]="Datasets/example/pant_short_merged"
)

for cat in top_short top_long pant_long pant_short; do
    log "==== Static-best $cat ===="
    PORTFOLIO_HYBRID_CATEGORIES="__none__" \
    PORTFOLIO_USE_MULTICAM=1 \
    PORTFOLIO_DEFAULT_TOP_SHORT=aug_50k \
    PORTFOLIO_DEFAULT_TOP_LONG=aug_90k \
    PORTFOLIO_DEFAULT_PANT_LONG=aug_90k \
    PORTFOLIO_DEFAULT_PANT_SHORT=aug_50k \
    ROUTER_GT_CATEGORY="$cat" \
        ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
            --policy_type portfolio_router \
            --policy_path /home/arjun/lehome-challenge/outputs/classifier/garment_classifier.pt \
            --dataset_root "${DATASETS[$cat]}" \
            --task_description "Fold a garment with bimanual robot arms" \
            --garment_type "$cat" \
            --max_steps 600 \
            --num_episodes 5 \
            --enable_cameras \
            --device cpu \
            --headless \
            > "/tmp/static_best_${cat}.log" 2>&1
    rc=$?
    log "Static-best $cat done (exit=$rc)"
    grep -E "Success Rate:|Average Return|fallback route|  (Top|Pant)_" "/tmp/static_best_${cat}.log" \
        | grep -v "Saved video\|Switching\|GarmentEnv" \
        | tail -24 \
        | tee -a "$PROGRESS"
    echo "" | tee -a "$PROGRESS"
done

log "==== STATIC-BEST EVAL DONE ===="
