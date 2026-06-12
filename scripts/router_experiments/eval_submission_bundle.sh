#!/bin/bash
# Evaluate the exact packaged submission policy (`submission/policy.py` +
# `submission/policies/`) through a local adapter, without building Docker.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CAT="${1:-}"
if [[ -z "$CAT" ]]; then
    echo "Usage: bash $0 <top_short|top_long|pant_long|pant_short> [pant_long_stabilizer_mode]"
    exit 1
fi
if [[ "${2:-}" != "" ]]; then
    export LEHOME_PANT_LONG_STABILIZER="$2"
fi

case "$CAT" in
    top_short) DATASET="Datasets/example/top_short_merged" ;;
    top_long) DATASET="Datasets/example/top_long_merged" ;;
    pant_long) DATASET="Datasets/example/pant_long_merged" ;;
    pant_short) DATASET="Datasets/example/pant_short_merged" ;;
    *) echo "Unknown category: $CAT"; exit 1 ;;
esac

MODE_SUFFIX=""
if [[ "${LEHOME_PANT_LONG_STABILIZER:-}" != "" ]]; then
    MODE_SUFFIX="_pantstab_${LEHOME_PANT_LONG_STABILIZER}"
fi
LOG="/tmp/submission_bundle_${CAT}${MODE_SUFFIX}.log"
echo "=== Submission bundle eval: ${CAT} ==="
echo "Policy: submission/policy.py"
echo "Policies root: submission/policies"
echo "Pant-long stabilizer: ${LEHOME_PANT_LONG_STABILIZER:-default}"
echo "Log: $LOG"

LEHOME_POLICY_DEVICE=cpu \
POLICIES_ROOT="$PWD/submission/policies" \
    ./third_party/IsaacLab/isaaclab.sh -p -m scripts.eval \
        --policy_type submission_bundle \
        --policy_path "$PWD/submission/policies/classifier/garment_classifier.pt" \
        --dataset_root "$DATASET" \
        --task_description "Fold a garment with bimanual robot arms" \
        --garment_type "$CAT" \
        --max_steps 600 \
        --num_episodes 5 \
        --enable_cameras \
        --device cpu \
        --headless \
        2>&1 | tee "$LOG"

echo ""
echo "=== Summary: ${CAT} ==="
grep -aE "Overall Summary|Total Episodes|Success Rate|  (Top|Pant)_" "$LOG" | tail -80
