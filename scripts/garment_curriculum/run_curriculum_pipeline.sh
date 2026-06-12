#!/bin/bash
# Orchestrate the aggressive morphology-curriculum workflow.
#
# This script is intentionally sequential. Isaac Sim is unstable when multiple
# sims run in parallel on this machine.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
case "$CATEGORY" in
    top_short|pant_long) ;;
    *)
        echo "Usage: $0 <top_short|pant_long>"
        echo "Stages can be skipped with: SKIP_VARIANTS=1 SKIP_RL=1 SKIP_HARVEST=1 SKIP_MERGE=1 SKIP_FINETUNE=1 SKIP_EVAL=1"
        exit 2
        ;;
esac

if [[ "${SKIP_VARIANTS:-0}" != "1" ]]; then
    python scripts/garment_curriculum/make_seen_variants.py --categories "$CATEGORY"
fi

if [[ "${SKIP_RL:-0}" != "1" ]]; then
    bash scripts/garment_curriculum/train_residual_curriculum.sh "$CATEGORY"
fi

if [[ "${SKIP_HARVEST:-0}" != "1" ]]; then
    if [[ -z "${RESIDUAL_CKPT:-}" ]]; then
        echo "Set RESIDUAL_CKPT=/path/to/checkpoint.pt for harvest stage."
        exit 1
    fi
    bash scripts/garment_curriculum/harvest_residual_success_demos.sh "$CATEGORY" "$RESIDUAL_CKPT"
fi

if [[ "${SKIP_MERGE:-0}" != "1" ]]; then
    bash scripts/garment_curriculum/merge_curriculum_demos.sh "$CATEGORY"
fi

if [[ "${SKIP_FINETUNE:-0}" != "1" ]]; then
    bash scripts/garment_curriculum/finetune_curriculum_act.sh "$CATEGORY" "${EXTRA_STEPS:-1000}"
fi

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    bash scripts/garment_curriculum/eval_curriculum_checkpoints.sh "$CATEGORY"
fi
