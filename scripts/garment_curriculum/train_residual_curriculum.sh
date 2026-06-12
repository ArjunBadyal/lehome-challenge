#!/bin/bash
# Train a residual SAC helper on curriculum variants.
#
# The residual is a demo generator only. It must not be packaged for final
# submission unless a separate full packaged eval proves it is safe.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
VARIANT_LIST="${2:-}"

case "$CATEGORY" in
    top_short)
        ACT_CKPT="${ACT_CKPT:-outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model}"
        DATASET_ROOT="${DATASET_ROOT:-Datasets/example/top_short_merged}"
        DEFAULT_LIST="outputs/garment_curriculum/eval_lists/top_short_variants.txt"
        ;;
    pant_long)
        ACT_CKPT="${ACT_CKPT:-outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model}"
        DATASET_ROOT="${DATASET_ROOT:-Datasets/example/pant_long_merged}"
        DEFAULT_LIST="outputs/garment_curriculum/eval_lists/pant_long_variants.txt"
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long> [variant_list]"
        exit 2
        ;;
esac

VARIANT_LIST="${VARIANT_LIST:-$DEFAULT_LIST}"
if [[ ! -f "$VARIANT_LIST" ]]; then
    echo "Variant list not found: $VARIANT_LIST"
    echo "Run: python scripts/garment_curriculum/make_seen_variants.py"
    exit 1
fi

MAX_TRAIN_GARMENTS="${MAX_TRAIN_GARMENTS:-16}"
mapfile -t GARMENTS < <(
    python - "$VARIANT_LIST" "$MAX_TRAIN_GARMENTS" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
limit = int(sys.argv[2])
items = [line.strip() for line in path.read_text().splitlines() if line.strip()]
if limit <= 0 or limit >= len(items):
    chosen = items
else:
    # Even spacing avoids taking only the first source garment when the list is
    # grouped by source. Preserve order and de-duplicate rounded indices.
    idxs = []
    seen = set()
    for i in range(limit):
        idx = round(i * (len(items) - 1) / max(limit - 1, 1))
        if idx not in seen:
            seen.add(idx)
            idxs.append(idx)
    chosen = [items[i] for i in idxs]
for item in chosen:
    print(item)
PY
)
if [[ "${#GARMENTS[@]}" -eq 0 ]]; then
    echo "No garments in $VARIANT_LIST"
    exit 1
fi

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-30000}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.04}"
RESIDUAL_SCALE_MAX="${RESIDUAL_SCALE_MAX:-0.14}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-5000}"
BC_PRETRAIN_EPOCHS="${BC_PRETRAIN_EPOCHS:-0}"
LOG_DIR="${LOG_DIR:-outputs/rl/residual_curriculum}"
RUN_NAME="${RUN_NAME:-${CATEGORY}_curriculum_$(date +%Y%m%d_%H%M%S)}"
LOG="/tmp/train_residual_curriculum_${CATEGORY}.log"

echo "=== Residual curriculum train: $CATEGORY ==="
echo "Garments (${#GARMENTS[@]}): ${GARMENTS[*]}"
echo "ACT:      $ACT_CKPT"
echo "Dataset:  $DATASET_ROOT"
echo "Steps:    $TOTAL_TIMESTEPS"
echo "BC pretrain epochs: $BC_PRETRAIN_EPOCHS"
echo "Log:      $LOG"

./third_party/IsaacLab/isaaclab.sh -p -m scripts.train_residual_sac \
    --task LeHome-BiSO101-Direct-Garment-v2 \
    --train_garments "${GARMENTS[@]}" \
    --garment_version Release \
    --num_envs 1 \
    --act_checkpoint "$ACT_CKPT" \
    --dataset_root "$DATASET_ROOT" \
    --total_timesteps "$TOTAL_TIMESTEPS" \
    --residual_scale "$RESIDUAL_SCALE" \
    --residual_scale_max "$RESIDUAL_SCALE_MAX" \
    --residual_mask 3 4 5 9 10 11 \
    --gate_steps 80 \
    --bc_pretrain_epochs "$BC_PRETRAIN_EPOCHS" \
    --learning_starts 1000 \
    --eval_every 0 \
    --checkpoint_freq "$CHECKPOINT_FREQ" \
    --log_dir "$LOG_DIR" \
    --run_name "$RUN_NAME" \
    --rl_device "${RL_DEVICE:-cuda:0}" \
    --device cpu \
    --headless \
    2>&1 | tee "$LOG"
