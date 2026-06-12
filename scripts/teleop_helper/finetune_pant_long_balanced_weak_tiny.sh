#!/bin/bash
# Tiny Pant-Long fine-tune on balanced Seen replay.
#
# The prior +5K / narrow-demo runs overfit. This script intentionally uses:
#   - base aug90 checkpoint as the start point
#   - balanced replay dataset, not demo-only data
#   - low LR
#   - checkpoints every 250 steps
#
# Usage:
#   bash scripts/teleop_helper/finetune_pant_long_balanced_weak_tiny.sh [extra_steps]
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

EXTRA_STEPS="${1:-1000}"
LR="${LR:-1e-6}"
START_STEP=90000
TARGET_STEP=$((START_STEP + EXTRA_STEPS))

START_DIR="outputs/train/act_pant_long_aug/checkpoints/090000"
OUT_DIR="outputs/train/act_pant_long_balanced_weak_tiny"
DATASET_ROOT="Datasets/teleop_merged_clean_balanced_weak/pant_long_merged"
START_CKPT_OUT="${OUT_DIR}/checkpoints/$(printf '%06d' "$START_STEP")"
CFG="${START_CKPT_OUT}/pretrained_model/train_config.json"

if [[ ! -d "$START_DIR" ]]; then
    echo "Missing start checkpoint: $START_DIR"
    exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Missing merged dataset: $DATASET_ROOT"
    exit 1
fi
if [[ -e "$OUT_DIR" && ! -d "$START_CKPT_OUT" ]]; then
    echo "Output exists in unexpected state: $OUT_DIR"
    exit 1
fi

echo "Setting up Pant-Long balanced weak tiny fine-tune: ${START_STEP} -> ${TARGET_STEP}, LR=${LR}"
mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "$START_DIR" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

python - <<EOF
import json
cfg_path = "$CFG"
cfg = json.load(open(cfg_path))
cfg["dataset"]["root"] = "$DATASET_ROOT"
cfg["dataset"]["repo_id"] = "pant_long_balanced_weak_seen"
cfg["dataset"]["image_transforms"]["enable"] = True
cfg["output_dir"] = "$OUT_DIR"
cfg["resume"] = True
cfg["checkpoint_path"] = "${OUT_DIR}/checkpoints/last"
cfg["steps"] = $TARGET_STEP
cfg["save_freq"] = 250
if "policy" in cfg:
    cfg["policy"]["optimizer_lr"] = float("$LR")
    cfg["policy"]["optimizer_lr_backbone"] = float("$LR")
if "optimizer" in cfg:
    cfg["optimizer"]["lr"] = float("$LR")
json.dump(cfg, open(cfg_path, "w"), indent=4)
print(
    "train_config: "
    f"dataset={cfg['dataset']['root']}, "
    f"steps={cfg['steps']}, save_freq={cfg['save_freq']}, "
    f"lr={cfg.get('optimizer', {}).get('lr')}"
)
EOF

LOG="/tmp/finetune_pant_long_balanced_weak_tiny.log"
echo "Launching lerobot-train; log=$LOG"
lerobot-train --config_path="$CFG" --resume=true 2>&1 | tee "$LOG"
