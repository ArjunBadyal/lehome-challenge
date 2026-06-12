#!/bin/bash
# Tiny ACT fine-tune on base + curriculum demos.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
EXTRA_STEPS="${2:-1000}"
LR="${LR:-1e-6}"

case "$CATEGORY" in
    top_short)
        START_DIR="outputs/train/act_top_short_aug/checkpoints/055000"
        START_STEP=55000
        DATASET_ROOT="${DATASET_ROOT:-Datasets/garment_curriculum/merged/top_short_merged}"
        OUT_DIR="${OUT_DIR:-outputs/train/act_top_short_curriculum}"
        REPO_ID="top_short_curriculum"
        ;;
    pant_long)
        START_DIR="outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250"
        START_STEP=90250
        DATASET_ROOT="${DATASET_ROOT:-Datasets/garment_curriculum/merged/pant_long_merged}"
        OUT_DIR="${OUT_DIR:-outputs/train/act_pant_long_curriculum}"
        REPO_ID="pant_long_curriculum"
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long> [extra_steps]"
        exit 2
        ;;
esac

TARGET_STEP=$((START_STEP + EXTRA_STEPS))
START_CKPT_OUT="${OUT_DIR}/checkpoints/$(printf '%06d' "$START_STEP")"
CFG="${START_CKPT_OUT}/pretrained_model/train_config.json"

if [[ ! -d "$START_DIR" ]]; then
    echo "Missing start checkpoint: $START_DIR"
    exit 1
fi
python scripts/garment_curriculum/validate_lerobot_dataset.py "$DATASET_ROOT" --min-episodes 10 >/dev/null

mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "$START_DIR" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

python - <<EOF
import json
cfg_path = "$CFG"
cfg = json.load(open(cfg_path))
cfg["dataset"]["root"] = "$DATASET_ROOT"
cfg["dataset"]["repo_id"] = "$REPO_ID"
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
print(f"train_config: dataset={cfg['dataset']['root']}, steps={cfg['steps']}, save_freq={cfg['save_freq']}, lr={cfg.get('optimizer', {}).get('lr')}")
EOF

LOG="/tmp/finetune_${CATEGORY}_curriculum.log"
echo "Launching lerobot-train; log=$LOG"
lerobot-train --config_path="$CFG" --resume=true 2>&1 | tee "$LOG"
