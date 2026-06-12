#!/bin/bash
# Fine-tune ACT from checkpoint weights with a fresh optimizer.
#
# This deliberately avoids --resume=true. The previous "low-LR" runs restored
# optimizer state and continued at the old effective LR; this script loads model
# weights through policy.pretrained_path and creates a new optimizer/scheduler.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
EXTRA_STEPS="${2:-1000}"

case "$CATEGORY" in
    top_short)
        BASE_PRETRAINED="${BASE_PRETRAINED:-outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model}"
        DATASET_ROOT="${DATASET_ROOT:-Datasets/garment_curriculum/merged/top_short_merged}"
        OUT_DIR="${OUT_DIR:-outputs/train/act_top_short_variant_fresh}"
        REPO_ID="${REPO_ID:-top_short_variant_fresh}"
        ;;
    pant_long)
        BASE_PRETRAINED="${BASE_PRETRAINED:-outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model}"
        DATASET_ROOT="${DATASET_ROOT:-Datasets/garment_curriculum/merged/pant_long_merged}"
        OUT_DIR="${OUT_DIR:-outputs/train/act_pant_long_variant_fresh}"
        REPO_ID="${REPO_ID:-pant_long_variant_fresh}"
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long> [extra_steps]"
        exit 2
        ;;
esac

LR="${LR:-5e-7}"
BACKBONE_LR="${BACKBONE_LR:-0.0}"
SAVE_FREQ="${SAVE_FREQ:-250}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ ! -d "$BASE_PRETRAINED" ]]; then
    echo "Missing pretrained checkpoint: $BASE_PRETRAINED"
    exit 1
fi
python scripts/garment_curriculum/validate_lerobot_dataset.py "$DATASET_ROOT" --min-episodes 10 >/dev/null

mkdir -p "$OUT_DIR"
CFG="$OUT_DIR/train_config_fresh.json"

python - <<EOF
import json
from pathlib import Path

base = Path("$BASE_PRETRAINED")
cfg = json.load(open(base / "train_config.json"))
cfg["dataset"]["root"] = "$DATASET_ROOT"
cfg["dataset"]["repo_id"] = "$REPO_ID"
cfg["dataset"]["image_transforms"]["enable"] = True
cfg["output_dir"] = "$OUT_DIR"
cfg["resume"] = False
cfg["checkpoint_path"] = None
cfg["steps"] = int("$EXTRA_STEPS")
cfg["save_freq"] = int("$SAVE_FREQ")
cfg["batch_size"] = int("$BATCH_SIZE")
cfg["save_checkpoint"] = True
cfg["job_name"] = "$REPO_ID"
cfg.setdefault("eval_freq", 0)
cfg["eval_freq"] = 0

policy = cfg.setdefault("policy", {})
policy["pretrained_path"] = str(base)
policy["optimizer_lr"] = float("$LR")
policy["optimizer_lr_backbone"] = float("$BACKBONE_LR")

if "optimizer" in cfg:
    cfg["optimizer"]["lr"] = float("$LR")

Path("$CFG").write_text(json.dumps(cfg, indent=4) + "\\n")
print("train_config:", "$CFG")
print("dataset:", cfg["dataset"]["root"])
print("pretrained:", policy["pretrained_path"])
print("steps:", cfg["steps"], "save_freq:", cfg["save_freq"])
print("lr:", policy["optimizer_lr"], "backbone_lr:", policy["optimizer_lr_backbone"])
EOF

LOG="/tmp/finetune_${CATEGORY}_variant_fresh.log"
echo "Launching fresh-optimizer lerobot-train; log=$LOG"
lerobot-train --config_path="$CFG" --resume=false 2>&1 | tee "$LOG"
