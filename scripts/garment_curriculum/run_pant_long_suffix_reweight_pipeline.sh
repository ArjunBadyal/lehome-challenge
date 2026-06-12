#!/bin/bash
# Conservative Pant-Long suffix-reweight experiment.
#
# This keeps the v7 Pant-Long checkpoint as the start point, duplicates only
# critical late frames from successful broad replay episodes, merges them back
# into the full replay set, then fine-tunes in tiny +250-step increments.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

SOURCE_ROOT="${SOURCE_ROOT:-Datasets/teleop_merged_clean_balanced_weak/pant_long_merged}"
SUFFIX_ROOT="${SUFFIX_ROOT:-Datasets/garment_curriculum/suffix_reweight/pant_long_suffix_80f_80ep}"
MERGED_ROOT="${MERGED_ROOT:-Datasets/garment_curriculum/merged/pant_long_suffix_reweight}"
OUT_DIR="${OUT_DIR:-outputs/train/act_pant_long_suffix_reweight}"

SUFFIX_FRAMES="${SUFFIX_FRAMES:-80}"
MAX_EPISODES="${MAX_EPISODES:-80}"
REPEAT="${REPEAT:-1}"
EXTRA_STEPS="${EXTRA_STEPS:-1000}"
LR="${LR:-1e-6}"
RUN_EVAL="${RUN_EVAL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

START_DIR="outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250"
START_STEP=90250
TARGET_STEP=$((START_STEP + EXTRA_STEPS))
START_CKPT_OUT="${OUT_DIR}/checkpoints/$(printf '%06d' "$START_STEP")"
CFG="${START_CKPT_OUT}/pretrained_model/train_config.json"

echo "[suffix] source=$SOURCE_ROOT"
echo "[suffix] suffix_root=$SUFFIX_ROOT frames=$SUFFIX_FRAMES episodes=$MAX_EPISODES repeat=$REPEAT"
echo "[suffix] merged_root=$MERGED_ROOT"
echo "[suffix] out_dir=$OUT_DIR steps=${START_STEP}->${TARGET_STEP} lr=$LR"

if [[ ! -d "$SOURCE_ROOT" ]]; then
    echo "Missing source dataset: $SOURCE_ROOT" >&2
    exit 1
fi
if [[ ! -d "$START_DIR" ]]; then
    echo "Missing start checkpoint: $START_DIR" >&2
    exit 1
fi

python scripts/garment_curriculum/validate_lerobot_dataset.py "$SOURCE_ROOT" --min-episodes 50

python scripts/garment_curriculum/create_suffix_reweight_dataset.py \
    --source-root "$SOURCE_ROOT" \
    --output-root "$SUFFIX_ROOT" \
    --repo-id pant_long_critical_suffix \
    --suffix-frames "$SUFFIX_FRAMES" \
    --max-episodes "$MAX_EPISODES" \
    --repeat "$REPEAT" \
    --overwrite

python scripts/garment_curriculum/validate_lerobot_dataset.py "$SUFFIX_ROOT" --min-episodes 1

python scripts/teleop_helper/clean_merge_lerobot.py \
    --output-root "$MERGED_ROOT" \
    --repo-id pant_long_suffix_reweight \
    --sources "$SOURCE_ROOT" "$SUFFIX_ROOT" \
    --overwrite

python scripts/garment_curriculum/validate_lerobot_dataset.py "$MERGED_ROOT" --min-episodes 50

if [[ "$SKIP_TRAIN" == "1" ]]; then
    echo "[suffix] SKIP_TRAIN=1; stopping after merged dataset creation."
    exit 0
fi

mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "$START_DIR" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

python - <<EOF
import json
cfg_path = "$CFG"
cfg = json.load(open(cfg_path))
cfg["dataset"]["root"] = "$MERGED_ROOT"
cfg["dataset"]["repo_id"] = "pant_long_suffix_reweight"
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
    f"steps={cfg['steps']}, "
    f"save_freq={cfg['save_freq']}, "
    f"lr={cfg.get('optimizer', {}).get('lr')}"
)
EOF

LOG="/tmp/finetune_pant_long_suffix_reweight.log"
echo "[suffix] launching lerobot-train; log=$LOG"
lerobot-train --config_path="$CFG" --resume=true 2>&1 | tee "$LOG"

if [[ "$RUN_EVAL" == "1" ]]; then
    RUN_DIR="$OUT_DIR" bash scripts/garment_curriculum/eval_curriculum_checkpoints.sh pant_long
fi
