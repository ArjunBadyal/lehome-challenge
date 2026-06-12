#!/bin/bash
# Retrain a category specialist with the teleop-merged dataset.
# Usage: bash scripts/teleop_helper/retrain.sh <category> [--steps 70000]
#
# Resumes from the current best checkpoint per category, with image augmentation
# enabled. Saves every 5K steps. After training, eval each checkpoint on 5-ep CPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
EXTRA_STEPS="${2:-15000}"  # default: train for +15K more steps

if [[ -z "$CATEGORY" ]]; then
    echo "Usage: bash $0 <category> [extra_steps]"
    echo "  category ∈ {top_short, top_long, pant_long, pant_short}"
    exit 1
fi

# Map category → starting checkpoint and target steps
case "$CATEGORY" in
    top_short)
        START_DIR="outputs/train/act_top_short_aug/checkpoints/055000"
        OUT_DIR="outputs/train/act_top_short_teleop"
        START_STEP=55000
        ;;
    top_long)
        START_DIR="outputs/train/act_top_long_aug/checkpoints/090000"
        OUT_DIR="outputs/train/act_top_long_teleop"
        START_STEP=90000
        ;;
    pant_long)
        START_DIR="outputs/train/act_pant_long_aug/checkpoints/090000"
        OUT_DIR="outputs/train/act_pant_long_teleop"
        START_STEP=90000
        ;;
    pant_short)
        START_DIR="golden_checkpoints/pant_short_45k"
        OUT_DIR="outputs/train/act_pant_short_teleop"
        START_STEP=45000
        ;;
    *) echo "Invalid category: $CATEGORY"; exit 1 ;;
esac

TARGET_STEP=$((START_STEP + EXTRA_STEPS))
DATASET_ROOT="Datasets/teleop_merged/${CATEGORY}_merged"
START_CKPT_OUT="${OUT_DIR}/checkpoints/$(printf '%06d' "$START_STEP")"

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Merged dataset not found: $DATASET_ROOT"
    echo "Run: bash scripts/teleop_helper/merge_teleop.sh $CATEGORY"
    exit 1
fi
if [[ ! -d "$START_DIR" ]]; then
    echo "Starting checkpoint not found: $START_DIR"
    exit 1
fi

# Copy starting checkpoint into the new output dir + edit train_config
echo "Setting up retrain: ${CATEGORY} (${START_STEP} -> ${TARGET_STEP})"
mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "$START_DIR" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

# Edit train_config.json to point at new dataset, enable aug, target steps
CFG="${START_CKPT_OUT}/pretrained_model/train_config.json"
python <<EOF
import json
cfg = json.load(open("$CFG"))
cfg["dataset"]["root"] = "$DATASET_ROOT"
cfg["dataset"]["repo_id"] = "${CATEGORY}_with_teleop"
cfg["dataset"]["image_transforms"]["enable"] = True
cfg["output_dir"] = "$OUT_DIR"
cfg["resume"] = True
cfg["checkpoint_path"] = "${OUT_DIR}/checkpoints/last"
cfg["steps"] = $TARGET_STEP
cfg["save_freq"] = 5000
json.dump(cfg, open("$CFG", "w"), indent=4)
print(f"train_config.json updated: dataset={cfg['dataset']['root']}, steps={cfg['steps']}")
EOF

echo ""
echo "Launching retrain (this will take a few hours on GPU)..."
nohup lerobot-train \
    --config_path="$CFG" \
    --resume=true \
    > "/tmp/retrain_${CATEGORY}_teleop.log" 2>&1 &
RETRAIN_PID=$!
echo "  pid=$RETRAIN_PID, log=/tmp/retrain_${CATEGORY}_teleop.log"
echo ""
echo "When complete, eval each checkpoint with: bash scripts/teleop_helper/eval_checkpoints.sh $CATEGORY"
