#!/bin/bash
# Overnight fine-tune of top_short_aug 55K on the 10 harvested demos.
# Saves every 1K steps for +5K total so we have a sweep of checkpoints to
# compare in the morning.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

START_DIR="outputs/train/act_top_short_aug/checkpoints/055000"
OUT_DIR="outputs/train/act_top_short_finetune_seen9"
DATASET_ROOT="Datasets/teleop_recovery/top_short_seen_9_act_harvest/010"
START_STEP=55000
TARGET_STEP=60000
START_CKPT_OUT="${OUT_DIR}/checkpoints/$(printf '%06d' "$START_STEP")"

mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "$START_DIR" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

CFG="${START_CKPT_OUT}/pretrained_model/train_config.json"
python <<EOF
import json
cfg = json.load(open("$CFG"))
cfg["dataset"]["root"] = "$DATASET_ROOT"
cfg["dataset"]["repo_id"] = "top_short_seen9_finetune"
cfg["dataset"]["image_transforms"]["enable"] = True
cfg["output_dir"] = "$OUT_DIR"
cfg["resume"] = True
cfg["checkpoint_path"] = "${OUT_DIR}/checkpoints/last"
cfg["steps"] = $TARGET_STEP
cfg["save_freq"] = 1000  # frequent saves so we can pick best checkpoint
json.dump(cfg, open("$CFG", "w"), indent=4)
print(f"train_config: dataset={cfg['dataset']['root']}, steps={cfg['steps']}, save_freq={cfg['save_freq']}")
EOF

echo "Launching top_short fine-tune (55K -> 60K, +5K steps, save every 1K)"
echo "  output: $OUT_DIR"
echo "  log:    /tmp/finetune_top_short.log"
nohup lerobot-train \
    --config_path="$CFG" \
    --resume=true \
    > "/tmp/finetune_top_short.log" 2>&1 &
PID=$!
echo "  pid=$PID"
disown $PID
