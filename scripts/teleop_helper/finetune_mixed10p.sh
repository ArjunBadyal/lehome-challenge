#!/bin/bash
# Fine-tune category specialists on clean base+harvest merged datasets.
#
# Uses Datasets/teleop_merged_clean_10p/<category>_merged, where the harvested
# recovery demos are repeated 3x (~10% of episodes). This is intentionally safer
# than the failed demo-only fine-tune, which caused catastrophic forgetting.
#
# Usage:
#   bash scripts/teleop_helper/finetune_mixed10p.sh top_short [extra_steps]
#   bash scripts/teleop_helper/finetune_mixed10p.sh pant_long [extra_steps]
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
EXTRA_STEPS="${2:-3000}"

case "$CATEGORY" in
    top_short)
        START_DIR="outputs/train/act_top_short_aug/checkpoints/055000"
        OUT_DIR="outputs/train/act_top_short_mixed10p_seen9"
        DATASET_ROOT="Datasets/teleop_merged_clean_10p/top_short_merged"
        START_STEP=55000
        ;;
    pant_long)
        # Use the current submission specialist as the starting point.
        START_DIR="outputs/train/act_pant_long_aug/checkpoints/090000"
        OUT_DIR="outputs/train/act_pant_long_mixed10p_seen3"
        DATASET_ROOT="Datasets/teleop_merged_clean_10p/pant_long_merged"
        START_STEP=90000
        ;;
    *)
        echo "Usage: bash $0 {top_short|pant_long} [extra_steps]"
        exit 1
        ;;
esac

TARGET_STEP=$((START_STEP + EXTRA_STEPS))
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

echo "Setting up mixed fine-tune: $CATEGORY ${START_STEP} -> ${TARGET_STEP}"
mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "$START_DIR" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

python - <<EOF
import json
cfg_path = "$CFG"
cfg = json.load(open(cfg_path))
cfg["dataset"]["root"] = "$DATASET_ROOT"
cfg["dataset"]["repo_id"] = "${CATEGORY}_mixed10p_harvest"
cfg["dataset"]["image_transforms"]["enable"] = True
cfg["output_dir"] = "$OUT_DIR"
cfg["resume"] = True
cfg["checkpoint_path"] = "${OUT_DIR}/checkpoints/last"
cfg["steps"] = $TARGET_STEP
cfg["save_freq"] = 500
if "policy" in cfg:
    cfg["policy"]["optimizer_lr"] = 5e-6
    cfg["policy"]["optimizer_lr_backbone"] = 5e-6
if "optimizer" in cfg:
    cfg["optimizer"]["lr"] = 5e-6
json.dump(cfg, open(cfg_path, "w"), indent=4)
print(f"train_config: dataset={cfg['dataset']['root']}, steps={cfg['steps']}, save_freq={cfg['save_freq']}, optimizer_lr={cfg.get('optimizer', {}).get('lr')}")
EOF

LOG="/tmp/finetune_${CATEGORY}_mixed10p.log"
echo "Launching lerobot-train; log=$LOG"
nohup lerobot-train --config_path="$CFG" --resume=true > "$LOG" 2>&1 &
echo "pid=$!"
