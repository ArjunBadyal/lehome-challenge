#!/bin/bash
# Overnight orchestrator:
#   1. Wait for top_short fine-tune (already running) to finish
#   2. Re-collect 5 pant_long demos (the prior 5 had a corrupted parquet)
#   3. Fine-tune pant_long on those demos
# All sequential to avoid GPU contention with training.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate

LOG=/tmp/overnight_queue.log
exec >>"$LOG" 2>&1
echo "=== OVERNIGHT QUEUE START $(date) ==="

# Step 1: wait for top_short fine-tune
echo "[step 1] waiting for top_short finetune (lerobot-train) to exit..."
while pgrep -f "act_top_short_finetune_seen9" >/dev/null 2>&1 || \
      pgrep -f "lerobot-train" | xargs -I{} grep -l "act_top_short_finetune_seen9" /proc/{}/cmdline 2>/dev/null | grep -q .; do
    sleep 60
done
# Belt-and-braces: any lerobot-train still alive? wait it out.
while pgrep -f "lerobot-train" >/dev/null 2>&1; do
    sleep 60
done
echo "[step 1] top_short finetune complete at $(date)"

# Step 2: re-collect pant_long demos (5 successes target)
echo "[step 2] launching pant_long harvester (target 5)..."
rm -f /tmp/lehome_ik_command.json /tmp/lehome_ik_status.json
NUM_EPISODE=5 bash scripts/teleop_helper/harvest_act_pant_long_seen_3.sh
echo "[step 2] pant_long harvester completed at $(date)"

# Find the latest finalized harvester dir
HARVEST_ROOT="Datasets/teleop_recovery/pant_long_seen_3_act_harvest"
LATEST=""
for d in $(ls -t "$HARVEST_ROOT" 2>/dev/null); do
    if [[ -f "$HARVEST_ROOT/$d/meta/episodes/chunk-000/file-000.parquet" ]]; then
        LATEST="$HARVEST_ROOT/$d"
        break
    fi
done
if [[ -z "$LATEST" ]]; then
    echo "[step 2 FAIL] no finalized pant_long harvest dir found; aborting"
    exit 1
fi
echo "[step 2] using harvest dir: $LATEST"

# Strip any depth columns
.venv/bin/python <<EOF
import pyarrow.parquet as pq
import json
from pathlib import Path
root = Path("$LATEST")
info_path = root / "meta" / "info.json"
info = json.loads(info_path.read_text())
if "observation.top_depth" in info.get("features", {}):
    del info["features"]["observation.top_depth"]
    info_path.write_text(json.dumps(info, indent=4))
    print("stripped depth from info.json")
for p in (root / "data").rglob("*.parquet"):
    t = pq.read_table(str(p))
    if "observation.top_depth" in t.column_names:
        pq.write_table(t.drop("observation.top_depth"), str(p))
        print(f"stripped depth from {p}")
EOF

# Step 3: fine-tune pant_long
echo "[step 3] launching pant_long fine-tune..."
START_DIR="golden_checkpoints/pant_long_80k/pretrained_model"
OUT_DIR="outputs/train/act_pant_long_finetune_seen3"
START_STEP=80000
TARGET_STEP=85000
START_CKPT_OUT="${OUT_DIR}/checkpoints/$(printf '%06d' "$START_STEP")"

mkdir -p "${OUT_DIR}/checkpoints"
cp -rn "golden_checkpoints/pant_long_80k" "$START_CKPT_OUT"
chmod -R u+w "$START_CKPT_OUT"
ln -sfn "$(printf '%06d' "$START_STEP")" "${OUT_DIR}/checkpoints/last"

CFG="${START_CKPT_OUT}/pretrained_model/train_config.json"
.venv/bin/python <<EOF
import json
cfg = json.load(open("$CFG"))
cfg["dataset"]["root"] = "$LATEST"
cfg["dataset"]["repo_id"] = "pant_long_seen3_finetune"
cfg["dataset"]["image_transforms"]["enable"] = True
cfg["output_dir"] = "$OUT_DIR"
cfg["resume"] = True
cfg["checkpoint_path"] = "${OUT_DIR}/checkpoints/last"
cfg["steps"] = $TARGET_STEP
cfg["save_freq"] = 1000
json.dump(cfg, open("$CFG", "w"), indent=4)
print(f"pant_long config: dataset={cfg['dataset']['root']}, steps={cfg['steps']}")
EOF

lerobot-train --config_path="$CFG" --resume=true
echo "[step 3] pant_long fine-tune complete at $(date)"

echo "=== OVERNIGHT QUEUE END $(date) ==="
