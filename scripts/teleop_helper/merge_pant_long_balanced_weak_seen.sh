#!/bin/bash
# Build a clean Pant-Long replay dataset with broad Seen coverage.
#
# Source mix:
#   - existing clean diverse Pant-Long merge: base + small Seen_0/Seen_3 additions
#   - newly harvested weak Seen demos: Seen_9, Seen_8, Seen_6 when available
#
# This avoids the previous failure mode where one target garment dominated the
# replay data and the policy forgot holdouts.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

BASE_ROOT="${BASE_ROOT:-Datasets/teleop_merged_clean_diverse/pant_long_merged}"
WEAK_BASES="${WEAK_BASES:-Datasets/teleop_recovery/pant_long_weak_seen_act_harvest Datasets/teleop_recovery/pant_long_weak_seen_retracted_act_harvest}"
EVAL_SUCCESS_BASES="${EVAL_SUCCESS_BASES:-Datasets/eval_success/pant_long_aug90_seen_success}"
OUT_ROOT="${OUT_ROOT:-Datasets/teleop_merged_clean_balanced_weak/pant_long_merged}"
REPO_ID="${REPO_ID:-pant_long_balanced_weak_seen}"

if [[ ! -d "$BASE_ROOT" ]]; then
    echo "Missing base replay dataset: $BASE_ROOT"
    exit 1
fi

sources=("$BASE_ROOT")

datasets_for() {
    local garment="$1"
    for weak_base in ${WEAK_BASES}; do
        find "${weak_base}/${garment}" -mindepth 2 -maxdepth 3 -type f -path '*/meta/info.json' 2>/dev/null \
            | sort \
            | while read -r info; do
                dirname "$(dirname "$info")"
            done
    done
}

for garment in Pant_Long_Seen_9 Pant_Long_Seen_8 Pant_Long_Seen_6; do
    found=0
    while IFS= read -r ds; do
        [[ -z "$ds" ]] && continue
        if [[ ! -f "$ds/meta/tasks.parquet" ]]; then
            echo "Skipping incomplete harvest dir (no tasks.parquet): $ds"
            continue
        fi
        found=1
        sources+=("$ds")
    done < <(datasets_for "$garment")
    if [[ "$found" -eq 0 ]]; then
        echo "Skipping missing weak-seen harvest: $garment"
    fi
done

for eval_base in ${EVAL_SUCCESS_BASES}; do
    while IFS= read -r info; do
        [[ -z "$info" ]] && continue
        sources+=("$(dirname "$(dirname "$info")")")
    done < <(find "$eval_base" -mindepth 2 -maxdepth 3 -type f -path '*/meta/info.json' 2>/dev/null | sort)
done

echo "Building balanced Pant-Long merge:"
echo "Weak bases:"
printf '  %s\n' ${WEAK_BASES}
echo "Eval-success bases:"
printf '  %s\n' ${EVAL_SUCCESS_BASES}
echo "Sources:"
printf '  %s\n' "${sources[@]}"
echo "Output: $OUT_ROOT"

python scripts/teleop_helper/clean_merge_lerobot.py \
    --output-root "$OUT_ROOT" \
    --repo-id "$REPO_ID" \
    --sources "${sources[@]}" \
    --overwrite

python - <<'PY'
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("Datasets/teleop_merged_clean_balanced_weak/pant_long_merged")
ds = LeRobotDataset("pant_long_balanced_weak_seen", root=root)
print(f"merged episodes={ds.num_episodes} frames={ds.num_frames}")
PY
