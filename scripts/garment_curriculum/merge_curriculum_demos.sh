#!/bin/bash
# Merge base ACT replay with validated curriculum demos.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

CATEGORY="${1:-}"
case "$CATEGORY" in
    top_short)
        BASE_DATASET="${BASE_DATASET:-Datasets/example/top_short_merged}"
        DEMO_BASE="${DEMO_BASE:-Datasets/garment_curriculum/residual_success/top_short}"
        OUT_ROOT="${OUT_ROOT:-Datasets/garment_curriculum/merged/top_short_merged}"
        REPO_ID="top_short_curriculum"
        ;;
    pant_long)
        BASE_DATASET="${BASE_DATASET:-Datasets/teleop_merged_clean_balanced_weak/pant_long_merged}"
        DEMO_BASE="${DEMO_BASE:-Datasets/garment_curriculum/residual_success/pant_long}"
        OUT_ROOT="${OUT_ROOT:-Datasets/garment_curriculum/merged/pant_long_merged}"
        REPO_ID="pant_long_curriculum"
        ;;
    *)
        echo "Usage: $0 <top_short|pant_long>"
        exit 2
        ;;
esac

if [[ ! -d "$BASE_DATASET" ]]; then
    echo "Missing base dataset: $BASE_DATASET"
    exit 1
fi

sources=("$BASE_DATASET")
while IFS= read -r ds; do
    [[ -f "$ds/meta/tasks.parquet" ]] || continue
    python scripts/garment_curriculum/validate_lerobot_dataset.py "$ds" >/dev/null
    sources+=("$ds")
done < <(find "$DEMO_BASE" -mindepth 2 -maxdepth 2 -type d -name '[0-9][0-9][0-9]' 2>/dev/null | sort)

if [[ "${#sources[@]}" -le 1 ]]; then
    echo "No valid curriculum demo datasets found under $DEMO_BASE"
    exit 1
fi

echo "=== Merge curriculum demos: $CATEGORY ==="
printf '  %s\n' "${sources[@]}"

python scripts/teleop_helper/clean_merge_lerobot.py \
    --output-root "$OUT_ROOT" \
    --repo-id "$REPO_ID" \
    --overwrite \
    --sources "${sources[@]}"

python scripts/garment_curriculum/validate_lerobot_dataset.py "$OUT_ROOT" --min-episodes 10
echo "Merged dataset: $OUT_ROOT"
