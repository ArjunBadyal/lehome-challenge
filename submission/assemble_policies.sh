#!/bin/bash
# Assemble the policies/ directory that goes into the Docker image.
#
# This script copies checkpoints and dataset metadata from the workspace into
# ./policies/ with the layout the Dockerfile expects. Run from submission/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$ROOT/.."
OUT="$ROOT/policies"

echo "[assemble] Writing to $OUT"
rm -rf "$OUT"
mkdir -p "$OUT"

# ---------------- Classifier ----------------
mkdir -p "$OUT/classifier"
cp "$REPO_ROOT/outputs/classifier/garment_classifier.pt" "$OUT/classifier/"
echo "[assemble] copied classifier"

# ---------------- Specialists ---------------
declare -A SPECIALIST_DIRS=(
    # Submission stack v7 (Pant-Long balanced-weak 090250, 2026-04-30):
    #   - Top-Short / Top-Long: single image-aug specialist (v4 stack)
    #   - Pant-Long: balanced-weak tiny fine-tune at 090250
    #   - Pant-Short: static aug60. This replaced the noisy kNN portfolio
    #     after eval showed aug60 static at 81.67% vs v5 portfolio at 73.33%.
    ["top_short"]="outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model"
    ["top_long"]="outputs/train/act_top_long_aug/checkpoints/090000/pretrained_model"
    ["pant_long"]="outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model"
    ["pant_short"]="outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model"
)

# Optional legacy Pant-Short portfolio candidates. Disabled by default for v7
# because the final policy uses static aug60 and does not load these files.
COPY_PANTSHORT_PORTFOLIO="${COPY_PANTSHORT_PORTFOLIO:-0}"
declare -A PANTSHORT_PORTFOLIO_DIRS=(
    ["aug_50k"]="outputs/train/act_pant_short_aug/checkpoints/050000/pretrained_model"
    ["aug_55k"]="outputs/train/act_pant_short_aug/checkpoints/055000/pretrained_model"
    ["aug_60k"]="outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model"
    ["aug_65k"]="outputs/train/act_pant_short_aug/checkpoints/065000/pretrained_model"
)
for cat in top_long top_short pant_long pant_short; do
    src="$REPO_ROOT/${SPECIALIST_DIRS[$cat]}"
    dst="$OUT/specialists/$cat/pretrained_model"
    if [ -d "$src" ]; then
        mkdir -p "$(dirname "$dst")"
        cp -r "$src" "$dst"
        echo "[assemble] copied specialist: $cat"
    else
        echo "[assemble] WARNING: missing specialist source: $src"
    fi
done

# ---------------- Pant-Short portfolio extras ----------
if [ "$COPY_PANTSHORT_PORTFOLIO" = "1" ]; then
    for label in "${!PANTSHORT_PORTFOLIO_DIRS[@]}"; do
        src="$REPO_ROOT/${PANTSHORT_PORTFOLIO_DIRS[$label]}"
        dst="$OUT/portfolio/pant_short/${label}/pretrained_model"
        if [ -d "$src" ]; then
            mkdir -p "$(dirname "$dst")"
            cp -r "$src" "$dst"
            echo "[assemble] copied portfolio: pant_short/$label"
        else
            echo "[assemble] WARNING: missing portfolio src: $src"
        fi
    done

    # Portfolio metadata: seen-garment embeddings + per-garment best-ckpt lookup
    mkdir -p "$OUT/portfolio"
    for f in seen_embeddings_multicam.npz best_checkpoints.json; do
        src="$REPO_ROOT/outputs/portfolio_router/$f"
        if [ -f "$src" ]; then
            cp "$src" "$OUT/portfolio/$f"
            echo "[assemble] copied portfolio metadata: $f"
        fi
    done
else
    echo "[assemble] skipped legacy Pant-Short portfolio extras"
fi

# ---------------- Unified fallback ----------
unified_src="$REPO_ROOT/outputs/train/act_all_cats/checkpoints/last/pretrained_model"
if [ -d "$unified_src" ]; then
    mkdir -p "$OUT/unified"
    cp -r "$unified_src" "$OUT/unified/pretrained_model"
    echo "[assemble] copied unified fallback"
else
    echo "[assemble] WARNING: missing unified source: $unified_src"
fi

# ---------------- Dataset metadata ----------
# LeRobotDatasetMetadata needs meta/info.json, meta/stats.json, meta/tasks.parquet.
# We copy only meta/ (not data/ or videos/) — 10s of MBs instead of GBs.
for ds in top_long_merged top_short_merged pant_long_merged pant_short_merged four_types_merged; do
    src="$REPO_ROOT/Datasets/example/$ds/meta"
    dst="$OUT/datasets/$ds/meta"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        cp -r "$src"/* "$dst/"
        echo "[assemble] copied dataset meta: $ds"
    else
        echo "[assemble] WARNING: missing dataset meta: $src"
    fi
done

echo ""
echo "[assemble] Summary:"
du -sh "$OUT"/*
echo ""
echo "[assemble] Done. Build with: docker build -t lehome-submission ."
