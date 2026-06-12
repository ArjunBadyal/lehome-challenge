#!/bin/bash
# Run the exact packaged submission policy on all four garment categories.
set -euo pipefail

cd "$(dirname "$0")/../.."

CATEGORIES=(top_short top_long pant_long pant_short)

echo "=== Submission bundle full eval ==="
echo "Started: $(date -Is)"
for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo "### ${cat}"
    bash scripts/router_experiments/eval_submission_bundle.sh "$cat"
done

echo ""
echo "=== Aggregate from logs ==="
python - <<'PY'
import re
from pathlib import Path

total_success = 0
total_episodes = 0

for cat in ["top_short", "top_long", "pant_long", "pant_short"]:
    log = Path(f"/tmp/submission_bundle_{cat}.log")
    text = log.read_text(errors="ignore") if log.exists() else ""
    matches = re.findall(r"Success Rate:\s+(\d+)/(\d+)\s+\(([0-9.]+)%\)", text)
    if not matches:
        print(f"{cat}: missing success-rate line ({log})")
        continue
    succ, eps, pct = matches[-1]
    succ_i, eps_i = int(succ), int(eps)
    total_success += succ_i
    total_episodes += eps_i
    print(f"{cat}: {succ_i}/{eps_i} = {float(pct):.2f}%")

if total_episodes:
    print(f"TOTAL: {total_success}/{total_episodes} = {100.0 * total_success / total_episodes:.2f}%")
PY
echo "Finished: $(date -Is)"
