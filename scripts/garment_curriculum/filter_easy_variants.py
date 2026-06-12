#!/usr/bin/env python3
"""Filter curriculum variant manifests to the easy scale/yaw subset.

The easy subset is intentionally narrow: small uniform scale changes and small
yaw offsets. These variants are meant for ACT self-imitation harvesting, not
for broad morphology exploration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/garment_curriculum/seen_variant_manifest.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/garment_curriculum/eval_lists"),
    )
    parser.add_argument("--categories", nargs="+", default=["top_short", "pant_long"])
    parser.add_argument("--scales", nargs="+", type=float, default=[0.96, 1.04])
    parser.add_argument("--yaws", nargs="+", type=float, default=[-5.0, 5.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scale_set = {round(v, 6) for v in args.scales}
    yaw_set = {round(v, 6) for v in args.yaws}
    wanted_categories = set(args.categories)
    by_category: dict[str, list[str]] = {cat: [] for cat in args.categories}

    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            category = row["category"]
            if category not in wanted_categories:
                continue
            if round(float(row["scale_factor"]), 6) not in scale_set:
                continue
            if round(float(row["yaw_offset_deg"]), 6) not in yaw_set:
                continue
            by_category.setdefault(category, []).append(row["variant"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for category, names in by_category.items():
        path = args.out_dir / f"{category}_easy_variants.txt"
        unique = sorted(set(names))
        path.write_text("\n".join(unique) + ("\n" if unique else ""))
        print(f"WROTE {path} ({len(unique)} variants)")


if __name__ == "__main__":
    main()
