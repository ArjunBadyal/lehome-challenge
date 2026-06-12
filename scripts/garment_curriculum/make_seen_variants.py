#!/usr/bin/env python3
"""Create training-only Seen garment scale/yaw curriculum variants.

The variants are JSON-only wrappers around the original USD assets. They use
uniform scale factors so the existing success checker remains valid: challenge
thresholds are multiplied by `init_scale[0]`.

Default output:
  Assets/objects/Challenge_Garment/Release/<Category>/<VariantName>/*.json
  outputs/garment_curriculum/seen_variant_manifest.jsonl
  outputs/garment_curriculum/eval_lists/<category>_variants.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCES = {
    "top_short": [
        "Top_Short_Seen_0",
        "Top_Short_Seen_2",
        "Top_Short_Seen_5",
        "Top_Short_Seen_9",
    ],
    "pant_long": [
        "Pant_Long_Seen_0",
        "Pant_Long_Seen_3",
        "Pant_Long_Seen_8",
        "Pant_Long_Seen_9",
    ],
}

CATEGORY_DIR = {
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(DEFAULT_SOURCES),
        default=sorted(DEFAULT_SOURCES),
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional explicit garment names. If set, all must match one category.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path("Assets/objects/Challenge_Garment"),
    )
    parser.add_argument("--version", default="Release")
    parser.add_argument(
        "--scale-factors",
        nargs="+",
        type=float,
        default=[0.96, 1.04, 0.92, 1.08],
        help="Uniform scale factors relative to the source JSON scale.",
    )
    parser.add_argument(
        "--yaw-offsets",
        nargs="+",
        type=float,
        default=[-10.0, -5.0, 5.0, 10.0],
        help="Degrees added to both min/max z Euler reset ranges.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/garment_curriculum/seen_variant_manifest.jsonl"),
    )
    parser.add_argument(
        "--eval-list-dir",
        type=Path,
        default=Path("outputs/garment_curriculum/eval_lists"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def category_from_garment(garment_name: str) -> str:
    if garment_name.startswith("Top_Short_"):
        return "top_short"
    if garment_name.startswith("Pant_Long_"):
        return "pant_long"
    raise ValueError(f"Unsupported curriculum garment: {garment_name}")


def source_dir(base_path: Path, version: str, garment_name: str) -> Path:
    category = CATEGORY_DIR[category_from_garment(garment_name)]
    return base_path / version / category / garment_name


def load_single_json(root: Path) -> tuple[Path, dict]:
    files = sorted(root.glob("*.json"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one JSON in {root}, found {len(files)}")
    return files[0], json.loads(files[0].read_text())


def scale_tag(scale_factor: float) -> str:
    return f"s{int(round(scale_factor * 100)):03d}"


def yaw_tag(yaw: float) -> str:
    prefix = "p" if yaw >= 0 else "m"
    return f"yaw{prefix}{int(round(abs(yaw))):02d}"


def variant_name(source: str, scale_factor: float, yaw: float) -> str:
    return f"{source}_Curr_{scale_tag(scale_factor)}_{yaw_tag(yaw)}"


def shifted_range(values: Iterable[float], yaw: float) -> list[float]:
    out = [float(v) for v in values]
    if len(out) != 6:
        raise ValueError(f"Rotation range must have 6 values, got {out}")
    out[2] += yaw
    out[5] += yaw
    return [round(v, 6) for v in out]


def make_variant_config(source_name: str, src_cfg: dict, scale_factor: float, yaw: float) -> dict:
    cfg = dict(src_cfg)
    src_scale = [float(x) for x in src_cfg["scale"]]
    if len(set(round(x, 8) for x in src_scale)) != 1:
        raise ValueError(f"{source_name} is not uniformly scaled already: {src_scale}")
    cfg["scale"] = [round(float(x) * scale_factor, 6) for x in src_scale]
    cfg["initial_rot_range"] = shifted_range(src_cfg["initial_rot_range"], yaw)
    cfg["soft_reset_rot_range"] = shifted_range(src_cfg["soft_reset_rot_range"], yaw)
    cfg["_curriculum_source"] = source_name
    cfg["_curriculum_uniform_scale_factor"] = scale_factor
    cfg["_curriculum_yaw_offset_deg"] = yaw
    return cfg


def main() -> None:
    args = parse_args()
    manifest_rows: list[dict] = []
    by_category: dict[str, list[str]] = {cat: [] for cat in args.categories}

    if args.sources:
        sources_by_cat: dict[str, list[str]] = {}
        for garment in args.sources:
            sources_by_cat.setdefault(category_from_garment(garment), []).append(garment)
    else:
        sources_by_cat = {
            cat: DEFAULT_SOURCES[cat]
            for cat in args.categories
        }

    for category, sources in sources_by_cat.items():
        category_dir = CATEGORY_DIR[category]
        for source_name in sources:
            src_dir = source_dir(args.base_path, args.version, source_name)
            src_json, src_cfg = load_single_json(src_dir)
            for scale_factor in args.scale_factors:
                for yaw in args.yaw_offsets:
                    name = variant_name(source_name, scale_factor, yaw)
                    dst_dir = args.base_path / args.version / category_dir / name
                    dst_json = dst_dir / f"{src_json.stem}_{scale_tag(scale_factor)}_{yaw_tag(yaw)}.json"
                    row = {
                        "category": category,
                        "source": source_name,
                        "variant": name,
                        "scale_factor": scale_factor,
                        "yaw_offset_deg": yaw,
                        "json": str(dst_json),
                    }
                    manifest_rows.append(row)
                    by_category.setdefault(category, []).append(name)

                    if args.dry_run:
                        print(f"DRY {source_name} -> {name}")
                        continue
                    if dst_dir.exists():
                        if not args.overwrite:
                            print(f"SKIP exists: {dst_dir}")
                            continue
                        shutil.rmtree(dst_dir)
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    cfg = make_variant_config(source_name, src_cfg, scale_factor, yaw)
                    dst_json.write_text(json.dumps(cfg, indent=4) + "\n")
                    print(f"WROTE {dst_json}")

    if not args.dry_run:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")
        args.eval_list_dir.mkdir(parents=True, exist_ok=True)
        for category, variants in by_category.items():
            if not variants:
                continue
            path = args.eval_list_dir / f"{category}_variants.txt"
            path.write_text("\n".join(sorted(set(variants))) + "\n")
            print(f"WROTE {path} ({len(set(variants))} variants)")
        print(f"WROTE {args.manifest} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
