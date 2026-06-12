#!/usr/bin/env python3
"""Create simple safetensors weight soups for compatible LeRobot checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_weighted(value: str) -> tuple[float, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Use WEIGHT:CHECKPOINT_PRETRAINED_MODEL")
    weight_s, path_s = value.split(":", 1)
    weight = float(weight_s)
    path = Path(path_s)
    if not (path / "model.safetensors").exists():
        raise argparse.ArgumentTypeError(f"missing model.safetensors under {path}")
    return weight, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-template",
        type=Path,
        required=True,
        help="Checkpoint pretrained_model dir to copy config/processors from.",
    )
    parser.add_argument(
        "members",
        nargs="+",
        type=parse_weighted,
        help="Weighted member as WEIGHT:path/to/pretrained_model.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {args.output}")
        shutil.rmtree(args.output)
    shutil.copytree(args.base_template, args.output)

    total = sum(weight for weight, _ in args.members)
    if total <= 0:
        raise SystemExit("Total soup weight must be positive")
    weights = [(weight / total, path) for weight, path in args.members]

    accum: dict[str, torch.Tensor] = {}
    expected_keys: set[str] | None = None
    for norm_weight, path in weights:
        tensors = load_file(path / "model.safetensors")
        keys = set(tensors)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            missing = sorted(expected_keys - keys)
            extra = sorted(keys - expected_keys)
            raise SystemExit(f"Incompatible tensors for {path}: missing={missing[:5]} extra={extra[:5]}")
        for key, tensor in tensors.items():
            value = tensor.float() * norm_weight
            accum[key] = value if key not in accum else accum[key] + value

    save_file(accum, args.output / "model.safetensors")
    metadata = {
        "members": [
            {"weight": weight, "path": str(path)}
            for weight, path in args.members
        ],
        "normalized_members": [
            {"weight": weight, "path": str(path)}
            for weight, path in weights
        ],
    }
    (args.output / "soup_metadata.json").write_text(json.dumps(metadata, indent=4) + "\n")
    print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
