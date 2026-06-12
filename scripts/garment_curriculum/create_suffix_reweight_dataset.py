#!/usr/bin/env python3
"""Create a small critical-suffix replay dataset from successful demos.

The output is a proper LeRobot dataset containing only the last N frames from a
uniform sample of source episodes. It is intended to be merged back with the
source dataset so ACT sees late fold/release frames slightly more often without
flooding training with narrow new demos.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", default="critical_suffix_reweight")
    parser.add_argument("--suffix-frames", type=int, default=80)
    parser.add_argument("--max-episodes", type=int, default=80)
    parser.add_argument("--min-episode-len", type=int, default=120)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--episode-indices",
        type=str,
        default=None,
        help="Optional comma-separated source episode indices. Overrides uniform sampling.",
    )
    return parser.parse_args()


def episode_table(root: Path):
    files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {root / 'meta/episodes'}")
    return pa.concat_tables([pq.read_table(path) for path in files]).to_pandas()


def choose_episodes(df, args: argparse.Namespace) -> list[int]:
    valid = df[df["length"].astype(int) >= args.min_episode_len].copy()
    if args.episode_indices:
        requested = [int(x) for x in args.episode_indices.split(",") if x.strip()]
        present = set(valid["episode_index"].astype(int).tolist())
        chosen = [idx for idx in requested if idx in present]
    else:
        eps = sorted(valid["episode_index"].astype(int).tolist())
        if args.max_episodes <= 0 or args.max_episodes >= len(eps):
            chosen = eps
        else:
            rng = np.random.default_rng(args.seed)
            chosen = sorted(rng.choice(eps, size=args.max_episodes, replace=False).tolist())
    if not chosen:
        raise ValueError("No episodes selected for suffix reweighting")
    return chosen


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def normalize_features(features: dict) -> dict:
    out = {}
    for key, spec in features.items():
        spec = dict(spec)
        if "shape" in spec:
            spec["shape"] = tuple(spec["shape"])
        out[key] = spec
    return out


def frame_value(key: str, value):
    arr = tensor_to_numpy(value)
    if key.startswith("observation.images.") and hasattr(arr, "shape"):
        # LeRobot returns decoded images as CHW tensors; the writer accepts HWC.
        if len(arr.shape) == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
    return arr


def main() -> None:
    args = parse_args()
    src = args.source_root.resolve()
    out = args.output_root.resolve()

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite to replace")
        shutil.rmtree(out)

    info = json.loads((src / "meta/info.json").read_text())
    features = normalize_features(info["features"])
    frame_feature_keys = [k for k in features.keys() if k not in DEFAULT_FEATURES]

    ds = LeRobotDataset(repo_id="source", root=src)
    episodes = episode_table(src)
    chosen = choose_episodes(episodes, args)

    out_ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=ds.fps,
        root=out,
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )

    written_eps = 0
    written_frames = 0
    chosen_rows = episodes.set_index(episodes["episode_index"].astype(int))
    for repeat_idx in range(args.repeat):
        for ep_idx in chosen:
            row = chosen_rows.loc[ep_idx]
            start = int(row["dataset_from_index"])
            stop = int(row["dataset_to_index"])
            length = stop - start
            suffix_len = min(args.suffix_frames, length)
            suffix_start = stop - suffix_len

            for global_idx in range(suffix_start, stop):
                item = ds[global_idx]
                frame = {key: frame_value(key, item[key]) for key in frame_feature_keys if key in item}
                frame["task"] = item.get("task", "fold the garment on the table")
                out_ds.add_frame(frame)
            out_ds.save_episode()
            written_eps += 1
            written_frames += suffix_len

    out_ds.finalize()

    manifest = {
        "source_root": str(src),
        "output_root": str(out),
        "suffix_frames": args.suffix_frames,
        "max_episodes": args.max_episodes,
        "repeat": args.repeat,
        "selected_source_episodes": chosen,
        "written_episodes": written_eps,
        "written_frames": written_frames,
    }
    (out / "meta/suffix_reweight_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
