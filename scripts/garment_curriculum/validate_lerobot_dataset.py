#!/usr/bin/env python3
"""Validate local LeRobot datasets used for curriculum fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--min-episodes", type=int, default=1)
    parser.add_argument("--require-videos", action="store_true", default=True)
    return parser.parse_args()


def parquet_rows(path: Path) -> int:
    return pq.read_table(path).num_rows


def validate(root: Path, min_episodes: int, require_videos: bool) -> tuple[bool, str]:
    info_path = root / "meta/info.json"
    tasks_path = root / "meta/tasks.parquet"
    if not info_path.exists():
        return False, f"missing {info_path}"
    if not tasks_path.exists():
        return False, f"missing {tasks_path}"

    info = json.loads(info_path.read_text())
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    if total_episodes < min_episodes:
        return False, f"episodes {total_episodes} < {min_episodes}"
    if total_frames <= 0:
        return False, "total_frames <= 0"

    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not data_files:
        return False, "no data parquet files"
    if not episode_files:
        return False, "no episode metadata parquet files"

    rows = sum(parquet_rows(path) for path in data_files)
    episode_rows = sum(parquet_rows(path) for path in episode_files)
    if rows != total_frames:
        return False, f"data rows {rows} != info.total_frames {total_frames}"
    if episode_rows != total_episodes:
        return False, f"episode rows {episode_rows} != info.total_episodes {total_episodes}"

    if require_videos:
        video_roots = sorted((root / "videos").glob("observation.images.*"))
        if not video_roots:
            return False, "no video roots"
        for video_root in video_roots:
            if not list(video_root.glob("chunk-*/*.mp4")):
                return False, f"missing videos under {video_root}"

    return True, f"episodes={total_episodes} frames={total_frames}"


def main() -> None:
    args = parse_args()
    failed = 0
    for root in args.roots:
        ok, msg = validate(root, args.min_episodes, args.require_videos)
        print(f"{'OK' if ok else 'FAIL'} {root}: {msg}")
        failed += int(not ok)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
