#!/usr/bin/env python3
"""Safely merge local LeRobot datasets without video concatenation.

The upstream LeRobot aggregate helper concatenates MP4 files and rewrites
episode timestamp offsets. On these challenge datasets that produced metadata
windows past the end of the concatenated video, causing training crashes like:

    Invalid frame index=12009 ... must be less than 10908

This merger is intentionally conservative:
  * data parquet files are copied to unique destination file indices;
  * video files are copied to unique destination file indices, never concatenated;
  * episode metadata is rewritten to point at those copied files while preserving
    per-source timestamps.

It is meant for local base+harvest fine-tuning datasets, not Hub publishing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import load_stats, load_tasks, write_stats, write_tasks


VIDEO_KEYS = (
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--sources", required=True, nargs="+", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output root before writing. Use only for generated merge outputs.",
    )
    return parser.parse_args()


def load_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta/info.json").read_text())


def write_info(root: Path, info: dict[str, Any]) -> None:
    out = root / "meta/info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=4) + "\n")


def parquet_files(root: Path, subdir: str) -> list[Path]:
    return sorted((root / subdir).glob("chunk-*/*.parquet"))


def video_files(root: Path, video_key: str) -> list[Path]:
    return sorted((root / "videos" / video_key).glob("chunk-*/*.mp4"))


def parse_chunk_file(path: Path) -> tuple[int, int]:
    return int(path.parent.name.split("-")[1]), int(path.stem.split("-")[1])


def validate_compatible(sources: list[Path]) -> None:
    infos = [load_info(src) for src in sources]
    first = infos[0]
    for src, info in zip(sources[1:], infos[1:], strict=False):
        for key in ("fps", "robot_type", "data_path", "video_path"):
            if info.get(key) != first.get(key):
                raise ValueError(f"{src} has incompatible {key}")
        # Recorded recovery datasets may contain extra observation channels
        # (notably observation.top_depth). Training should use the base schema,
        # so extras are dropped later in copy_data_files; common base features
        # must still match exactly.
        src_features = info.get("features", {})
        for feature_name, feature_spec in first.get("features", {}).items():
            src_spec = src_features.get(feature_name)
            if src_spec is None:
                raise ValueError(
                    f"{src} has incompatible feature {feature_name}: "
                    f"{src_spec} != {feature_spec}"
                )
            ref = {k: v for k, v in feature_spec.items() if k != "info"}
            cand = {k: v for k, v in src_spec.items() if k != "info"}
            if cand != ref:
                raise ValueError(
                    f"{src} has incompatible feature {feature_name}: "
                    f"{src_spec} != {feature_spec}"
                )


def merge_tasks(sources: list[Path]) -> tuple[pd.DataFrame, list[dict[int, int]]]:
    names: OrderedDict[str, None] = OrderedDict()
    src_tasks: list[pd.DataFrame] = []
    for src in sources:
        tasks = load_tasks(src)
        src_tasks.append(tasks)
        for name in tasks.index.tolist():
            names.setdefault(str(name), None)

    merged = pd.DataFrame({"task_index": range(len(names))}, index=list(names))

    maps: list[dict[int, int]] = []
    for tasks in src_tasks:
        mapping: dict[int, int] = {}
        for task_name, row in tasks.iterrows():
            mapping[int(row["task_index"])] = int(merged.loc[task_name, "task_index"])
        maps.append(mapping)
    return merged, maps


def copy_videos(
    source: Path,
    output: Path,
    video_key: str,
    next_file_index: int,
) -> tuple[dict[tuple[int, int], tuple[int, int]], int]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for src_file in video_files(source, video_key):
        src_chunk, src_file_idx = parse_chunk_file(src_file)
        dst_chunk, dst_file_idx = 0, next_file_index
        dst_file = output / "videos" / video_key / "chunk-000" / f"file-{dst_file_idx:03d}.mp4"
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        mapping[(src_chunk, src_file_idx)] = (dst_chunk, dst_file_idx)
        next_file_index += 1
    return mapping, next_file_index


def copy_data_files(
    source: Path,
    output: Path,
    next_file_index: int,
    episode_offset: int,
    frame_offset: int,
    task_map: dict[int, int],
    keep_columns: set[str],
) -> tuple[dict[tuple[int, int], tuple[int, int]], int]:
    mapping: dict[tuple[int, int], tuple[int, int]] = {}
    for src_file in parquet_files(source, "data"):
        src_chunk, src_file_idx = parse_chunk_file(src_file)
        table = pq.read_table(src_file)
        df = table.to_pandas()
        extra_cols = [c for c in df.columns if c not in keep_columns]
        if extra_cols:
            df = df.drop(columns=extra_cols)
        df["episode_index"] = df["episode_index"].astype("int64") + episode_offset
        df["index"] = df["index"].astype("int64") + frame_offset
        df["task_index"] = df["task_index"].map(lambda x: task_map[int(x)]).astype("int64")

        dst_chunk, dst_file_idx = 0, next_file_index
        dst_file = output / "data" / "chunk-000" / f"file-{dst_file_idx:03d}.parquet"
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst_file, index=False)
        mapping[(src_chunk, src_file_idx)] = (dst_chunk, dst_file_idx)
        next_file_index += 1
    return mapping, next_file_index


def merge_episode_metadata(
    source: Path,
    episode_offset: int,
    frame_offset: int,
    data_map: dict[tuple[int, int], tuple[int, int]],
    video_maps: dict[str, dict[tuple[int, int], tuple[int, int]]],
    task_map: dict[int, int],
) -> pd.DataFrame:
    frames = [pq.read_table(p).to_pandas() for p in parquet_files(source, "meta/episodes")]
    df = pd.concat(frames, ignore_index=True)

    df["episode_index"] = df["episode_index"].astype("int64") + episode_offset
    df["dataset_from_index"] = df["dataset_from_index"].astype("int64") + frame_offset
    df["dataset_to_index"] = df["dataset_to_index"].astype("int64") + frame_offset
    df["meta/episodes/chunk_index"] = 0
    df["meta/episodes/file_index"] = 0

    for i, row in df.iterrows():
        src_data = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        df.at[i, "data/chunk_index"] = data_map[src_data][0]
        df.at[i, "data/file_index"] = data_map[src_data][1]

        for key, mapping in video_maps.items():
            src_video = (
                int(row[f"videos/{key}/chunk_index"]),
                int(row[f"videos/{key}/file_index"]),
            )
            dst_chunk, dst_file = mapping[src_video]
            df.at[i, f"videos/{key}/chunk_index"] = dst_chunk
            df.at[i, f"videos/{key}/file_index"] = dst_file

    # Stats columns for task_index are kept as source metadata; the actual frame
    # rows above have already been remapped, which is what training consumes.
    if "stats/task_index/min" in df.columns:
        df["stats/task_index/min"] = df["stats/task_index/min"].map(
            lambda x: [task_map[int(v)] for v in x] if isinstance(x, list) else x
        )
        df["stats/task_index/max"] = df["stats/task_index/max"].map(
            lambda x: [task_map[int(v)] for v in x] if isinstance(x, list) else x
        )
    return df


def patch_stale_data_file_refs(source: Path, data_map: dict[tuple[int, int], tuple[int, int]]) -> None:
    """Map stale source data file refs to the only real copied parquet.

    Some provided challenge datasets have meta/episodes rows with
    data/file_index spread across 0..N, while physically all data rows live in
    data/chunk-000/file-000.parquet. LeRobot can still load them because it
    reads all data parquets directly, but a metadata-preserving merge must not
    propagate references to nonexistent source files.
    """
    if len(data_map) != 1:
        return
    fallback = next(iter(data_map.values()))
    frames = [pq.read_table(p).to_pandas() for p in parquet_files(source, "meta/episodes")]
    df = pd.concat(frames, ignore_index=True)
    for _, row in df.iterrows():
        src_data = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        data_map.setdefault(src_data, fallback)


def merge_garment_info(sources: list[Path], output: Path) -> None:
    merged: dict[str, dict[str, Any]] = {}
    for src in sources:
        path = src / "meta/garment_info.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for garment, episodes in data.items():
            out_eps = merged.setdefault(garment, {})
            next_idx = max([int(k) for k in out_eps.keys()] + [-1]) + 1
            for _, payload in sorted(episodes.items(), key=lambda kv: int(kv[0])):
                out_eps[str(next_idx)] = payload
                next_idx += 1
    if merged:
        out = output / "meta/garment_info.json"
        out.write_text(json.dumps(merged, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    sources = [p.resolve() for p in args.sources]
    output = args.output_root.resolve()

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite for generated merge outputs")
        shutil.rmtree(output)

    validate_compatible(sources)
    merged_tasks, task_maps = merge_tasks(sources)

    output.mkdir(parents=True, exist_ok=True)
    write_tasks(merged_tasks, output)

    base_info = load_info(sources[0])
    keep_columns = set(base_info["features"].keys())

    episode_offset = 0
    frame_offset = 0
    next_data_file = 0
    next_video_file = {key: 0 for key in VIDEO_KEYS}
    episode_frames: list[pd.DataFrame] = []
    stats_list = []

    for src_idx, src in enumerate(sources):
        info = load_info(src)
        stats = load_stats(src)
        if stats is not None:
            stats_list.append(stats)

        video_maps: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
        for key in VIDEO_KEYS:
            mapping, next_video_file[key] = copy_videos(src, output, key, next_video_file[key])
            video_maps[key] = mapping

        data_map, next_data_file = copy_data_files(
            src,
            output,
            next_data_file,
            episode_offset,
            frame_offset,
            task_maps[src_idx],
            keep_columns,
        )
        patch_stale_data_file_refs(src, data_map)

        episode_frames.append(
            merge_episode_metadata(
                src,
                episode_offset,
                frame_offset,
                data_map,
                video_maps,
                task_maps[src_idx],
            )
        )

        episode_offset += int(info["total_episodes"])
        frame_offset += int(info["total_frames"])

    episodes = pd.concat(episode_frames, ignore_index=True)
    ep_out = output / "meta/episodes/chunk-000/file-000.parquet"
    ep_out.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(ep_out, index=False)

    info = dict(base_info)
    info["total_episodes"] = episode_offset
    info["total_frames"] = frame_offset
    info["total_tasks"] = len(merged_tasks)
    info["splits"] = {"train": f"0:{episode_offset}"}
    write_info(output, info)

    if stats_list:
        write_stats(aggregate_stats(stats_list), output)
    else:
        shutil.copy2(sources[0] / "meta/stats.json", output / "meta/stats.json")

    merge_garment_info(sources, output)
    print(f"Wrote {output}")
    print(f"episodes={episode_offset} frames={frame_offset} data_files={next_data_file}")
    for key in VIDEO_KEYS:
        print(f"{key}: video_files={next_video_file[key]}")


if __name__ == "__main__":
    main()
