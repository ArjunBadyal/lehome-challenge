"""Run hierarchical SAC across multiple garments by launching one trainer process per segment.

This avoids Isaac cloth reset/recreation hangs by never swapping garments inside
an already-running trainer process.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_SH = REPO_ROOT / "third_party" / "IsaacLab" / "isaaclab.sh"


def find_run_dir(log_root: Path, run_name: str) -> Path:
    matches = sorted(log_root.glob(f"*_{run_name}"))
    if not matches:
        raise FileNotFoundError(f"No run directory found for run_name={run_name!r} under {log_root}")
    return matches[-1]


def checkpoint_step(path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    metadata = ckpt.get("metadata", {})
    return int(ckpt.get("step", metadata.get("step", 0)))


def build_segment_command(args, garment_name: str, target_step: int, checkpoint: Path | None,
                          replay_buffer: Path | None) -> list[str]:
    cmd = [
        str(ISAACLAB_SH),
        "-p",
        "scripts/train_hierarchical_sac.py",
        "--task", args.task,
        "--garment_name", garment_name,
        "--garment_version", args.garment_version,
        "--act_checkpoint", args.act_checkpoint,
        "--dataset_root", args.dataset_root,
        "--chunk_size", str(args.chunk_size),
        "--residual_scale", str(args.residual_scale),
        "--residual_scale_max", str(args.residual_scale_max),
        "--residual_anneal_steps", str(args.residual_anneal_steps),
        "--hold_steps", str(args.hold_steps),
        "--sub_reward_weight", str(args.sub_reward_weight),
        "--episode_length_s", str(args.episode_length_s),
        "--early_stop_dense_score", str(args.early_stop_dense_score),
        "--early_stop_plateau_steps", str(args.early_stop_plateau_steps),
        "--early_stop_plateau_min_dense_score", str(args.early_stop_plateau_min_dense_score),
        "--early_stop_plateau_delta", str(args.early_stop_plateau_delta),
        "--reset_timeout_s", str(args.reset_timeout_s),
        "--stabilize_steps", str(args.stabilize_steps),
        "--env_recreate_interval", "0",
        "--total_timesteps", str(target_step),
        "--batch_size", str(args.batch_size),
        "--buffer_size", str(args.buffer_size),
        "--learning_starts", str(args.learning_starts),
        "--learning_rate", str(args.learning_rate),
        "--gamma", str(args.gamma),
        "--tau", str(args.tau),
        "--log_dir", args.log_dir,
        "--run_name", args.run_name,
        "--rl_device", args.rl_device,
        "--device", args.device,
        "--seed", str(args.seed),
        "--num_envs", "1",
    ]
    if args.headless:
        cmd.append("--headless")
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    if replay_buffer is not None and replay_buffer.exists():
        cmd.extend(["--replay_buffer_path", str(replay_buffer)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Out-of-process multi-garment hierarchical SAC scheduler.")
    parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v2")
    parser.add_argument("--train_garments", type=str, nargs="+", required=True)
    parser.add_argument("--garment_version", type=str, default="Release")
    parser.add_argument("--act_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--residual_scale", type=float, default=0.1)
    parser.add_argument("--residual_scale_max", type=float, default=0.15)
    parser.add_argument("--residual_anneal_steps", type=int, default=30000)
    parser.add_argument("--hold_steps", type=int, default=50)
    parser.add_argument("--sub_reward_weight", type=float, default=2.0)
    parser.add_argument("--episode_length_s", type=float, default=10.0)
    parser.add_argument("--early_stop_dense_score", type=float, default=0.0)
    parser.add_argument("--early_stop_plateau_steps", type=int, default=180)
    parser.add_argument("--early_stop_plateau_min_dense_score", type=float, default=0.8)
    parser.add_argument("--early_stop_plateau_delta", type=float, default=0.005)
    parser.add_argument("--reset_timeout_s", type=int, default=90)
    parser.add_argument("--stabilize_steps", type=int, default=20)
    parser.add_argument("--total_timesteps", type=int, default=50000)
    parser.add_argument("--segment_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=400000)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--log_dir", type=str, default="outputs/rl/hierarchical_sac")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume scheduled training from an existing trainer checkpoint.")
    parser.add_argument("--replay_buffer_path", type=str, default=None,
                        help="Replay buffer to pair with --checkpoint when resuming.")
    args = parser.parse_args()

    if len(args.train_garments) < 2:
        raise ValueError("Use train_hierarchical_sac.py directly for a single garment.")
    if args.segment_steps <= 0:
        raise ValueError("--segment_steps must be positive.")

    if args.run_name is None:
        args.run_name = f"multi_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    log_root = REPO_ROOT / args.log_dir / args.task
    checkpoint: Path | None = Path(args.checkpoint).resolve() if args.checkpoint else None
    replay_buffer: Path | None = Path(args.replay_buffer_path).resolve() if args.replay_buffer_path else None
    current_step = checkpoint_step(checkpoint) if checkpoint is not None else 0
    segment_index = current_step // args.segment_steps

    if checkpoint is not None:
        print(f"[INFO] Resuming schedule from checkpoint: {checkpoint}", flush=True)
        print(f"[INFO] Restored current_step={current_step}", flush=True)
        if replay_buffer is None:
            candidate = checkpoint.parent / "replay_buffer.npz"
            if candidate.exists():
                replay_buffer = candidate

    while current_step < args.total_timesteps:
        garment_name = args.train_garments[segment_index % len(args.train_garments)]
        target_step = min(current_step + args.segment_steps, args.total_timesteps)
        cmd = build_segment_command(args, garment_name, target_step, checkpoint, replay_buffer)
        child_env = os.environ.copy()
        if child_env.get("TERM") in (None, "", "dumb"):
            child_env["TERM"] = "xterm"
        print(
            f"[INFO] Segment {segment_index + 1}: garment={garment_name}, "
            f"target_step={target_step}/{args.total_timesteps}",
            flush=True,
        )
        print(f"[INFO] Launching: {shlex.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=REPO_ROOT, env=child_env)
        if completed.returncode != 0:
            print(f"[ERROR] Segment failed with exit code {completed.returncode}", flush=True)
            return completed.returncode

        run_dir = find_run_dir(log_root, args.run_name)
        checkpoint = run_dir / "model.pt"
        replay_buffer = run_dir / "replay_buffer.npz"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Expected checkpoint not found after segment: {checkpoint}")
        reached_step = checkpoint_step(checkpoint)
        if reached_step < target_step:
            raise RuntimeError(
                f"Segment on garment={garment_name} did not reach target_step={target_step}. "
                f"Checkpoint only reached step={reached_step}."
            )

        current_step = reached_step
        segment_index += 1

    print(f"[INFO] Completed scheduled training. Final checkpoint: {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
