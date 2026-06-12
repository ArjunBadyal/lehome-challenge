"""Hierarchical residual SAC: sub-policies for directional garment folding.

Architecture
------------
Every ``chunk_size`` steps:
    obs_dict (with camera images) → ACT (frozen) → action_chunk [chunk_size, 12]

Every step:
    1. task_metrics → condition_margins (5), checkpoint_positions (6×3)
    2. MetaController selects sub-policy k (fold_up/down/left/right or no_op)
    3. obs_47d = [state(12), act_action(12), margins(5), checkpoints(18)]
    4. sub_actors[k](obs_47d) → delta [12]
    5. final_action = clip(act_action + scale * delta, joint_limits)
"""

import argparse
import copy
import faulthandler
import os
import signal
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Hierarchical residual SAC for garment folding.")
parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v2")
parser.add_argument("--garment_name", type=str, default=None)
parser.add_argument("--train_garments", type=str, nargs="+", default=None)
parser.add_argument("--garment_version", type=str, default="Release")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)

# ACT
parser.add_argument("--act_checkpoint", type=str, required=True)
parser.add_argument("--dataset_root", type=str, required=True)
parser.add_argument("--chunk_size", type=int, default=100)

# Residual
parser.add_argument("--residual_scale", type=float, default=0.1)
parser.add_argument("--residual_scale_max", type=float, default=0.15)
parser.add_argument("--residual_anneal_steps", type=int, default=30000)
parser.add_argument("--residual_hidden", type=int, nargs="+", default=[128, 128])

# Hierarchical
parser.add_argument("--hold_steps", type=int, default=50,
                     help="Min steps before meta-controller re-evaluates.")
parser.add_argument("--sub_reward_weight", type=float, default=2.0,
                     help="Weight on sub-policy-specific reward bonus.")
parser.add_argument("--episode_length_s", type=float, default=10.0,
                     help="Episode duration in seconds.")
parser.add_argument("--early_stop_dense_score", type=float, default=0.0,
                     help="Training-only early stop when dense_score reaches this threshold; 0 disables.")
parser.add_argument("--early_stop_plateau_steps", type=int, default=180,
                     help="Training-only early stop when dense_score plateaus for this many steps; 0 disables.")
parser.add_argument("--early_stop_plateau_min_dense_score", type=float, default=0.8,
                     help="Minimum dense_score before plateau early stop can trigger.")
parser.add_argument("--early_stop_plateau_delta", type=float, default=0.005,
                     help="Minimum dense_score improvement to reset plateau tracking.")
parser.add_argument("--reset_timeout_s", type=int, default=90,
                     help="Timeout in seconds for reset/stabilization phases.")
parser.add_argument("--stabilize_steps", type=int, default=20,
                     help="Physics settle steps after each reset.")
parser.add_argument("--env_recreate_interval", type=int, default=4,
                     help="Recreate the whole env every N episodes to avoid reset stalls; 0 disables.")

# SAC
parser.add_argument("--total_timesteps", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=400000)
parser.add_argument("--learning_starts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)

# Logging
parser.add_argument("--log_dir", type=str, default="outputs/rl/hierarchical_sac")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint_freq", type=int, default=5000)
parser.add_argument("--log_freq", type=int, default=250)
parser.add_argument("--rl_device", type=str, default="cuda:0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--replay_buffer_path", type=str, default=None,
                     help="Optional replay buffer .npz to restore when resuming.")
parser.add_argument("--save_buffer", action="store_true", default=True)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Post-launch imports
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from torch.utils.tensorboard import SummaryWriter

import lehome.tasks.bedroom  # noqa: F401

from scripts.rl.hierarchical_model import (
    NUM_LEARNED, NO_OP, SUB_POLICY_NAMES,
    RuleBasedMetaController, SubPolicyBank, build_hierarchical_obs,
)
from scripts.rl.tagged_replay_buffer import TaggedReplayBuffer
from scripts.rl.act_wrapper import ACTChunkProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tensor_item(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0].item())
    return float(value)


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for sp, tp in zip(source.parameters(), target.parameters(), strict=True):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


def anneal_scale(step: int, min_s: float, max_s: float, anneal_steps: int) -> float:
    frac = min(step / max(anneal_steps, 1), 1.0)
    return min_s + (max_s - min_s) * frac


def compute_sub_reward(
    k: int,
    global_reward: float,
    prev_margins: np.ndarray,
    curr_margins: np.ndarray,
    weight: float,
) -> float:
    """Compute shaped reward for sub-policy k."""
    if k == NO_OP:
        return global_reward

    if k in (0, 1):  # fold_up, fold_down — target margin[1] (front/back)
        delta_margin = float(curr_margins[1] - prev_margins[1])
    elif k in (2, 3):  # fold_left, fold_right — target min(margin[0], margin[2])
        prev_worst = min(prev_margins[0], prev_margins[2])
        curr_worst = min(curr_margins[0], curr_margins[2])
        delta_margin = float(curr_worst - prev_worst)
    else:
        delta_margin = 0.0

    return global_reward + weight * delta_margin


def save_checkpoint(path: str, *, bank: SubPolicyBank, metadata: dict,
                    step: int, episode_index: int, activation_counts: np.ndarray):
    data = bank.state_dicts()
    checkpoint_metadata = dict(metadata)
    checkpoint_metadata["step"] = int(step)
    checkpoint_metadata["episode_index"] = int(episode_index)
    data["metadata"] = checkpoint_metadata
    data["step"] = int(step)
    data["episode_index"] = int(episode_index)
    data["activation_counts"] = np.asarray(activation_counts, dtype=np.int64)
    torch.save(data, path)


class ResetTimeoutError(RuntimeError):
    pass


def _raise_reset_timeout(signum, frame):
    raise ResetTimeoutError("reset timed out")


def run_with_timeout(timeout_s: int, label: str, fn):
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        return fn()

    timeout_s = max(int(timeout_s), 1)
    old_handler = signal.getsignal(signal.SIGALRM)
    faulthandler.dump_traceback_later(timeout_s, repeat=False)
    signal.signal(signal.SIGALRM, _raise_reset_timeout)
    signal.alarm(timeout_s)
    try:
        return fn()
    finally:
        signal.alarm(0)
        faulthandler.cancel_dump_traceback_later()
        signal.signal(signal.SIGALRM, old_handler)


def close_env_instance(env) -> None:
    if env is not None:
        env.close()


def make_env_cfg(base_env_cfg, garment_name: str):
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg.garment_name = garment_name
    return env_cfg


def create_env_instance(task: str, env_cfg, *, timeout_s: int = 0, label: str = "gym environment"):
    print(f"[INFO] Creating gym environment for garment={env_cfg.garment_name}...", flush=True)
    env = run_with_timeout(
        timeout_s,
        label,
        lambda: gym.make(task, cfg=env_cfg),
    )
    raw_env = env.unwrapped
    print("[INFO] Gym environment created.", flush=True)
    return env, raw_env


def reset_env_instance(env, raw_env, args, stabilize_fn, *, label: str):
    print(f"[INFO] {label}: reset start", flush=True)

    if hasattr(raw_env, "object") and raw_env.object is not None:
        if not hasattr(raw_env.object, "initial_points_positions"):
            run_with_timeout(
                args.reset_timeout_s,
                f"{label}: initialize_obs",
                raw_env.initialize_obs,
            )
            if hasattr(raw_env, "sim") and raw_env.sim is not None:
                def _warmup():
                    for _ in range(5):
                        raw_env.sim.step(render=True)
                run_with_timeout(args.reset_timeout_s, f"{label}: sim warmup", _warmup)

    obs_dict = run_with_timeout(
        args.reset_timeout_s,
        f"{label}: env.reset",
        lambda: env.reset()[0],
    )
    run_with_timeout(
        args.reset_timeout_s,
        f"{label}: stabilize_garment_after_reset",
        lambda: stabilize_fn(raw_env, args, num_steps=args.stabilize_steps),
    )

    state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)
    task_metrics = raw_env._get_task_metrics()
    prev_margins = task_metrics["condition_margins"].cpu().numpy().copy()

    print(f"[INFO] {label}: reset complete", flush=True)
    return obs_dict, state, prev_margins


def reset_with_recreate_fallback(task: str, base_env_cfg, garment_name: str, env, raw_env, args,
                                 stabilize_fn, *, label: str):
    try:
        obs_dict, state, prev_margins = reset_env_instance(
            env, raw_env, args, stabilize_fn, label=label
        )
        return env, raw_env, obs_dict, state, prev_margins
    except ResetTimeoutError:
        print(
            f"[WARN] {label}: reset timed out for garment={garment_name}. "
            "Recreating env and retrying once.",
            flush=True,
        )
        close_env_instance(env)
        fresh_env_cfg = make_env_cfg(base_env_cfg, garment_name)
        env, raw_env = create_env_instance(
            task, fresh_env_cfg,
            timeout_s=args.reset_timeout_s,
            label=f"{label}: recreate env garment={garment_name}",
        )
        obs_dict, state, prev_margins = reset_env_instance(
            env, raw_env, args, stabilize_fn, label=f"{label} retry"
        )
        return env, raw_env, obs_dict, state, prev_margins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env = None
    raw_env = None
    bank = None
    metadata = None
    writer = None
    custom_writer = None
    log_dir = None
    start_time = time.time()

    try:
        # ---- Environment ----
        print("[INFO] Loading task configs...", flush=True)
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")

        if args.num_envs != 1:
            raise ValueError("Hierarchical SAC trainer supports only --num_envs 1.")

        env_cfg.sim.device = args.device if args.device else env_cfg.sim.device
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.garment_version = args.garment_version

        # Build garment train list
        if args.train_garments:
            train_garments = args.train_garments
        elif args.garment_name:
            train_garments = [args.garment_name]
        else:
            raise ValueError("Provide --garment_name or --train_garments")
        if len(train_garments) > 1:
            raise ValueError(
                "train_hierarchical_sac.py only supports a single garment per process. "
                "Use scripts/train_hierarchical_schedule.py for multi-garment training."
            )
        env_cfg.garment_name = train_garments[0]
        env_cfg.terminate_on_success = True
        env_cfg.episode_length_s = args.episode_length_s
        # Avoid in-place cloth recreation on episode reset. Garment swaps recreate
        # the entire env instead, which is slower but much more stable.
        env_cfg.garment_reset_mode = "soft"
        base_env_cfg = copy.deepcopy(env_cfg)

        print(f"[INFO] Training garments ({len(train_garments)}): {train_garments}", flush=True)
        current_garment = train_garments[0]
        current_env_cfg = make_env_cfg(base_env_cfg, current_garment)
        env, raw_env = create_env_instance(
            args.task, current_env_cfg,
            timeout_s=args.reset_timeout_s,
            label=f"create env garment={current_garment}",
        )

        # Action bounds
        action_space = raw_env.single_action_space
        action_dim = int(np.prod(action_space.shape))  # 12
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        hierarchical_obs_dim = 12 + 12 + 5 + 18  # 47

        rl_device = torch.device(args.rl_device)
        action_low_dev = action_low.to(rl_device)
        action_high_dev = action_high.to(rl_device)

        # ---- Logging ----
        log_root = os.path.abspath(os.path.join(args.log_dir, args.task))
        if args.checkpoint:
            log_dir = str(Path(args.checkpoint).resolve().parent)
            os.makedirs(log_dir, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            safe_garment = (args.garment_name or train_garments[0]).replace("/", "_")
            run_name = args.run_name or safe_garment
            log_dir = os.path.join(log_root, f"{timestamp}_{run_name}")
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        command_path = Path(log_dir, "command.txt")
        if command_path.exists():
            with command_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
                handle.write(" ".join(sys.argv))
        else:
            command_path.write_text(" ".join(sys.argv), encoding="utf-8")
        writer = SummaryWriter(log_dir)
        custom_writer = SummaryWriter(os.path.join(log_dir, "custom_metrics"))
        print(f"[INFO] Logging to: {log_dir}", flush=True)

        # ---- ACT (frozen) ----
        print("[INFO] Loading ACT policy...", flush=True)
        act = ACTChunkProvider(
            policy_path=args.act_checkpoint,
            dataset_root=args.dataset_root,
            device=args.rl_device,
            chunk_size=args.chunk_size,
        )
        print(f"[INFO] ACT loaded. action_dim={act.action_dim}, chunk_size={act.chunk_size}", flush=True)

        # ---- Hierarchical sub-policies ----
        hidden_sizes = tuple(args.residual_hidden)
        bank = SubPolicyBank(
            num_subpolicies=NUM_LEARNED,
            obs_dim=hierarchical_obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=rl_device,
            learning_rate=args.learning_rate,
        )
        meta = RuleBasedMetaController(hold_steps=args.hold_steps)
        target_entropy = -float(action_dim)
        resume_step = 0
        episode_index = 0
        activation_counts = np.zeros(NUM_LEARNED + 1, dtype=np.int64)

        # ---- Resume ----
        if args.checkpoint:
            print(f"[INFO] Resuming from: {args.checkpoint}", flush=True)
            ckpt = torch.load(args.checkpoint, map_location=rl_device, weights_only=False)
            bank.load_state_dicts(ckpt)
            checkpoint_metadata = ckpt.get("metadata", {})
            resume_step = int(ckpt.get("step", checkpoint_metadata.get("step", 0)))
            episode_index = int(ckpt.get("episode_index", checkpoint_metadata.get("episode_index", 0)))
            restored_counts = ckpt.get("activation_counts")
            if restored_counts is not None:
                activation_counts = np.asarray(restored_counts, dtype=np.int64)
            print(
                f"[INFO] Restored training state: step={resume_step}, episode_index={episode_index}",
                flush=True,
            )

        metadata = {
            "obs_dim": hierarchical_obs_dim,
            "action_dim": action_dim,
            "hidden_sizes": list(hidden_sizes),
            "action_low": action_low.cpu(),
            "action_high": action_high.cpu(),
            "num_subpolicies": NUM_LEARNED,
            "hold_steps": args.hold_steps,
            "sub_reward_weight": args.sub_reward_weight,
            "residual_scale": args.residual_scale,
            "residual_scale_max": args.residual_scale_max,
            "episode_length_s": args.episode_length_s,
            "early_stop_dense_score": args.early_stop_dense_score,
            "early_stop_plateau_steps": args.early_stop_plateau_steps,
            "early_stop_plateau_min_dense_score": args.early_stop_plateau_min_dense_score,
            "early_stop_plateau_delta": args.early_stop_plateau_delta,
            "reset_timeout_s": args.reset_timeout_s,
            "stabilize_steps": args.stabilize_steps,
            "env_recreate_interval": args.env_recreate_interval,
            "act_checkpoint": args.act_checkpoint,
            "dataset_root": args.dataset_root,
            "chunk_size": args.chunk_size,
            "trainer": "hierarchical_residual_sac",
            "observation_key": "observation.state",
            "action_semantics": "absolute_joint_positions",
            "submission_safe": True,
            "task_id": args.task,
            "train_garments": train_garments,
            "garment_version": args.garment_version,
        }

        replay_buffer = TaggedReplayBuffer(args.buffer_size, hierarchical_obs_dim, action_dim)
        replay_buffer_path = None
        if args.replay_buffer_path:
            replay_buffer_path = args.replay_buffer_path
        elif args.checkpoint:
            candidate = os.path.join(log_dir, "replay_buffer.npz")
            if os.path.exists(candidate):
                replay_buffer_path = candidate
        if replay_buffer_path:
            restored = replay_buffer.load(replay_buffer_path)
            print(f"[INFO] Restored replay buffer ({restored} transitions): {replay_buffer_path}", flush=True)

        # ---- Initial reset ----
        from scripts.utils.common import stabilize_garment_after_reset
        env, raw_env, obs_dict, state, prev_margins = reset_with_recreate_fallback(
            args.task, base_env_cfg, current_garment, env, raw_env, args,
            stabilize_garment_after_reset, label=f"initial garment={current_garment}",
        )

        act.reset()
        meta.reset()
        act_chunk: torch.Tensor | None = None
        episodes_in_current_env = 1

        episode_reward = 0.0
        episode_length = 0
        recent_returns: deque[float] = deque(maxlen=20)
        episode_end_counts = {
            "success": 0,
            "dense_threshold": 0,
            "dense_plateau": 0,
            "max_episode_length": 0,
        }
        initial_metrics = raw_env._get_task_metrics()
        episode_best_dense_score = tensor_item(initial_metrics.get("dense_score"), 0.0)
        plateau_steps = 0

        if resume_step >= args.total_timesteps:
            print(
                f"[INFO] Checkpoint step {resume_step} already reached total_timesteps={args.total_timesteps}.",
                flush=True,
            )
            return

        print(f"[INFO] Starting hierarchical SAC training for {args.total_timesteps} steps...", flush=True)

        for step in range(resume_step + 1, args.total_timesteps + 1):
            chunk_step = (step - 1) % args.chunk_size

            # --- ACT re-plan every chunk_size steps ---
            if chunk_step == 0:
                obs_dict_full = raw_env._get_observations()
                act_obs = {}
                for key, val in obs_dict_full.items():
                    if key.startswith("observation."):
                        if isinstance(val, torch.Tensor):
                            act_obs[key] = val.detach().cpu().numpy()[0] if val.dim() > 1 else val.detach().cpu().numpy()
                        else:
                            act_obs[key] = np.asarray(val)
                act_chunk = act.get_chunk(act_obs)

            act_action = act_chunk[chunk_step]  # [12] on rl_device

            # --- Get task metrics for meta-controller ---
            task_metrics = raw_env._get_task_metrics()
            curr_margins = task_metrics["condition_margins"].cpu().numpy()
            checkpoint_pos = task_metrics["checkpoint_positions_cm"]  # (6, 3) tensor

            # --- Meta-controller selects sub-policy ---
            k = meta.select(curr_margins, checkpoint_pos.cpu().numpy())
            activation_counts[k] += 1

            # --- Build hierarchical observation ---
            h_obs = build_hierarchical_obs(
                state, act_action, task_metrics["condition_margins"],
                checkpoint_pos, rl_device,
            )

            # --- Sub-policy action ---
            scale = anneal_scale(step, args.residual_scale, args.residual_scale_max,
                                 args.residual_anneal_steps)

            if k == NO_OP:
                delta = torch.zeros(action_dim, device=rl_device)
            elif step <= args.learning_starts:
                delta = torch.empty(action_dim, device=rl_device).uniform_(-1.0, 1.0)
            else:
                delta = bank.act(k, h_obs.unsqueeze(0), deterministic=False).squeeze(0)

            final_action = act_action + scale * delta
            final_action = torch.clamp(final_action, action_low_dev, action_high_dev)

            # --- Step environment ---
            action_np = final_action.detach().cpu().numpy().astype(np.float32)
            action_tensor = torch.as_tensor(action_np[None], device=env_cfg.sim.device, dtype=torch.float32)
            raw_env.step(action_tensor)

            reward_tensor = raw_env._get_rewards()
            global_reward = tensor_item(reward_tensor)

            next_state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)

            # Next task metrics for sub-reward
            next_metrics = raw_env._get_task_metrics()
            next_margins = next_metrics["condition_margins"].cpu().numpy()
            next_dense_score = tensor_item(next_metrics.get("dense_score"), 0.0)

            # Sub-policy reward
            shaped_reward = compute_sub_reward(
                k, global_reward, prev_margins, next_margins, args.sub_reward_weight
            )

            # Episode termination
            if next_dense_score > episode_best_dense_score + args.early_stop_plateau_delta:
                episode_best_dense_score = next_dense_score
                plateau_steps = 0
            else:
                plateau_steps += 1

            done = bool(tensor_item(raw_env._get_success())) if hasattr(raw_env, '_get_success') else False
            end_reason = "success" if done else None
            if (
                not done
                and args.early_stop_dense_score > 0.0
                and next_dense_score >= args.early_stop_dense_score
            ):
                done = True
                end_reason = "dense_threshold"
            elif (
                not done
                and args.early_stop_plateau_steps > 0
                and next_dense_score >= args.early_stop_plateau_min_dense_score
                and plateau_steps >= args.early_stop_plateau_steps
            ):
                done = True
                end_reason = "dense_plateau"
            elif episode_length + 1 >= raw_env.max_episode_length:
                done = True
                end_reason = "max_episode_length"

            # Next observation
            next_act_action = act_chunk[min(chunk_step + 1, args.chunk_size - 1)]
            next_h_obs = build_hierarchical_obs(
                next_state, next_act_action, next_metrics["condition_margins"],
                next_metrics["checkpoint_positions_cm"], rl_device,
            )

            # Store transition (only for learned sub-policies)
            if k != NO_OP:
                h_obs_np = h_obs.cpu().numpy()
                delta_np = delta.detach().cpu().numpy()
                next_h_obs_np = next_h_obs.cpu().numpy()
                if not replay_buffer.add(h_obs_np, delta_np, shaped_reward, next_h_obs_np, done, k):
                    print(f"[WARN] NaN transition at step {step}", flush=True)

            episode_reward += global_reward
            episode_length += 1
            prev_margins = next_margins.copy()

            # --- SAC update for active sub-policy ---
            if k != NO_OP and step >= args.learning_starts:
                batch = replay_buffer.sample_for_subpolicy(k, args.batch_size, rl_device)
                if batch is not None:
                    alpha = bank.log_alphas[k].exp()
                    actor_k = bank.actors[k]
                    q1_k, q2_k = bank.q1s[k], bank.q2s[k]
                    q1t_k, q2t_k = bank.q1_targets[k], bank.q2_targets[k]

                    with torch.no_grad():
                        next_deltas, next_log_prob, _ = actor_k.sample(batch["next_obs"])
                        tgt_q = torch.min(q1t_k(batch["next_obs"], next_deltas),
                                          q2t_k(batch["next_obs"], next_deltas))
                        tgt_q = batch["rewards"] + (1.0 - batch["dones"]) * args.gamma * (
                            tgt_q - alpha * next_log_prob)

                    q1_loss = F.mse_loss(q1_k(batch["obs"], batch["actions"]), tgt_q)
                    q2_loss = F.mse_loss(q2_k(batch["obs"], batch["actions"]), tgt_q)
                    bank.q_optimizers[k].zero_grad(set_to_none=True)
                    (q1_loss + q2_loss).backward()
                    bank.q_optimizers[k].step()

                    new_deltas, log_prob, _ = actor_k.sample(batch["obs"])
                    q_pi = torch.min(q1_k(batch["obs"], new_deltas), q2_k(batch["obs"], new_deltas))
                    actor_loss = (alpha.detach() * log_prob - q_pi).mean()
                    bank.actor_optimizers[k].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    bank.actor_optimizers[k].step()

                    alpha_loss = -(bank.log_alphas[k] * (log_prob + target_entropy).detach()).mean()
                    bank.alpha_optimizers[k].zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    bank.alpha_optimizers[k].step()

                    soft_update(q1_k, q1t_k, args.tau)
                    soft_update(q2_k, q2t_k, args.tau)

                    if step % args.log_freq == 0:
                        name = SUB_POLICY_NAMES[k]
                        writer.add_scalar(f"train/{name}/q1_loss", float(q1_loss.item()), step)
                        writer.add_scalar(f"train/{name}/actor_loss", float(actor_loss.item()), step)
                        writer.add_scalar(f"train/{name}/alpha", float(alpha.item()), step)

            # --- Logging ---
            if step % args.log_freq == 0:
                writer.add_scalar("env/reward", global_reward, step)
                writer.add_scalar("residual/scale", scale, step)
                name = SUB_POLICY_NAMES[k]
                writer.add_scalar(f"meta/active_subpolicy", k, step)

                delta_mag = float(torch.abs(scale * delta).mean().item())
                act_mag = float(torch.abs(act_action).mean().item())
                writer.add_scalar("residual/delta_magnitude", delta_mag, step)
                writer.add_scalar("residual/act_magnitude", act_mag, step)
                if act_mag > 0:
                    writer.add_scalar("residual/delta_ratio", delta_mag / act_mag, step)

                # Per-sub-policy activation counts
                for i, sp_name in enumerate(SUB_POLICY_NAMES):
                    writer.add_scalar(f"meta/activations/{sp_name}", int(activation_counts[i]), step)
                    if i < NUM_LEARNED:
                        writer.add_scalar(f"buffer/{sp_name}_count",
                                          replay_buffer.count_for_subpolicy(i), step)

                # Dense score
                writer.add_scalar("env/dense_score", task_metrics.get("dense_score", 0.0), step)

                # Env metrics
                info = {}
                if hasattr(raw_env, 'extras') and isinstance(raw_env.extras, dict):
                    info = raw_env.extras
                for mkey in ["dense_score", "primary_score", "secondary_score", "success"]:
                    if mkey in info:
                        custom_writer.add_scalar(f"env/{mkey}", tensor_item(info[mkey]), step)

            state = next_state

            # --- Episode boundary ---
            if done:
                recent_returns.append(episode_reward)
                writer.add_scalar("episode/return", episode_reward, step)
                writer.add_scalar("episode/length", episode_length, step)
                writer.add_scalar("episode/index", episode_index, step)
                if recent_returns:
                    writer.add_scalar("episode/return_mean_20",
                                      float(np.mean(recent_returns)), step)
                writer.add_scalar("episode/best_dense_score", episode_best_dense_score, step)
                if end_reason is not None:
                    episode_end_counts[end_reason] += 1
                    writer.add_scalar(f"episode/end_count/{end_reason}", episode_end_counts[end_reason], step)

                episode_reward = 0.0
                episode_length = 0
                episode_index += 1

                print(
                    f"[INFO] Episode {episode_index}: garment={current_garment}, "
                    f"end_reason={end_reason}, best_dense_score={episode_best_dense_score:.3f}",
                    flush=True,
                )
                env, raw_env, obs_dict, state, prev_margins = reset_with_recreate_fallback(
                    args.task, base_env_cfg, current_garment, env, raw_env, args,
                    stabilize_garment_after_reset,
                    label=f"episode {episode_index} garment={current_garment}",
                )
                episodes_in_current_env += 1
                act.reset()
                meta.reset()
                act_chunk = None
                reset_metrics = raw_env._get_task_metrics()
                episode_best_dense_score = tensor_item(reset_metrics.get("dense_score"), 0.0)
                plateau_steps = 0

            # --- Checkpoint ---
            if step % args.checkpoint_freq == 0 or step == args.total_timesteps:
                ckpt_path = os.path.join(log_dir, f"checkpoint_{step:08d}.pt")
                save_checkpoint(
                    ckpt_path,
                    bank=bank,
                    metadata=metadata,
                    step=step,
                    episode_index=episode_index,
                    activation_counts=activation_counts,
                )
                print(f"[INFO] Saved checkpoint: {ckpt_path}", flush=True)

        # ---- Final save ----
        final_path = os.path.join(log_dir, "model.pt")
        save_checkpoint(
            final_path,
            bank=bank,
            metadata=metadata,
            step=args.total_timesteps,
            episode_index=episode_index,
            activation_counts=activation_counts,
        )
        print(f"[INFO] Final model: {final_path}", flush=True)

        if args.save_buffer:
            buf_path = os.path.join(log_dir, "replay_buffer.npz")
            replay_buffer.save(buf_path)
            print(f"[INFO] Saved replay buffer ({replay_buffer.size} transitions): {buf_path}", flush=True)

        # Print activation summary
        print("[INFO] Sub-policy activation summary:", flush=True)
        for i, name in enumerate(SUB_POLICY_NAMES):
            print(f"  {name}: {activation_counts[i]} steps", flush=True)

    except Exception as e:
        if bank is not None and log_dir and metadata is not None:
            try:
                emergency_path = os.path.join(log_dir, "emergency_checkpoint.pt")
                save_checkpoint(
                    emergency_path,
                    bank=bank,
                    metadata=metadata,
                    step=resume_step if "step" not in locals() else step,
                    episode_index=episode_index if "episode_index" in locals() else 0,
                    activation_counts=activation_counts if "activation_counts" in locals() else np.zeros(NUM_LEARNED + 1, dtype=np.int64),
                )
                print(f"[INFO] Emergency checkpoint: {emergency_path}", flush=True)
            except Exception as save_exc:
                print(f"[WARN] Failed to save emergency checkpoint: {save_exc!r}", flush=True)
        print(f"[ERROR] Training failed: {e!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if writer:
            writer.flush(); writer.close()
        if custom_writer:
            custom_writer.flush(); custom_writer.close()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[ERROR] train_hierarchical_sac.py failed: {exc!r}", flush=True)
        traceback.print_exc()
        raise
