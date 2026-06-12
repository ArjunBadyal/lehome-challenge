"""Encoder-feature residual SAC: submission-compliant residual using ACT encoder features.

Architecture
------------
Every ``chunk_size`` steps:
    obs_dict (with images) → ACT → action_chunk [chunk_size, 12]
                           → ACT encoder → features [512]

Every step:
    obs_536d = [state(12), act_action(12), encoder_features(512)]
    EncoderResidualActor(obs_536d) → delta [12]
    final_action = clip(act_action + scale * delta, joint_limits)

Submission-compliant: encoder features are derived from public observations
(3× RGB + joint state) via the frozen ACT encoder.
"""

import argparse
import os
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
parser = argparse.ArgumentParser(description="Encoder-feature residual SAC.")
parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v2")
parser.add_argument("--garment_name", type=str, required=True)
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
parser.add_argument("--residual_hidden", type=int, nargs="+", default=[256, 256])

# SAC
parser.add_argument("--total_timesteps", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=200000)
parser.add_argument("--learning_starts", type=int, default=500)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--init_alpha", type=float, default=0.2)

# Episode
parser.add_argument("--episode_length_s", type=float, default=6.67,
                     help="Episode duration matching eval --max_steps 600 at 90Hz.")

# Logging
parser.add_argument("--log_dir", type=str, default="outputs/rl/encoder_residual_sac")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint_freq", type=int, default=5000)
parser.add_argument("--log_freq", type=int, default=250)
parser.add_argument("--rl_device", type=str, default="cuda:0")
parser.add_argument("--checkpoint", type=str, default=None)
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

from scripts.rl.encoder_residual_model import EncoderResidualActor
from scripts.rl.sac_model import QNetwork
from scripts.rl.replay_buffer import ReplayBuffer
from scripts.rl.act_encoder_wrapper import ACTWithEncoderFeatures


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


def save_checkpoint(path: str, *, actor, q1, q2, q1_target, q2_target,
                    actor_opt, q_opt, alpha_opt, log_alpha, metadata):
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "q1_state_dict": q1.state_dict(),
        "q2_state_dict": q2.state_dict(),
        "q1_target_state_dict": q1_target.state_dict(),
        "q2_target_state_dict": q2_target.state_dict(),
        "actor_optimizer_state_dict": actor_opt.state_dict(),
        "q_optimizer_state_dict": q_opt.state_dict(),
        "alpha_optimizer_state_dict": alpha_opt.state_dict(),
        "log_alpha": log_alpha.detach().cpu(),
        "metadata": metadata,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env = None
    writer = None
    custom_writer = None
    start_time = time.time()

    try:
        # ---- Environment ----
        print("[INFO] Loading task configs...", flush=True)
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")

        if args.num_envs != 1:
            raise ValueError("Encoder residual SAC supports only --num_envs 1.")

        env_cfg.sim.device = args.device if args.device else env_cfg.sim.device
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.garment_name = args.garment_name
        env_cfg.garment_version = args.garment_version
        env_cfg.terminate_on_success = True
        env_cfg.episode_length_s = args.episode_length_s

        print("[INFO] Creating gym environment...", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        raw_env = env.unwrapped
        print("[INFO] Gym environment created.", flush=True)

        # Action bounds
        action_space = raw_env.single_action_space
        action_dim = int(np.prod(action_space.shape))  # 12
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)

        rl_device = torch.device(args.rl_device)
        action_low_dev = action_low.to(rl_device)
        action_high_dev = action_high.to(rl_device)

        # ---- Logging ----
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_garment = args.garment_name.replace("/", "_")
        run_name = args.run_name or safe_garment
        log_root = os.path.abspath(os.path.join(args.log_dir, args.task))
        log_dir = os.path.join(log_root, f"{timestamp}_{run_name}")
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        Path(log_dir, "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
        writer = SummaryWriter(log_dir)
        custom_writer = SummaryWriter(os.path.join(log_dir, "custom_metrics"))
        print(f"[INFO] Logging to: {log_dir}", flush=True)

        # ---- ACT with encoder features (frozen) ----
        print("[INFO] Loading ACT with encoder features...", flush=True)
        act = ACTWithEncoderFeatures(
            policy_path=args.act_checkpoint,
            dataset_root=args.dataset_root,
            device=args.rl_device,
            chunk_size=args.chunk_size,
        )
        encoder_dim = act.encoder_feature_dim  # 512
        obs_dim = 12 + 12 + encoder_dim  # 536
        print(f"[INFO] ACT loaded. encoder_dim={encoder_dim}, obs_dim={obs_dim}", flush=True)

        # ---- Encoder residual actor + Q-networks ----
        hidden_sizes = tuple(args.residual_hidden)
        actor = EncoderResidualActor(
            state_dim=12, action_dim=action_dim, encoder_dim=encoder_dim,
            hidden_sizes=hidden_sizes,
        ).to(rl_device)
        q1 = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q2 = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q1_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q2_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
        q_optimizer = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()), lr=args.learning_rate)
        log_alpha = torch.tensor(
            np.log(args.init_alpha), device=rl_device, dtype=torch.float32, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.learning_rate)
        target_entropy = -float(action_dim)

        # ---- Resume ----
        if args.checkpoint:
            print(f"[INFO] Resuming from: {args.checkpoint}", flush=True)
            ckpt = torch.load(args.checkpoint, map_location=rl_device)
            actor.load_state_dict(ckpt["actor_state_dict"])
            q1.load_state_dict(ckpt["q1_state_dict"])
            q2.load_state_dict(ckpt["q2_state_dict"])
            q1_target.load_state_dict(ckpt["q1_target_state_dict"])
            q2_target.load_state_dict(ckpt["q2_target_state_dict"])
            actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(ckpt["q_optimizer_state_dict"])
            alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
            log_alpha = ckpt["log_alpha"].to(device=rl_device).requires_grad_(True)

        metadata = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "encoder_dim": encoder_dim,
            "hidden_sizes": list(hidden_sizes),
            "action_low": action_low.cpu(),
            "action_high": action_high.cpu(),
            "residual_scale": args.residual_scale,
            "residual_scale_max": args.residual_scale_max,
            "act_checkpoint": args.act_checkpoint,
            "dataset_root": args.dataset_root,
            "chunk_size": args.chunk_size,
            "trainer": "encoder_residual_sac",
            "observation_key": "observation.state",
            "action_semantics": "absolute_joint_positions",
            "submission_safe": True,
            "task_id": args.task,
            "garment_name": args.garment_name,
            "garment_version": args.garment_version,
        }

        replay_buffer = ReplayBuffer(args.buffer_size, obs_dim, action_dim)

        # ---- Initial reset ----
        print("[INFO] Resetting environment...", flush=True)
        from scripts.utils.common import stabilize_garment_after_reset

        if hasattr(raw_env, 'object') and raw_env.object is not None:
            if not hasattr(raw_env.object, 'initial_points_positions'):
                raw_env.initialize_obs()
                if hasattr(raw_env, 'sim') and raw_env.sim is not None:
                    for _ in range(5):
                        raw_env.sim.step(render=True)

        obs_dict = env.reset()[0]
        stabilize_garment_after_reset(raw_env, args, num_steps=20)
        state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)

        act.reset()
        act_chunk: list[np.ndarray] = []
        encoder_features: torch.Tensor | None = None
        step_in_chunk = 0

        episode_reward = 0.0
        episode_length = 0
        episode_index = 0
        recent_returns: deque[float] = deque(maxlen=20)

        print(f"[INFO] Starting encoder residual SAC training for {args.total_timesteps} steps...", flush=True)

        for step in range(1, args.total_timesteps + 1):
            # --- ACT re-plan every chunk_size steps ---
            if step_in_chunk >= len(act_chunk) or len(act_chunk) == 0:
                obs_dict_full = raw_env._get_observations()
                act_obs = {}
                for key, val in obs_dict_full.items():
                    if key.startswith("observation."):
                        if isinstance(val, torch.Tensor):
                            act_obs[key] = val.detach().cpu().numpy()[0] if val.dim() > 1 else val.detach().cpu().numpy()
                        else:
                            act_obs[key] = np.asarray(val)

                # Get action chunk + encoder features
                act_chunk = []
                action_np, encoder_features = act.get_action_and_features(act_obs)
                act_chunk.append(action_np)
                # Fill the rest of the chunk
                for _ in range(args.chunk_size - 1):
                    action_np = act.lerobot_policy.select_action(act_obs)
                    act_chunk.append(action_np)
                step_in_chunk = 0

            act_action_np = act_chunk[step_in_chunk]
            step_in_chunk += 1

            act_action = torch.as_tensor(act_action_np, device=rl_device, dtype=torch.float32)
            state_tensor = torch.as_tensor(state, device=rl_device, dtype=torch.float32)

            # Build 536D observation: [state(12), act_action(12), encoder_features(512)]
            residual_obs = torch.cat([state_tensor, act_action, encoder_features.to(rl_device)])

            # --- Residual policy ---
            scale = anneal_scale(step, args.residual_scale, args.residual_scale_max,
                                 args.residual_anneal_steps)

            if step <= args.learning_starts:
                delta = torch.empty(action_dim, device=rl_device).uniform_(-1.0, 1.0)
            else:
                delta = actor.act(residual_obs.unsqueeze(0), deterministic=False).squeeze(0)

            final_action = act_action + scale * delta
            final_action = torch.clamp(final_action, action_low_dev, action_high_dev)

            # --- Step environment ---
            action_for_env = final_action.detach().cpu().numpy().astype(np.float32)
            action_tensor = torch.as_tensor(action_for_env[None], device=env_cfg.sim.device, dtype=torch.float32)
            raw_env.step(action_tensor)

            reward_tensor = raw_env._get_rewards()
            reward_value = tensor_item(reward_tensor)
            next_state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)

            # Episode termination
            done = bool(tensor_item(raw_env._get_success())) if hasattr(raw_env, '_get_success') else False
            if episode_length + 1 >= raw_env.max_episode_length:
                done = True

            # Next observation
            next_act_action_np = act_chunk[min(step_in_chunk, len(act_chunk) - 1)]
            next_act_action = torch.as_tensor(next_act_action_np, device=rl_device, dtype=torch.float32)
            next_state_tensor = torch.as_tensor(next_state, device=rl_device, dtype=torch.float32)
            next_residual_obs = torch.cat([next_state_tensor, next_act_action, encoder_features.to(rl_device)])

            # Store transition
            residual_obs_np = residual_obs.cpu().numpy()
            delta_np = delta.detach().cpu().numpy()
            next_residual_obs_np = next_residual_obs.cpu().numpy()
            if not replay_buffer.add(residual_obs_np, delta_np, reward_value, next_residual_obs_np, done):
                print(f"[WARN] NaN transition at step {step}", flush=True)

            episode_reward += reward_value
            episode_length += 1

            # --- SAC update ---
            if step >= args.learning_starts and replay_buffer.size >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size, rl_device)
                alpha = log_alpha.exp()

                with torch.no_grad():
                    next_deltas, next_log_prob, _ = actor.sample(batch["next_obs"])
                    tgt_q = torch.min(
                        q1_target(batch["next_obs"], next_deltas),
                        q2_target(batch["next_obs"], next_deltas),
                    )
                    tgt_q = batch["rewards"] + (1.0 - batch["dones"]) * args.gamma * (
                        tgt_q - alpha * next_log_prob)

                q1_loss = F.mse_loss(q1(batch["obs"], batch["actions"]), tgt_q)
                q2_loss = F.mse_loss(q2(batch["obs"], batch["actions"]), tgt_q)
                q_optimizer.zero_grad(set_to_none=True)
                (q1_loss + q2_loss).backward()
                torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
                q_optimizer.step()

                new_deltas, log_prob, _ = actor.sample(batch["obs"])
                q_pi = torch.min(q1(batch["obs"], new_deltas), q2(batch["obs"], new_deltas))
                actor_loss = (alpha.detach() * log_prob - q_pi).mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                actor_optimizer.step()

                alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_optimizer.step()

                soft_update(q1, q1_target, args.tau)
                soft_update(q2, q2_target, args.tau)

                if step % args.log_freq == 0:
                    writer.add_scalar("train/q1_loss", float(q1_loss.item()), step)
                    writer.add_scalar("train/actor_loss", float(actor_loss.item()), step)
                    writer.add_scalar("train/alpha", float(alpha), step)

            # --- Logging ---
            if step % args.log_freq == 0:
                writer.add_scalar("env/reward", reward_value, step)
                writer.add_scalar("residual/scale", scale, step)
                delta_mag = float(torch.abs(scale * delta).mean().item())
                act_mag = float(torch.abs(act_action).mean().item())
                writer.add_scalar("residual/delta_magnitude", delta_mag, step)
                writer.add_scalar("residual/act_magnitude", act_mag, step)
                if act_mag > 0:
                    writer.add_scalar("residual/delta_ratio", delta_mag / act_mag, step)

                info = {}
                if hasattr(raw_env, 'extras') and isinstance(raw_env.extras, dict):
                    info = raw_env.extras
                for key in ["dense_score", "primary_score", "secondary_score", "success"]:
                    if key in info:
                        custom_writer.add_scalar(f"env/{key}", tensor_item(info[key]), step)

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

                print(f"[INFO] Episode {episode_index}: return={episode_reward:.1f}, length={episode_length}", flush=True)

                episode_reward = 0.0
                episode_length = 0
                episode_index += 1

                obs_dict = env.reset()[0]
                stabilize_garment_after_reset(raw_env, args, num_steps=20)
                state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)
                act.reset()
                act_chunk = []
                step_in_chunk = 0

            # --- Checkpoint ---
            if step % args.checkpoint_freq == 0 or step == args.total_timesteps:
                ckpt_path = os.path.join(log_dir, f"checkpoint_{step:08d}.pt")
                save_checkpoint(
                    ckpt_path, actor=actor, q1=q1, q2=q2,
                    q1_target=q1_target, q2_target=q2_target,
                    actor_opt=actor_optimizer, q_opt=q_optimizer,
                    alpha_opt=alpha_optimizer, log_alpha=log_alpha,
                    metadata=metadata,
                )
                print(f"[INFO] Saved checkpoint: {ckpt_path}", flush=True)

        # ---- Final save ----
        final_path = os.path.join(log_dir, "model.pt")
        save_checkpoint(
            final_path, actor=actor, q1=q1, q2=q2,
            q1_target=q1_target, q2_target=q2_target,
            actor_opt=actor_optimizer, q_opt=q_optimizer,
            alpha_opt=alpha_optimizer, log_alpha=log_alpha,
            metadata=metadata,
        )
        print(f"[INFO] Final model: {final_path}", flush=True)

        if args.save_buffer:
            buf_path = os.path.join(log_dir, "replay_buffer.npz")
            replay_buffer.save(buf_path)
            print(f"[INFO] Saved replay buffer ({replay_buffer.size} transitions): {buf_path}", flush=True)

    except Exception as e:
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
        print(f"[ERROR] train_encoder_residual_sac.py failed: {exc!r}", flush=True)
        traceback.print_exc()
        raise
