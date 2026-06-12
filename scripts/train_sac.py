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
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl.replay_buffer import ReplayBuffer
from scripts.rl.sac_model import QNetwork, SquashedGaussianActor


def extract_policy_obs(obs_dict: dict) -> np.ndarray:
    policy_obs = obs_dict["policy"]
    if isinstance(policy_obs, torch.Tensor):
        return policy_obs.detach().cpu().numpy()[0].astype(np.float32, copy=False)
    return np.asarray(policy_obs, dtype=np.float32)[0]


def tensor_item(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0].item())
    return float(value)


parser = argparse.ArgumentParser(description="Train a custom SAC policy on the LeHome garment folding task.")
parser.add_argument(
    "--task",
    type=str,
    default="LeHome-BiSO101-Direct-Garment-SAC-v0",
    help="Gym task id.",
)
parser.add_argument("--garment_name", type=str, required=True, help="Garment asset name to train on.")
parser.add_argument("--garment_version", type=str, default="Release", help="Garment version to use.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--total_timesteps", type=int, default=None, help="Override total SAC timesteps.")
parser.add_argument("--log_dir", type=str, default="outputs/rl/sac", help="Root directory for SAC runs.")
parser.add_argument("--run_name", type=str, default=None, help="Optional run name suffix.")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from a custom SAC checkpoint (.pt).")
parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Checkpoint frequency in env steps.")
parser.add_argument("--log_freq", type=int, default=250, help="TensorBoard logging frequency in env steps.")
parser.add_argument(
    "--rl_device",
    type=str,
    default="cuda:0",
    help="Torch device for the SAC model.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record training videos.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in environment steps.")
parser.add_argument("--video_interval", type=int, default=5000, help="Video trigger interval in environment steps.")
parser.add_argument("--demo_path", type=str, default=None, help="Path to LeRobot dataset root for demo pre-filling.")
parser.add_argument("--demo_reward", type=float, default=0.0, help="Constant reward assigned to demo transitions.")
parser.add_argument("--demo_max_episodes", type=int, default=None, help="Max demo episodes to load (None = all).")
parser.add_argument("--save_buffer", action="store_true", default=True, help="Save replay buffer to disk after training.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import lehome.tasks.bedroom  # noqa: F401


def save_checkpoint(
    path: str,
    actor: SquashedGaussianActor,
    q1: QNetwork,
    q2: QNetwork,
    q1_target: QNetwork,
    q2_target: QNetwork,
    actor_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    alpha_optimizer: torch.optim.Optimizer,
    log_alpha: torch.Tensor,
    metadata: dict,
) -> None:
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "q1_state_dict": q1.state_dict(),
            "q2_state_dict": q2.state_dict(),
            "q1_target_state_dict": q1_target.state_dict(),
            "q2_target_state_dict": q2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "q_optimizer_state_dict": q_optimizer.state_dict(),
            "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
            "log_alpha": log_alpha.detach().cpu(),
            "metadata": metadata,
        },
        path,
    )


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
        target_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def main():
    env = None
    vec_env = None
    writer = None
    custom_writer = None
    start_time = time.time()
    try:
        print("[INFO] Loading task configs...", flush=True)
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args.task, "sb3_sac_cfg_entry_point")

        if args.num_envs != 1:
            raise ValueError("Custom SAC trainer currently supports only --num_envs 1.")

        env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        env_cfg.garment_name = args.garment_name
        env_cfg.garment_version = args.garment_version

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_garment_name = args.garment_name.replace("/", "_")
        run_name = args.run_name or safe_garment_name
        log_root_path = os.path.abspath(os.path.join(args.log_dir, args.task))
        log_dir = os.path.join(log_root_path, f"{timestamp}_{run_name}")
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

        print(f"[INFO] Logging experiment in directory: {log_root_path}", flush=True)
        print(f"Exact experiment name requested from command line: {os.path.basename(log_dir)}", flush=True)
        print(f"[INFO] Simulation device: {env_cfg.sim.device}", flush=True)
        print(f"[INFO] RL device: {args.rl_device}", flush=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        Path(log_dir, "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")

        print("[INFO] Creating gym environment...", flush=True)
        env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        print("[INFO] Gym environment created.", flush=True)

        if args.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args.video_interval == 0,
                "video_length": args.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        vec_env = Sb3VecEnvWrapper(env, fast_variant=False)
        obs_dim = int(np.prod(env.unwrapped.single_observation_space["policy"].shape))
        action_space = env.unwrapped.single_action_space
        action_dim = int(np.prod(action_space.shape))
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        rl_device = torch.device(args.rl_device)

        hidden_sizes = tuple(agent_cfg.get("policy_kwargs", {}).get("net_arch", [256, 256, 256]))
        total_timesteps = int(args.total_timesteps or agent_cfg.get("n_timesteps", 300000))
        batch_size = int(agent_cfg.get("batch_size", 256))
        buffer_size = int(agent_cfg.get("buffer_size", 100000))
        learning_starts = int(agent_cfg.get("learning_starts", 2000))
        gamma = float(agent_cfg.get("gamma", 0.99))
        tau = float(agent_cfg.get("tau", 0.005))
        learning_rate = float(agent_cfg.get("learning_rate", 3e-4))
        train_freq = int(agent_cfg.get("train_freq", 1))
        gradient_steps = int(agent_cfg.get("gradient_steps", 1))
        target_entropy_cfg = agent_cfg.get("target_entropy", "auto")
        target_entropy = -float(action_dim) if target_entropy_cfg == "auto" else float(target_entropy_cfg)
        ent_coef_cfg = str(agent_cfg.get("ent_coef", "auto_0.2"))
        init_alpha = float(ent_coef_cfg.split("_", 1)[1]) if ent_coef_cfg.startswith("auto_") else float(ent_coef_cfg)

        actor = SquashedGaussianActor(obs_dim, action_dim, action_low, action_high, hidden_sizes).to(rl_device)
        q1 = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q2 = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q1_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q2_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(rl_device)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
        q_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=learning_rate)
        log_alpha = torch.tensor(np.log(init_alpha), device=rl_device, dtype=torch.float32, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=learning_rate)

        metadata = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_sizes": list(hidden_sizes),
            "action_low": action_low.cpu(),
            "action_high": action_high.cpu(),
            "target_entropy": target_entropy,
            "observation_key": "observation.state",
            "action_semantics": "absolute_joint_positions",
            "submission_safe": True,
            "task_id": args.task,
            "garment_name": args.garment_name,
            "garment_version": args.garment_version,
        }

        if args.checkpoint:
            print(f"[INFO] Loading checkpoint from: {args.checkpoint}", flush=True)
            checkpoint = torch.load(args.checkpoint, map_location=rl_device)
            actor.load_state_dict(checkpoint["actor_state_dict"])
            q1.load_state_dict(checkpoint["q1_state_dict"])
            q2.load_state_dict(checkpoint["q2_state_dict"])
            q1_target.load_state_dict(checkpoint["q1_target_state_dict"])
            q2_target.load_state_dict(checkpoint["q2_target_state_dict"])
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
            log_alpha = checkpoint["log_alpha"].to(device=rl_device).requires_grad_(True)

        writer = SummaryWriter(log_dir)
        custom_writer = SummaryWriter(os.path.join(log_dir, "custom_metrics"))
        print("[INFO] TensorBoard logs:", flush=True)
        print(f"  Main: {log_dir}", flush=True)
        print(f"  Custom metrics: {os.path.join(log_dir, 'custom_metrics')}", flush=True)
        print(f"  Launch with: tensorboard --logdir {os.path.abspath(args.log_dir)}", flush=True)

        replay_buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        # Pre-fill replay buffer with demonstrations
        if args.demo_path:
            from scripts.rl.demo_loader import load_demo_transitions

            print(f"[INFO] Loading demonstrations from: {args.demo_path}", flush=True)
            demo_data = load_demo_transitions(
                args.demo_path,
                demo_reward=args.demo_reward,
                max_episodes=args.demo_max_episodes,
            )
            n_demo = replay_buffer.add_batch(demo_data)
            print(f"[INFO] Pre-filled replay buffer with {n_demo} demo transitions", flush=True)
            writer.add_scalar("demo/transitions_loaded", n_demo, 0)

            # Skip random exploration if we have enough demos
            if replay_buffer.size >= learning_starts:
                print(
                    f"[INFO] Demo buffer ({replay_buffer.size}) >= learning_starts ({learning_starts}), "
                    "training will begin immediately",
                    flush=True,
                )

        print("[INFO] Running initial env.reset()...", flush=True)
        obs = vec_env.reset()[0].astype(np.float32, copy=False)
        print(f"[INFO] Initial observation shape: {obs.shape}", flush=True)

        episode_reward = 0.0
        episode_length = 0
        episode_index = 0
        recent_returns: deque[float] = deque(maxlen=20)

        for step in range(1, total_timesteps + 1):
            if step <= learning_starts and replay_buffer.size < learning_starts:
                action = np.random.uniform(action_space.low, action_space.high).astype(np.float32)
            else:
                obs_tensor = torch.as_tensor(obs[None], device=rl_device, dtype=torch.float32)
                action = actor.act(obs_tensor, deterministic=False).detach().cpu().numpy()[0].astype(np.float32)

            next_obs_batch, reward_batch, done_batch, infos = vec_env.step(action[None])
            next_obs = next_obs_batch[0].astype(np.float32, copy=False)
            reward_value = float(reward_batch[0])
            done = bool(done_batch[0])
            info = infos[0] if infos else {}

            if not replay_buffer.add(obs, action, reward_value, next_obs, done):
                print(f"[WARN] Skipped NaN transition at step {step}", flush=True)
            episode_reward += reward_value
            episode_length += 1

            if step >= learning_starts and replay_buffer.size >= batch_size and step % train_freq == 0:
                for grad_step in range(gradient_steps):
                    batch = replay_buffer.sample(batch_size, rl_device)
                    alpha = log_alpha.exp()

                    with torch.no_grad():
                        next_actions, next_log_prob, _ = actor.sample(batch["next_obs"])
                        target_q = torch.min(
                            q1_target(batch["next_obs"], next_actions),
                            q2_target(batch["next_obs"], next_actions),
                        )
                        target_q = batch["rewards"] + (1.0 - batch["dones"]) * gamma * (target_q - alpha * next_log_prob)

                    q1_loss = F.mse_loss(q1(batch["obs"], batch["actions"]), target_q)
                    q2_loss = F.mse_loss(q2(batch["obs"], batch["actions"]), target_q)
                    q_optimizer.zero_grad(set_to_none=True)
                    (q1_loss + q2_loss).backward()
                    q_optimizer.step()

                    new_actions, log_prob, _ = actor.sample(batch["obs"])
                    q_pi = torch.min(q1(batch["obs"], new_actions), q2(batch["obs"], new_actions))
                    actor_loss = (alpha.detach() * log_prob - q_pi).mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_optimizer.step()

                    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                    alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    soft_update(q1, q1_target, tau)
                    soft_update(q2, q2_target, tau)

                if step % args.log_freq == 0:
                    writer.add_scalar("train/q1_loss", float(q1_loss.item()), step)
                    writer.add_scalar("train/q2_loss", float(q2_loss.item()), step)
                    writer.add_scalar("train/actor_loss", float(actor_loss.item()), step)
                    writer.add_scalar("train/alpha_loss", float(alpha_loss.item()), step)
                    writer.add_scalar("train/alpha", float(log_alpha.exp().item()), step)
                    writer.add_scalar("train/replay_size", replay_buffer.size, step)

            if step % args.log_freq == 0:
                for key in [
                    "dense_score",
                    "primary_score",
                    "secondary_score",
                    "condition_reward",
                    "progress_reward",
                    "action_penalty",
                    "joint_velocity_penalty",
                    "success_bonus",
                    "condition_margin_min",
                    "success",
                    "invalid_state",
                ]:
                    if key in info:
                        custom_writer.add_scalar(f"env/{key}", tensor_item(info[key]), step)
                writer.add_scalar("env/reward", reward_value, step)

            obs = next_obs

            if done:
                recent_returns.append(episode_reward)
                writer.add_scalar("episode/return", episode_reward, step)
                writer.add_scalar("episode/length", episode_length, step)
                writer.add_scalar("episode/index", episode_index, step)
                if "episode" in info and isinstance(info["episode"], dict):
                    for key, value in info["episode"].items():
                        writer.add_scalar(f"episode/{key}", float(value), step)
                if recent_returns:
                    writer.add_scalar("episode/return_mean_20", float(np.mean(recent_returns)), step)

                episode_reward = 0.0
                episode_length = 0
                episode_index += 1
                obs = vec_env.reset()[0].astype(np.float32, copy=False)

            if step % args.checkpoint_freq == 0 or step == total_timesteps:
                checkpoint_path = os.path.join(log_dir, f"checkpoint_{step:08d}.pt")
                save_checkpoint(
                    checkpoint_path,
                    actor,
                    q1,
                    q2,
                    q1_target,
                    q2_target,
                    actor_optimizer,
                    q_optimizer,
                    alpha_optimizer,
                    log_alpha,
                    metadata,
                )
                print(f"[INFO] Saved checkpoint: {checkpoint_path}", flush=True)

        final_model_path = os.path.join(log_dir, "model.pt")
        save_checkpoint(
            final_model_path,
            actor,
            q1,
            q2,
            q1_target,
            q2_target,
            actor_optimizer,
            q_optimizer,
            alpha_optimizer,
            log_alpha,
            metadata,
        )
        print("[INFO] SAC learn() finished.", flush=True)
        print(f"[INFO] Final model: {final_model_path}", flush=True)

        if args.save_buffer:
            buffer_path = os.path.join(log_dir, "replay_buffer.npz")
            replay_buffer.save(buffer_path)
            print(f"[INFO] Saved replay buffer ({replay_buffer.size} transitions) to: {buffer_path}", flush=True)
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        if custom_writer is not None:
            custom_writer.flush()
            custom_writer.close()
        print(f"Training time: {round(time.time() - start_time, 2)} seconds", flush=True)
        if vec_env is not None:
            vec_env.close()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"[ERROR] train_sac.py failed: {exc!r}", flush=True)
        traceback.print_exc()
        raise
