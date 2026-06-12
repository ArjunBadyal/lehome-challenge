"""Residual SAC fine-tuning of a frozen ACT policy.

Architecture
------------
Every ``chunk_size`` steps:
    obs_dict (with camera images) → ACT → action_chunk [chunk_size, 12]

Every step:
    state [12] + act_action [12] → Residual MLP → delta [12]
    final_action = clip(act_action + scale * delta, joint_low, joint_high)
    env.step(final_action) → reward

The ACT policy is **frozen**; only the residual actor and Q-networks are trained.
"""

import argparse
import logging
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
parser = argparse.ArgumentParser(description="Residual SAC fine-tuning of a frozen ACT policy.")
parser.add_argument("--task", type=str, default="LeHome-BiSO101-Direct-Garment-v2")
parser.add_argument("--garment_name", type=str, default=None,
                     help="Single garment to train on (legacy). Use --train_garments for multi-garment.")
parser.add_argument("--train_garments", type=str, nargs="+", default=None,
                     help="List of garment names to cycle through during training.")
parser.add_argument("--garment_version", type=str, default="Release")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)

# ACT
parser.add_argument("--act_checkpoint", type=str, required=True,
                     help="Path to pretrained ACT checkpoint (LeRobot format).")
parser.add_argument("--dataset_root", type=str, required=True,
                     help="Path to LeRobot dataset root (for ACT metadata).")
parser.add_argument("--chunk_size", type=int, default=100,
                     help="ACT action chunk length.")

# Residual
parser.add_argument("--residual_scale", type=float, default=0.1,
                     help="Initial residual scale (anneals to --residual_scale_max).")
parser.add_argument("--residual_scale_max", type=float, default=0.3,
                     help="Max residual scale after annealing.")
parser.add_argument("--residual_anneal_steps", type=int, default=30000,
                     help="Steps over which residual_scale anneals from min to max.")
parser.add_argument("--residual_hidden", type=int, nargs="+", default=[128, 128],
                     help="Hidden layer sizes for residual MLP.")

# SAC hyper-parameters
parser.add_argument("--total_timesteps", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--buffer_size", type=int, default=200000)
parser.add_argument("--learning_starts", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--init_alpha", type=float, default=0.2)

# ----- Surgical residual controls -----
# Masked residual: restrict which joints the residual is allowed to touch.
# Default mask: indices 3,4,5 (L wrist_flex, wrist_roll, gripper) and 9,10,11
# (R wrist_flex, wrist_roll, gripper). Shoulders / elbows left to ACT.
parser.add_argument(
    "--residual_mask", type=int, nargs="+",
    default=[3, 4, 5, 9, 10, 11],
    help="Indices (0..11) of joints the residual can write to. Others are zeroed.",
)
# Late-phase gate: residual active only after `--gate_steps` sim steps into the
# episode OR once any gripper state has crossed into the 'closed' regime.
parser.add_argument("--gate_steps", type=int, default=100,
                     help="Apply residual only after this many steps into an episode.")
parser.add_argument("--gate_gripper_threshold", type=float, default=-0.05,
                     help="Gripper state (rad) below which we consider 'grasped/closing'.")
# BC pretraining from demos
parser.add_argument("--bc_pretrain_epochs", type=int, default=5,
                     help="Epochs of MSE pretraining on demo deltas. 0 to skip.")
parser.add_argument("--bc_delta_clip", type=float, default=0.05,
                     help="Clip |a_demo - a_base| per joint before using as BC target.")
parser.add_argument("--bc_max_frames", type=int, default=20000,
                     help="Max number of demo frames to use for BC pretraining.")
parser.add_argument("--bc_lr", type=float, default=5e-4,
                     help="Learning rate for BC pretraining of the residual.")
# BC-anchor on SAC actor loss: actor_loss += lambda_bc * ||delta_online - delta_demo_nn||^2
parser.add_argument("--lambda_bc", type=float, default=0.1,
                     help="Weight of BC anchor in SAC actor loss.")
# Mini-suite eval for checkpoint selection
parser.add_argument("--eval_every", type=int, default=2000,
                     help="Run the mini-suite eval every N steps. 0 = disabled.")
parser.add_argument("--eval_garments", type=str, nargs="+",
                     default=["Pant_Long_Seen_3", "Pant_Long_Seen_5",
                              "Pant_Long_Seen_9", "Pant_Long_Unseen_0"],
                     help="Garments for the mini-suite checkpoint selection.")
parser.add_argument("--eval_episodes_per_garment", type=int, default=2)
# Hard abort gate
parser.add_argument("--abort_if_below", type=float, default=0.55,
                     help="Abort if mini-suite success rate is below this at --abort_step.")
parser.add_argument("--abort_step", type=int, default=6000)

# Logging / checkpointing
parser.add_argument("--log_dir", type=str, default="outputs/rl/residual_sac")
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint_freq", type=int, default=5000)
parser.add_argument("--log_freq", type=int, default=250)
parser.add_argument("--rl_device", type=str, default="cuda:0")
parser.add_argument("--checkpoint", type=str, default=None,
                     help="Resume from a residual-SAC checkpoint.")
parser.add_argument("--save_buffer", action="store_true", default=True)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # ACT needs camera images

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Post-launch imports (IsaacLab requirement)
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from torch.utils.tensorboard import SummaryWriter

import lehome.tasks.bedroom  # noqa: F401  — register gym envs

from scripts.rl.residual_model import ResidualActor
from scripts.rl.sac_model import QNetwork
from scripts.rl.replay_buffer import ReplayBuffer
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


def anneal_scale(step: int, min_scale: float, max_scale: float, anneal_steps: int) -> float:
    frac = min(step / max(anneal_steps, 1), 1.0)
    return min_scale + (max_scale - min_scale) * frac


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
        # The environment's success checker logs every probe at INFO. During
        # residual RL that can produce thousands of lines and materially slow
        # CPU simulation through I/O. Keep trainer metrics visible, suppress
        # per-probe cloth distances.
        logging.disable(logging.INFO)
        logging.getLogger("lehome.tasks.bedroom.garment_bi_v2").setLevel(logging.WARNING)
        logging.getLogger("lehome.assets.object.Garment").setLevel(logging.WARNING)

        # ---- Environment ----
        print("[INFO] Loading task configs...", flush=True)
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")

        if args.num_envs != 1:
            raise ValueError("Residual SAC trainer currently supports only --num_envs 1.")

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
        env_cfg.garment_name = train_garments[0]  # initial garment

        # End episodes early on success and shorten max episode length for RL
        env_cfg.terminate_on_success = True
        env_cfg.episode_length_s = 10  # 900 steps at 90Hz
        # Use "recreate" mode so garment swap actually rebuilds the cloth
        env_cfg.garment_reset_mode = "recreate"

        print("[INFO] Creating gym environment...", flush=True)
        env = gym.make(args.task, cfg=env_cfg)
        raw_env = env.unwrapped
        print("[INFO] Gym environment created.", flush=True)

        # Action bounds from the environment
        action_space = raw_env.single_action_space
        action_dim = int(np.prod(action_space.shape))  # 12
        action_low = torch.as_tensor(action_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32)
        state_dim = action_dim  # joint state is same dim as action (12)
        residual_obs_dim = state_dim + action_dim  # 24

        rl_device = torch.device(args.rl_device)
        action_low_dev = action_low.to(rl_device)
        action_high_dev = action_high.to(rl_device)

        # Masked residual: 1.0 on writable joints, 0.0 elsewhere.
        residual_mask = np.zeros(action_dim, dtype=np.float32)
        for idx in args.residual_mask:
            if 0 <= idx < action_dim:
                residual_mask[idx] = 1.0
        residual_mask_dev = torch.as_tensor(residual_mask, device=rl_device)
        print(f"[INFO] Residual mask (writable joints): {args.residual_mask}", flush=True)
        print(f"[INFO] Late-phase gate: gate_steps={args.gate_steps}, "
              f"gripper_threshold={args.gate_gripper_threshold}", flush=True)

        # ---- Logging ----
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_garment = (args.garment_name or train_garments[0]).replace("/", "_")
        run_name = args.run_name or safe_garment
        log_root = os.path.abspath(os.path.join(args.log_dir, args.task))
        log_dir = os.path.join(log_root, f"{timestamp}_{run_name}")
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        Path(log_dir, "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
        writer = SummaryWriter(log_dir)
        custom_writer = SummaryWriter(os.path.join(log_dir, "custom_metrics"))
        print(f"[INFO] Logging to: {log_dir}", flush=True)

        print(f"[INFO] Training garments ({len(train_garments)}): {train_garments}", flush=True)

        # ---- ACT (frozen) ----
        print("[INFO] Loading ACT policy...", flush=True)
        act = ACTChunkProvider(
            policy_path=args.act_checkpoint,
            dataset_root=args.dataset_root,
            device=args.rl_device,
            chunk_size=args.chunk_size,
        )
        print(f"[INFO] ACT loaded. action_dim={act.action_dim}, chunk_size={act.chunk_size}", flush=True)

        # ---- Residual actor + Q-networks ----
        hidden_sizes = tuple(args.residual_hidden)
        actor = ResidualActor(obs_dim=residual_obs_dim, action_dim=action_dim,
                              hidden_sizes=hidden_sizes).to(rl_device)
        q1 = QNetwork(residual_obs_dim, action_dim, hidden_sizes).to(rl_device)
        q2 = QNetwork(residual_obs_dim, action_dim, hidden_sizes).to(rl_device)
        q1_target = QNetwork(residual_obs_dim, action_dim, hidden_sizes).to(rl_device)
        q2_target = QNetwork(residual_obs_dim, action_dim, hidden_sizes).to(rl_device)
        q1_target.load_state_dict(q1.state_dict())
        q2_target.load_state_dict(q2.state_dict())

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
        q_optimizer = torch.optim.Adam(
            list(q1.parameters()) + list(q2.parameters()), lr=args.learning_rate)
        log_alpha = torch.tensor(
            np.log(args.init_alpha), device=rl_device, dtype=torch.float32, requires_grad=True)
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.learning_rate)
        target_entropy = -float(action_dim)

        # ---- Checkpoint resume ----
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
            "obs_dim": residual_obs_dim,
            "action_dim": action_dim,
            "hidden_sizes": list(hidden_sizes),
            "action_low": action_low.cpu(),
            "action_high": action_high.cpu(),
            "residual_scale": args.residual_scale,
            "residual_scale_max": args.residual_scale_max,
            "act_checkpoint": args.act_checkpoint,
            "dataset_root": args.dataset_root,
            "chunk_size": args.chunk_size,
            "trainer": "residual_sac",
            "observation_key": "observation.state",
            "action_semantics": "absolute_joint_positions",
            "submission_safe": True,
            "task_id": args.task,
            "garment_name": train_garments[0],
            "train_garments": train_garments,
            "garment_version": args.garment_version,
        }

        replay_buffer = ReplayBuffer(args.buffer_size, residual_obs_dim, action_dim)

        # ---- BC pretraining from demo deltas ----
        # For each demo (obs, action_t, state_t) pair we compute the frozen-ACT
        # action from the demo observation, then MSE-fit the residual to
        # delta* = clip(a_demo - a_base, -bc_delta_clip, +bc_delta_clip) * mask.
        # This gives the residual a sane starting point instead of random noise.
        bc_obs: list[np.ndarray] = []
        bc_target: list[np.ndarray] = []
        if args.bc_pretrain_epochs > 0:
            print(f"[INFO] Building BC pretraining data "
                  f"(up to {args.bc_max_frames} frames)...", flush=True)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            demo_ds = LeRobotDataset(repo_id="lehome", root=args.dataset_root)
            n_frames = min(len(demo_ds), args.bc_max_frames)
            # Sample evenly across the dataset
            indices = np.linspace(0, len(demo_ds) - 1, num=n_frames, dtype=np.int64)
            # A fresh ACT wrapper that re-plans EVERY frame (select_action).
            # We don't want chunk caching here — each demo frame is independent.
            from scripts.eval_policy.lerobot_policy import LeRobotPolicy
            bc_act_policy = LeRobotPolicy(
                policy_path=args.act_checkpoint,
                dataset_root=args.dataset_root,
                task_description="Fold a garment with bimanual robot arms",
                device=args.rl_device,
            )
            for i, idx in enumerate(indices):
                frame = demo_ds[int(idx)]
                # Build obs dict in the same shape LeRobotPolicy expects
                obs_for_act: dict[str, np.ndarray] = {}
                for key in frame:
                    if not key.startswith("observation."):
                        continue
                    val = frame[key]
                    if isinstance(val, torch.Tensor):
                        val_np = val.detach().cpu().numpy()
                    else:
                        val_np = np.asarray(val)
                    obs_for_act[key] = val_np
                bc_act_policy.reset()
                try:
                    a_base = bc_act_policy.select_action(obs_for_act)
                except Exception as exc:
                    # Skip transient demo frames that ACT rejects.
                    if i == 0:
                        print(f"[WARN] BC first-frame ACT failed: {exc}", flush=True)
                    continue
                a_demo = frame["action"].detach().cpu().numpy()
                state_np = frame["observation.state"].detach().cpu().numpy()
                # Residual observation = [state, a_base]; target = masked clipped delta.
                delta_star = np.clip(
                    a_demo - a_base, -args.bc_delta_clip, args.bc_delta_clip
                ).astype(np.float32)
                delta_star = delta_star * residual_mask
                bc_obs.append(np.concatenate([state_np, a_base]).astype(np.float32))
                bc_target.append(delta_star)

            if bc_obs:
                bc_obs_t = torch.as_tensor(np.stack(bc_obs), device=rl_device)
                bc_target_t = torch.as_tensor(np.stack(bc_target), device=rl_device)
                print(f"[INFO] BC dataset: {bc_obs_t.shape[0]} transitions", flush=True)
                bc_optimizer = torch.optim.Adam(actor.parameters(), lr=args.bc_lr)
                batch = 256
                num = bc_obs_t.shape[0]
                for epoch in range(args.bc_pretrain_epochs):
                    perm = torch.randperm(num, device=rl_device)
                    epoch_loss = 0.0
                    for start in range(0, num, batch):
                        idx = perm[start:start + batch]
                        mean, _ = actor(bc_obs_t[idx])
                        pred = torch.tanh(mean) * residual_mask_dev
                        loss = ((pred - bc_target_t[idx]) ** 2).mean()
                        bc_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        bc_optimizer.step()
                        epoch_loss += loss.item() * idx.shape[0]
                    print(f"[BC] epoch {epoch + 1}/{args.bc_pretrain_epochs}  "
                          f"loss={epoch_loss / num:.6f}", flush=True)
                # Keep the demo tensors for the BC-anchor in SAC updates.
                _bc_cached_obs = bc_obs_t
                _bc_cached_target = bc_target_t
            else:
                print("[WARN] BC dataset empty — skipping pretrain and anchor.", flush=True)
                _bc_cached_obs = None
                _bc_cached_target = None
        else:
            _bc_cached_obs = None
            _bc_cached_target = None

        # ---- Initial reset + garment stabilization ----
        print("[INFO] Initializing garment and resetting environment...", flush=True)
        from scripts.utils.common import stabilize_garment_after_reset

        # Initialize garment particles before first reset (required for cloth physics)
        if hasattr(raw_env, 'object') and raw_env.object is not None:
            if not hasattr(raw_env.object, 'initial_points_positions'):
                raw_env.initialize_obs()
                # Run a few sim steps to settle physics
                if hasattr(raw_env, 'sim') and raw_env.sim is not None:
                    for _ in range(5):
                        raw_env.sim.step(render=True)

        obs_dict = env.reset()[0]
        stabilize_garment_after_reset(raw_env, args, num_steps=20)

        state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)

        act.reset()
        act_chunk: torch.Tensor | None = None

        episode_reward = 0.0
        episode_length = 0
        episode_index = 0
        recent_returns: deque[float] = deque(maxlen=20)

        print(f"[INFO] Starting residual SAC training for {args.total_timesteps} steps...", flush=True)

        for step in range(1, args.total_timesteps + 1):
            chunk_step = (step - 1) % args.chunk_size

            # --- ACT re-plan every chunk_size steps OR after episode reset ---
            # act_chunk can be None if the previous step reset the episode.
            if chunk_step == 0 or act_chunk is None:
                obs_dict_full = raw_env._get_observations()
                # Build numpy observation dict for ACT
                act_obs = {}
                for key, val in obs_dict_full.items():
                    if key.startswith("observation."):
                        if isinstance(val, torch.Tensor):
                            act_obs[key] = val.detach().cpu().numpy()[0] if val.dim() > 1 else val.detach().cpu().numpy()
                        else:
                            act_obs[key] = np.asarray(val)
                act_chunk = act.get_chunk(act_obs)  # [chunk_size, action_dim] on rl_device

            act_action = act_chunk[chunk_step]  # [action_dim] tensor on rl_device
            state_tensor = torch.as_tensor(state, device=rl_device, dtype=torch.float32)
            residual_obs = torch.cat([state_tensor, act_action])  # [24]

            # --- Residual policy ---
            scale = anneal_scale(step, args.residual_scale, args.residual_scale_max,
                                 args.residual_anneal_steps)

            if step <= args.learning_starts:
                # Random exploration: small random delta
                delta = torch.empty(action_dim, device=rl_device).uniform_(-1.0, 1.0)
            else:
                delta = actor.act(residual_obs.unsqueeze(0), deterministic=False).squeeze(0)

            # --- Masked residual: zero out joints not in the mask ---
            delta = delta * residual_mask_dev

            # --- Late-phase gate: disable residual until one of
            #     (a) `gate_steps` episode steps have passed, OR
            #     (b) a gripper has entered the 'closing' regime.
            gripper_state_l = float(state[5])
            gripper_state_r = float(state[11])
            gripper_closed = (gripper_state_l < args.gate_gripper_threshold
                              or gripper_state_r < args.gate_gripper_threshold)
            if episode_length < args.gate_steps and not gripper_closed:
                delta = torch.zeros_like(delta)

            final_action = act_action + scale * delta
            final_action = torch.clamp(final_action, action_low_dev, action_high_dev)

            # --- Step environment ---
            action_np = final_action.detach().cpu().numpy().astype(np.float32)
            action_tensor = torch.as_tensor(action_np[None], device=env_cfg.sim.device, dtype=torch.float32)
            raw_env.step(action_tensor)

            reward_tensor = raw_env._get_rewards()
            reward_value = tensor_item(reward_tensor)
            next_state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)

            # Check for episode termination
            done = bool(tensor_item(raw_env._get_success())) if hasattr(raw_env, '_get_success') else False
            if episode_length + 1 >= raw_env.max_episode_length:
                done = True

            # Next residual obs (for buffer)
            next_act_action = act_chunk[min(chunk_step + 1, args.chunk_size - 1)]
            next_state_tensor = torch.as_tensor(next_state, device=rl_device, dtype=torch.float32)
            next_residual_obs = torch.cat([next_state_tensor, next_act_action]).cpu().numpy()

            # Store transition
            residual_obs_np = residual_obs.cpu().numpy()
            delta_np = delta.detach().cpu().numpy()
            if not replay_buffer.add(residual_obs_np, delta_np, reward_value, next_residual_obs, done):
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
                q_optimizer.step()

                new_deltas, log_prob, _ = actor.sample(batch["obs"])
                q_pi = torch.min(q1(batch["obs"], new_deltas), q2(batch["obs"], new_deltas))
                actor_loss = (alpha.detach() * log_prob - q_pi).mean()

                # BC anchor: keep the residual near the demo-correction distribution.
                # We sample a mini-batch of demo transitions every actor update.
                if (args.lambda_bc > 0.0
                        and _bc_cached_obs is not None
                        and _bc_cached_target is not None):
                    bc_n = _bc_cached_obs.shape[0]
                    bc_idx = torch.randint(0, bc_n, (batch["obs"].shape[0],),
                                           device=rl_device)
                    bc_mean, _ = actor(_bc_cached_obs[bc_idx])
                    bc_pred = torch.tanh(bc_mean) * residual_mask_dev
                    bc_loss = ((bc_pred - _bc_cached_target[bc_idx]) ** 2).mean()
                    actor_loss = actor_loss + args.lambda_bc * bc_loss

                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_optimizer.step()

                alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                alpha_optimizer.step()

                soft_update(q1, q1_target, args.tau)
                soft_update(q2, q2_target, args.tau)

                if step % args.log_freq == 0:
                    writer.add_scalar("train/q1_loss", float(q1_loss.item()), step)
                    writer.add_scalar("train/q2_loss", float(q2_loss.item()), step)
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

                # Env metrics from info
                info = {}
                if hasattr(raw_env, 'extras') and isinstance(raw_env.extras, dict):
                    info = raw_env.extras
                for key in ["dense_score", "primary_score", "secondary_score",
                            "condition_reward", "progress_reward", "success"]:
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

                episode_reward = 0.0
                episode_length = 0
                episode_index += 1

                # Cycle garment for next episode
                next_garment = train_garments[episode_index % len(train_garments)]
                env_cfg.garment_name = next_garment
                raw_env.cfg.garment_name = next_garment
                print(f"[INFO] Episode {episode_index}: garment={next_garment}", flush=True)

                # Reset (recreate mode rebuilds the cloth for the new garment)
                obs_dict = env.reset()[0]
                stabilize_garment_after_reset(raw_env, args, num_steps=20)
                state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)
                act.reset()
                act_chunk = None  # Force re-plan on next iteration

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

            # --- Mini-suite eval (every --eval_every steps) ---
            if (args.eval_every > 0 and step % args.eval_every == 0
                    and step >= args.learning_starts):
                # Save current training-loop bookkeeping; we'll restart the
                # episode after the eval since switching garments forces a
                # reset anyway.
                mini_current_garment = env_cfg.garment_name
                mini_successes = 0
                mini_total = 0
                actor.eval()
                try:
                    for eval_gar in args.eval_garments:
                        for _ep in range(args.eval_episodes_per_garment):
                            env_cfg.garment_name = eval_gar
                            raw_env.cfg.garment_name = eval_gar
                            if hasattr(raw_env, "switch_garment"):
                                try:
                                    raw_env.switch_garment(eval_gar, args.garment_version)
                                except Exception:
                                    pass
                            _ = env.reset()[0]
                            stabilize_garment_after_reset(raw_env, args, num_steps=20)
                            s = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)
                            act.reset()
                            ep_chunk: torch.Tensor | None = None
                            ep_success = False
                            for ep_step in range(raw_env.max_episode_length):
                                cs = ep_step % args.chunk_size
                                if cs == 0 or ep_chunk is None:
                                    od = raw_env._get_observations()
                                    ao = {}
                                    for key, val in od.items():
                                        if key.startswith("observation."):
                                            if isinstance(val, torch.Tensor):
                                                ao[key] = val.detach().cpu().numpy()[0] if val.dim() > 1 else val.detach().cpu().numpy()
                                            else:
                                                ao[key] = np.asarray(val)
                                    ep_chunk = act.get_chunk(ao)
                                a_base = ep_chunk[cs]
                                s_t = torch.as_tensor(s, device=rl_device, dtype=torch.float32)
                                ro = torch.cat([s_t, a_base])
                                d = actor.act(ro.unsqueeze(0), deterministic=True).squeeze(0)
                                d = d * residual_mask_dev
                                gate_l = float(s[5]) < args.gate_gripper_threshold
                                gate_r = float(s[11]) < args.gate_gripper_threshold
                                if ep_step < args.gate_steps and not (gate_l or gate_r):
                                    d = torch.zeros_like(d)
                                fa = torch.clamp(
                                    a_base + args.residual_scale_max * d,
                                    action_low_dev, action_high_dev,
                                )
                                at = torch.as_tensor(
                                    fa.detach().cpu().numpy()[None],
                                    device=env_cfg.sim.device, dtype=torch.float32,
                                )
                                raw_env.step(at)
                                s = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)
                                if hasattr(raw_env, "_get_success"):
                                    try:
                                        if bool(tensor_item(raw_env._get_success())):
                                            ep_success = True
                                            break
                                    except Exception:
                                        pass
                            mini_successes += int(ep_success)
                            mini_total += 1
                finally:
                    actor.train()
                rate = mini_successes / max(mini_total, 1)
                print(f"[MINI-SUITE] step {step}: {mini_successes}/{mini_total} = {rate:.3f}", flush=True)
                writer.add_scalar("eval/mini_suite_success_rate", rate, step)
                # Hard abort gate
                if step >= args.abort_step and rate < args.abort_if_below:
                    print(f"[ABORT] mini-suite {rate:.3f} < {args.abort_if_below} at step "
                          f"{step} — stopping training as planned.", flush=True)
                    break
                # Force episode reset for next iteration.
                env_cfg.garment_name = mini_current_garment
                raw_env.cfg.garment_name = mini_current_garment
                if hasattr(raw_env, "switch_garment"):
                    try:
                        raw_env.switch_garment(mini_current_garment, args.garment_version)
                    except Exception:
                        pass
                _ = env.reset()[0]
                stabilize_garment_after_reset(raw_env, args, num_steps=20)
                state = raw_env._get_joint_position_tensor().detach().cpu().numpy()[0].astype(np.float32)
                act.reset()
                act_chunk = None
                episode_reward = 0.0
                episode_length = 0

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
        print(f"[ERROR] train_residual_sac.py failed: {exc!r}", flush=True)
        traceback.print_exc()
        raise
