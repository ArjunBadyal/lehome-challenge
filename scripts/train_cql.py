"""Offline CQL (Conservative Q-Learning) trainer.

Trains a policy from demonstration data + reward-labeled online transitions
without requiring a simulator.  Runs entirely on GPU.

Usage:
    python scripts/train_cql.py \
        --demo_path Datasets/example/top_long_merged \
        --online_buffer_paths outputs/rl/sac/.../replay_buffer.npz \
        --alpha_cql 5.0 --total_steps 50000 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl.demo_loader import load_demo_transitions
from scripts.rl.replay_buffer import ReplayBuffer
from scripts.rl.sac_model import QNetwork, SquashedGaussianActor

# SO101 dual-arm joint limits (radians), 6 joints × 2 arms = 12
_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
DEFAULT_ACTION_LOW = torch.tensor(
    [np.deg2rad(_JOINT_LIMITS_DEG[j][0]) for _ in range(2) for j in _JOINT_NAMES],
    dtype=torch.float32,
)
DEFAULT_ACTION_HIGH = torch.tensor(
    [np.deg2rad(_JOINT_LIMITS_DEG[j][1]) for _ in range(2) for j in _JOINT_NAMES],
    dtype=torch.float32,
)


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    for sp, tp in zip(source.parameters(), target.parameters(), strict=True):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)


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


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    start_time = time.time()

    # --- 1. Load data ---
    print("[INFO] Loading demonstration data...", flush=True)
    demo_data = load_demo_transitions(
        args.demo_path,
        demo_reward=args.demo_reward,
        max_episodes=args.demo_max_episodes,
    )
    obs_dim = demo_data["obs"].shape[1]
    action_dim = demo_data["actions"].shape[1]

    demo_buffer = ReplayBuffer(len(demo_data["obs"]) + 1, obs_dim, action_dim)
    n_demo = demo_buffer.add_batch(demo_data)
    print(f"[INFO] Loaded {n_demo} demo transitions (reward={args.demo_reward})", flush=True)

    online_buffer = ReplayBuffer(500_000, obs_dim, action_dim)
    for buf_path in args.online_buffer_paths or []:
        n = online_buffer.load(buf_path)
        print(f"[INFO] Loaded {n} online transitions from {buf_path}", flush=True)

    total = demo_buffer.size + online_buffer.size
    print(f"[INFO] Total data: {demo_buffer.size} demo + {online_buffer.size} online = {total}", flush=True)
    if total == 0:
        raise ValueError("No data loaded. Provide --demo_path and/or --online_buffer_paths.")

    # --- 2. Build networks ---
    action_low = DEFAULT_ACTION_LOW.clone()
    action_high = DEFAULT_ACTION_HIGH.clone()
    hidden_sizes = tuple(args.hidden_sizes)

    if args.checkpoint:
        print(f"[INFO] Loading checkpoint: {args.checkpoint}", flush=True)
        ckpt = torch.load(args.checkpoint, map_location=device)
        meta = ckpt["metadata"]
        action_low = torch.as_tensor(meta["action_low"], dtype=torch.float32)
        action_high = torch.as_tensor(meta["action_high"], dtype=torch.float32)
        hidden_sizes = tuple(meta["hidden_sizes"])

    actor = SquashedGaussianActor(obs_dim, action_dim, action_low, action_high, hidden_sizes).to(device)
    q1 = QNetwork(obs_dim, action_dim, hidden_sizes).to(device)
    q2 = QNetwork(obs_dim, action_dim, hidden_sizes).to(device)
    q1_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(device)
    q2_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    if args.checkpoint:
        actor.load_state_dict(ckpt["actor_state_dict"])
        q1.load_state_dict(ckpt["q1_state_dict"])
        q2.load_state_dict(ckpt["q2_state_dict"])
        q1_target.load_state_dict(ckpt["q1_target_state_dict"])
        q2_target.load_state_dict(ckpt["q2_target_state_dict"])
        print("[INFO] Loaded weights from checkpoint", flush=True)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    q_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    log_alpha = torch.tensor(np.log(0.2), device=device, dtype=torch.float32, requires_grad=True)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(action_dim)

    action_low_dev = action_low.to(device)
    action_high_dev = action_high.to(device)

    # --- 3. Logging ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"cql_{timestamp}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

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
        "trainer": "cql",
        "alpha_cql": args.alpha_cql,
        "demo_reward": args.demo_reward,
    }

    print(f"[INFO] CQL training: {args.total_steps} steps, alpha_cql={args.alpha_cql}", flush=True)
    print(f"[INFO] Logging to: {log_dir}", flush=True)
    print(f"[INFO] Device: {device}", flush=True)

    writer.add_scalar("data/demo_transitions", demo_buffer.size, 0)
    writer.add_scalar("data/online_transitions", online_buffer.size, 0)

    # --- 4. Training loop ---
    N = args.num_cql_samples

    for step in range(1, args.total_steps + 1):
        # 4a. Sample mixed batch
        if online_buffer.size > 0:
            demo_bs = int(args.batch_size * args.demo_ratio)
            online_bs = args.batch_size - demo_bs
            demo_batch = demo_buffer.sample(demo_bs, device)
            online_batch = online_buffer.sample(online_bs, device)
            batch = {k: torch.cat([demo_batch[k], online_batch[k]], dim=0) for k in demo_batch}
        else:
            batch = demo_buffer.sample(args.batch_size, device)

        alpha = log_alpha.exp()

        # 4b. Bellman target
        with torch.no_grad():
            next_a, next_logp, _ = actor.sample(batch["next_obs"])
            target_q = torch.min(
                q1_target(batch["next_obs"], next_a),
                q2_target(batch["next_obs"], next_a),
            )
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * args.gamma * (target_q - alpha * next_logp)

        # 4c. Q loss = Bellman + CQL penalty
        q1_pred = q1(batch["obs"], batch["actions"])
        q2_pred = q2(batch["obs"], batch["actions"])
        bellman_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        # CQL: sample random + policy actions, penalize OOD Q-values
        B = batch["obs"].shape[0]
        obs_rep = batch["obs"].unsqueeze(1).expand(-1, N, -1).reshape(B * N, obs_dim)

        # Random actions uniform in action space
        rand_actions = torch.rand(B * N, action_dim, device=device)
        rand_actions = rand_actions * (action_high_dev - action_low_dev) + action_low_dev

        # Policy actions
        policy_actions, policy_logp, _ = actor.sample(obs_rep)

        q1_rand = q1(obs_rep, rand_actions).reshape(B, N, 1)
        q2_rand = q2(obs_rep, rand_actions).reshape(B, N, 1)
        q1_pi = q1(obs_rep, policy_actions).reshape(B, N, 1) - policy_logp.reshape(B, N, 1)
        q2_pi = q2(obs_rep, policy_actions).reshape(B, N, 1) - policy_logp.reshape(B, N, 1)

        q1_cat = torch.cat([q1_rand, q1_pi], dim=1)  # (B, 2N, 1)
        q2_cat = torch.cat([q2_rand, q2_pi], dim=1)

        cql_q1_ood = torch.logsumexp(q1_cat, dim=1).mean()
        cql_q2_ood = torch.logsumexp(q2_cat, dim=1).mean()
        cql_loss = (cql_q1_ood - q1_pred.mean()) + (cql_q2_ood - q2_pred.mean())

        q_loss = bellman_loss + args.alpha_cql * cql_loss

        q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), args.grad_clip)
        q_optimizer.step()

        # 4d. Actor update
        new_a, logp, _ = actor.sample(batch["obs"])
        q_pi = torch.min(q1(batch["obs"], new_a), q2(batch["obs"], new_a))
        actor_loss = (alpha.detach() * logp - q_pi).mean()

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
        actor_optimizer.step()

        # 4e. Alpha update
        alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
        alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optimizer.step()

        # 4f. Target soft update
        soft_update(q1, q1_target, args.tau)
        soft_update(q2, q2_target, args.tau)

        # 4g. Logging
        if step % args.log_freq == 0:
            writer.add_scalar("train/bellman_loss", bellman_loss.item(), step)
            writer.add_scalar("train/cql_loss", cql_loss.item(), step)
            writer.add_scalar("train/q_loss_total", q_loss.item(), step)
            writer.add_scalar("train/actor_loss", actor_loss.item(), step)
            writer.add_scalar("train/alpha_loss", alpha_loss.item(), step)
            writer.add_scalar("train/alpha", alpha.item(), step)
            writer.add_scalar("train/q1_mean", q1_pred.mean().item(), step)
            writer.add_scalar("train/q2_mean", q2_pred.mean().item(), step)
            print(
                f"step:{step} bellman:{bellman_loss.item():.4f} cql:{cql_loss.item():.4f} "
                f"actor:{actor_loss.item():.4f} alpha:{alpha.item():.4f} "
                f"q1:{q1_pred.mean().item():.4f}",
                flush=True,
            )

        # 4h. Checkpointing
        if step % args.checkpoint_freq == 0 or step == args.total_steps:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{step:08d}.pt")
            save_checkpoint(
                ckpt_path, actor, q1, q2, q1_target, q2_target,
                actor_optimizer, q_optimizer, alpha_optimizer, log_alpha, metadata,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}", flush=True)

    # Save final model
    final_path = os.path.join(log_dir, "model.pt")
    save_checkpoint(
        final_path, actor, q1, q2, q1_target, q2_target,
        actor_optimizer, q_optimizer, alpha_optimizer, log_alpha, metadata,
    )
    print(f"[INFO] Final model: {final_path}", flush=True)

    writer.flush()
    writer.close()
    elapsed = time.time() - start_time
    print(f"[INFO] CQL training complete. Time: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline CQL trainer (no simulator required)")
    parser.add_argument("--demo_path", type=str, required=True, help="Path to LeRobot dataset root.")
    parser.add_argument("--online_buffer_paths", type=str, nargs="*", default=None, help="Paths to .npz replay buffers from online SAC runs.")
    parser.add_argument("--demo_reward", type=float, default=0.1, help="Reward assigned to demo transitions.")
    parser.add_argument("--demo_max_episodes", type=int, default=None, help="Max demo episodes to load.")
    parser.add_argument("--demo_ratio", type=float, default=0.5, help="Fraction of each batch from demo data.")
    parser.add_argument("--alpha_cql", type=float, default=5.0, help="CQL conservatism coefficient.")
    parser.add_argument("--num_cql_samples", type=int, default=10, help="Number of random/policy actions for CQL penalty.")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total gradient updates.")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network soft update rate.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 256, 256], help="MLP hidden layer sizes.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Warm-start from SAC/CQL checkpoint.")
    parser.add_argument("--log_dir", type=str, default="outputs/rl/cql", help="Output directory.")
    parser.add_argument("--run_name", type=str, default=None, help="Run name.")
    parser.add_argument("--checkpoint_freq", type=int, default=5000, help="Checkpoint frequency.")
    parser.add_argument("--log_freq", type=int, default=100, help="Log frequency.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
