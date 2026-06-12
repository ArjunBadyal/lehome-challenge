"""Load LeRobot demonstration datasets into SAC replay buffer format.

Converts (observation.state, action) trajectories from LeRobot Parquet
files into (obs, action, reward, next_obs, done) transitions suitable
for pre-filling an off-policy replay buffer.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_demo_transitions(
    dataset_root: str,
    *,
    demo_reward: float = 0.0,
    max_episodes: int | None = None,
) -> dict[str, np.ndarray]:
    """Load demonstration data and convert to SAC replay buffer transitions.

    Args:
        dataset_root: Path to the LeRobot dataset root (containing data/ and meta/).
        demo_reward: Constant reward assigned to each demo transition.
            Expert demos are high-quality so a small positive reward (or 0)
            is reasonable.  The Q-network will learn the actual values.
        max_episodes: If set, only load this many episodes.

    Returns:
        Dictionary with keys ``obs``, ``actions``, ``rewards``,
        ``next_obs``, ``dones`` — each a numpy float32 array ready for
        ``ReplayBuffer.add()``.
    """
    dataset_root = Path(dataset_root)
    parquet_files = sorted(glob.glob(str(dataset_root / "data" / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True).sort_values(["episode_index", "frame_index"])

    episodes = sorted(df["episode_index"].unique())
    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []
    all_next_obs: list[np.ndarray] = []
    all_dones: list[np.ndarray] = []

    for ep_idx in episodes:
        ep = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.stack(ep["observation.state"].values).astype(np.float32)
        actions = np.stack(ep["action"].values).astype(np.float32)

        n = len(states)
        if n < 2:
            continue

        # Each transition: (s_t, a_t, r, s_{t+1}, done)
        obs = states[:-1]  # (N-1, obs_dim)
        next_obs = states[1:]  # (N-1, obs_dim)
        acts = actions[:-1]  # (N-1, action_dim)
        rewards = np.full((n - 1,), demo_reward, dtype=np.float32)
        dones = np.zeros(n - 1, dtype=np.float32)
        dones[-1] = 1.0  # last transition in episode is terminal

        all_obs.append(obs)
        all_actions.append(acts)
        all_rewards.append(rewards)
        all_next_obs.append(next_obs)
        all_dones.append(dones)

    return {
        "obs": np.concatenate(all_obs),
        "actions": np.concatenate(all_actions),
        "rewards": np.concatenate(all_rewards),
        "next_obs": np.concatenate(all_next_obs),
        "dones": np.concatenate(all_dones),
    }


def prefill_replay_buffer(replay_buffer, demo_data: dict[str, np.ndarray]) -> int:
    """Add demo transitions to an existing ReplayBuffer.

    Args:
        replay_buffer: A ReplayBuffer instance with an ``add()`` method.
        demo_data: Output of ``load_demo_transitions()``.

    Returns:
        Number of transitions added.
    """
    n = len(demo_data["obs"])
    for i in range(n):
        replay_buffer.add(
            obs=demo_data["obs"][i],
            action=demo_data["actions"][i],
            reward=float(demo_data["rewards"][i]),
            next_obs=demo_data["next_obs"][i],
            done=bool(demo_data["dones"][i]),
        )
    return n
