"""Replay buffer with NaN filtering and disk persistence."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> bool:
        """Add a transition. Returns False and skips if any value is NaN."""
        if (
            not np.isfinite(obs).all()
            or not np.isfinite(action).all()
            or not np.isfinite(reward)
            or not np.isfinite(next_obs).all()
        ):
            return False
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return True

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=device),
            "actions": torch.as_tensor(self.actions[indices], device=device),
            "rewards": torch.as_tensor(self.rewards[indices], device=device),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=device),
            "dones": torch.as_tensor(self.dones[indices], device=device),
        }

    def save(self, path: str) -> None:
        """Save valid portion of buffer to compressed .npz file."""
        np.savez_compressed(
            path,
            obs=self.obs[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            next_obs=self.next_obs[: self.size],
            dones=self.dones[: self.size],
        )

    def load(self, path: str) -> int:
        """Load transitions from .npz and append to buffer. Returns count added."""
        data = np.load(path)
        n = len(data["obs"])
        added = 0
        for i in range(n):
            if self.add(
                data["obs"][i],
                data["actions"][i],
                float(data["rewards"][i, 0]),
                data["next_obs"][i],
                bool(data["dones"][i, 0]),
            ):
                added += 1
        return added

    def add_batch(self, demo_data: dict[str, np.ndarray]) -> int:
        """Bulk-add from demo_loader output dict. Returns count added."""
        n = len(demo_data["obs"])
        added = 0
        for i in range(n):
            if self.add(
                demo_data["obs"][i],
                demo_data["actions"][i],
                float(demo_data["rewards"][i]),
                demo_data["next_obs"][i],
                bool(demo_data["dones"][i]),
            ):
                added += 1
        return added
