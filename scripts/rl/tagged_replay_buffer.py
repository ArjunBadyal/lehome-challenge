"""Replay buffer with sub-policy tagging for hierarchical RL."""

from __future__ import annotations

import numpy as np
import torch


class TaggedReplayBuffer:
    """Replay buffer that tags each transition with a sub-policy index.

    Supports filtered sampling: ``sample_for_subpolicy(k, ...)`` returns
    only transitions collected by sub-policy *k*.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.sub_policy_ids = np.full(self.capacity, -1, dtype=np.int8)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        sub_policy_id: int,
    ) -> bool:
        """Add a transition tagged with a sub-policy index. Returns False on NaN."""
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
        self.sub_policy_ids[self.ptr] = sub_policy_id
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return True

    def count_for_subpolicy(self, k: int) -> int:
        """Number of transitions tagged with sub-policy k."""
        return int((self.sub_policy_ids[:self.size] == k).sum())

    def sample_for_subpolicy(
        self, k: int, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor] | None:
        """Sample a batch of transitions for sub-policy k only.

        Returns None if fewer than ``batch_size`` transitions exist for k.
        """
        mask = self.sub_policy_ids[:self.size] == k
        valid_indices = np.where(mask)[0]
        if len(valid_indices) < batch_size:
            return None

        chosen = np.random.choice(valid_indices, size=batch_size, replace=False)
        return {
            "obs": torch.as_tensor(self.obs[chosen], device=device),
            "actions": torch.as_tensor(self.actions[chosen], device=device),
            "rewards": torch.as_tensor(self.rewards[chosen], device=device),
            "next_obs": torch.as_tensor(self.next_obs[chosen], device=device),
            "dones": torch.as_tensor(self.dones[chosen], device=device),
        }

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            obs=self.obs[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_obs=self.next_obs[:self.size],
            dones=self.dones[:self.size],
            sub_policy_ids=self.sub_policy_ids[:self.size],
        )

    def load(self, path: str) -> int:
        data = np.load(path)
        size = min(len(data["obs"]), self.capacity)
        if size == 0:
            self.ptr = 0
            self.size = 0
            self.sub_policy_ids.fill(-1)
            return 0

        self.obs[:size] = data["obs"][:size]
        self.actions[:size] = data["actions"][:size]
        self.rewards[:size] = data["rewards"][:size]
        self.next_obs[:size] = data["next_obs"][:size]
        self.dones[:size] = data["dones"][:size]
        self.sub_policy_ids[:size] = data["sub_policy_ids"][:size]
        if size < self.capacity:
            self.sub_policy_ids[size:] = -1
        self.size = size
        self.ptr = size % self.capacity
        return size
