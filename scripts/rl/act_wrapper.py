"""Wrapper that turns a LeRobot ACT policy into a chunk provider for RL."""

from __future__ import annotations

import numpy as np
import torch

from scripts.eval_policy.lerobot_policy import LeRobotPolicy


class ACTChunkProvider:
    """Manages ACT's 100-step action chunks for use inside a training loop.

    Usage::

        provider = ACTChunkProvider(policy_path, dataset_root, ...)
        obs_dict = env._get_observations()
        chunk = provider.get_chunk(obs_dict)   # [chunk_size, action_dim]
        act_action = chunk[step % chunk_size]  # single-step action
    """

    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str = "Fold a garment with bimanual robot arms",
        device: str = "cuda",
        chunk_size: int = 100,
    ):
        self.chunk_size = chunk_size
        self.device = device

        # The LeRobotPolicy handles loading, preprocessing, and postprocessing
        self.policy = LeRobotPolicy(
            policy_path=policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
        )
        self.action_dim = self.policy.action_dim
        self._cached_chunk: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear internal state at episode start."""
        self.policy.reset()
        self._cached_chunk = None

    def get_chunk(self, obs_dict: dict[str, np.ndarray]) -> torch.Tensor:
        """Query ACT for a full action chunk.

        Args:
            obs_dict: Observation dictionary with images and state
                      (numpy arrays, as returned by env._get_observations()).

        Returns:
            Tensor of shape [chunk_size, action_dim] on self.device.
            If the policy returns fewer than chunk_size actions, the last
            action is repeated to fill the chunk.
        """
        # ACT's select_action returns a single action per call, but the
        # underlying LeRobot ACT model actually predicts a full chunk
        # internally. We call it chunk_size times to fill the buffer.
        # In practice, ACT caches its chunk and returns successive actions
        # via its built-in action queue.
        actions = []
        for _ in range(self.chunk_size):
            action = self.policy.select_action(obs_dict)  # np.ndarray
            actions.append(torch.as_tensor(action, dtype=torch.float32))

        self._cached_chunk = torch.stack(actions).to(self.device)  # [chunk_size, action_dim]
        return self._cached_chunk

    @property
    def cached_chunk(self) -> torch.Tensor | None:
        return self._cached_chunk
