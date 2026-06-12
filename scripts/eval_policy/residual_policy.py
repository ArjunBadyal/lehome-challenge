"""Eval policy combining a frozen ACT base with a trained residual MLP."""

from __future__ import annotations

import numpy as np
import torch

from scripts.rl.residual_model import ResidualActor

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("residual")
class ResidualPolicy(BasePolicy):
    """Loads an ACT checkpoint + residual SAC checkpoint for evaluation.

    At each step:
        1. ACT produces an action from the full observation (images + state).
        2. The residual MLP takes [state, act_action] and outputs a delta.
        3. final_action = clip(act_action + scale * delta, joint_low, joint_high)

    Args:
        model_path:  Path to residual SAC checkpoint (.pt) containing
                     ``actor_state_dict`` and ``metadata``.
        device:      Torch device for inference.
        deterministic: Use deterministic residual actions (mean of Gaussian).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        deterministic: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.deterministic = deterministic

        # Load residual checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        metadata = checkpoint["metadata"]

        if metadata.get("trainer") != "residual_sac":
            raise ValueError(
                f"Expected trainer='residual_sac', got {metadata.get('trainer')!r}"
            )

        self.residual_scale = float(metadata.get("residual_scale_max",
                                                   metadata.get("residual_scale", 0.1)))
        self.chunk_size = int(metadata.get("chunk_size", 100))

        action_low = torch.as_tensor(metadata["action_low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(metadata["action_high"], dtype=torch.float32, device=self.device)
        self.action_low = action_low
        self.action_high = action_high

        # Build residual actor
        self.residual = ResidualActor(
            obs_dim=int(metadata["obs_dim"]),
            action_dim=int(metadata["action_dim"]),
            hidden_sizes=tuple(metadata["hidden_sizes"]),
        ).to(self.device)
        self.residual.load_state_dict(checkpoint["actor_state_dict"])
        self.residual.eval()

        # Build ACT policy
        act_checkpoint = metadata["act_checkpoint"]
        dataset_root = metadata["dataset_root"]
        self.act_policy = LeRobotPolicy(
            policy_path=act_checkpoint,
            dataset_root=dataset_root,
            task_description="Fold a garment with bimanual robot arms",
            device=device,
        )

        # Internal state
        self._step_in_chunk = 0
        self._act_chunk: list[np.ndarray] = []

    def reset(self):
        """Reset internal state at episode start."""
        self.act_policy.reset()
        self._step_in_chunk = 0
        self._act_chunk = []

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Generate a residual-corrected action.

        Args:
            observation: Dict with ``observation.state`` (float32) and
                         camera images (uint8).

        Returns:
            Action array (float32) of absolute joint positions.
        """
        # Re-plan ACT chunk when needed
        if self._step_in_chunk >= len(self._act_chunk):
            self._act_chunk = []
            for _ in range(self.chunk_size):
                action = self.act_policy.select_action(observation)
                self._act_chunk.append(action)
            self._step_in_chunk = 0

        act_action_np = self._act_chunk[self._step_in_chunk]
        self._step_in_chunk += 1

        # Get state
        state = np.asarray(observation["observation.state"], dtype=np.float32)

        # Residual correction
        with torch.no_grad():
            act_action_t = torch.as_tensor(act_action_np, device=self.device, dtype=torch.float32)
            state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            residual_obs = torch.cat([state_t, act_action_t]).unsqueeze(0)  # [1, 24]
            delta = self.residual.act(residual_obs, deterministic=self.deterministic).squeeze(0)

            final = act_action_t + self.residual_scale * delta
            final = torch.clamp(final, self.action_low, self.action_high)

        return final.cpu().numpy().astype(np.float32)
