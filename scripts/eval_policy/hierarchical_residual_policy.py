"""Eval policy combining ACT + hierarchical sub-policy residuals."""

from __future__ import annotations

import numpy as np
import torch

from scripts.rl.hierarchical_model import (
    NUM_LEARNED, NO_OP, RuleBasedMetaController, SubPolicyBank, build_hierarchical_obs,
)
from scripts.rl.residual_model import ResidualActor

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("hierarchical_residual")
class HierarchicalResidualPolicy(BasePolicy):
    """Loads ACT + hierarchical sub-policy residuals for evaluation.

    At each step:
        1. ACT produces an action chunk (every chunk_size steps).
        2. Meta-controller selects a sub-policy based on condition_margins.
        3. Selected sub-policy produces a delta correction.
        4. final_action = clip(act_action + scale * delta, joint_limits)

    Requires ``condition_margins`` and ``checkpoint_positions_cm`` in the
    observation dict (added by the env when task metrics are available).
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

        checkpoint = torch.load(model_path, map_location=self.device)
        metadata = checkpoint["metadata"]

        if metadata.get("trainer") != "hierarchical_residual_sac":
            raise ValueError(
                f"Expected trainer='hierarchical_residual_sac', got {metadata.get('trainer')!r}"
            )

        self.residual_scale = float(metadata.get("residual_scale_max",
                                                   metadata.get("residual_scale", 0.1)))
        self.chunk_size = int(metadata.get("chunk_size", 100))
        self.hold_steps = int(metadata.get("hold_steps", 50))

        action_low = torch.as_tensor(metadata["action_low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(metadata["action_high"], dtype=torch.float32, device=self.device)
        self.action_low = action_low
        self.action_high = action_high

        # Build sub-policy bank
        self.bank = SubPolicyBank(
            num_subpolicies=int(metadata.get("num_subpolicies", NUM_LEARNED)),
            obs_dim=int(metadata["obs_dim"]),
            action_dim=int(metadata["action_dim"]),
            hidden_sizes=tuple(metadata["hidden_sizes"]),
            device=self.device,
        )
        self.bank.load_state_dicts(checkpoint)
        for actor in self.bank.actors:
            actor.eval()

        # Meta-controller
        self.meta = RuleBasedMetaController(hold_steps=self.hold_steps)

        # ACT policy
        self.act_policy = LeRobotPolicy(
            policy_path=metadata["act_checkpoint"],
            dataset_root=metadata["dataset_root"],
            task_description="Fold a garment with bimanual robot arms",
            device=device,
        )

        # Internal state
        self._step_in_chunk = 0
        self._act_chunk: list[np.ndarray] = []

    def reset(self):
        self.act_policy.reset()
        self.meta.reset()
        self._step_in_chunk = 0
        self._act_chunk = []

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        # Re-plan ACT chunk when needed
        if self._step_in_chunk >= len(self._act_chunk):
            self._act_chunk = []
            for _ in range(self.chunk_size):
                action = self.act_policy.select_action(observation)
                self._act_chunk.append(action)
            self._step_in_chunk = 0

        act_action_np = self._act_chunk[self._step_in_chunk]
        self._step_in_chunk += 1

        state = np.asarray(observation["observation.state"], dtype=np.float32)

        # Get task metrics for meta-controller
        # These should be in the observation dict (added by env)
        condition_margins = observation.get("condition_margins")
        checkpoint_positions_cm = observation.get("checkpoint_positions_cm")

        if condition_margins is None or checkpoint_positions_cm is None:
            # Fallback: no task metrics available, use flat residual from sub-policy 0
            with torch.no_grad():
                act_action_t = torch.as_tensor(act_action_np, device=self.device, dtype=torch.float32)
                state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
                # Build a dummy 47D obs with zeros for missing fields
                dummy_margins = torch.zeros(5, device=self.device)
                dummy_checkpoints = torch.zeros(6, 3, device=self.device)
                h_obs = build_hierarchical_obs(state, act_action_t, dummy_margins, dummy_checkpoints, self.device)
                delta = self.bank.act(0, h_obs.unsqueeze(0), deterministic=self.deterministic).squeeze(0)
                final = act_action_t + self.residual_scale * delta
                final = torch.clamp(final, self.action_low, self.action_high)
            return final.cpu().numpy().astype(np.float32)

        # Convert to numpy if tensors
        if isinstance(condition_margins, torch.Tensor):
            condition_margins = condition_margins.cpu().numpy()
        condition_margins = np.asarray(condition_margins, dtype=np.float32)

        if isinstance(checkpoint_positions_cm, torch.Tensor):
            checkpoint_positions_cm_np = checkpoint_positions_cm.cpu().numpy()
        else:
            checkpoint_positions_cm_np = np.asarray(checkpoint_positions_cm, dtype=np.float32)

        # Meta-controller
        k = self.meta.select(condition_margins, checkpoint_positions_cm_np.reshape(6, 3))

        with torch.no_grad():
            act_action_t = torch.as_tensor(act_action_np, device=self.device, dtype=torch.float32)
            margins_t = torch.as_tensor(condition_margins, device=self.device, dtype=torch.float32)
            checkpoints_t = torch.as_tensor(checkpoint_positions_cm_np, device=self.device, dtype=torch.float32).reshape(6, 3)

            h_obs = build_hierarchical_obs(state, act_action_t, margins_t, checkpoints_t, self.device)

            if k == NO_OP:
                delta = torch.zeros(act_action_t.shape, device=self.device)
            else:
                delta = self.bank.act(k, h_obs.unsqueeze(0), deterministic=self.deterministic).squeeze(0)

            final = act_action_t + self.residual_scale * delta
            final = torch.clamp(final, self.action_low, self.action_high)

        return final.cpu().numpy().astype(np.float32)
