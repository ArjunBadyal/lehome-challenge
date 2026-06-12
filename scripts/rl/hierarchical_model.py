"""Hierarchical sub-policy bank and rule-based meta-controller for garment folding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from .residual_model import ResidualActor
from .sac_model import QNetwork

# Sub-policy indices
FOLD_UP = 0
FOLD_DOWN = 1
FOLD_LEFT = 2
FOLD_RIGHT = 3
NO_OP = 4

SUB_POLICY_NAMES = ["fold_up", "fold_down", "fold_left", "fold_right", "no_op"]
NUM_LEARNED = 4  # NO_OP is hardcoded


class RuleBasedMetaController:
    """Selects a sub-policy based on garment checkpoint geometry.

    Picks the sub-policy targeting the worst unsatisfied primary condition.
    Holds the selection for ``hold_steps`` to prevent oscillation.
    """

    def __init__(self, hold_steps: int = 50):
        self.hold_steps = hold_steps
        self._current_id: int = NO_OP
        self._steps_held: int = 0

    def reset(self) -> None:
        self._current_id = NO_OP
        self._steps_held = 0

    def select(
        self,
        condition_margins: np.ndarray,  # (5,)
        checkpoint_positions_cm: np.ndarray,  # (6, 3)
    ) -> int:
        """Return the sub-policy index to use this step."""
        self._steps_held += 1

        # Only re-evaluate after hold period
        if self._steps_held < self.hold_steps and self._current_id != NO_OP:
            return self._current_id

        # If secondary conditions severely violated (badly over-compressed), back off.
        # Threshold -0.5 avoids triggering NO_OP at episode start when garment
        # is naturally bunched (margins slightly negative is normal).
        if condition_margins[3] < -0.5 or condition_margins[4] < -0.5:
            self._current_id = NO_OP
            self._steps_held = 0
            return NO_OP

        # Find worst primary condition (most negative margin)
        primary_margins = condition_margins[:3]
        worst_idx = int(np.argmin(primary_margins))

        if worst_idx == 0:
            # Sleeves: dist(p0, p4) — fold whichever side is further from center
            # Use x-coordinate to decide direction
            center_x = checkpoint_positions_cm[:, 0].mean()
            if checkpoint_positions_cm[0, 0] > center_x:
                new_id = FOLD_LEFT  # bring p0 leftward
            else:
                new_id = FOLD_RIGHT  # bring p4 rightward
        elif worst_idx == 1:
            # Front/back: dist(p2, p3) — fold based on which point is higher (z-axis)
            if checkpoint_positions_cm[2, 2] > checkpoint_positions_cm[3, 2]:
                new_id = FOLD_DOWN  # bring p2 down toward p3
            else:
                new_id = FOLD_UP  # bring p3 up toward p2
        elif worst_idx == 2:
            # Sides: dist(p1, p5) — same lateral logic as sleeves
            center_x = checkpoint_positions_cm[:, 0].mean()
            if checkpoint_positions_cm[1, 0] > center_x:
                new_id = FOLD_LEFT
            else:
                new_id = FOLD_RIGHT
        else:
            new_id = NO_OP

        if new_id != self._current_id:
            self._steps_held = 0
        self._current_id = new_id
        return new_id


class SubPolicyBank(nn.Module):
    """Holds N learned sub-policy actors with their Q-networks.

    Each sub-policy is a ResidualActor (squashed Gaussian MLP)
    with its own pair of Q-networks and SAC entropy parameter.
    """

    def __init__(
        self,
        num_subpolicies: int = NUM_LEARNED,
        obs_dim: int = 47,
        action_dim: int = 12,
        hidden_sizes: Sequence[int] = (128, 128),
        device: torch.device | str = "cpu",
        learning_rate: float = 3e-4,
    ):
        super().__init__()
        self.num_subpolicies = num_subpolicies
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Build per-sub-policy networks
        self.actors: nn.ModuleList = nn.ModuleList()
        self.q1s: nn.ModuleList = nn.ModuleList()
        self.q2s: nn.ModuleList = nn.ModuleList()
        self.q1_targets: nn.ModuleList = nn.ModuleList()
        self.q2_targets: nn.ModuleList = nn.ModuleList()
        self.log_alphas: list[torch.Tensor] = []
        self.actor_optimizers: list[torch.optim.Adam] = []
        self.q_optimizers: list[torch.optim.Adam] = []
        self.alpha_optimizers: list[torch.optim.Adam] = []

        for _ in range(num_subpolicies):
            actor = ResidualActor(obs_dim, action_dim, hidden_sizes).to(self.device)
            q1 = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
            q2 = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
            q1_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
            q2_target = QNetwork(obs_dim, action_dim, hidden_sizes).to(self.device)
            q1_target.load_state_dict(q1.state_dict())
            q2_target.load_state_dict(q2.state_dict())

            log_alpha = torch.tensor(
                np.log(0.2), device=self.device, dtype=torch.float32, requires_grad=True
            )

            self.actors.append(actor)
            self.q1s.append(q1)
            self.q2s.append(q2)
            self.q1_targets.append(q1_target)
            self.q2_targets.append(q2_target)
            self.log_alphas.append(log_alpha)
            self.actor_optimizers.append(torch.optim.Adam(actor.parameters(), lr=learning_rate))
            self.q_optimizers.append(
                torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=learning_rate)
            )
            self.alpha_optimizers.append(torch.optim.Adam([log_alpha], lr=learning_rate))

    def act(self, k: int, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action from sub-policy k."""
        return self.actors[k].act(obs, deterministic=deterministic)

    def state_dicts(self) -> dict:
        """Serialize all sub-policy weights."""
        return {
            "sub_actors": {i: a.state_dict() for i, a in enumerate(self.actors)},
            "q1s": {i: q.state_dict() for i, q in enumerate(self.q1s)},
            "q2s": {i: q.state_dict() for i, q in enumerate(self.q2s)},
            "q1_targets": {i: q.state_dict() for i, q in enumerate(self.q1_targets)},
            "q2_targets": {i: q.state_dict() for i, q in enumerate(self.q2_targets)},
            "log_alphas": {i: la.detach().cpu() for i, la in enumerate(self.log_alphas)},
            "actor_optimizers": {i: opt.state_dict() for i, opt in enumerate(self.actor_optimizers)},
            "q_optimizers": {i: opt.state_dict() for i, opt in enumerate(self.q_optimizers)},
            "alpha_optimizers": {i: opt.state_dict() for i, opt in enumerate(self.alpha_optimizers)},
        }

    def load_state_dicts(self, data: dict) -> None:
        """Deserialize all sub-policy weights."""
        for i in range(self.num_subpolicies):
            self.actors[i].load_state_dict(data["sub_actors"][i])
            self.q1s[i].load_state_dict(data["q1s"][i])
            self.q2s[i].load_state_dict(data["q2s"][i])
            self.q1_targets[i].load_state_dict(data["q1_targets"][i])
            self.q2_targets[i].load_state_dict(data["q2_targets"][i])
            self.log_alphas[i] = data["log_alphas"][i].to(self.device).requires_grad_(True)
            # Re-create alpha optimizer with new tensor
            self.alpha_optimizers[i] = torch.optim.Adam(
                [self.log_alphas[i]], lr=self.alpha_optimizers[i].defaults["lr"]
            )
            if "actor_optimizers" in data:
                self.actor_optimizers[i].load_state_dict(data["actor_optimizers"][i])
            if "q_optimizers" in data:
                self.q_optimizers[i].load_state_dict(data["q_optimizers"][i])
            if "alpha_optimizers" in data:
                self.alpha_optimizers[i].load_state_dict(data["alpha_optimizers"][i])


def build_hierarchical_obs(
    joint_state: np.ndarray,  # (12,)
    act_action: torch.Tensor,  # (12,) on rl_device
    condition_margins: torch.Tensor,  # (5,)
    checkpoint_positions_cm: torch.Tensor,  # (6, 3)
    device: torch.device,
) -> torch.Tensor:
    """Build the 47D observation for a sub-policy."""
    state_t = torch.as_tensor(joint_state, device=device, dtype=torch.float32)
    margins_t = condition_margins.to(device).float()
    # Scale checkpoint positions to ~[-5, 5] range
    checkpoints_flat = (checkpoint_positions_cm.reshape(-1) / 100.0).to(device).float()
    return torch.cat([state_t, act_action.to(device), margins_t, checkpoints_flat])
