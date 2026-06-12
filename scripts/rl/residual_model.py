"""Residual actor for correcting a base (ACT) policy via SAC."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from .sac_model import LOG_STD_MIN, LOG_STD_MAX, build_mlp


class ResidualActor(nn.Module):
    """Squashed-Gaussian residual policy.

    Input:  [joint_state (12), act_action (12)] = 24D
    Output: delta action in [-1, 1]^12 (scaled externally by residual_scale)

    Reuses the same tanh-squashing and log-prob math as SquashedGaussianActor,
    but output is always in [-1, 1] (no action_low/high rescaling — the caller
    adds ``act_action + scale * delta`` and clips to joint limits).
    """

    def __init__(
        self,
        obs_dim: int = 24,
        action_dim: int = 12,
        hidden_sizes: Sequence[int] = (128, 128),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.backbone = build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

        # Initialise near-zero mean so initial residual is ~0
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # Output in [-1, 1] — no rescaling
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean)
        return y_t, log_prob, mean_action

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            _, _, action = self.sample(obs)
            return action
        action, _, _ = self.sample(obs)
        return action
