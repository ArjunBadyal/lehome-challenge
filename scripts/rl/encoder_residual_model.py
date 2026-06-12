"""Residual actor that uses ACT encoder features as input.

Input:  [joint_state(12), act_action(12), encoder_features(512)] = 536D
Output: delta action in [-1, 1]^12

Submission-compliant: encoder features are derived from public observations
(3× RGB + joint state) via the frozen ACT encoder, not from privileged
simulator signals.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

from .sac_model import LOG_STD_MIN, LOG_STD_MAX


class EncoderResidualActor(nn.Module):
    """Residual policy conditioned on ACT encoder features.

    Uses a larger hidden layer to handle the 536D input, but keeps
    the output small (12D delta) and zero-initialized.
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 12,
        encoder_dim: int = 512,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__()
        obs_dim = state_dim + action_dim + encoder_dim  # 536
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.encoder_dim = encoder_dim

        # Encoder feature compression (512 → 64) to reduce parameter count
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(),
        )
        compressed_dim = state_dim + action_dim + 64  # 88

        # Main policy network
        layers: list[nn.Module] = []
        last_dim = compressed_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std_head = nn.Linear(last_dim, action_dim)

        # Zero-init mean so initial residual ≈ 0
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Split obs into components
        state_action = obs[..., :24]  # [state(12), act_action(12)]
        encoder_feat = obs[..., 24:]  # [encoder_features(512)]

        # Compress encoder features
        compressed = self.encoder_proj(encoder_feat)
        x = torch.cat([state_action, compressed], dim=-1)

        hidden = self.backbone(x)
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
