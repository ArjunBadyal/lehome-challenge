from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(nn.ReLU())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        hidden_sizes: Sequence[int],
    ):
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1])
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.register_buffer("action_low", action_low.clone())
        self.register_buffer("action_high", action_high.clone())
        self.register_buffer("action_scale", 0.5 * (action_high - action_low))
        self.register_buffer("action_bias", 0.5 * (action_high + action_low))

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
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1.0 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            _, _, action = self.sample(obs)
            return action
        action, _, _ = self.sample(obs)
        return action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))
