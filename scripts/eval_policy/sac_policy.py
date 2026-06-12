from __future__ import annotations

import numpy as np
import torch

from scripts.rl.sac_model import SquashedGaussianActor

from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("sac")
class SacPolicy(BasePolicy):
    def __init__(self, model_path: str, device: str = "cuda", deterministic: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.deterministic = deterministic

        checkpoint = torch.load(model_path, map_location=self.device)
        metadata = checkpoint["metadata"]
        if not metadata.get("submission_safe", False):
            raise ValueError(
                "SAC checkpoint is not marked submission-safe. Expected a model trained on "
                "'observation.state' with standard absolute joint actions."
            )
        if metadata.get("observation_key") != "observation.state":
            raise ValueError(
                f"SAC checkpoint expects observation key {metadata.get('observation_key')!r}, "
                "but evaluation only supports 'observation.state'."
            )
        if metadata.get("action_semantics") != "absolute_joint_positions":
            raise ValueError(
                f"SAC checkpoint uses unsupported action semantics {metadata.get('action_semantics')!r}."
            )
        action_low = torch.as_tensor(metadata["action_low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(metadata["action_high"], dtype=torch.float32, device=self.device)

        self.actor = SquashedGaussianActor(
            obs_dim=int(metadata["obs_dim"]),
            action_dim=int(metadata["action_dim"]),
            action_low=action_low,
            action_high=action_high,
            hidden_sizes=tuple(metadata["hidden_sizes"]),
        ).to(self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.actor.eval()

    def reset(self):
        pass

    def select_action(self, observation):
        if "observation.state" not in observation:
            raise KeyError("SAC policy expects 'observation.state' in the observation dictionary.")

        state = np.asarray(observation["observation.state"], dtype=np.float32)
        with torch.no_grad():
            state_tensor = torch.as_tensor(state[None], device=self.device, dtype=torch.float32)
            action = self.actor.act(state_tensor, deterministic=self.deterministic)
        return action.squeeze(0).cpu().numpy().astype(np.float32)
