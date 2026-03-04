"""SAC policy wrapper for competition evaluation submission."""

import numpy as np
import torch
from typing import Dict

from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("sac")
class SACPolicy(BasePolicy):
    """Evaluation wrapper that loads a trained SAC checkpoint and runs deterministic inference."""

    def __init__(self, model_path: str = None, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.device = device

        if model_path is None:
            raise ValueError("model_path is required for SACPolicy")

        from scripts.sac.config import SACConfig
        from scripts.sac.encoder import ObservationEncoder
        from scripts.sac.model import Actor

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config = ckpt["config"]

        self.config = config
        self.camera_keys = config.camera_keys
        self.image_size = config.image_size

        encoder = ObservationEncoder(
            camera_keys=config.camera_keys,
            state_dim=config.state_dim,
            image_size=config.image_size,
            hidden_dim=config.image_encoder_hidden_dim,
            spatial_num_features=config.spatial_num_features,
            latent_dim=config.latent_dim,
        )
        self.actor = Actor(encoder, config)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.to(device)
        self.actor.eval()

        # action scaling
        self.action_low = np.array(config.action_low, dtype=np.float32)
        self.action_high = np.array(config.action_high, dtype=np.float32)
        self.action_mid = (self.action_high + self.action_low) / 2.0
        self.action_half_range = (self.action_high - self.action_low) / 2.0

        print(f"[SACPolicy] Loaded checkpoint from {model_path}, device={device}")

    def reset(self):
        pass

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        from scripts.sac.utils import preprocess_obs, obs_dict_to_tensors

        obs = preprocess_obs(observation, self.image_size, self.camera_keys)
        obs_t = obs_dict_to_tensors(obs, self.camera_keys, self.device)
        tanh_action = self.actor.get_det_action(obs_t).squeeze(0)
        env_action = self.action_mid + tanh_action * self.action_half_range
        return env_action.astype(np.float32)
