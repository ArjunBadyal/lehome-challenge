"""Eval policy combining ACT + encoder-feature residual."""

from __future__ import annotations

import numpy as np
import torch

from scripts.rl.encoder_residual_model import EncoderResidualActor
from scripts.rl.act_encoder_wrapper import ACTWithEncoderFeatures

from .base_policy import BasePolicy
from .registry import PolicyRegistry


@PolicyRegistry.register("encoder_residual")
class EncoderResidualPolicy(BasePolicy):
    """Loads ACT + encoder-feature residual for evaluation.

    Submission-compliant: encoder features derived from public observations.
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 deterministic: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device(device)
        self.deterministic = deterministic

        checkpoint = torch.load(model_path, map_location=self.device)
        metadata = checkpoint["metadata"]

        if metadata.get("trainer") != "encoder_residual_sac":
            raise ValueError(
                f"Expected trainer='encoder_residual_sac', got {metadata.get('trainer')!r}"
            )

        self.residual_scale = float(metadata.get("residual_scale_max",
                                                   metadata.get("residual_scale", 0.1)))
        self.chunk_size = int(metadata.get("chunk_size", 100))

        action_low = torch.as_tensor(metadata["action_low"], dtype=torch.float32, device=self.device)
        action_high = torch.as_tensor(metadata["action_high"], dtype=torch.float32, device=self.device)
        self.action_low = action_low
        self.action_high = action_high

        encoder_dim = int(metadata.get("encoder_dim", 512))

        self.residual = EncoderResidualActor(
            state_dim=12,
            action_dim=int(metadata["action_dim"]),
            encoder_dim=encoder_dim,
            hidden_sizes=tuple(metadata["hidden_sizes"]),
        ).to(self.device)
        self.residual.load_state_dict(checkpoint["actor_state_dict"])
        self.residual.eval()

        # ACT with encoder features
        self.act = ACTWithEncoderFeatures(
            policy_path=metadata["act_checkpoint"],
            dataset_root=metadata["dataset_root"],
            device=device,
            chunk_size=self.chunk_size,
        )

    def reset(self):
        self.act.reset()

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        action_np, encoder_features = self.act.get_action_and_features(observation)
        state = np.asarray(observation["observation.state"], dtype=np.float32)

        with torch.no_grad():
            act_action_t = torch.as_tensor(action_np, device=self.device, dtype=torch.float32)
            state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
            residual_obs = torch.cat([state_t, act_action_t, encoder_features.to(self.device)])
            delta = self.residual.act(residual_obs.unsqueeze(0), deterministic=self.deterministic).squeeze(0)
            final = act_action_t + self.residual_scale * delta
            final = torch.clamp(final, self.action_low, self.action_high)

        return final.cpu().numpy().astype(np.float32)
