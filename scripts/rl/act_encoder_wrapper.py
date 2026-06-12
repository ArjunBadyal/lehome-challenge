"""Wrapper that extracts ACT encoder features alongside actions.

Provides a 512-dim latent vector summarizing the visual + proprioceptive
scene, derived entirely from public observations (3× RGB + joint state).
Submission-compliant: no privileged simulator signals.
"""

from __future__ import annotations

import numpy as np
import torch

from scripts.eval_policy.lerobot_policy import LeRobotPolicy


class ACTWithEncoderFeatures:
    """Wraps LeRobotPolicy to expose the ACT encoder's latent token.

    At each step, ``get_action_and_features()`` returns both the ACT action
    and a 512-dim encoder feature vector. The features come from the first
    token of the transformer encoder output (the VAE latent / CLS token),
    which fuses all 3 camera images + joint state.

    These features are derived entirely from public observations and can be
    used as residual policy input without violating submission constraints.
    """

    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str = "Fold a garment with bimanual robot arms",
        device: str = "cuda",
        chunk_size: int = 100,
    ):
        self.device = torch.device(device)
        self.chunk_size = chunk_size

        self.lerobot_policy = LeRobotPolicy(
            policy_path=policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
        )
        self.action_dim = self.lerobot_policy.action_dim

        # Get reference to the underlying ACT model
        self._act_model = self.lerobot_policy.policy.model
        self._act_policy = self.lerobot_policy.policy

        # Determine encoder feature dim from the model config
        self.encoder_feature_dim = self._act_model.config.dim_model  # 512

        # Chunk management
        self._action_chunk: list[np.ndarray] = []
        self._feature_cache: torch.Tensor | None = None
        self._step_in_chunk = 0

    def reset(self) -> None:
        self.lerobot_policy.reset()
        self._action_chunk = []
        self._feature_cache = None
        self._step_in_chunk = 0

    def _extract_encoder_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the ACT encoder and return the latent token (first token).

        This replicates the encoding portion of ACT.forward() without
        running the decoder or predicting actions.
        """
        model = self._act_model
        batch_size = batch["observation.state"].shape[0]

        # Build encoder input tokens (same logic as ACT.forward lines 440-488)
        encoder_in_tokens = []

        # VAE latent token — at inference, use zeros (no VAE sampling)
        cls_embed = torch.zeros(
            1, batch_size, model.config.dim_model,
            dtype=torch.float32, device=self.device,
        )
        encoder_in_tokens.append(cls_embed.squeeze(0))

        # 1D feature positional embeddings
        encoder_in_pos_embed = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Robot state token
        if model.config.robot_state_feature:
            encoder_in_tokens.append(
                model.encoder_robot_state_input_proj(batch["observation.state"])
            )

        # Image tokens
        if model.config.image_features:
            import einops
            obs_images = [batch[key] for key in model.config.image_features]
            for img in obs_images:
                cam_features = model.backbone(img)["feature_map"]
                cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(
                    dtype=cam_features.dtype
                )
                cam_features = model.encoder_img_feat_input_proj(cam_features)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)

        # Run encoder
        encoder_out = model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # Return the first token (latent/CLS) as the feature vector: (batch, 512)
        return encoder_out[0]  # shape: (batch_size, dim_model)

    @torch.no_grad()
    def get_action_and_features(
        self, observation: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Get the next ACT action and encoder features.

        Re-plans (full forward pass + feature extraction) when the chunk
        is exhausted. On intermediate steps, returns cached features from
        the last re-plan.

        Returns:
            action: numpy float32 array (action_dim,)
            features: tensor (encoder_feature_dim,) on self.device
        """
        if self._step_in_chunk >= len(self._action_chunk):
            # Re-plan: get full chunk + extract features
            self._action_chunk = []

            # Process observation for ACT
            batch_obs = self.lerobot_policy._process_observation(
                self.lerobot_policy._filter_observations(
                    observation, self.lerobot_policy.input_features
                ) if self.lerobot_policy.input_features else observation
            )

            # Extract encoder features
            self._feature_cache = self._extract_encoder_features(batch_obs).squeeze(0)

            # Get action chunk via normal ACT pipeline
            # Reset to force full re-plan
            self._act_policy.reset()
            for _ in range(self.chunk_size):
                action = self.lerobot_policy.select_action(observation)
                self._action_chunk.append(action)

            self._step_in_chunk = 0

        action = self._action_chunk[self._step_in_chunk]
        self._step_in_chunk += 1

        return action, self._feature_cache
