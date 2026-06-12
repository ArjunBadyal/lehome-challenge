from __future__ import annotations
from collections import deque
from typing import Any, Dict, List
from collections.abc import Sequence

import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from pxr import UsdShade, Sdf, UsdGeom
import omni.kit.commands
from isaacsim.core.utils.prims import is_prim_path_valid
import isaacsim.core.utils.prims as prims_utils

from lehome.tasks.bedroom.garment_bi_cfg_v2 import GarmentEnvCfg
from lehome.utils.success_checker_chanllege import success_checker_garment_fold
from lehome.utils.depth_to_pointcloud import generate_pointcloud_from_data
from lehome.assets.scenes.bedroom import MARBLE_BEDROOM_USD_PATH
from lehome.devices.action_process import preprocess_device_action
from lehome.assets.object.Garment import GarmentObject
from lehome.tasks.bedroom.challenge_garment_loader import ChallengeGarmentLoader
import logging
from lehome.utils.logger import get_logger

# Create logger for this module with DEBUG level
logger = get_logger(__name__)


class GarmentEnv(DirectRLEnv):
    cfg: GarmentEnvCfg

    def __init__(self, cfg: GarmentEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.action_scale = self.cfg.action_scale
        self.object = None  # Will be created in _setup_scene

        # Cache for distance-based reward (to handle step_interval decorator)
        self._last_computed_reward = 0.0
        # State for potential-based reward shaping:
        # Keep the previous step's sigmoid-potentials over margins so we can
        # return reward = alpha * delta(min_phi) + beta * sum(delta_phi_i)
        # plus explicit release and asymmetry penalties.
        self._prev_phis: list[float] | None = None
        self._prev_min_phi: float | None = None
        # Gripper action from the previous step — used for release-penalty.
        self._prev_gripper_action = None

        self.garment_loader = ChallengeGarmentLoader(cfg.garment_cfg_base_path)
        self.garment_config = self.garment_loader.load_garment_config(
            cfg.garment_name, cfg.garment_version
        )
        self.particle_config = OmegaConf.load(cfg.particle_cfg_path)

        if cfg.use_random_seed:
            # Use random seed (no fixed seed)
            self.garment_rng = np.random.RandomState()
        else:
            # Use fixed seed from config
            self.garment_rng = np.random.RandomState(cfg.random_seed)

        cfg.viewer = cfg.viewer.replace(
            eye=(0, -1.2, 1.3),
            lookat=(0, 6.4, -2.8),
        )
        super().__init__(cfg, render_mode, **kwargs)
        self.left_joint_pos = self.left_arm.data.joint_pos
        self.right_joint_pos = self.right_arm.data.joint_pos

    def _setup_scene(self):
        self.left_arm = Articulation(self.cfg.left_robot)
        self.right_arm = Articulation(self.cfg.right_robot)
        self.top_camera = TiledCamera(self.cfg.top_camera)
        self.left_camera = TiledCamera(self.cfg.left_wrist)
        self.right_camera = TiledCamera(self.cfg.right_wrist)
        cfg = sim_utils.UsdFileCfg(usd_path=f"{MARBLE_BEDROOM_USD_PATH}")
        cfg.func(
            "/World/Scene",
            cfg,
            translation=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 0.0),
        )

        # Create garment object with selected asset
        self._create_garment_object()

        # add articulation to scene
        self.scene.articulations["left_arm"] = self.left_arm
        self.scene.articulations["right_arm"] = self.right_arm
        self.scene.sensors["top_camera"] = self.top_camera
        self.scene.sensors["left_camera"] = self.left_camera
        self.scene.sensors["right_camera"] = self.right_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_garment_object(self):
        """
        Create a new GarmentObject with the currently selected asset.
        """
        if self.object is not None:
            self._delete_garment_object()

        # Generate prim_path based on garment_name, default to "Cloth" if not specified
        garment_name = getattr(self.cfg, "garment_name", None)
        if garment_name and garment_name.strip():
            prim_name = garment_name.strip()
        else:
            prim_name = "Cloth"

        prim_path = f"/World/Object/{prim_name}"

        try:
            if is_prim_path_valid(prim_path):
                logger.debug(
                    f"[GarmentEnv] Prim path {prim_path} still exists, deleting before creation"
                )
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                if hasattr(self, "sim") and self.sim is not None:
                    for _ in range(5):
                        self.sim.step(render=True)
                if is_prim_path_valid(prim_path):
                    logger.warning(
                        f"[GarmentEnv] WARNING: Prim path {prim_path} still exists after deletion attempt!"
                    )
                else:
                    logger.debug(
                        f"[GarmentEnv] Prim path {prim_path} successfully deleted"
                    )
        except Exception as e:
            logger.debug(
                f"[GarmentEnv] Could not delete existing prim (may not exist): {e}"
            )

        # Create new garment object
        try:
            logger.debug(
                f"[GarmentEnv] Creating GarmentObject at prim_path: {prim_path}"
            )
            self.object = GarmentObject(
                prim_path=prim_path,
                particle_config=self.particle_config,
                garment_config=self.garment_config,
                rng=self.garment_rng,
            )
            logger.info("[GarmentEnv] GarmentObject created successfully")
        except Exception as e:
            logger.error(f"[GarmentEnv] Failed to create GarmentObject: {e}")
            raise RuntimeError(f"Failed to create GarmentObject: {e}") from e

        # Validate created object
        self._validate_created_object()

        self.texture_cfg = self.particle_config.objects.get("texture_randomization", {})
        self.light_cfg = self.particle_config.objects.get("light_randomization", {})
        logger.debug(
            f"[GarmentEnv] Loaded texture_cfg: {bool(self.texture_cfg)}, light_cfg: {bool(self.light_cfg)}"
        )

    def _validate_created_object(self):
        """
        Validate that the GarmentObject was created successfully and has required attributes.

        Raises:
            RuntimeError: If object validation fails
        """
        logger.debug("[GarmentEnv] Validating created GarmentObject...")

        if self.object is None:
            raise RuntimeError("GarmentObject creation returned None")

        required_attrs = [
            "usd_prim_path",
            "mesh_prim_path",
            "particle_system_path",
            "particle_material_path",
        ]

        for attr in required_attrs:
            if not hasattr(self.object, attr):
                raise RuntimeError(f"GarmentObject missing required attribute: {attr}")

            attr_value = getattr(self.object, attr)
            if attr_value is None:
                raise RuntimeError(f"GarmentObject attribute {attr} is None")

        prim_paths_to_check = [
            ("usd_prim_path", self.object.usd_prim_path),
            ("mesh_prim_path", self.object.mesh_prim_path),
        ]

        for path_name, path_value in prim_paths_to_check:
            if not is_prim_path_valid(path_value):
                logger.warning(
                    f"[GarmentEnv] Prim path {path_name} '{path_value}' is not valid in stage. "
                    "This may be expected if the prim hasn't been added yet."
                )
            else:
                logger.debug(
                    f"[GarmentEnv] Prim path {path_name} '{path_value}' is valid"
                )

        logger.debug("[GarmentEnv] GarmentObject validation passed")

    def _delete_garment_object(self):
        """Delete the current garment object from the stage.

        This method ensures complete cleanup of the garment object, including:
        - USD prim deletion
        - Particle system cleanup
        - All child prims removal
        """
        if self.object is None:
            return
        
        # bug: stuck while eval
        from isaacsim.core.api import World
        world = World.instance()
        # bug: stuck while eval
        was_playing = world.is_playing()
        if was_playing:
            world.pause()
            
        try:
            # Try to get prim_path from object first, then fallback to garment_name-based path
            if hasattr(self.object, "usd_prim_path") and self.object.usd_prim_path:
                prim_path = self.object.usd_prim_path
            else:
                # Fallback: generate prim_path based on garment_name, same logic as creation
                garment_name = getattr(self.cfg, "garment_name", None)
                if garment_name and garment_name.strip():
                    prim_name = garment_name.strip()
                else:
                    prim_name = "Cloth"
                prim_path = f"/World/Object/{prim_name}"

            if hasattr(self.object, "particle_system_path"):
                particle_path = self.object.particle_system_path
                try:
                    if is_prim_path_valid(particle_path):
                        omni.kit.commands.execute("DeletePrims", paths=[particle_path])
                        logger.debug(
                            f"[GarmentEnv] Deleted particle system at {particle_path}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[GarmentEnv] Failed to delete particle system: {e}"
                    )

            if is_prim_path_valid(prim_path):
                omni.kit.commands.execute("DeletePrims", paths=[prim_path])
                logger.debug(f"[GarmentEnv] Deleted garment prim at {prim_path}")
            else:
                logger.warning(
                    f"[GarmentEnv] Prim path {prim_path} is not valid, skipping deletion"
                )

        except Exception as e:
            logger.warning(f"[GarmentEnv] Failed to delete garment object: {e}")
            import traceback

            traceback.print_exc()
        # bug: stuck while eval
        if was_playing:
            world.play()
        self.object = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        self.left_arm.set_joint_position_target(self.actions[:, :6])
        self.right_arm.set_joint_position_target(self.actions[:, 6:])

    def _get_joint_position_tensor(self) -> torch.Tensor:
        """Concatenated (left 6 + right 6) joint positions. Used by residual RL."""
        return torch.cat(
            [self.left_arm.data.joint_pos[:, :6], self.right_arm.data.joint_pos[:, :6]],
            dim=-1,
        )

    def _get_observations(self) -> dict:
        action = self.actions.squeeze(0)
        left_joint_pos = torch.cat(
            [self.left_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        right_joint_pos = torch.cat(
            [self.right_joint_pos[:, i].unsqueeze(1) for i in range(6)], dim=-1
        )
        joint_pos = torch.cat([left_joint_pos, right_joint_pos], dim=1)
        joint_pos = joint_pos.squeeze(0)
        top_camera_rgb = self.top_camera.data.output["rgb"]
        top_camera_depth = self.top_camera.data.output["depth"].squeeze()
        left_camera_rgb = self.left_camera.data.output["rgb"]
        right_camera_rgb = self.right_camera.data.output["rgb"]

        # Convert depth from meters to millimeters (uint16)
        # Range: 0-65535 mm (0-65.535 m), precision: 1 mm
        depth_np = top_camera_depth.cpu().detach().numpy().copy()
        depth_mm = np.clip(depth_np * 1000, 0, 65535).astype(np.uint16)

        observations = {
            "action": action.cpu().detach().numpy(),
            "observation.state": joint_pos.cpu().detach().numpy(),
            "observation.images.top_rgb": top_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.left_rgb": left_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.images.right_rgb": right_camera_rgb.cpu()
            .detach()
            .numpy()
            .squeeze(),
            "observation.top_depth": depth_mm,
        }
        return observations

    def _get_workspace_pointcloud(
        self, env_index: int = 0, num_points: int = 2048, use_fps: bool = False
    ):
        """
        Retrive workspace pointcloud from specified env_id (Robot Right Arm Base Frame)。

        Args:
            env_index (int)
            num_points (int)
            use_fps (bool)

        Returns:
            points (np.ndarray)
            colors (np.ndarray) (0-255)
        """
        top_camera_rgb_tensor = self.top_camera.data.output["rgb"]
        top_camera_depth_tensor = self.top_camera.data.output["depth"]

        depth_img = top_camera_depth_tensor[env_index].clone().cpu().numpy().squeeze()

        # Converting to float32 and to meter
        depth_img = depth_img.astype(np.float32) / 1000.0

        # RGB Tensor shape (Num_Envs, H, W, 4) -> (H, W, 3/4)
        rgb_img = top_camera_rgb_tensor[env_index].clone().cpu().numpy()

        pointclouds = generate_pointcloud_from_data(
            rgb_image=rgb_img,
            depth_image=depth_img,
            num_points=num_points,
            use_fps=use_fps,
        )
        print(f"[Info] Generated pointcloud shape: {pointclouds.shape}")

        return pointclouds

    def _get_rewards(self) -> torch.Tensor:
        """Potential-based shaped reward for residual RL fine-tuning.

        R_step = alpha * (phi(min_m_t) - phi(min_m_{t-1}))        # push on worst blocker
               + beta  * sum_i w_i * (phi(m_i_t) - phi(m_i_{t-1})) # rewards overall progress
               - gamma * release_penalty                           # opening gripper mid-fold
               - eta   * asymmetry_penalty                         # pair-imbalance (pants)

        where:
          m_i     = signed margin of condition i (positive = satisfied, cm)
          phi(m)  = sigmoid(m / tau), tau = 1 cm
          alpha = 1.0, beta = 0.25, gamma = 0.05, eta = 0.05

        Because the reward is the DELTA of a potential function, the optimal
        policy is invariant to this shaping (Ng & Russell 1999), so an RL
        agent can't hack the reward by cycling through high-potential states
        without actually reducing the worst-margin. Success still returns
        a sparse +1.0 bonus that dominates.
        """
        # Check if object is valid
        if self.object is None:
            return 0.0
        if not hasattr(self.object, "_cloth_prim_view"):
            return 0.0

        garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
        result = success_checker_garment_fold(self.object, garment_type)

        # step_interval decorator may return False on skipped ticks
        if not isinstance(result, dict):
            return self._last_computed_reward

        success = result.get("success", False)
        details = result.get("details", {})

        if success:
            # Sparse success bonus — always dominates the shaping.
            self._last_computed_reward = 1.0
            return 1.0

        if not details:
            return 0.0

        # Shaping constants
        tau = 1.0       # centimetres — conditions use cm units already
        alpha = 1.0     # weight on worst-margin delta
        beta = 0.25     # weight on per-condition delta sum
        gamma = 0.05    # weight on release penalty
        eta = 0.05      # weight on asymmetry penalty

        def _phi(m: float) -> float:
            # sigmoid with tau scaling
            if m >= 20.0:  # avoid overflow
                return 1.0
            if m <= -20.0:
                return 0.0
            return 1.0 / (1.0 + np.exp(-m / tau))

        # Compute signed margins and per-condition potentials.
        margins: list[float] = []
        pair_left_margin: float | None = None
        pair_right_margin: float | None = None

        for cond_info in details.values():
            value = float(cond_info.get("value", 0.0))
            threshold = float(cond_info.get("threshold", 0.0))
            description = str(cond_info.get("description", ""))
            is_less_than = "<=" in description

            if is_less_than:
                margin = threshold - value  # positive when within
            else:
                margin = value - threshold

            margins.append(margin)

            if "p[0], p[4]" in description:
                pair_left_margin = margin
            elif "p[1], p[5]" in description:
                pair_right_margin = margin

        if not margins:
            return 0.0

        phis = [_phi(m) for m in margins]
        min_phi = min(phis)

        # Potential-based shaping: reward is delta-of-potential since previous step.
        if self._prev_phis is not None and len(self._prev_phis) == len(phis):
            delta_min_phi = min_phi - (self._prev_min_phi or 0.0)
            delta_sum_phi = sum(phis) - sum(self._prev_phis)
        else:
            delta_min_phi = 0.0
            delta_sum_phi = 0.0
        self._prev_phis = phis
        self._prev_min_phi = min_phi

        # Asymmetry penalty — only for garment categories that have both
        # p[0]-p[4] and p[1]-p[5] pair conditions (all 4 supported types do).
        if pair_left_margin is not None and pair_right_margin is not None:
            asymmetry_penalty = abs(pair_left_margin - pair_right_margin)
        else:
            asymmetry_penalty = 0.0

        # Release penalty — fires only when fold is clearly incomplete
        # (min_phi < 0.5 means some condition is still on the "not satisfied"
        # side of its threshold) AND the gripper is being commanded open.
        release_penalty = 0.0
        try:
            current_actions = self.actions[0].detach().cpu().numpy()
            if self._prev_gripper_action is not None:
                gripper_delta_left = float(current_actions[5] - self._prev_gripper_action[0])
                gripper_delta_right = float(current_actions[11] - self._prev_gripper_action[1])
                opening = max(gripper_delta_left, gripper_delta_right)
                if opening > 0.02 and min_phi < 0.5:
                    release_penalty = 1.0
            self._prev_gripper_action = (
                float(current_actions[5]),
                float(current_actions[11]),
            )
        except Exception:
            # `self.actions` may not be initialised on the very first tick.
            pass

        total = (
            alpha * delta_min_phi
            + beta * delta_sum_phi
            - gamma * release_penalty
            - eta * asymmetry_penalty
        )
        # Bound the per-step shaping; success bonus is outside this path.
        total = float(max(-1.0, min(1.0, total)))
        self._last_computed_reward = total
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _check_success(self) -> bool:
        """Check success based on garment type."""
        if self.object is None:
            return False

        if not hasattr(self.object, "_cloth_prim_view"):
            return False

        garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
        result = success_checker_garment_fold(self.object, garment_type)

        if isinstance(result, dict):
            return result.get("success", False)
        else:
            return bool(result)

    def _get_success(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.object is None:
            success = False
            result = None
        elif not hasattr(self.object, "_cloth_prim_view"):
            success = False
            result = None
        else:
            garment_type = self.garment_loader.get_garment_type(self.cfg.garment_name)
            result = success_checker_garment_fold(self.object, garment_type)

            if isinstance(result, dict):
                logger.info(
                    f"[Success Check] Garment type: {result.get('garment_type', 'unknown')}, Thresholds: {result.get('thresholds', [])}"
                )

                details = result.get("details", {})
                for key, condition_info in details.items():
                    status = "✓" if condition_info.get("passed", False) else "✗"
                    logger.info(
                        f"  {condition_info.get('description', '')} -> {status}"
                    )

                success = result.get("success", False)
                logger.info(
                    f"[Success Check] Final result: {'Success ✓' if success else 'Failed ✗'}"
                )
            else:
                success = bool(result)

        if isinstance(success, bool):
            success_tensor = torch.tensor(
                [success] * len(self.episode_length_buf), device=self.device
            )
        else:
            success_tensor = torch.zeros_like(self.episode_length_buf, dtype=torch.bool)
        episode_success = success_tensor
        return episode_success

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.left_arm._ALL_INDICES
        super()._reset_idx(env_ids)

        # Reset cached reward on episode reset
        self._last_computed_reward = 0.0
        self._prev_phis = None
        self._prev_min_phi = None
        self._prev_gripper_action = None

        left_joint_pos = self.left_arm.data.default_joint_pos[env_ids]
        right_joint_pos = self.right_arm.data.default_joint_pos[env_ids]
        self.left_arm.write_joint_position_to_sim(
            left_joint_pos, joint_ids=None, env_ids=env_ids
        )
        self.right_arm.write_joint_position_to_sim(
            right_joint_pos, joint_ids=None, env_ids=env_ids
        )

        # Reset the garment object
        if self.object is not None:
            self.object.reset()

        # Apply randomization if enabled in config
        if self.texture_cfg.get("enable", False):
            self._randomize_table038_texture()

        if self.light_cfg.get("enable", False):
            self._randomize_light()

    def _randomize_table038_texture(self):
        """Randomize Table038 texture based on config."""
        if not self.texture_cfg.get("enable", False):
            return

        folder = self.texture_cfg.get("folder", "")
        if not os.path.isabs(folder):
            folder = os.path.join(os.getcwd(), folder)

        min_id = int(self.texture_cfg.get("min_id", 1))
        max_id = int(self.texture_cfg.get("max_id", 1))
        shader_path = self.texture_cfg.get("prim_path", "")

        if not folder or not os.path.exists(folder):
            print(f"[Reset][Warn] Texture folder not found: {folder}")
            return
        if not shader_path:
            print("[Reset][Warn] No prim_path provided for texture randomization")
            return

        stage = self.scene.stage
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            print(f"[Reset][Warn] Shader prim not found at {shader_path}")
            return

        shader = UsdShade.Shader(shader_prim)
        idx = random.randint(min_id, max_id)
        tex_path = os.path.join(folder, f"{idx}.png")

        tex_input = shader.GetInput("file") or shader.GetInput("diffuse_texture")
        if not tex_input:
            print("[Reset][Warn] No texture input found on shader")
            return

        tex_input.Set(Sdf.AssetPath(tex_path))
        # print(f"[Reset] Texture randomized -> {tex_path}")

    def _randomize_light(self):
        """Randomize DomeLight attributes based on config."""
        if not self.light_cfg.get("enable", False):
            return

        prim_path = self.light_cfg.get("prim_path", "/World/Light")
        intensity_range = self.light_cfg.get("intensity_range", [800, 2000])
        color_range = self.light_cfg.get("color_range", [0.0, 1.0])

        stage = self.scene.stage
        light_prim = stage.GetPrimAtPath(prim_path)
        if not light_prim.IsValid():
            print(f"[Reset][Warn] Light prim not found at {prim_path}")
            return

        intensity = random.uniform(*intensity_range)
        color = tuple[float, float, float](
            random.uniform(color_range[0], color_range[1]) for _ in range(3)
        )

        light_prim.GetAttribute("inputs:intensity").Set(intensity)
        light_prim.GetAttribute("inputs:color").Set(color)

        # print(f"[Reset] Light randomized -> intensity={intensity:.1f}, color={color}")

    def preprocess_device_action(
        self, action: dict[str, Any], teleop_device
    ) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

    def initialize_obs(self):
        self.object.initialize()

    def get_all_pose(self):
        return self.object.get_all_pose()

    def set_all_pose(self, pose):
        self.object.set_all_pose(pose)

    def switch_garment(self, garment_name: str, garment_version: str = None):
        """Switch to a different garment without recreating the environment.

        This method allows reusing the same environment instance for different garments,
        which is much faster than closing and recreating the environment.

        Args:
            garment_name: Name of the garment to switch to (e.g., "Top_Long_Seen_0")
            garment_version: Version of the garment ("Release" or "Holdout"),
                            defaults to current cfg.garment_version
        """
        logger.info(
            f"[GarmentEnv] Switching garment to: {garment_name} (version: {garment_version})"
        )

        if self.object is not None:
            self._delete_garment_object()
            logger.info("[GarmentEnv] Old garment object deleted")

        if garment_version is None:
            garment_version = self.cfg.garment_version

        # Update config
        self.cfg.garment_name = garment_name
        self.cfg.garment_version = garment_version

        # Reload garment configuration
        self.garment_config = self.garment_loader.load_garment_config(
            garment_name, garment_version
        )
        logger.debug(f"[GarmentEnv] Garment config reloaded for {garment_name}")

        # solve particle ditorition
        logger.debug(
            f"[GarmentEnv] Running physics steps to clean up old particle system..."
        )
        cleanup_steps = 20

        if hasattr(self, "sim") and self.sim is not None:
            for i in range(cleanup_steps):
                try:
                    self.sim.step(render=True)
                    # Log progress every 5 steps
                    if (i + 1) % 5 == 0:
                        logger.debug(
                            f"[GarmentEnv] Cleanup progress: {i+1}/{cleanup_steps}"
                        )
                except Exception as e:
                    logger.warning(f"[GarmentEnv] Error during cleanup step {i+1}: {e}")
                    # Continue with next step
                    continue
            logger.debug(f"[GarmentEnv] Cleanup physics steps completed")
        else:
            logger.warning(f"[GarmentEnv] sim not available, skipping cleanup steps")

        # create new garment object
        self._create_garment_object()
        logger.debug(f"[GarmentEnv] New garment object created for {garment_name}")
        logger.debug(
            f"[GarmentEnv] Running initial physics steps to register prim in stage..."
        )
        initial_steps = 5
        if hasattr(self, "sim") and self.sim is not None:
            for i in range(initial_steps):
                try:
                    self.sim.step(render=True)
                except Exception as e:
                    logger.warning(f"[GarmentEnv] Error during initial step {i+1}: {e}")
            logger.debug(f"[GarmentEnv] Initial physics steps completed")
        else:
            logger.warning(f"[GarmentEnv] sim not available, skipping initial steps")
        if hasattr(self, "render"):
            try:
                self.render()
                logger.debug(f"[GarmentEnv] Render called after initial physics steps")
            except Exception as e:
                logger.warning(
                    f"[GarmentEnv] Error during render after initial steps: {e}"
                )

        try:
            self.initialize_obs()
            logger.debug(
                f"[GarmentEnv] Observation system initialized for {garment_name}"
            )
            if hasattr(self, "render"):
                try:
                    self.render()
                    logger.debug(
                        f"[GarmentEnv] Render called after observation initialization"
                    )
                except Exception as e:
                    logger.debug(
                        f"[GarmentEnv] Error during render after observation init: {e}"
                    )
        except Exception as e:
            logger.warning(
                f"[GarmentEnv] Failed to initialize observations (may be expected): {e}"
            )

    def cleanup(self):
        """Cleanup method (defensive programming).

        Note: When environments are fully closed and recreated (as in eval.py),
        this cleanup is not strictly necessary since _create_garment_object()
        already handles checking and cleaning up existing prims when creating
        a new environment. However, this method is kept as a safety measure
        for cases where the same environment instance might be reused.
        """
        logger.debug("[GarmentEnv] Starting cleanup...")

        # Delete garment object if it exists
        if self.object is not None:
            self._delete_garment_object()
            logger.debug("[GarmentEnv] Garment object cleaned up")

        # Clear references
        self.object = None
        # Note: Don't clear garment_config and particle_config as they might be needed
        # if the environment is reset rather than recreated

        logger.debug("[GarmentEnv] Cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup on deletion."""
        try:
            if hasattr(self, "object") and self.object is not None:
                self.cleanup()
        except Exception:
            # Ignore errors during destruction
            pass
