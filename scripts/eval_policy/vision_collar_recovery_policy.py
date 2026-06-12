"""ACT + closed-loop vision-guided collar-fold recovery suffix.

Phase 2 of the procedural-data plan (internal notes).

Replaces the open-loop scripted suffix (`scripted_collar_recovery_policy.py`)
with a closed-loop visual servo:
  1. ACT specialist runs normally for the early phase.
  2. Trigger condition fires (stationarity OR fixed step count).
  3. The suffix uses the top RGB camera to:
       - segment the garment silhouette
       - detect top-left / top-right corner landmarks (proxy for shoulders)
       - detect left and right gripper positions
       - use proportional control to nudge each gripper toward the corresponding
         shoulder corner without releasing cloth ACT had grasped
       - then close grippers (if not already), lift slightly, and bring grippers
         toward the garment centroid (the collar fold target)
       - re-detect every step; abort and hold if detection fails

All inputs are public observations (top_rgb, top_depth, observation.state).
No check_point / privileged data online.

Action / state convention: 12D bimanual SO-101
  index 0..4:   left arm  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
  index 5:      left gripper
  index 6..10:  right arm
  index 11:     right gripper
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple

import numpy as np

from lehome.utils.logger import get_logger
from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry
from scripts.cv_collar_pol.segment_and_landmark import (
    segment_garment,
    compute_landmarks,
    detect_grippers,
    Landmarks,
)

logger = get_logger(__name__)


# Joint indices in 12D bimanual state/action.
LEFT_SHOULDER_PAN = 0
LEFT_SHOULDER_LIFT = 1
LEFT_ELBOW = 2
LEFT_WRIST_FLEX = 3
LEFT_WRIST_ROLL = 4
LEFT_GRIPPER_IDX = 5
RIGHT_SHOULDER_PAN = 6
RIGHT_SHOULDER_LIFT = 7
RIGHT_ELBOW = 8
RIGHT_WRIST_FLEX = 9
RIGHT_WRIST_ROLL = 10
RIGHT_GRIPPER_IDX = 11

LEFT_ARM_IDX = [LEFT_SHOULDER_PAN, LEFT_SHOULDER_LIFT, LEFT_ELBOW,
                LEFT_WRIST_FLEX, LEFT_WRIST_ROLL]
RIGHT_ARM_IDX = [RIGHT_SHOULDER_PAN, RIGHT_SHOULDER_LIFT, RIGHT_ELBOW,
                 RIGHT_WRIST_FLEX, RIGHT_WRIST_ROLL]

GRIPPER_OPEN = 0.5
GRIPPER_CLOSED = -0.15

# Proportional gains (rad per normalized image error).
# Conservative — starts small to avoid overshoot.
K_SHOULDER_PAN = 0.30
K_SHOULDER_LIFT = 0.10
K_WRIST_FLEX = 0.10

# Safety: max delta per policy step
MAX_DELTA_PER_STEP = 0.04

# Image dimensions (assumed 480x640; verified in compute_landmarks)
IMG_W = 640
IMG_H = 480

SUFFIX_VERSION = "vision_collar_v1"


# Suffix phase plan (in suffix-step counts)
PHASE_APPROACH_DURATION = 50   # servo grippers toward top-corners
PHASE_PINCH_DURATION    = 20   # close grippers
PHASE_FOLD_DURATION     = 50   # servo toward centroid (bring shoulders together)
PHASE_RELEASE_DURATION  = 30   # release + hold
PHASE_SETTLE_DURATION   = 50   # hold for cloth physics
TOTAL_SUFFIX_DURATION   = (PHASE_APPROACH_DURATION + PHASE_PINCH_DURATION
                            + PHASE_FOLD_DURATION + PHASE_RELEASE_DURATION
                            + PHASE_SETTLE_DURATION)


def _phase_at(suffix_step: int) -> str:
    if suffix_step < PHASE_APPROACH_DURATION:
        return "approach"
    s = suffix_step - PHASE_APPROACH_DURATION
    if s < PHASE_PINCH_DURATION:
        return "pinch"
    s -= PHASE_PINCH_DURATION
    if s < PHASE_FOLD_DURATION:
        return "fold"
    s -= PHASE_FOLD_DURATION
    if s < PHASE_RELEASE_DURATION:
        return "release"
    return "settle"


@PolicyRegistry.register("act_with_vision_collar_recovery")
class VisionCollarRecoveryPolicy(BasePolicy):
    """ACT specialist + closed-loop vision-guided collar recovery suffix."""

    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str = "Fold a garment with bimanual robot arms",
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()

        self.act = LeRobotPolicy(
            policy_path=policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
        )

        self.trigger_step = int(os.environ.get(
            "RECOVERY_TRIGGER_STEP", kwargs.get("trigger_step", 350)
        ))
        self.stationary_window = int(os.environ.get(
            "RECOVERY_STATIONARY_WIN", kwargs.get("stationary_window", 30)
        ))
        self.stationary_thr = float(os.environ.get(
            "RECOVERY_STATIONARY_THR", kwargs.get("stationary_thr", 0.02)
        ))

        self.reset()
        logger.info(
            f"[VisionCollarRecoveryPolicy] trigger_step={self.trigger_step}, "
            f"stationary_window={self.stationary_window}, "
            f"stationary_thr={self.stationary_thr}, version={SUFFIX_VERSION}"
        )

    def reset(self):
        self.act.reset()
        self.step_count = 0
        self.in_suffix = False
        self.suffix_start_step: Optional[int] = None
        self.snapshot_action: Optional[np.ndarray] = None
        self._prev_state: Optional[np.ndarray] = None
        self._stationary_count = 0
        self._last_action: Optional[np.ndarray] = None
        # Detection history for stability checks
        self._lm_history: list = []

    def _is_stationary(self, state: np.ndarray) -> bool:
        if self._prev_state is None:
            self._prev_state = state.copy()
            return False
        diff = float(np.abs(state - self._prev_state).max())
        self._prev_state = state.copy()
        return diff < self.stationary_thr

    def _check_trigger(self, state: np.ndarray) -> bool:
        if self.in_suffix:
            return True
        if self.step_count >= self.trigger_step:
            return True
        if self._is_stationary(state):
            self._stationary_count += 1
            if self._stationary_count >= self.stationary_window:
                logger.info(
                    f"[vision_recovery] stationarity trigger at step {self.step_count}"
                )
                return True
        else:
            self._stationary_count = 0
        return False

    def _detect(self, top_rgb: np.ndarray) -> Optional[Landmarks]:
        """Run segmentation + landmark detection on a top-RGB frame.

        Returns None if detection failed. Caller should hold position in that case.
        Note: env provides RGB as (H, W, 3), but cv2 expects BGR. The training
        pipeline normalizes from RGB; the saved videos write BGR. We accept
        whatever the env gives — the segmentation thresholds work approximately
        either way for white/yellow detection.
        """
        try:
            mask = segment_garment(top_rgb)
            lm = compute_landmarks(mask)
            if lm is not None:
                lg, rg = detect_grippers(top_rgb)
                lm.left_gripper = lg
                lm.right_gripper = rg
            return lm
        except Exception as exc:
            logger.warning(f"[vision_recovery] detection error: {exc}")
            return None

    def _compute_servo_action(
        self,
        anchor: np.ndarray,
        state: np.ndarray,
        lm: Landmarks,
        phase: str,
    ) -> np.ndarray:
        """Generate a target joint-position action for the current phase.

        Uses pixel-space proportional control. Mapping:
          - target_x normalized to [-0.5, 0.5] of image width
          - shoulder_pan delta proportional to (target_x - gripper_x_norm)
          - shoulder_lift / wrist_flex deltas based on target_y
        """
        action = anchor.copy()  # start from snapshot as baseline

        # Phase-specific targets in pixel space:
        if phase == "approach":
            # Each gripper to its corresponding top-corner (shoulder)
            target_left = lm.top_left
            target_right = lm.top_right
            grip_l_target = anchor[LEFT_GRIPPER_IDX]   # don't change yet
            grip_r_target = anchor[RIGHT_GRIPPER_IDX]
        elif phase == "pinch":
            # Hold position; close grippers
            target_left = lm.top_left
            target_right = lm.top_right
            grip_l_target = GRIPPER_CLOSED
            grip_r_target = GRIPPER_CLOSED
        elif phase == "fold":
            # Drive both grippers toward centroid (the centerline)
            target_left = lm.centroid
            target_right = lm.centroid
            grip_l_target = GRIPPER_CLOSED
            grip_r_target = GRIPPER_CLOSED
        elif phase == "release":
            # Hold near centroid, open grippers
            target_left = lm.centroid
            target_right = lm.centroid
            grip_l_target = GRIPPER_OPEN
            grip_r_target = GRIPPER_OPEN
        else:  # settle
            target_left = lm.centroid
            target_right = lm.centroid
            grip_l_target = GRIPPER_OPEN
            grip_r_target = GRIPPER_OPEN

        # Compute pixel-space errors (only meaningful if grippers are detected).
        if lm.left_gripper is not None:
            err_lx = (target_left[0] - lm.left_gripper[0]) / IMG_W   # in [-1, 1]
            err_ly = (target_left[1] - lm.left_gripper[1]) / IMG_H
            d_pan_l = -K_SHOULDER_PAN * err_lx    # left arm: + shoulder_pan moves gripper RIGHT
            # Negative because left gripper is on the LEFT side of image; positive shoulder_pan
            # rotates the arm to its left (decreasing image x).
            d_lift_l = K_SHOULDER_LIFT * err_ly   # downward error → lower shoulder
            d_wrist_l = K_WRIST_FLEX * err_ly
        else:
            d_pan_l = d_lift_l = d_wrist_l = 0.0

        if lm.right_gripper is not None:
            err_rx = (target_right[0] - lm.right_gripper[0]) / IMG_W
            err_ry = (target_right[1] - lm.right_gripper[1]) / IMG_H
            d_pan_r = +K_SHOULDER_PAN * err_rx
            d_lift_r = K_SHOULDER_LIFT * err_ry
            d_wrist_r = K_WRIST_FLEX * err_ry
        else:
            d_pan_r = d_lift_r = d_wrist_r = 0.0

        # Clip per-step deltas
        for d in [d_pan_l, d_lift_l, d_wrist_l, d_pan_r, d_lift_r, d_wrist_r]:
            pass  # ensure no NaN; handled below
        d_pan_l = float(np.clip(d_pan_l, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP))
        d_lift_l = float(np.clip(d_lift_l, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP))
        d_wrist_l = float(np.clip(d_wrist_l, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP))
        d_pan_r = float(np.clip(d_pan_r, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP))
        d_lift_r = float(np.clip(d_lift_r, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP))
        d_wrist_r = float(np.clip(d_wrist_r, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP))

        # Apply deltas relative to CURRENT MEASURED STATE (not accumulating).
        # This gives bounded corrections: action = state + small_correction,
        # so even if the robot doesn't track perfectly, we never drift past
        # MAX_DELTA_PER_STEP from where the joints actually are.
        # Additionally, clamp the resulting action to anchor ± 0.6 rad to
        # prevent the controller from moving far from where ACT left things.
        action = state.copy()
        action[LEFT_SHOULDER_PAN] = state[LEFT_SHOULDER_PAN] + d_pan_l
        action[LEFT_SHOULDER_LIFT] = state[LEFT_SHOULDER_LIFT] + d_lift_l
        action[LEFT_WRIST_FLEX] = state[LEFT_WRIST_FLEX] + d_wrist_l
        action[LEFT_GRIPPER_IDX] = grip_l_target
        action[RIGHT_SHOULDER_PAN] = state[RIGHT_SHOULDER_PAN] + d_pan_r
        action[RIGHT_SHOULDER_LIFT] = state[RIGHT_SHOULDER_LIFT] + d_lift_r
        action[RIGHT_WRIST_FLEX] = state[RIGHT_WRIST_FLEX] + d_wrist_r
        action[RIGHT_GRIPPER_IDX] = grip_r_target

        # Clamp arms to anchor ± 0.6 rad envelope (tight enough to avoid
        # destroying ACT's progress; loose enough to allow the corrective fold)
        ENVELOPE = 0.6
        for idx in LEFT_ARM_IDX + RIGHT_ARM_IDX:
            action[idx] = float(np.clip(action[idx],
                                         anchor[idx] - ENVELOPE,
                                         anchor[idx] + ENVELOPE))
        return action.astype(np.float32)

    def _hold_action(self, state: np.ndarray) -> np.ndarray:
        """Fallback: command current measured state (zero motion)."""
        return state.astype(np.float32).copy()

    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32)

        if not self.in_suffix and self._check_trigger(state):
            self.in_suffix = True
            self.suffix_start_step = self.step_count
            self.snapshot_action = state.copy()
            logger.info(
                f"[vision_recovery] entered suffix at step {self.step_count}"
            )

        if self.in_suffix:
            suffix_step = self.step_count - self.suffix_start_step
            phase = _phase_at(suffix_step)

            top_rgb = observation.get("observation.images.top_rgb")
            lm: Optional[Landmarks] = None
            if top_rgb is not None:
                rgb_np = np.asarray(top_rgb)
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8) if rgb_np.max() <= 1.0 else rgb_np.astype(np.uint8)
                # Env provides RGB; cv2 wants BGR. The segmenter is robust to the
                # swap (yellow detection in HSV is roughly invariant for our gain
                # range), but for correctness we flip to BGR.
                if rgb_np.shape[-1] == 3:
                    rgb_bgr = rgb_np[..., ::-1].copy()
                else:
                    rgb_bgr = rgb_np
                lm = self._detect(rgb_bgr)

            if lm is None:
                action = self._hold_action(state)
            else:
                self._lm_history.append(lm)
                if self.snapshot_action is None:
                    self.snapshot_action = state.copy()
                action = self._compute_servo_action(
                    self.snapshot_action, state, lm, phase
                )
        else:
            action = self.act.select_action(observation).astype(np.float32)

        self._last_action = action.copy()
        self.step_count += 1
        return action
