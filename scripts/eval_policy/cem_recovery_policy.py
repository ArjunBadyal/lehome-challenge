"""ACT + knot-interpolated recovery suffix for CEM trajectory optimization.

The recovery suffix is parameterized by KNOTS (6 knots × 6 joints = 36 floats).
Knots are linearly interpolated over the recovery horizon. The 6 controllable
joints are L+R for shoulder_lift, elbow_flex, wrist_flex (no shoulder_pan to
avoid the "separating-garment" failure mode seen in earlier hand-tuned attempts).
Grippers are forced tight throughout the recovery and released at the end.

The CEM driver loads a knot vector from /tmp/cem_knots.npy and the trigger
step from CEM_TRIGGER_STEP env var. Each rollout = one full ACT prefix + one
suffix realization.

This policy is used ONLY OFFLINE to generate successful repaired trajectories.
The final submission stack does not include this policy.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np

from lehome.utils.logger import get_logger
from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


# Joint indices (12D bimanual SO-101)
LEFT_SHOULDER_LIFT = 1
LEFT_ELBOW = 2
LEFT_WRIST_FLEX = 3
LEFT_GRIPPER_IDX = 5
RIGHT_SHOULDER_LIFT = 7
RIGHT_ELBOW = 8
RIGHT_WRIST_FLEX = 9
RIGHT_GRIPPER_IDX = 11

# 6 controllable joints, in fixed order
CONTROL_JOINTS = [
    LEFT_SHOULDER_LIFT,
    LEFT_ELBOW,
    LEFT_WRIST_FLEX,
    RIGHT_SHOULDER_LIFT,
    RIGHT_ELBOW,
    RIGHT_WRIST_FLEX,
]
N_CONTROL = len(CONTROL_JOINTS)

GRIPPER_OPEN = 0.5
GRIPPER_CLOSED = -0.15
GRIPPER_TIGHT = -0.20

# Rate limit: same cap as PolicyStabilizer's calibrated values
# (95th percentile of per-step joint |Δ| in demo data).
MAX_DELTA_ARM = 0.111
MAX_DELTA_GRIPPER = 0.104
PER_JOINT_MAX = np.array([
    MAX_DELTA_ARM, MAX_DELTA_ARM, MAX_DELTA_ARM, MAX_DELTA_ARM, MAX_DELTA_ARM,
    MAX_DELTA_GRIPPER,
    MAX_DELTA_ARM, MAX_DELTA_ARM, MAX_DELTA_ARM, MAX_DELTA_ARM, MAX_DELTA_ARM,
    MAX_DELTA_GRIPPER,
], dtype=np.float32)

# Absolute joint bounds (radians) — derived from observed demo state ranges.
# Keep arms within plausible reachable space to prevent cloth physics blowup.
JOINT_LO = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -0.3,
                     -2.0, -2.0, -2.0, -2.0, -2.0, -0.3], dtype=np.float32)
JOINT_HI = np.array([+2.0, +2.0, +2.0, +2.0, +2.0, +0.6,
                     +2.0, +2.0, +2.0, +2.0, +2.0, +0.6], dtype=np.float32)


def _interp_knots(knots: np.ndarray, n_steps: int) -> np.ndarray:
    """Linear-interpolate (n_knots, n_joints) knots into (n_steps, n_joints) trajectory."""
    n_knots = len(knots)
    if n_knots == 1:
        return np.tile(knots[0], (n_steps, 1))
    times = np.linspace(0, n_knots - 1, n_steps)
    out = np.zeros((n_steps, knots.shape[1]), dtype=np.float32)
    for joint in range(knots.shape[1]):
        out[:, joint] = np.interp(times, np.arange(n_knots), knots[:, joint])
    return out


@PolicyRegistry.register("cem_recovery_top_short")
class CEMRecoveryTopShort(BasePolicy):
    """ACT + knot-driven recovery suffix.

    Knots loaded from CEM_KNOTS_FILE (default /tmp/cem_knots.npy).
    Trigger step from CEM_TRIGGER_STEP (default 220).
    Recovery horizon from CEM_HORIZON (default 100).
    """

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
        self.trigger_step = int(os.environ.get("CEM_TRIGGER_STEP", 220))
        self.horizon = int(os.environ.get("CEM_HORIZON", 100))
        self.release_steps = int(os.environ.get("CEM_RELEASE_STEPS", 15))
        self.knots_file = os.environ.get("CEM_KNOTS_FILE", "/tmp/cem_knots.npy")
        self.knots = None
        if os.path.exists(self.knots_file):
            self.knots = np.load(self.knots_file).astype(np.float32)
            assert self.knots.shape[1] == N_CONTROL, f"Expected knots shape (n,{N_CONTROL}), got {self.knots.shape}"
        self.suffix_actions = None  # filled at trigger time
        self.reset()
        logger.info(
            f"[CEMRecoveryTopShort] trigger={self.trigger_step} "
            f"horizon={self.horizon} release={self.release_steps} "
            f"knots_file={self.knots_file} knots={'loaded' if self.knots is not None else 'None'}"
        )

    def reset(self):
        self.act.reset()
        self.step_count = 0
        self.in_suffix = False
        self.suffix_start_step: Optional[int] = None
        self.snapshot_state: Optional[np.ndarray] = None
        self.suffix_actions = None
        self.prev_action: Optional[np.ndarray] = None

    def _build_suffix_actions(self, anchor: np.ndarray) -> np.ndarray:
        """Build the (horizon + release_steps, 12) action sequence.

        Knots specify the controllable-joint trajectory as DELTAS from anchor.
        Each knot is a 6-vector applied as anchor[joint] + knot[i].
        Non-controllable joints stay at anchor. Grippers are tight throughout
        horizon, ramped to open during release.
        """
        if self.knots is None:
            return np.tile(anchor, (self.horizon + self.release_steps, 1))
        # Knots are now interpreted as DELTAS from anchor. The CEM seed knots
        # may originally be in absolute target space; the driver should now
        # extract them as deltas from a corresponding-step anchor.
        traj_ctrl = _interp_knots(self.knots, self.horizon)  # (horizon, N_CONTROL)
        actions = np.tile(anchor, (self.horizon + self.release_steps, 1)).astype(np.float32)
        for i, joint in enumerate(CONTROL_JOINTS):
            actions[:self.horizon, joint] = anchor[joint] + traj_ctrl[:, i]
        actions[:self.horizon, LEFT_GRIPPER_IDX] = GRIPPER_TIGHT
        actions[:self.horizon, RIGHT_GRIPPER_IDX] = GRIPPER_TIGHT
        for k in range(self.release_steps):
            alpha = (k + 1) / self.release_steps
            grip = GRIPPER_TIGHT + (GRIPPER_OPEN - GRIPPER_TIGHT) * alpha
            actions[self.horizon + k, LEFT_GRIPPER_IDX] = grip
            actions[self.horizon + k, RIGHT_GRIPPER_IDX] = grip
            for i, joint in enumerate(CONTROL_JOINTS):
                actions[self.horizon + k, joint] = anchor[joint] + traj_ctrl[-1, i]
        return actions

    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32)
        # State-aware trigger: only fire if grippers are CLOSED on cloth.
        # If ACT has open grippers at trigger_step, it's either not ready (still
        # approaching) or has released the cloth — either way our recovery would
        # do nothing useful. Skip and let ACT continue.
        if not self.in_suffix and self.step_count >= self.trigger_step:
            grippers_closed = state[5] < -0.05 and state[11] < -0.05
            # Hard timeout: 50 steps grace, then fire regardless
            timeout = self.step_count >= self.trigger_step + 50
            if grippers_closed or timeout:
                self.in_suffix = True
                self.suffix_start_step = self.step_count
                self.snapshot_state = state.copy()
                self.suffix_actions = self._build_suffix_actions(self.snapshot_state)
                logger.info(
                    f"[cem_recovery] trigger fired at step {self.step_count}, "
                    f"L_grip={state[5]:.3f} R_grip={state[11]:.3f}, "
                    f"suffix_len={len(self.suffix_actions)}, timeout={timeout}"
                )
        if self.in_suffix:
            suffix_step = self.step_count - self.suffix_start_step
            if suffix_step < len(self.suffix_actions):
                action = self.suffix_actions[suffix_step].copy()
            else:
                action = self.suffix_actions[-1].copy()
                action[LEFT_GRIPPER_IDX] = GRIPPER_OPEN
                action[RIGHT_GRIPPER_IDX] = GRIPPER_OPEN
        else:
            action = self.act.select_action(observation).astype(np.float32)

        # Rate-limit per-step deltas + clamp to absolute joint bounds (prevents
        # cloth physics divergence that would otherwise produce nonsense scores).
        if self.prev_action is not None:
            delta = action - self.prev_action
            delta = np.clip(delta, -PER_JOINT_MAX, PER_JOINT_MAX)
            action = self.prev_action + delta
        action = np.clip(action, JOINT_LO, JOINT_HI)
        self.prev_action = action.copy()
        self.step_count += 1
        return action.astype(np.float32)
