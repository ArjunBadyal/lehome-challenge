"""ACT + scripted collar-fold recovery suffix.

Wraps an existing Top-Short ACT specialist. Runs ACT normally for the first
part of the episode; when ACT has been near-stationary for a window of steps
OR a fixed step count has elapsed, switches to a hand-coded recovery
suffix that pinches the collar/shoulder region, lifts, brings inward, and
releases.

Phase 1 of the scripted-collar-recovery plan (internal notes). This is a
proof-of-life policy:
the suffix uses joint-delta moves from wherever ACT left the arms, not absolute
poses, so it does not depend on knowing the garment's pose. Public observations
only — no check_point ground truth at runtime.

Action / state convention (12D bimanual SO-101):
  index 0..4:   left arm  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
  index 5:      left gripper
  index 6..10:  right arm
  index 11:     right gripper
"""

from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np

from lehome.utils.logger import get_logger
from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

logger = get_logger(__name__)


# Joint indices in the 12D state/action vector (SO-101 bimanual)
LEFT_ARM_IDX = [0, 1, 2, 3, 4]
LEFT_GRIPPER_IDX = 5
RIGHT_ARM_IDX = [6, 7, 8, 9, 10]
RIGHT_GRIPPER_IDX = 11

# Gripper conventions (from PolicyStabilizer constants):
#   open      ≳ 0.3
#   closed    ≲ -0.12 (on cloth) / -0.15 (empty hard stop)
GRIPPER_OPEN = 0.5
GRIPPER_CLOSED = -0.15

# Suffix scheduling. All times in *policy-step* counts (post-trigger).
# Designed for a 600-step episode with trigger at ~step 350.
SUFFIX_PHASES = [
    # (label,             duration_steps, description)
    ("compress_inward",    40, "shoulder_pan inward while keeping anchor gripper state"),
    ("lift_slightly",      20, "small wrist_flex up to gather cloth"),
    ("settle",             40, "hold position (no release - keep cloth gathered)"),
]
# Total suffix duration = 100 steps. v2 design: start from ACT's final state,
# compress shoulders inward without releasing cloth. Conservative — won't undo
# ACT's good work; will only add inward compression on top.

SUFFIX_VERSION = "top_short_collar_v2"


def _smoothstep(t: float) -> float:
    """Smooth interpolation factor in [0, 1]; t in [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@PolicyRegistry.register("act_with_collar_recovery")
class ScriptedCollarRecoveryPolicy(BasePolicy):
    """ACT specialist + hand-coded collar-fold suffix.

    Args (passed by PolicyRegistry.create from CLI):
        policy_path: Path to the wrapped ACT specialist's pretrained_model dir
        dataset_root: Dataset root for ACT metadata
        task_description: Forwarded to ACT
        device: cpu or cuda for ACT inference
        trigger_step: Step at which to switch from ACT to scripted suffix
                      (default 350; override via env STABILIZER_RECOVERY_TRIGGER_STEP)
        stationary_window: # of consecutive low-velocity steps to also trigger
                           (default 30; env STABILIZER_RECOVERY_STATIONARY_WIN)
        stationary_thr:    L_inf joint-velocity threshold for "stationary"
                           (default 0.02; env STABILIZER_RECOVERY_STATIONARY_THR)

    The wrapped ACT specialist must accept 12D bimanual actions / 12D state.
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

        self.trigger_step = int(os.environ.get(
            "RECOVERY_TRIGGER_STEP", kwargs.get("trigger_step", 350)
        ))
        self.stationary_window = int(os.environ.get(
            "RECOVERY_STATIONARY_WIN", kwargs.get("stationary_window", 30)
        ))
        self.stationary_thr = float(os.environ.get(
            "RECOVERY_STATIONARY_THR", kwargs.get("stationary_thr", 0.02)
        ))
        # Per-phase deltas. v2: conservative — keep ACT's anchor gripper state.
        # Strategy: ACT may have already closed grippers on cloth at trigger
        # time; releasing it here destroys good progress. So we KEEP the anchor
        # gripper state and only adjust arm joints.
        self._phase_deltas = {
            # phase: (arm_delta_l, arm_delta_r, gripper_mode) where gripper_mode
            # is "anchor" (use anchor state's gripper) or "open"/"close" override
            "compress_inward":   (np.array([+0.25, -0.05, 0.0,  0.05, 0.0]), np.array([-0.25, -0.05, 0.0,  0.05, 0.0]), "anchor"),
            "lift_slightly":     (np.array([0.0,  -0.10, -0.05, -0.05, 0.0]), np.array([0.0,  -0.10, -0.05, -0.05, 0.0]), "anchor"),
            "settle":            (np.zeros(5),                                np.zeros(5),                                "anchor"),
        }

        # State machine (per-episode)
        self.reset()

        logger.info(
            f"[ScriptedCollarRecoveryPolicy] trigger_step={self.trigger_step}, "
            f"stationary_window={self.stationary_window}, "
            f"stationary_thr={self.stationary_thr}, version={SUFFIX_VERSION}"
        )

    def reset(self):
        self.act.reset()
        self.step_count = 0
        self.in_suffix = False
        self.suffix_start_step = None
        # State at trigger time, snapshotted to compute target joints
        self.snapshot_action: np.ndarray | None = None
        # Stationarity history
        self._prev_state: np.ndarray | None = None
        self._stationary_count = 0
        # Action used last step (for rate-limit-friendly suffix)
        self._last_action: np.ndarray | None = None

    def _is_stationary(self, state: np.ndarray) -> bool:
        if self._prev_state is None:
            self._prev_state = state.copy()
            return False
        diff = np.abs(state - self._prev_state).max()
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
                    f"[recovery] stationarity trigger fired at step {self.step_count} "
                    f"(stationary {self._stationary_count} steps)"
                )
                return True
        else:
            self._stationary_count = 0
        return False

    def _phase_at(self, suffix_step: int) -> tuple[str, float]:
        """Return (phase_name, fraction-into-phase in [0,1]) for the given suffix step."""
        cum = 0
        for label, dur, _desc in SUFFIX_PHASES:
            if suffix_step < cum + dur:
                frac = (suffix_step - cum) / max(1, dur)
                return label, frac
            cum += dur
        # Past end of suffix: hold last phase
        return SUFFIX_PHASES[-1][0], 1.0

    def _suffix_action(self, suffix_step: int, state: np.ndarray) -> np.ndarray:
        """Compute the joint-position action for the given suffix step.

        Approach: use the snapshot action at trigger time as an anchor; apply
        per-phase deltas gradually using a smoothstep envelope. The action
        returned is *target joint positions*, matching the action convention
        expected by the env.
        """
        if self.snapshot_action is None:
            self.snapshot_action = state.copy()
        anchor = self.snapshot_action.copy()

        phase, frac = self._phase_at(suffix_step)
        arm_l_delta, arm_r_delta, grip_mode = self._phase_deltas[phase]

        # Compute cumulative deltas: sum of all completed phase deltas plus
        # the current phase scaled by smoothstep(frac).
        cum_l = np.zeros(5)
        cum_r = np.zeros(5)
        cum_step = 0
        for label, dur, _desc in SUFFIX_PHASES:
            l_d, r_d, _ = self._phase_deltas[label]
            if suffix_step >= cum_step + dur:
                cum_l = cum_l + l_d
                cum_r = cum_r + r_d
            else:
                f = (suffix_step - cum_step) / max(1, dur)
                cum_l = cum_l + l_d * _smoothstep(f)
                cum_r = cum_r + r_d * _smoothstep(f)
                break
            cum_step += dur

        action = anchor.copy()
        action[LEFT_ARM_IDX] = anchor[LEFT_ARM_IDX] + cum_l
        action[RIGHT_ARM_IDX] = anchor[RIGHT_ARM_IDX] + cum_r

        # Gripper: v2 keeps the anchor gripper state ("anchor" mode).
        # If a phase explicitly says "open" or "close" we override; otherwise
        # the gripper is left at whatever ACT set at trigger time. This avoids
        # the v1 mistake of releasing cloth that ACT had already grasped.
        cum_step = 0
        chosen_mode = "anchor"
        for label, dur, _desc in SUFFIX_PHASES:
            _, _, mode = self._phase_deltas[label]
            if suffix_step < cum_step + dur:
                chosen_mode = mode
                break
            cum_step += dur
        else:
            chosen_mode = SUFFIX_PHASES[-1][0] if False else "anchor"

        if chosen_mode == "anchor":
            action[LEFT_GRIPPER_IDX] = anchor[LEFT_GRIPPER_IDX]
            action[RIGHT_GRIPPER_IDX] = anchor[RIGHT_GRIPPER_IDX]
        elif chosen_mode == "close":
            action[LEFT_GRIPPER_IDX] = GRIPPER_CLOSED
            action[RIGHT_GRIPPER_IDX] = GRIPPER_CLOSED
        elif chosen_mode == "open":
            action[LEFT_GRIPPER_IDX] = GRIPPER_OPEN
            action[RIGHT_GRIPPER_IDX] = GRIPPER_OPEN

        return action.astype(np.float32)

    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32)

        if not self.in_suffix and self._check_trigger(state):
            self.in_suffix = True
            self.suffix_start_step = self.step_count
            # Snapshot the joint state at trigger time as the suffix anchor.
            # We use *state* (current measured joints) rather than last action
            # so that even if ACT was issuing actions ahead of where the arms
            # actually are, the suffix continues from physical reality.
            self.snapshot_action = state.copy()
            logger.info(
                f"[recovery] entered suffix at step {self.step_count}, "
                f"anchor state min/max: {state.min():.3f}/{state.max():.3f}"
            )

        if self.in_suffix:
            suffix_step = self.step_count - self.suffix_start_step
            action = self._suffix_action(suffix_step, state)
        else:
            action = self.act.select_action(observation).astype(np.float32)

        self._last_action = action.copy()
        self.step_count += 1
        return action
