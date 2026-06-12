"""PolicyStabilizer: eval-time wrapper that smooths actions, delays throws, and retries failed grasps.

Wraps any BasePolicy with three interventions:
  1. Action rate limiting — caps per-step joint delta to prevent throwing
  2. Gripper release delay — holds gripper closed while arm is moving fast
  3. Grasp retry — detects failed grasps and triggers open-perturb-close sequence

All interventions are eval-time only; no retraining needed. Submission-compliant:
uses only public observations (state + images) and modifies actions.
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque

from .base_policy import BasePolicy
from .registry import PolicyRegistry


# Joint indices in the 12D state/action vector (SO-101 bimanual)
LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11
LEFT_ARM_INDICES = [0, 1, 2, 3, 4]  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
RIGHT_ARM_INDICES = [6, 7, 8, 9, 10]

# Calibrated thresholds from demo data (see scripts/calibrate_stabilizer.py)
# Note: gripper state values are in range [-0.155, +0.115] when commanded closed.
# Grippers closed on cloth cluster around -0.12 to -0.14 (stopped early by cloth).
# Grippers closed empty cluster around -0.15 (hit hard stop).
# Gap between clusters is only ~0.03, so bimodality is weak → grasp retry disabled by default.
GRIPPER_OPEN_THRESHOLD = 0.3             # > this means open
GRIPPER_CLOSED_EMPTY = -0.148            # very negative = empty close (hard stop)
GRIPPER_CLOSED_WITH_CLOTH = -0.125       # less negative = cloth held (early stop)
ARM_HIGH_VELOCITY = 0.111                # calibrated: 90th %ile of arm speed at release
MAX_JOINT_DELTA_ARM = 0.111              # calibrated: 95th %ile of per-step arm joint |Δ|
MAX_JOINT_DELTA_GRIPPER = 0.104          # calibrated: 95th %ile of per-step gripper |Δ|
STABILITY_WINDOW = 15                    # steps for stability check
STABILITY_TOLERANCE = 0.01               # |delta| must be below this for stability


class PolicyStabilizer(BasePolicy):
    """Eval-time stabilizer wrapping a base policy.

    Flags:
      - rate_limit: cap per-step joint delta to max_joint_delta
      - delay_throws: delay gripper release during high arm velocity
      - retry_grasps: detect failed grasps, override for retry
    """

    def __init__(
        self,
        base_policy: BasePolicy,
        max_joint_delta_arm: float = MAX_JOINT_DELTA_ARM,
        max_joint_delta_gripper: float = MAX_JOINT_DELTA_GRIPPER,
        arm_vel_threshold: float = ARM_HIGH_VELOCITY,
        gripper_empty_threshold: float = GRIPPER_CLOSED_EMPTY,
        retry_open_steps: int = 10,
        retry_perturb_steps: int = 5,
        retry_perturb_amplitude: float = 0.05,
        max_retries_per_arm: int = 2,
        rate_limit: bool = True,
        delay_throws: bool = True,
        retry_grasps: bool = False,  # disabled by default: clusters not cleanly separable
        release_hold_steps: int = 0,
    ):
        super().__init__()
        self.base = base_policy

        # Per-joint rate limits: 5 arm joints + gripper per side
        self.per_joint_max_delta = np.array([
            max_joint_delta_arm,      # L shoulder pan
            max_joint_delta_arm,      # L shoulder lift
            max_joint_delta_arm,      # L elbow
            max_joint_delta_arm,      # L wrist flex
            max_joint_delta_arm,      # L wrist roll
            max_joint_delta_gripper,  # L gripper
            max_joint_delta_arm,      # R shoulder pan
            max_joint_delta_arm,      # R shoulder lift
            max_joint_delta_arm,      # R elbow
            max_joint_delta_arm,      # R wrist flex
            max_joint_delta_arm,      # R wrist roll
            max_joint_delta_gripper,  # R gripper
        ], dtype=np.float32)

        self.arm_vel_threshold = arm_vel_threshold
        self.gripper_empty_threshold = gripper_empty_threshold
        self.retry_open_steps = retry_open_steps
        self.retry_perturb_steps = retry_perturb_steps
        self.retry_perturb_amplitude = retry_perturb_amplitude
        self.max_retries_per_arm = max_retries_per_arm

        self.rate_limit = rate_limit
        self.delay_throws = delay_throws
        self.retry_grasps = retry_grasps
        self.release_hold_steps = int(release_hold_steps)

        # Action-log counters (for per-episode diagnostics)
        self.n_rate_clips = 0
        self.n_delayed_releases = 0
        self.n_retries = {"left": 0, "right": 0}
        self.first_retry_step = None
        self.step_count = 0

        # Per-episode state
        self.release_hold_countdown = {"left": 0, "right": 0}
        self.reset()

    def reset(self):
        self.base.reset()
        self.prev_action: np.ndarray | None = None
        self.prev_state: np.ndarray | None = None

        # Gripper history for stability + grasp-outcome detection
        self.gripper_history = {
            "left": deque(maxlen=STABILITY_WINDOW),
            "right": deque(maxlen=STABILITY_WINDOW),
        }
        self.gripper_cmd_history = {
            "left": deque(maxlen=STABILITY_WINDOW),
            "right": deque(maxlen=STABILITY_WINDOW),
        }
        # Has this arm's grasp already been checked (only check once per close attempt)
        self.grasp_checked = {"left": False, "right": False}
        # Was the most recent grasp attempt a failure?
        self.pending_retry = {"left": False, "right": False}
        self.retry_countdown = {"left": 0, "right": 0}
        self.retries_used = {"left": 0, "right": 0}
        # Phase: 0 = waiting for close command; 1 = gripper is closing; 2 = checked
        self.grasp_phase = {"left": 0, "right": 0}
        self.release_hold_countdown = {"left": 0, "right": 0}

    def select_action(self, observation):
        state = np.asarray(observation["observation.state"], dtype=np.float32)

        # Update gripper history with current state
        self.gripper_history["left"].append(state[LEFT_GRIPPER_IDX])
        self.gripper_history["right"].append(state[RIGHT_GRIPPER_IDX])

        # Get base action
        raw_action = self.base.select_action(observation).astype(np.float32)
        action = raw_action.copy()

        # 1. Grasp retry detection & override
        if self.retry_grasps:
            for arm, gidx, arm_idx in [
                ("left", LEFT_GRIPPER_IDX, LEFT_ARM_INDICES),
                ("right", RIGHT_GRIPPER_IDX, RIGHT_ARM_INDICES),
            ]:
                # Track the commanded gripper value
                self.gripper_cmd_history[arm].append(action[gidx])

                # Check if we're in active retry mode
                if self.retry_countdown[arm] > 0:
                    # First retry_open_steps: force gripper open
                    # Next retry_perturb_steps: add noise to arm joints
                    total_countdown = self.retry_countdown[arm]
                    if total_countdown > self.retry_perturb_steps:
                        # Open gripper phase
                        action[gidx] = 0.5
                    else:
                        # Perturb phase
                        noise = np.random.uniform(
                            -self.retry_perturb_amplitude,
                            self.retry_perturb_amplitude,
                            size=len(arm_idx),
                        )
                        action[arm_idx] = state[arm_idx] + noise
                        action[gidx] = 0.5
                    self.retry_countdown[arm] -= 1
                    continue

                # Detect failed grasp: commanded closed for stability_window AND gripper actually closed empty
                if self._detect_failed_grasp(arm) and self.retries_used[arm] < self.max_retries_per_arm:
                    self.retry_countdown[arm] = self.retry_open_steps + self.retry_perturb_steps
                    self.retries_used[arm] += 1
                    # Reset history so we don't keep detecting the same failure
                    self.gripper_history[arm].clear()
                    self.gripper_cmd_history[arm].clear()

        # 2. Delay throws: if commanded to open gripper but arm is moving fast, keep closed.
        #    If release_hold_steps > 0, continue holding the gripper for that many steps
        #    after the trigger fires, even if the arm has slowed down.
        if self.delay_throws and self.prev_state is not None:
            arm_vel_left = np.abs(state[LEFT_ARM_INDICES] - self.prev_state[LEFT_ARM_INDICES]).max()
            arm_vel_right = np.abs(state[RIGHT_ARM_INDICES] - self.prev_state[RIGHT_ARM_INDICES]).max()

            prev_left_g = self.prev_action[LEFT_GRIPPER_IDX] if self.prev_action is not None else state[LEFT_GRIPPER_IDX]
            prev_right_g = self.prev_action[RIGHT_GRIPPER_IDX] if self.prev_action is not None else state[RIGHT_GRIPPER_IDX]

            for arm, gidx, prev_g, arm_vel in [
                ("left", LEFT_GRIPPER_IDX, prev_left_g, arm_vel_left),
                ("right", RIGHT_GRIPPER_IDX, prev_right_g, arm_vel_right),
            ]:
                opening = action[gidx] > prev_g + 0.1
                if opening and arm_vel > self.arm_vel_threshold:
                    action[gidx] = prev_g
                    self.n_delayed_releases += 1
                    if self.release_hold_steps > 0:
                        self.release_hold_countdown[arm] = self.release_hold_steps
                elif self.release_hold_countdown[arm] > 0 and opening:
                    action[gidx] = prev_g
                    self.n_delayed_releases += 1
                    self.release_hold_countdown[arm] -= 1
                elif self.release_hold_countdown[arm] > 0:
                    self.release_hold_countdown[arm] -= 1

        # 3. Rate limit: cap per-step joint delta (per-joint caps)
        if self.rate_limit and self.prev_action is not None:
            delta = action - self.prev_action
            # Per-joint cap
            delta_clipped = np.clip(delta, -self.per_joint_max_delta, self.per_joint_max_delta)
            # Count how many joints were clipped this step
            if (np.abs(delta) > self.per_joint_max_delta).any():
                self.n_rate_clips += 1
            action = self.prev_action + delta_clipped

        self.prev_action = action.copy()
        self.prev_state = state.copy()
        self.step_count += 1

        return action.astype(np.float32)

    def get_stats(self) -> dict:
        """Return counters useful for per-episode logging."""
        return {
            "step_count": int(self.step_count),
            "n_rate_clips": int(self.n_rate_clips),
            "n_delayed_releases": int(self.n_delayed_releases),
            "n_retries_left": int(self.retries_used["left"]),
            "n_retries_right": int(self.retries_used["right"]),
            "first_retry_step": int(self.first_retry_step) if self.first_retry_step is not None else -1,
        }

    def _detect_failed_grasp(self, arm: str) -> bool:
        """Detect a failed grasp: commanded close AND gripper is stable at empty-closed value."""
        if len(self.gripper_history[arm]) < STABILITY_WINDOW:
            return False
        if len(self.gripper_cmd_history[arm]) < STABILITY_WINDOW:
            return False

        # Commanded to close: all recent commands are near 0
        cmds = np.array(self.gripper_cmd_history[arm])
        if cmds.mean() > 0.1:  # not commanding closed
            return False

        # Gripper is stable: variance in window is low
        hist = np.array(self.gripper_history[arm])
        if hist.std() > STABILITY_TOLERANCE:
            return False

        # Final value below empty threshold = empty gripper
        if hist.mean() < self.gripper_empty_threshold:
            return True

        return False


def _apply_numeric_env_overrides(flag_kwargs: dict) -> None:
    """Parse STABILIZER_* env vars for numeric knobs and fold into flag_kwargs.

    Vars:
      STABILIZER_RATE_LIMIT_SCALE   - float, multiplies MAX_JOINT_DELTA_ARM
      STABILIZER_GRIPPER_RATE_SCALE - float, multiplies MAX_JOINT_DELTA_GRIPPER
      STABILIZER_VEL_THRESHOLD      - float, replaces ARM_HIGH_VELOCITY
      STABILIZER_RELEASE_HOLD_STEPS - int, hold gripper closed N extra steps after trigger
    """
    import os
    rate_scale = float(os.environ.get("STABILIZER_RATE_LIMIT_SCALE", "1.0"))
    grip_scale = float(os.environ.get("STABILIZER_GRIPPER_RATE_SCALE", "1.0"))
    flag_kwargs.setdefault("max_joint_delta_arm", MAX_JOINT_DELTA_ARM * rate_scale)
    flag_kwargs.setdefault("max_joint_delta_gripper", MAX_JOINT_DELTA_GRIPPER * grip_scale)
    flag_kwargs["_rate_scale"] = rate_scale
    flag_kwargs["_grip_scale"] = grip_scale
    vel_thr = os.environ.get("STABILIZER_VEL_THRESHOLD")
    if vel_thr is not None:
        flag_kwargs.setdefault("arm_vel_threshold", float(vel_thr))
    hold = os.environ.get("STABILIZER_RELEASE_HOLD_STEPS")
    if hold is not None:
        flag_kwargs.setdefault("release_hold_steps", int(hold))


@PolicyRegistry.register("stabilized_lerobot")
class StabilizedLeRobotPolicy(BasePolicy):
    """Convenience wrapper: loads a LeRobotPolicy and applies PolicyStabilizer.

    Flags (set via environment variables for easy CLI override):
      STABILIZER_RATE_LIMIT=1|0      default 1
      STABILIZER_DELAY_THROWS=1|0    default 1
      STABILIZER_RETRY_GRASPS=1|0    default 0 (clusters not separable)
    """

    def __init__(
        self,
        policy_path: str,
        dataset_root: str,
        task_description: str = "Fold a garment with bimanual robot arms",
        device: str = "cuda",
        **stabilizer_kwargs,
    ):
        super().__init__()
        import os
        from .lerobot_policy import LeRobotPolicy

        base = LeRobotPolicy(
            policy_path=policy_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device,
        )

        # Allow env-var overrides
        flag_kwargs = dict(stabilizer_kwargs)
        flag_kwargs.setdefault("rate_limit", os.environ.get("STABILIZER_RATE_LIMIT", "1") == "1")
        flag_kwargs.setdefault("delay_throws", os.environ.get("STABILIZER_DELAY_THROWS", "1") == "1")
        flag_kwargs.setdefault("retry_grasps", os.environ.get("STABILIZER_RETRY_GRASPS", "0") == "1")
        _apply_numeric_env_overrides(flag_kwargs)

        print(f"[StabilizedLeRobotPolicy] rate_limit={flag_kwargs['rate_limit']} "
              f"delay_throws={flag_kwargs['delay_throws']} "
              f"retry_grasps={flag_kwargs['retry_grasps']} "
              f"rate_scale={flag_kwargs.get('_rate_scale', 1.0):.2f} "
              f"grip_scale={flag_kwargs.get('_grip_scale', 1.0):.2f} "
              f"vel_thr={flag_kwargs.get('arm_vel_threshold', ARM_HIGH_VELOCITY):.3f} "
              f"hold={flag_kwargs.get('release_hold_steps', 0)}", flush=True)
        flag_kwargs.pop('_rate_scale', None)
        flag_kwargs.pop('_grip_scale', None)

        self.wrapper = PolicyStabilizer(base, **flag_kwargs)

    def reset(self):
        self.wrapper.reset()

    def select_action(self, observation):
        return self.wrapper.select_action(observation)

    def get_stats(self) -> dict:
        return self.wrapper.get_stats()


@PolicyRegistry.register("stabilized_router")
class StabilizedRouterPolicy(BasePolicy):
    """RouterPolicy wrapped with PolicyStabilizer.

    Classifies the garment from the top_rgb image at episode start, delegates
    action selection to the category-specific specialist, and applies the
    rate-limit / delay-throw / retry-grasp interventions on top.

    This is the submission-compliant path: no garment-type CLI flag is
    consulted; the classifier runs on the same public observations the
    challenge server provides.
    """

    def __init__(
        self,
        policy_path: str | None = None,
        model_path: str | None = None,
        dataset_root: str = "",
        task_description: str = "Fold a garment with bimanual robot arms",
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        **stabilizer_kwargs,
    ):
        super().__init__()
        import os
        from .router_policy import RouterPolicy

        # evaluation.py passes policy_path as model_path for non-lerobot types
        classifier_path = policy_path or model_path
        router = RouterPolicy(
            model_path=classifier_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        flag_kwargs = dict(stabilizer_kwargs)
        flag_kwargs.setdefault("rate_limit", os.environ.get("STABILIZER_RATE_LIMIT", "1") == "1")
        flag_kwargs.setdefault("delay_throws", os.environ.get("STABILIZER_DELAY_THROWS", "1") == "1")
        flag_kwargs.setdefault("retry_grasps", os.environ.get("STABILIZER_RETRY_GRASPS", "0") == "1")
        _apply_numeric_env_overrides(flag_kwargs)

        print(f"[StabilizedRouterPolicy] rate_limit={flag_kwargs['rate_limit']} "
              f"delay_throws={flag_kwargs['delay_throws']} "
              f"retry_grasps={flag_kwargs['retry_grasps']} "
              f"rate_scale={flag_kwargs.get('_rate_scale', 1.0):.2f} "
              f"grip_scale={flag_kwargs.get('_grip_scale', 1.0):.2f} "
              f"vel_thr={flag_kwargs.get('arm_vel_threshold', ARM_HIGH_VELOCITY):.3f} "
              f"hold={flag_kwargs.get('release_hold_steps', 0)}", flush=True)
        flag_kwargs.pop('_rate_scale', None)
        flag_kwargs.pop('_grip_scale', None)

        self.wrapper = PolicyStabilizer(router, **flag_kwargs)

    def reset(self):
        self.wrapper.reset()

    def select_action(self, observation):
        return self.wrapper.select_action(observation)

    def get_stats(self) -> dict:
        return self.wrapper.get_stats()
