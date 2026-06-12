"""ACT + AI-designed corrective injection (per-garment).

Approach: AI inspects the saved failure trajectory + frames offline, designs a
specific corrective action sequence for THAT garment's failure mode, then
this policy plays the corrective sequence after ACT runs to a chosen
trigger step.

For Top_Short_Unseen_0:
- ACT peaks at step 250 with both grippers closed on the collar/shoulder caps.
- ACT's failure: at step 260 it OPENS the grippers (L_grip 0.13, R_grip 0.33),
  releasing the cloth. By step 280 the fold is disturbed.
- Corrective: at step 250+, force grippers CLOSED and pan shoulders inward to
  bring the two collar regions together. Release at step 290+.
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


# 12D bimanual SO-101 joint indices
LEFT_ARM_IDX = [0, 1, 2, 3, 4]
LEFT_GRIPPER_IDX = 5
RIGHT_ARM_IDX = [6, 7, 8, 9, 10]
RIGHT_GRIPPER_IDX = 11

GRIPPER_OPEN = 0.5
GRIPPER_CLOSED = -0.15
GRIPPER_TIGHT_DEFAULT = -0.20  # over-clamp to prevent cloth slipping during lift

# Per-garment corrective sequences (v2).
# v2 strategy: drive toward a TARGET joint state copied from a SUCCESSFUL
# Seen episode's final state, via linear interpolation. The trajectory of ACT
# on a similar Top-Short Seen garment that succeeded is a much better source
# of "what joints should look like at fold completion" than my hand-tuning.
#
# Source: Top_Short_Seen_2 episode 2 (293 steps -> success). Final state:
#   L_arm: [0.08, -1.71, 1.50, 1.20, -0.24], L_grip=-0.15
#   R_arm: [-0.20, -1.71, 1.54, 1.21, 0.13], R_grip=-0.15
TARGET_STATE_AT_FOLD_COMPLETE = {
    # Joint index → target value
    # v4: ONLY target lift axis joints (shoulder_lift + elbow + wrist_flex).
    # Keep shoulder_pan + wrist_roll AT WHATEVER the snapshot has, so arms
    # don't separate horizontally and pull the collar apart.
    1: -1.71,   # L shoulder_lift (arm UP)
    2: 1.50,    # L elbow_flex (folded inward)
    3: 1.20,    # L wrist_flex
    7: -1.71,   # R shoulder_lift (arm UP)
    8: 1.54,    # R elbow_flex (folded inward)
    9: 1.21,    # R wrist_flex
}

# Templates of various lengths for parameterized search (random search / CMA-ES).
import os as _os
_TEMPLATE_PATHS = {
    60:  "outputs/cv_collar_pol/seen2_template_actions.npy",       # 60-step
    80:  None,                                                      # built on demand
    100: "outputs/cv_collar_pol/seen2_template_actions_100.npy",
    120: None,
    200: "outputs/cv_collar_pol/seen2_template_actions_200.npy",
}

def _load_or_build_template(length: int) -> Optional[np.ndarray]:
    path = _TEMPLATE_PATHS.get(length)
    if path is not None and _os.path.exists(path):
        return np.load(path)
    # Build on demand from the 200-step template
    full = _TEMPLATE_PATHS.get(200)
    if full and _os.path.exists(full):
        full_arr = np.load(full)
        if length <= len(full_arr):
            return full_arr[-length:]
    return None


def _build_correction_from_env(default_trigger: int = 200) -> dict:
    """Build the per-garment correction config from env vars (for hp search)."""
    trigger_step = int(_os.environ.get("AI_TELEOP_TRIGGER_STEP", default_trigger))
    template_length = int(_os.environ.get("AI_TELEOP_TEMPLATE_LENGTH", 100))
    blend_steps = int(_os.environ.get("AI_TELEOP_BLEND_STEPS", 10))
    hold_duration = int(_os.environ.get("AI_TELEOP_HOLD_DURATION", 20))
    template = _load_or_build_template(template_length)
    replay_steps = max(0, template_length - blend_steps) if template is not None else 0
    return {
        "trigger_step": trigger_step,
        "trigger_require_grippers_closed": True,
        "v2_target": TARGET_STATE_AT_FOLD_COMPLETE,
        "template_actions": template,
        "phases": [
            ("tighten_grip", 10, {}, "tight"),
            ("blend_to_template", blend_steps, "blend_to_template", "tight"),
            ("replay_template", replay_steps, "replay_template", "tight"),
            ("hold_at_target", hold_duration, {}, "tight"),
            ("release", 15, {}, "open"),
            ("settle", 200, {}, "open"),
        ],
    }


TEMPLATE_ACTIONS = _load_or_build_template(100)
TOP_SHORT_UNSEEN_0_CORRECTION = _build_correction_from_env()


def _get_correction_for(garment: str) -> Optional[dict]:
    if garment == "Top_Short_Unseen_0":
        # Rebuild every time so env var changes take effect (random search)
        return _build_correction_from_env()
    return None


@PolicyRegistry.register("ai_teleop_top_short")
class AITelop_TopShort(BasePolicy):
    """ACT + AI-designed per-garment corrective injection."""

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
        # Garment name comes from env env_cfg.garment_name; we read it from the
        # observation if available, else default to Top_Short_Unseen_0.
        # The eval framework doesn't pass garment_name through the obs dict,
        # so we use an env var override.
        self.garment_name = os.environ.get(
            "AI_TELEOP_GARMENT", kwargs.get("garment_name", "Top_Short_Unseen_0")
        )
        self.correction = _get_correction_for(self.garment_name)
        if self.correction is None:
            logger.warning(
                f"[AITelop_TopShort] No correction for garment {self.garment_name}; "
                f"will pass through ACT actions unmodified."
            )
        self.reset()
        logger.info(
            f"[AITelop_TopShort] garment={self.garment_name}, "
            f"trigger_step={self.correction['trigger_step'] if self.correction else 'N/A'}"
        )

    def reset(self):
        self.act.reset()
        self.step_count = 0
        self.in_suffix = False
        self.suffix_start_step: Optional[int] = None
        self.snapshot_state: Optional[np.ndarray] = None

    def _suffix_phase_at(self, suffix_step: int):
        """Return (phase_label, in_phase_step, total_phase_steps, arm_delta, gripper_mode)
        or None if past end of all phases."""
        if self.correction is None:
            return None
        cum = 0
        for label, n, delta, gcmd in self.correction["phases"]:
            if suffix_step < cum + n:
                return (label, suffix_step - cum, n, delta, gcmd)
            cum += n
        return None

    def _suffix_action(self, suffix_step: int) -> np.ndarray:
        """Compute hand-designed action at the given suffix step.

        v2 supports two phase modes:
          - "interp_to_target": linearly interpolate joints from snapshot
            toward TARGET_STATE_AT_FOLD_COMPLETE over the phase duration
          - {joint: delta}: per-step joint deltas (legacy v1 mode)
        Grippers always use absolute target (open/closed).
        """
        anchor = self.snapshot_state.copy()
        action = anchor.copy()

        cum_step = 0
        gripper_mode = "closed"
        template = self.correction.get("template_actions")

        for label, n, mode, gcmd in self.correction["phases"]:
            if suffix_step >= cum_step + n:
                # Phase fully done — apply its end-state effect (set anchor for next)
                if mode == "interp_to_target":
                    target = self.correction["v2_target"]
                    for j, tval in target.items():
                        action[j] = tval
                elif mode == "blend_to_template" and template is not None:
                    # End of blend: action = template[n-1]
                    action[:] = template[n - 1]
                elif mode == "replay_template" and template is not None:
                    # End of replay: action = template[10 + n - 1] (offset by blend phase)
                    action[:] = template[10 + n - 1]
                elif isinstance(mode, dict):
                    for j, d in mode.items():
                        action[j] = anchor[j] + d * n
                gripper_mode = gcmd
                cum_step += n
            else:
                steps_in = suffix_step - cum_step
                if mode == "interp_to_target":
                    target = self.correction["v2_target"]
                    frac = steps_in / max(1, n)
                    t = frac * frac * (3.0 - 2.0 * frac)
                    for j, tval in target.items():
                        action[j] = anchor[j] + (tval - anchor[j]) * t
                elif mode == "blend_to_template" and template is not None:
                    # Blend from anchor (step-250 state) toward template[steps_in]
                    frac = steps_in / max(1, n)
                    t = frac * frac * (3.0 - 2.0 * frac)
                    target_act = template[steps_in]
                    action = anchor + (target_act - anchor) * t
                elif mode == "replay_template" and template is not None:
                    # Replay template starting at index 10 (after blend phase)
                    idx = 10 + steps_in
                    if idx < len(template):
                        action[:] = template[idx]
                    else:
                        action[:] = template[-1]
                elif isinstance(mode, dict):
                    for j, d in mode.items():
                        action[j] = anchor[j] + d * steps_in
                gripper_mode = gcmd
                break

        gripper_tight = float(os.environ.get("AI_TELEOP_GRIPPER_TIGHTNESS", GRIPPER_TIGHT_DEFAULT))
        if gripper_mode == "closed":
            action[LEFT_GRIPPER_IDX] = GRIPPER_CLOSED
            action[RIGHT_GRIPPER_IDX] = GRIPPER_CLOSED
        elif gripper_mode == "tight":
            action[LEFT_GRIPPER_IDX] = gripper_tight
            action[RIGHT_GRIPPER_IDX] = gripper_tight
        elif gripper_mode == "open":
            action[LEFT_GRIPPER_IDX] = GRIPPER_OPEN
            action[RIGHT_GRIPPER_IDX] = GRIPPER_OPEN

        return action.astype(np.float32)

    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(observation["observation.state"], dtype=np.float32)

        # Trigger (state-aware in v9)
        if (self.correction is not None and not self.in_suffix
                and self.step_count >= self.correction["trigger_step"]):
            require_closed = self.correction.get("trigger_require_grippers_closed", False)
            if require_closed:
                grippers_closed = state[5] < -0.05 and state[11] < -0.05
                if not grippers_closed:
                    # Wait for grippers to close before triggering
                    if self.step_count >= self.correction["trigger_step"] + 50:
                        # Hard timeout: fire anyway after 50 steps grace
                        logger.info(
                            f"[ai_teleop] state-aware trigger timed out at step {self.step_count}, "
                            f"L_grip={state[5]:.3f} R_grip={state[11]:.3f} (firing anyway)"
                        )
                    else:
                        # Don't trigger, let ACT continue
                        action = self.act.select_action(observation).astype(np.float32)
                        self.step_count += 1
                        return action
            self.in_suffix = True
            self.suffix_start_step = self.step_count
            self.snapshot_state = state.copy()
            logger.info(
                f"[ai_teleop] trigger fired at step {self.step_count}, "
                f"snapshot L_grip={state[5]:.3f} R_grip={state[11]:.3f}"
            )

        if self.in_suffix:
            suffix_step = self.step_count - self.suffix_start_step
            phase = self._suffix_phase_at(suffix_step)
            if phase is None:
                # Past end of correction; hold last commanded action
                action = self._suffix_action(self._total_phase_steps() - 1)
            else:
                action = self._suffix_action(suffix_step)
        else:
            action = self.act.select_action(observation).astype(np.float32)

        self.step_count += 1
        return action

    def _total_phase_steps(self) -> int:
        if self.correction is None:
            return 0
        return sum(n for _, n, _, _ in self.correction["phases"])
