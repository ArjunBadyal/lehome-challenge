"""Scripted check-point oracle for fold demo recording (training-time only).

At episode start, snapshot the cloth's `check_point` world positions from sim,
then drive a fixed bimanual phase machine via IK to fold source-points to
their destination locations:

  * top-short-sleeve : hem (4,5) folded up onto collar (0,1)
  * long-pant         : waist (0,1) folded down onto knee (4,5)

Success is evaluated by the standard challenge checker after a settle phase;
successful episodes are saved as LeRobot demos, failures are discarded and
re-attempted.

Compliance: check_points are read only at demo-recording time. The trained
policy at evaluation time still observes only state + RGB + depth.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# This venv has a shadow `pinocchio` stub (a coverage tool) at the venv root
# that masks the real Pinocchio kinematics library (installed under cmeel).
# Fix: make sure the real cmeel-bundled pinocchio resolves first, and evict
# any already-cached stub. Must happen BEFORE we touch RobotKinematics.
def _ensure_real_pinocchio() -> None:
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cmeel_pin = os.path.join(
        sys.prefix, "lib", py_ver, "site-packages",
        "cmeel.prefix", "lib", py_ver, "site-packages",
    )
    if not os.path.isdir(os.path.join(cmeel_pin, "pinocchio")):
        return
    # Force cmeel.prefix to position 0 even if already present further down
    # (the IsaacLab launcher inserts it after the venv root, where the stub
    # `pinocchio` package shadows the real one).
    while cmeel_pin in sys.path:
        sys.path.remove(cmeel_pin)
    sys.path.insert(0, cmeel_pin)
    for mod in list(sys.modules.keys()):
        if mod == "pinocchio" or mod.startswith("pinocchio."):
            del sys.modules[mod]
    try:
        import pinocchio as pin  # noqa: F401
    except Exception:
        pass


_ensure_real_pinocchio()

import numpy as np

from lehome.utils.logger import get_logger
from lehome.utils.success_checker_chanllege import (
    get_object_particle_position,
    check_top_sleeve,
    check_pant_long,
)

logger = get_logger("oracle_fold")

# Gripper joint commands (rad). Match dataset_record.py's GRIPPER_CLOSED.
GRIPPER_OPEN = 0.6
GRIPPER_CLOSED = -0.18

LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11

# Phase plan: list of (name, end_step). Each phase ends at end_step (exclusive).
# HOME swings both arms from their outward rest pose to a forward-facing
# neutral pose so IK has a good warm start for the cloth target.
_PHASES: List[Tuple[str, int]] = [
    ("HOME",             60),
    ("APPROACH",        120),
    ("DESCEND",         170),
    ("CLOSE",           210),
    ("PINCH",           240),
    ("LIFT",            280),
    ("FOLD",            370),
    ("DESCEND_RELEASE", 400),
    ("OPEN",            430),
    ("RETRACT",         470),
    ("SETTLE",          570),
]
TOTAL_STEPS = _PHASES[-1][1]


def _category_from_garment_type(garment_type: str) -> str:
    if garment_type in ("top-short-sleeve", "top-long-sleeve"):
        return "top_short"  # same fold pattern works for both top variants
    if garment_type == "long-pant":
        return "pant_long"
    if garment_type == "short-pant":
        return "pant_short"
    raise ValueError(f"oracle does not support garment_type={garment_type}")


class OracleFolder:
    """Scripted bimanual fold via cloth-state ground truth.

    Use:
      folder = OracleFolder(env, args)            # at policy-init time
      folder.reset()                              # after each env.reset()
      action = folder.next_action(obs, flags)     # each sim step
      # folder will set flags['success'] or flags['remove'] when its plan
      # has fully executed.
    """

    def __init__(self, env, args, category: Optional[str] = None):
        self.env = env
        self.args = args
        from lehome.utils import RobotKinematics
        self.solver = RobotKinematics(
            getattr(args, "ee_urdf_path", None) or "Assets/robots/so101_new_calib.urdf",
            target_frame_name="gripper_frame_link",
            joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
        )
        if category is None:
            try:
                garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
                category = _category_from_garment_type(garment_type)
            except Exception as e:
                raise ValueError(f"Could not auto-detect oracle category: {e}")
        if category not in ("top_short", "pant_long"):
            raise ValueError(
                f"OracleFolder supports top_short / pant_long; got {category}"
            )
        self.category = category
        # Source and destination check_point indices for the fold:
        if category == "pant_long":
            # Fold WAIST (0,1) DOWN onto KNEE (4,5)
            self.src_idx = (0, 1)
            self.dst_idx = (4, 5)
        else:  # top_short
            # Fold HEM (4,5) UP onto COLLAR (0,1)
            self.src_idx = (4, 5)
            self.dst_idx = (0, 1)
        self.checker = check_top_sleeve if category == "top_short" else check_pant_long
        self.reset()

    def reset(self) -> None:
        self.step_idx = 0
        self.checkpoint_snapshot: Optional[np.ndarray] = None  # (N, 3) meters
        self.episode_done = False
        self.success_flag: Optional[bool] = None
        self.current_phase_name: Optional[str] = None
        self.phase_start_step = 0
        self.phase_start_state: Optional[np.ndarray] = None
        self.phase_target_state: Optional[np.ndarray] = None
        self.phase_grip_mode: str = "OPEN"
        self.last_logged_phase: Optional[str] = None

    def _snapshot_checkpoints(self) -> bool:
        try:
            obj = self.env.object
            check_points = list(obj.check_points)
            positions_cm = get_object_particle_position(obj, check_points)
            if positions_cm is None:
                return False
            positions_m = np.asarray(positions_cm, dtype=np.float32) / 100.0
            self.checkpoint_snapshot = positions_m
            return True
        except Exception as e:
            logger.warning(f"[Oracle] check_point snapshot failed: {e}")
            return False

    def _current_phase(self) -> Tuple[str, int, int]:
        prev_end = 0
        for name, end_step in _PHASES:
            if self.step_idx < end_step:
                return name, prev_end, end_step
            prev_end = end_step
        return _PHASES[-1][0], prev_end, TOTAL_STEPS

    def _ik_pair(
        self,
        L_world: np.ndarray,
        R_world: np.ndarray,
        start_state: np.ndarray,
        grip_mode: str,
    ) -> np.ndarray:
        from lehome.utils import compute_joints_from_world_point
        target = start_state.copy().astype(np.float32)
        grip_angle = GRIPPER_CLOSED if grip_mode == "CLOSED" else GRIPPER_OPEN
        # Use a forward-facing neutral warm start when the current pose is
        # the rest pose (shoulder_pan ≈ ±1.14 rad pointing outward), which
        # otherwise pins IK in a local minimum.
        def _warm(joints6: np.ndarray) -> np.ndarray:
            warm = joints6.copy()
            if abs(float(warm[0])) > 0.8:
                warm[0:5] = 0.0
            return warm
        try:
            jl = compute_joints_from_world_point(
                self.solver, self.env, "left", L_world.astype(np.float32),
                current_joints=_warm(start_state[0:6]),
                state_unit="rad", gripper_angle=float(grip_angle),
            )
            if jl is not None:
                target[0:6] = np.asarray(jl, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[Oracle] IK left failed: {e}")
        try:
            jr = compute_joints_from_world_point(
                self.solver, self.env, "right", R_world.astype(np.float32),
                current_joints=_warm(start_state[6:12]),
                state_unit="rad", gripper_angle=float(grip_angle),
            )
            if jr is not None:
                target[6:12] = np.asarray(jr, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[Oracle] IK right failed: {e}")
        return target

    def _phase_targets(
        self, phase_name: str, start_state: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """Return (target 12D rad, gripper mode in {'OPEN', 'CLOSED', 'KEEP'})."""
        snap = self.checkpoint_snapshot
        L_src = snap[self.src_idx[0]]
        R_src = snap[self.src_idx[1]]
        L_dst = snap[self.dst_idx[0]]
        R_dst = snap[self.dst_idx[1]]

        # IK targets the `gripper_frame_link` (wrist origin), which sits ~5 cm
        # ABOVE the actual jaw tips. Click-IK adds the same +5 cm to land
        # the jaws AT the clicked depth. Use the same offset here so jaw tips
        # actually reach the cloth surface at descend time.
        FRAME_OFFSET = float(getattr(self.args, "click_ik_z_offset", 0.05))
        Z_APPROACH   = FRAME_OFFSET + 0.08
        Z_DESCEND    = FRAME_OFFSET - 0.005  # jaws ~5 mm below surface (around cloth)
        Z_LIFT       = FRAME_OFFSET + 0.06
        Z_RELEASE    = FRAME_OFFSET + 0.03
        Z_RETRACT    = FRAME_OFFSET + 0.18

        if phase_name == "HOME":
            # Forward-facing neutral pose; gives IK a sane warm start for the
            # cloth target (rest-pose shoulder_pan = ±1.14 rad sends IK into
            # a local minimum that doesn't point the arm at the table).
            target = start_state.copy().astype(np.float32)
            target[0:5]  = 0.0   # left arm: 5 joints to neutral
            target[6:11] = 0.0   # right arm: 5 joints to neutral
            return target, "OPEN"
        if phase_name == "APPROACH":
            L = L_src + np.array([0, 0, Z_APPROACH])
            R = R_src + np.array([0, 0, Z_APPROACH])
            return self._ik_pair(L, R, start_state, "OPEN"), "OPEN"
        if phase_name == "DESCEND":
            L = L_src + np.array([0, 0, Z_DESCEND])
            R = R_src + np.array([0, 0, Z_DESCEND])
            return self._ik_pair(L, R, start_state, "OPEN"), "OPEN"
        if phase_name == "CLOSE":
            # Hold position; gripper closes
            return start_state.copy(), "CLOSED"
        if phase_name == "PINCH":
            # Stay at grasp height with closed grippers — let cloth pinch settle
            return start_state.copy(), "CLOSED"
        if phase_name == "LIFT":
            L = L_src + np.array([0, 0, Z_LIFT])
            R = R_src + np.array([0, 0, Z_LIFT])
            return self._ik_pair(L, R, start_state, "CLOSED"), "CLOSED"
        if phase_name == "FOLD":
            L = np.array([L_dst[0], L_dst[1], L_src[2] + Z_LIFT], dtype=np.float32)
            R = np.array([R_dst[0], R_dst[1], R_src[2] + Z_LIFT], dtype=np.float32)
            return self._ik_pair(L, R, start_state, "CLOSED"), "CLOSED"
        if phase_name == "DESCEND_RELEASE":
            L = np.array([L_dst[0], L_dst[1], L_src[2] + Z_RELEASE], dtype=np.float32)
            R = np.array([R_dst[0], R_dst[1], R_src[2] + Z_RELEASE], dtype=np.float32)
            return self._ik_pair(L, R, start_state, "CLOSED"), "CLOSED"
        if phase_name == "OPEN":
            return start_state.copy(), "OPEN"
        if phase_name == "RETRACT":
            L = np.array([L_dst[0], L_dst[1], L_src[2] + Z_RETRACT], dtype=np.float32)
            R = np.array([R_dst[0], R_dst[1], R_src[2] + Z_RETRACT], dtype=np.float32)
            return self._ik_pair(L, R, start_state, "OPEN"), "OPEN"
        # SETTLE or unknown: hold position, grippers open
        return start_state.copy(), "OPEN"

    def next_action(
        self, observations: Dict[str, Any], flags: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        if self.episode_done:
            # Don't issue more actions after we've signaled success/remove.
            return None

        state = np.asarray(observations.get("observation.state"), dtype=np.float32)
        if state.shape[0] < 12:
            return None

        # First call: snapshot check_points
        if self.checkpoint_snapshot is None:
            if not self._snapshot_checkpoints():
                flags["remove"] = True
                self.episode_done = True
                logger.warning("[Oracle] could not read check_points; discarding episode")
                return state.copy()
            logger.info(
                f"[Oracle] category={self.category} src={self.src_idx} dst={self.dst_idx} "
                f"L_src={self.checkpoint_snapshot[self.src_idx[0]].tolist()} "
                f"R_src={self.checkpoint_snapshot[self.src_idx[1]].tolist()} "
                f"L_dst={self.checkpoint_snapshot[self.dst_idx[0]].tolist()} "
                f"R_dst={self.checkpoint_snapshot[self.dst_idx[1]].tolist()}"
            )

        phase_name, _, phase_end = self._current_phase()

        # On phase change, recompute target from current state and snapshot
        if phase_name != self.current_phase_name:
            self.current_phase_name = phase_name
            self.phase_start_step = self.step_idx
            self.phase_start_state = state.copy()
            self.phase_target_state, self.phase_grip_mode = self._phase_targets(
                phase_name, state
            )
            if phase_name != self.last_logged_phase:
                # Read actual gripper world positions for diagnostic
                try:
                    L_now = self.env.left_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
                    R_now = self.env.right_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
                    logger.info(
                        f"[Oracle] -> {phase_name} step={self.step_idx} "
                        f"end={phase_end} grip={self.phase_grip_mode} "
                        f"L_now=({L_now[0]:.3f},{L_now[1]:.3f},{L_now[2]:.3f}) "
                        f"R_now=({R_now[0]:.3f},{R_now[1]:.3f},{R_now[2]:.3f})"
                    )
                except Exception:
                    logger.info(
                        f"[Oracle] -> {phase_name} step={self.step_idx} "
                        f"end={phase_end} grip={self.phase_grip_mode}"
                    )
                self.last_logged_phase = phase_name

        # Linear interp in joint space within phase
        phase_len = max(1, phase_end - self.phase_start_step)
        local_step = self.step_idx - self.phase_start_step + 1
        alpha = float(min(1.0, local_step / phase_len))
        action = self.phase_start_state + alpha * (
            self.phase_target_state - self.phase_start_state
        )

        # Override grippers per phase
        if self.phase_grip_mode == "OPEN":
            action[LEFT_GRIPPER_IDX] = GRIPPER_OPEN
            action[RIGHT_GRIPPER_IDX] = GRIPPER_OPEN
        elif self.phase_grip_mode == "CLOSED":
            action[LEFT_GRIPPER_IDX] = GRIPPER_CLOSED
            action[RIGHT_GRIPPER_IDX] = GRIPPER_CLOSED

        self.step_idx += 1

        # End-of-plan: evaluate success and signal save/discard
        if self.step_idx >= TOTAL_STEPS and not self.episode_done:
            self.episode_done = True
            self.success_flag = self._evaluate_success()
            if self.success_flag:
                flags["success"] = True
                logger.info("[Oracle] EPISODE SUCCESS - will save")
            else:
                flags["remove"] = True
                logger.info("[Oracle] EPISODE FAIL - will discard")

        return action.astype(np.float32)

    def _evaluate_success(self) -> bool:
        try:
            obj = self.env.object
            check_points = list(obj.check_points)
            current_scale = float(obj.init_scale[0])
            success_distance = [d * current_scale for d in obj.success_distance]
            p = get_object_particle_position(obj, check_points)
            if p is None:
                return False
            ok, details = self.checker(p, success_distance)
            for key, info in details.items():
                status = "OK" if info.get("passed") else "FAIL"
                logger.info(f"[Oracle eval] {info.get('description', key)} -> {status}")
            return bool(ok)
        except Exception as e:
            logger.warning(f"[Oracle] success eval failed: {e}")
            return False
