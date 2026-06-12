"""Dataset recording utility functions for teleoperation data collection."""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union, List
import gymnasium as gym
import numpy as np
import torch

from isaacsim.simulation_app import SimulationApp
from isaaclab.envs import DirectRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lehome.devices import (
    Se3Keyboard,
    SO101Leader,
    BiSO101Leader,
    BiKeyboard,
)
from lehome.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from lehome.utils.record import (
    get_next_experiment_path_with_gap,
    append_episode_initial_pose,
)
from lehome.utils.logger import get_logger

from .common import stabilize_garment_after_reset

logger = get_logger(__name__)

LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11
GRIPPER_CLOSED = -0.18  # near USD lower joint limit (-10 deg = -0.1745 rad)
GRIPPER_OPEN = 0.50


def ensure_real_pinocchio() -> None:
    """Prefer the cmeel-bundled Pinocchio over the venv root stub package."""
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cmeel_pin = os.path.join(
        sys.prefix,
        "lib",
        py_ver,
        "site-packages",
        "cmeel.prefix",
        "lib",
        py_ver,
        "site-packages",
    )
    if not os.path.isdir(os.path.join(cmeel_pin, "pinocchio")):
        return
    while cmeel_pin in sys.path:
        sys.path.remove(cmeel_pin)
    sys.path.insert(0, cmeel_pin)
    for mod in list(sys.modules.keys()):
        if mod == "pinocchio" or mod.startswith("pinocchio."):
            del sys.modules[mod]


def _success_progress_metrics(result: Any) -> Dict[str, float]:
    """Extract lightweight progress metrics from the official success checker."""
    metrics = {
        "passed_count": 0.0,
        "total_count": 0.0,
        "close_passed_count": 0.0,
        "close_total_count": 0.0,
        "best_close_ratio": float("inf"),
        "worst_close_ratio": float("inf"),
    }
    if not isinstance(result, dict):
        return metrics
    details = result.get("details", {}) or {}
    metrics["total_count"] = float(len(details))
    for condition in details.values():
        if bool(condition.get("passed", False)):
            metrics["passed_count"] += 1.0
        description = str(condition.get("description", ""))
        if "<=" not in description:
            continue
        metrics["close_total_count"] += 1.0
        if bool(condition.get("passed", False)):
            metrics["close_passed_count"] += 1.0
        threshold = float(condition.get("threshold", 0.0) or 0.0)
        value = float(condition.get("value", float("inf")))
        if threshold > 1e-6 and np.isfinite(value):
            ratio = value / threshold
            metrics["best_close_ratio"] = min(
                metrics["best_close_ratio"],
                ratio,
            )
            metrics["worst_close_ratio"] = (
                ratio
                if not np.isfinite(metrics["worst_close_ratio"])
                else max(metrics["worst_close_ratio"], ratio)
            )
    return metrics


def _parse_early_restart_schedule(value: str) -> list[dict[str, float]]:
    """Parse staged early-restart rules.

    Format: "step:min_passed:best_close:max_close_passed", comma-separated.
    Empty fields mean "do not check"; examples:
      160:2:2.8:0   -> restart if passed<2 OR best_close>2.8
      220:3::1      -> restart if passed<3 OR close_passed<=1
    """
    rules: list[dict[str, float]] = []
    for raw_part in str(value or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = part.split(":")
        if len(fields) > 4:
            logger.warning("[EarlyRestart] invalid schedule rule %r", part)
            continue
        fields += [""] * (4 - len(fields))
        try:
            rule = {
                "step": float(int(fields[0])),
                "min_passed": float(int(fields[1])) if fields[1] else -1.0,
                "best_close_ratio": float(fields[2]) if fields[2] else float("inf"),
                "min_close_passed": float(int(fields[3])) if fields[3] else -1.0,
            }
        except ValueError:
            logger.warning("[EarlyRestart] invalid schedule rule %r", part)
            continue
        rules.append(rule)
    return sorted(rules, key=lambda r: r["step"])


def _format_success_details(result: Any, max_items: int = 5) -> str:
    """Compact one-line summary of official success-check distances."""
    if not isinstance(result, dict):
        return "no success details"
    parts = []
    for condition in (result.get("details", {}) or {}).values():
        description = str(condition.get("description", ""))
        value = float(condition.get("value", float("nan")))
        threshold = float(condition.get("threshold", float("nan")))
        passed = bool(condition.get("passed", False))
        symbol = "ok" if passed else "fail"
        if description:
            parts.append(f"{description}: {value:.2f}/{threshold:.2f} {symbol}")
        else:
            parts.append(f"{value:.2f}/{threshold:.2f} {symbol}")
        if len(parts) >= max_items:
            break
    return "; ".join(parts) if parts else "no success details"


def _success_details_payload(result: Any) -> Dict[str, Any]:
    """JSON-serializable snapshot of raw success checker state."""
    payload: Dict[str, Any] = {
        "success": bool(result.get("success", False)) if isinstance(result, dict) else False,
        "garment_type": result.get("garment_type", "unknown") if isinstance(result, dict) else "unknown",
        "thresholds": result.get("thresholds", []) if isinstance(result, dict) else [],
        "points": result.get("points", []) if isinstance(result, dict) else [],
        "conditions": {},
    }
    if isinstance(result, dict):
        for name, condition in (result.get("details", {}) or {}).items():
            payload["conditions"][name] = {
                "description": str(condition.get("description", "")),
                "value": float(condition.get("value", float("nan"))),
                "threshold": float(condition.get("threshold", float("nan"))),
                "passed": bool(condition.get("passed", False)),
            }
    return payload


def _write_score_probe(
    args: argparse.Namespace,
    *,
    episode_index: int,
    attempt_index: int,
    episode_step: int,
    outcome: str,
    result: Dict[str, Any],
) -> None:
    """Append a raw cloth-score probe to JSONL for success/failure comparison."""
    path = str(getattr(args, "score_probe_log", "") or "")
    if not path:
        return
    metrics = _success_progress_metrics(result)
    record = {
        "time": time.time(),
        "garment_name": getattr(args, "garment_name", None),
        "episode_index": int(episode_index),
        "attempt_index": int(attempt_index),
        "episode_step": int(episode_step),
        "outcome": outcome,
        "passed_count": int(metrics["passed_count"]),
        "total_count": int(metrics["total_count"]),
        "close_passed_count": int(metrics["close_passed_count"]),
        "close_total_count": int(metrics["close_total_count"]),
        "best_close_ratio": float(metrics["best_close_ratio"]),
        "worst_close_ratio": float(metrics["worst_close_ratio"]),
        **_success_details_payload(result),
    }
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning("[ScoreProbe] failed writing %s: %s", path, e)


def _parse_probe_steps(value: str) -> set[int]:
    steps: set[int] = set()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            steps.add(int(part))
        except ValueError:
            logger.warning("[ScoreProbe] ignoring invalid probe step %r", part)
    return steps


def _raw_garment_success_result(env: DirectRLEnv) -> Dict[str, Any]:
    """Run the garment success calculation without the step-interval wrapper.

    `env._get_success()` returns only a tensor boolean, which is right for the
    environment API but not enough for early-restart decisions. This helper
    mirrors the official checker and returns the condition details.
    """
    if getattr(env, "object", None) is None or not hasattr(env.object, "_cloth_prim_view"):
        return {
            "success": False,
            "garment_type": "unknown",
            "thresholds": [],
            "points": [],
            "details": {},
        }
    try:
        from lehome.utils.success_checker_chanllege import (
            get_object_particle_position,
            check_top_sleeve,
            check_pant_short,
            check_pant_long,
        )

        garment_type = env.garment_loader.get_garment_type(env.cfg.garment_name)
        check_point_indices = env.object.check_points
        raw_success_distance = env.object.success_distance
        current_scale = float(env.object.init_scale[0])
        success_distance = [d * current_scale for d in raw_success_distance]
        points = get_object_particle_position(env.object, check_point_indices)
        if points is None:
            return {
                "success": False,
                "garment_type": garment_type,
                "thresholds": success_distance,
                "points": [],
                "details": {},
            }
        if garment_type in {"top-long-sleeve", "top-short-sleeve"}:
            success, details = check_top_sleeve(points, success_distance)
        elif garment_type == "short-pant":
            success, details = check_pant_short(points, success_distance)
        elif garment_type == "long-pant":
            success, details = check_pant_long(points, success_distance)
        else:
            success, details = False, {}
        return {
            "success": bool(success),
            "garment_type": garment_type,
            "thresholds": success_distance,
            "points": points,
            "details": details,
        }
    except Exception as e:
        logger.warning("[EarlyRestart] raw success detail check failed: %s", e)
        return {
            "success": False,
            "garment_type": "unknown",
            "thresholds": [],
            "points": [],
            "details": {},
        }


def _gripper_release_gate(
    env: DirectRLEnv,
    observations: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    """Check that an auto-harvest success includes release/retract.

    The official task success checker only evaluates cloth geometry. During
    demo harvesting that can save a rollout while the robot is still holding the
    garment. This optional gate requires open grippers and, if configured, a
    minimum gripper-to-cloth clearance before auto-save.
    """
    if not bool(getattr(args, "auto_save_require_release", False)):
        return True, "release gate disabled"

    state = observations.get("observation.state")
    if state is None:
        return False, "missing observation.state"
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    if state_np.shape[0] <= RIGHT_GRIPPER_IDX:
        return False, f"state too short for grippers: {state_np.shape[0]}"

    open_threshold = float(getattr(args, "auto_save_min_gripper_open", 0.20))
    left_grip = float(state_np[LEFT_GRIPPER_IDX])
    right_grip = float(state_np[RIGHT_GRIPPER_IDX])
    if left_grip < open_threshold or right_grip < open_threshold:
        return (
            False,
            f"grippers not open enough L={left_grip:.3f} R={right_grip:.3f} "
            f"threshold={open_threshold:.3f}",
        )

    clearance = float(getattr(args, "auto_save_min_gripper_cloth_distance", 0.0))
    if clearance <= 0:
        return (
            True,
            f"grippers open L={left_grip:.3f} R={right_grip:.3f}; clearance disabled",
        )

    try:
        cloth_points, _, _, _ = env.object.get_current_mesh_points()
        cloth_np = np.asarray(cloth_points, dtype=np.float32)
        left_pos = (
            env.left_arm.data.body_link_pos_w[0, -1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        right_pos = (
            env.right_arm.data.body_link_pos_w[0, -1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        left_min = float(np.linalg.norm(cloth_np - left_pos[None, :], axis=1).min())
        right_min = float(np.linalg.norm(cloth_np - right_pos[None, :], axis=1).min())
    except Exception as e:
        return False, f"clearance check failed: {e}"

    if left_min < clearance or right_min < clearance:
        return (
            False,
            f"grippers too close to cloth L={left_min:.3f}m R={right_min:.3f}m "
            f"threshold={clearance:.3f}m",
        )
    return (
        True,
        f"released and clear L_grip={left_grip:.3f} R_grip={right_grip:.3f} "
        f"L_dist={left_min:.3f}m R_dist={right_min:.3f}m",
    )


def validate_task_and_device(args: argparse.Namespace) -> None:
    """Validate that task name matches the teleop device configuration.

    Args:
        args: Command-line arguments containing task and teleop_device.

    Raises:
        ValueError: If task is not specified.
        AssertionError: If task and device configuration mismatch.
    """
    if args.task is None:
        raise ValueError("Please specify --task.")
    if "Bi" in args.task:
        assert (
            args.teleop_device == "bi-so101leader"
            or args.teleop_device == "bi-keyboard"
        ), "Only support bi-so101leader or bi-keyboard for bi-arm task"
    else:
        assert (
            args.teleop_device == "so101leader" or args.teleop_device == "keyboard"
        ), "Only support so101leader or keyboard for single-arm task"


def create_teleop_interface(
    env: DirectRLEnv, args: argparse.Namespace
) -> Union[Se3Keyboard, SO101Leader, BiSO101Leader, BiKeyboard]:
    """Create teleoperation interface based on device type.

    Args:
        env: Environment instance.
        args: Command-line arguments containing teleop_device and related config.

    Returns:
        Teleoperation interface instance.

    Raises:
        ValueError: If teleop_device is invalid.
    """
    if args.teleop_device == "keyboard":
        return Se3Keyboard(env, sensitivity=0.25 * args.sensitivity)
    if args.teleop_device == "so101leader":
        return SO101Leader(env, port=args.port, recalibrate=args.recalibrate)
    if args.teleop_device == "bi-so101leader":
        return BiSO101Leader(
            env,
            left_port=args.left_arm_port,
            right_port=args.right_arm_port,
            recalibrate=args.recalibrate,
        )
    if args.teleop_device == "bi-keyboard":
        return BiKeyboard(env, sensitivity=0.25 * args.sensitivity)
    raise ValueError(
        f"Invalid device interface '{args.teleop_device}'. "
        f"Supported: 'keyboard', 'so101leader', 'bi-so101leader', 'bi-keyboard'."
    )


def register_teleop_callbacks(
    teleop_interface: Any,
    recording_enabled: bool = False,
    args: Optional[argparse.Namespace] = None,
) -> Dict[str, bool]:
    """Register callback functions for teleoperation control keys.

    Key bindings:
        S: Start recording
        N: Mark current episode as successful (only active during recording)
        D: Discard current episode and re-record (only active during recording)
        X: Restart current episode without saving (only active during recording)
        C: Assisted mode only, force both grippers closed while ACT moves arms
        V: Assisted mode only, force both grippers open/released while ACT moves arms
        Z: Assisted mode only, return gripper timing to ACT
        ESC: Abort entire recording process and clear buffer

    Args:
        teleop_interface: Teleoperation interface instance.
        recording_enabled: Whether recording is enabled. If False, N/D keys are
            disabled in idle phase.

    Returns:
        Dictionary of status flags for recording control.
    """
    flags = {
        "start": False,  # S: Start recording
        "success": False,  # N: Success/early termination of current episode
        "remove": False,  # D: Discard current episode
        "restart": False,  # X: Reset current episode without saving
        "abort": False,  # ESC: Abort entire recording process, clear buffer
        "manual": False,  # M: manual takeover in assisted recording
        "policy_paused": False,  # P: hold current pose instead of policy playback
        "gripper_override": None,  # None | "closed" | "open" in assisted ACT mode
        "auto_fold_request": False,  # G: bimanual fold-to-centroid trigger
        "visual_repair_request": False,  # E: visual landmark IK grasp/fold/release macro
        "unattended_auto": False,  # True: ignore manual save/control callbacks
        "keyboard_locked_after_pause": False,  # Safe mode: ignore keys after P/C/V pause
        "last_motion_state": None,  # most recent unpaused observation.state
        "last_motion_delta": None,  # recent actual joint-state motion direction
    }

    def ignored_in_unattended_auto(key_name: str) -> bool:
        if not flags.get("unattended_auto"):
            return False
        logger.info("[%s] Ignored in unattended auto-collection mode", key_name)
        return True

    def ignored_in_safe_assist(key_name: str) -> bool:
        if not bool(getattr(args, "safe_assist_hotkeys", False)):
            return False
        if flags.get("keyboard_locked_after_pause"):
            if key_name == "R":
                return False
            logger.info("[%s] Ignored while keyboard locked after pause", key_name)
            return True
        if key_name not in {"N", "D", "M", "E", "G", "ESC"}:
            return False
        logger.info("[%s] Ignored in safe assist mode", key_name)
        return True

    def on_start():
        if ignored_in_safe_assist("S"):
            return
        if ignored_in_unattended_auto("S"):
            return
        flags["start"] = True
        # Make keyboard takeover usable without requiring a separate B press.
        if hasattr(teleop_interface, "started"):
            teleop_interface.started = True
        if hasattr(teleop_interface, "_started"):
            teleop_interface._started = True
        logger.info("[S] Recording started!")

    def on_success():
        if ignored_in_safe_assist("N"):
            return
        if ignored_in_unattended_auto("N"):
            return
        if not recording_enabled or not flags["start"]:
            # Ignore N key in idle phase (before recording starts)
            logger.debug("[N] Ignored (recording not started yet)")
            return
        flags["success"] = True
        logger.info("[N] Mark the current episode as successful.")

    def on_remove():
        if ignored_in_safe_assist("D"):
            return
        if ignored_in_unattended_auto("D"):
            return
        if not recording_enabled or not flags["start"]:
            # Ignore D key in idle phase (before recording starts)
            logger.debug("[D] Ignored (recording not started yet)")
            return
        flags["remove"] = True
        logger.info("[D] Discard the current episode and re-record.")

    def on_restart():
        if ignored_in_safe_assist("X"):
            return
        if ignored_in_unattended_auto("X"):
            return
        if not recording_enabled or not flags["start"]:
            logger.debug("[X] Ignored (recording not started yet)")
            return
        flags["restart"] = True
        logger.info("[X] Restart current episode without saving.")

    def on_abort():
        if ignored_in_safe_assist("ESC"):
            return
        if ignored_in_unattended_auto("ESC"):
            return
        flags["abort"] = True
        logger.warning("[ESC] Abort recording, clearing the current episode buffer...")

    def on_manual_toggle():
        if ignored_in_safe_assist("M"):
            return
        if ignored_in_unattended_auto("M"):
            return
        flags["manual"] = not flags["manual"]
        logger.info(f"[M] Manual takeover {'ON' if flags['manual'] else 'OFF'}")

    def on_policy_pause_toggle():
        if ignored_in_safe_assist("P"):
            return
        if ignored_in_unattended_auto("P"):
            return
        if bool(getattr(args, "safe_assist_hotkeys", False)):
            flags["policy_paused"] = True
            flags["keyboard_locked_after_pause"] = True
            logger.info("[P] Simulation PAUSED; keyboard locked until LiveIK resume")
            return
        flags["policy_paused"] = not flags["policy_paused"]
        logger.info(f"[P] Simulation {'PAUSED' if flags['policy_paused'] else 'RESUMED'}")

    def on_resume_policy():
        if ignored_in_safe_assist("R"):
            return
        if ignored_in_unattended_auto("R"):
            return
        flags["manual"] = False
        flags["policy_paused"] = False
        flags["gripper_override"] = None
        flags["keyboard_locked_after_pause"] = False
        logger.info("[R] Simulation resumed; manual takeover OFF; ACT grippers ON")

    def on_grippers_closed():
        if ignored_in_safe_assist("C"):
            return
        if ignored_in_unattended_auto("C"):
            return
        flags["gripper_override"] = "closed"
        flags["policy_paused"] = True
        flags["keyboard_locked_after_pause"] = bool(getattr(args, "safe_assist_hotkeys", False))
        logger.info("[C] Gripper override: CLOSED; simulation paused for IK/manual placement")

    def on_grippers_open():
        if ignored_in_safe_assist("V"):
            return
        if ignored_in_unattended_auto("V"):
            return
        flags["gripper_override"] = "open"
        flags["policy_paused"] = True
        flags["keyboard_locked_after_pause"] = bool(getattr(args, "safe_assist_hotkeys", False))
        logger.info("[V] Gripper override: OPEN/RELEASE; simulation paused for placement")

    def on_grippers_act():
        if ignored_in_safe_assist("Z"):
            return
        if ignored_in_unattended_auto("Z"):
            return
        flags["gripper_override"] = None
        logger.info("[Z] Gripper override cleared; ACT controls grippers")

    def on_auto_fold():
        if ignored_in_safe_assist("G"):
            return
        if ignored_in_unattended_auto("G"):
            return
        # Only meaningful when grippers are forced closed (cloth grasped)
        if flags.get("gripper_override") != "closed":
            logger.info("[G] Ignored: press C first to confirm cloth is grasped")
            return
        flags["auto_fold_request"] = True
        flags["policy_paused"] = True
        logger.info("[G] Auto-fold requested: both arms toward garment centroid")

    def on_visual_repair():
        if ignored_in_safe_assist("E"):
            return
        if ignored_in_unattended_auto("E"):
            return
        flags["visual_repair_request"] = True
        flags["policy_paused"] = True
        logger.info("[E] Visual repair requested: landmark IK grasp/fold/release macro")

    teleop_interface.add_callback("S", on_start)
    teleop_interface.add_callback("N", on_success)
    teleop_interface.add_callback("D", on_remove)
    teleop_interface.add_callback("X", on_restart)
    teleop_interface.add_callback("ESCAPE", on_abort)
    teleop_interface.add_callback("M", on_manual_toggle)
    teleop_interface.add_callback("P", on_policy_pause_toggle)
    teleop_interface.add_callback("R", on_resume_policy)
    teleop_interface.add_callback("C", on_grippers_closed)
    teleop_interface.add_callback("V", on_grippers_open)
    teleop_interface.add_callback("Z", on_grippers_act)
    teleop_interface.add_callback("G", on_auto_fold)
    teleop_interface.add_callback("E", on_visual_repair)

    return flags


def create_assist_policy(args: argparse.Namespace):
    """Create an optional policy used for assisted demo recording."""
    if not getattr(args, "assist_policy_type", None):
        return None

    from scripts.eval_policy import PolicyRegistry

    policy_type = args.assist_policy_type
    kwargs = {
        "device": args.assist_policy_device,
        "task_description": args.task_description,
    }
    if policy_type in (
        "lerobot",
        "stabilized_lerobot",
        "act_with_collar_recovery",
        "act_with_vision_collar_recovery",
        "ai_teleop_top_short",
        "cem_recovery_top_short",
    ):
        if not args.assist_policy_path or not args.assist_dataset_root:
            raise ValueError(
                "--assist_policy_path and --assist_dataset_root are required "
                f"for assisted policy type {policy_type}"
            )
        kwargs.update(
            {
                "policy_path": args.assist_policy_path,
                "dataset_root": args.assist_dataset_root,
            }
        )
    else:
        if args.assist_policy_path:
            kwargs["model_path"] = args.assist_policy_path

    policy = PolicyRegistry.create(policy_type, **kwargs)
    logger.info(
        f"[Assist] Loaded policy type={policy_type}, "
        f"path={getattr(args, 'assist_policy_path', None)}, device={args.assist_policy_device}"
    )
    return policy


def apply_assisted_gripper_override(
    action_np: np.ndarray, flags: Dict[str, Any]
) -> np.ndarray:
    """Override only gripper joints while leaving ACT arm trajectory untouched."""
    override = flags.get("gripper_override")
    if override is None:
        return action_np
    action_np = np.asarray(action_np, dtype=np.float32).copy()
    if action_np.shape[0] <= RIGHT_GRIPPER_IDX:
        return action_np
    if override == "closed":
        action_np[LEFT_GRIPPER_IDX] = GRIPPER_CLOSED
        action_np[RIGHT_GRIPPER_IDX] = GRIPPER_CLOSED
    elif override == "open":
        action_np[LEFT_GRIPPER_IDX] = GRIPPER_OPEN
        action_np[RIGHT_GRIPPER_IDX] = GRIPPER_OPEN
    return action_np


def apply_auto_grip_timing(
    actions: torch.Tensor,
    args: argparse.Namespace,
    episode_step: int,
) -> torch.Tensor:
    """Force a late close/release gripper schedule for ACT harvest rollouts."""
    hold_start = int(getattr(args, "auto_grip_hold_start_step", -1))
    hold_end = int(getattr(args, "auto_grip_hold_end_step", -1))
    release_until = int(getattr(args, "auto_grip_release_until_step", -1))
    if actions is None or actions.shape[-1] <= RIGHT_GRIPPER_IDX:
        return actions

    mode = None
    value = None
    if hold_start >= 0 and hold_end > hold_start and hold_start <= episode_step < hold_end:
        mode = "closed"
        value = GRIPPER_CLOSED
    elif hold_end >= 0 and release_until > hold_end and hold_end <= episode_step < release_until:
        mode = "open"
        value = GRIPPER_OPEN
    if mode is None:
        return actions

    actions = actions.clone()
    actions[..., LEFT_GRIPPER_IDX] = float(value)
    actions[..., RIGHT_GRIPPER_IDX] = float(value)
    if episode_step in {hold_start, hold_end} or episode_step % 40 == 0:
        logger.info("[AutoGripTiming] step=%d forcing grippers %s", episode_step, mode)
    return actions


def force_gripper_joint_state_to_sim(env: DirectRLEnv, mode: Optional[str]) -> None:
    """Directly write gripper joint positions while physics is soft-paused."""
    if mode not in {"closed", "open"}:
        return
    value = GRIPPER_CLOSED if mode == "closed" else GRIPPER_OPEN
    for arm_name in ("left_arm", "right_arm"):
        arm = getattr(env, arm_name, None)
        if arm is None:
            continue
        try:
            joint_pos = arm.data.joint_pos.clone()
            joint_pos[:, -1] = value
            arm.write_joint_position_to_sim(joint_pos, joint_ids=None)
            arm.set_joint_position_target(joint_pos)
        except Exception as e:
            logger.debug("[Pause] failed to force %s gripper=%s: %s", arm_name, mode, e)


class ClickIKController:
    """Top-camera click-to-IK controller for assisted demo repair.

    This uses an OpenCV top-camera view rather than raw Isaac viewport clicks:
    top RGB pixel + top depth -> world point -> nearest arm IK -> interpolated
    joint-space move. It is intended for paused assisted recording after `C`
    or `V`, where ACT arm motion is held and the user places the cloth manually.
    """

    WINDOW_NAME = "Click-IK Top Camera"

    def __init__(self, env: DirectRLEnv, args: argparse.Namespace):
        self.env = env
        self.args = args
        self.enabled = bool(getattr(args, "enable_click_ik", False))
        self.cv2 = None
        self.window_enabled = False
        self.solver = None
        self.pending_click: Tuple[int, int] | None = None
        self.trajectory: List[np.ndarray] = []
        self.visual_repair_debug_active = False
        self.visual_repair_debug_step = 0
        self.visual_repair_plan_id = 0
        self.last_visual_repair_plan_len = 0
        self.command_file = Path(
            getattr(args, "ik_command_file", "/tmp/lehome_ik_command.json")
        )
        self.status_file = Path(
            getattr(args, "ik_status_file", "/tmp/lehome_ik_status.json")
        )

        if not self.enabled:
            return

        try:
            ensure_real_pinocchio()
            import cv2
            from lehome.utils import RobotKinematics

            self.cv2 = cv2
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
            try:
                cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.WINDOW_NAME, 640, 480)
                cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
                self.window_enabled = True
                logger.info(
                    "[ClickIK] GUI enabled. Press C/V to pause+grip, then click the top-camera window."
                )
            except Exception as e:
                self.window_enabled = False
                logger.warning(
                    "[ClickIK] GUI window unavailable (%s); headless visual repair still enabled.",
                    e,
                )
        except Exception as e:
            logger.error(f"[ClickIK] Disabled; initialization failed: {e}")
            self.enabled = False

    def _on_mouse(self, event, x, y, flags, param):
        if not self.enabled or self.cv2 is None:
            return
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.pending_click = (int(x), int(y))
            logger.info(f"[ClickIK] Click queued at pixel=({int(x)}, {int(y)})")

    def reset(self):
        self.pending_click = None
        self.trajectory.clear()
        self.visual_repair_debug_active = False
        self.visual_repair_debug_step = 0
        self.last_visual_repair_plan_len = 0

    def is_busy(self) -> bool:
        return bool(self.trajectory)

    def close(self):
        if self.enabled and self.cv2 is not None and self.window_enabled:
            try:
                self.cv2.destroyWindow(self.WINDOW_NAME)
            except Exception:
                pass

    def _write_ik_status(self, status: str, **payload: Any) -> None:
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"status": status, "time": time.time(), **payload}
            self.status_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[LiveIK] failed to write status: %s", e)

    def _save_snapshot(self, observations: Dict[str, Any]) -> bool:
        """Save paused top RGB/depth so external code can choose real cloth pixels."""
        rgb = observations.get("observation.images.top_rgb")
        depth = observations.get("observation.top_depth")
        if rgb is None:
            self._write_ik_status("error", cmd="snapshot", error="missing top_rgb")
            return False

        out_dir = Path("/tmp/lehome_snapshots")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        rgb_path = out_dir / f"snapshot_{stamp}_top_rgb.png"
        depth_path = out_dir / f"snapshot_{stamp}_top_depth.npy"

        frame = np.asarray(rgb)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if frame.max(initial=0) <= 1.0:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        wrote_png = False
        try:
            if self.cv2 is not None:
                self.cv2.imwrite(str(rgb_path), self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))
                wrote_png = True
        except Exception as e:
            logger.warning("[LiveIK] cv2 snapshot write failed: %s", e)
        if not wrote_png:
            # Portable binary PPM fallback; keep .ppm next to the requested png.
            rgb_path = rgb_path.with_suffix(".ppm")
            with rgb_path.open("wb") as f:
                f.write(f"P6\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode("ascii"))
                f.write(np.ascontiguousarray(frame).tobytes())

        depth_stats = None
        if depth is not None:
            depth_arr = np.asarray(depth)
            np.save(depth_path, depth_arr)
            finite = depth_arr[np.isfinite(depth_arr)]
            if finite.size:
                depth_stats = {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                }

        self._write_ik_status(
            "ok",
            cmd="snapshot",
            rgb_path=str(rgb_path),
            depth_path=str(depth_path) if depth is not None else None,
            rgb_shape=list(frame.shape),
            depth_stats=depth_stats,
        )
        logger.info("[LiveIK] snapshot saved: %s", rgb_path)
        return True

    def _valid_cloth_world_target(self, world: Optional[np.ndarray], label: str) -> bool:
        """Reject background pixels that reconstruct far above the table."""
        if world is None:
            self._write_ik_status("error", cmd="move", error=f"invalid depth for {label}")
            return False
        z = float(world[2])
        # Top-camera pixels on the garment/table reconstruct to roughly z=1.5-1.8
        # in this task's world frame because the IK target is the gripper frame,
        # not the cloth root. Reject only extreme depth artifacts.
        if not (1.25 <= z <= 1.95):
            self._write_ik_status(
                "error",
                cmd="move",
                error=f"{label} world z={z:.3f} is outside cloth/table range",
                world=world.tolist(),
            )
            logger.warning("[LiveIK] rejected %s target with implausible z=%.3f", label, z)
            return False
        return True

    def _consume_ik_command(
        self, observations: Dict[str, Any], flags: Dict[str, Any]
    ) -> bool:
        """Poll an external JSON command and, if present, convert it to actions.

        This is the soft-pause control path: the sim stops stepping physics while
        `policy_paused` is true, and an external process can inject IK commands
        without relying on GUI clicks. Physics resumes only while an IK trajectory
        is actively being executed.
        """
        if not self.command_file:
            return False
        if not self.command_file.exists():
            return False
        try:
            raw = self.command_file.read_text(encoding="utf-8")
            command = json.loads(raw)
        except Exception as e:
            self._write_ik_status("error", error=f"could not read command: {e}")
            return False
        try:
            self.command_file.unlink()
        except Exception:
            pass

        cmd = str(command.get("cmd", "move")).lower()
        if cmd in {"snapshot", "snap"}:
            return self._save_snapshot(observations)
        if cmd in {"pause", "hold"}:
            flags["policy_paused"] = True
            self._write_ik_status("ok", cmd=cmd, message="simulation paused")
            logger.info("[LiveIK] simulation paused by command")
            return True
        if cmd == "resume":
            flags["policy_paused"] = False
            flags["gripper_override"] = None
            flags["keyboard_locked_after_pause"] = False
            self._write_ik_status("ok", cmd=cmd, message="simulation resumed")
            logger.info("[LiveIK] simulation resumed by command")
            return True
        if cmd in {"restart", "reset"}:
            flags["restart"] = True
            self._write_ik_status("ok", cmd=cmd, message="episode restart requested")
            logger.info("[LiveIK] restart requested by command")
            return True
        if cmd in {"grip", "gripper"}:
            mode = str(command.get("mode", command.get("gripper", "closed"))).lower()
            if mode in {"closed", "close", "c"}:
                flags["gripper_override"] = "closed"
                flags["policy_paused"] = True
                flags["keyboard_locked_after_pause"] = True
            elif mode in {"open", "release", "v"}:
                flags["gripper_override"] = "open"
                flags["policy_paused"] = True
                flags["keyboard_locked_after_pause"] = True
            elif mode in {"act", "clear", "none", "z"}:
                flags["gripper_override"] = None
            else:
                self._write_ik_status("error", cmd=cmd, error=f"unknown gripper mode {mode}")
                return False
            self._write_ik_status(
                "ok", cmd=cmd, mode=flags.get("gripper_override") or "act"
            )
            logger.info("[LiveIK] gripper override=%s", flags.get("gripper_override"))
            return True
        if cmd in {"continue", "extend"}:
            state = np.asarray(observations.get("observation.state"), dtype=np.float32)
            delta = flags.get("last_motion_delta")
            if state.shape[0] < 12 or delta is None:
                self._write_ik_status(
                    "error",
                    cmd=cmd,
                    error="missing recent motion direction; let ACT move before pausing",
                )
                return False
            delta = np.asarray(delta, dtype=np.float32)
            if delta.shape[0] < 12 or float(np.linalg.norm(delta[:12])) < 1e-5:
                self._write_ik_status("error", cmd=cmd, error="recent motion direction is zero")
                return False

            gain = float(command.get("gain", 4.0))
            steps = max(1, int(command.get("steps", 30)))
            grip_mode = str(command.get("gripper", "closed")).lower()
            target = state.copy()
            # Continue only arm joints. Keep grippers explicit and symmetric.
            arm_delta = np.zeros_like(target)
            arm_delta[0:5] = delta[0:5]
            arm_delta[6:11] = delta[6:11]
            # Keep the extrapolation small enough to behave like a nudge, not a throw.
            max_joint_delta = float(command.get("max_joint_delta", 0.12))
            arm_delta = np.clip(arm_delta * gain, -max_joint_delta, max_joint_delta)
            target += arm_delta
            if grip_mode in {"closed", "close", "c"}:
                target[LEFT_GRIPPER_IDX] = GRIPPER_CLOSED
                target[RIGHT_GRIPPER_IDX] = GRIPPER_CLOSED
                flags["gripper_override"] = "closed"
            elif grip_mode in {"open", "release", "v"}:
                target[LEFT_GRIPPER_IDX] = GRIPPER_OPEN
                target[RIGHT_GRIPPER_IDX] = GRIPPER_OPEN
                flags["gripper_override"] = "open"
            elif flags.get("gripper_override") == "closed":
                target[LEFT_GRIPPER_IDX] = GRIPPER_CLOSED
                target[RIGHT_GRIPPER_IDX] = GRIPPER_CLOSED

            self.trajectory = [
                (state + (target - state) * (i / steps)).astype(np.float32)
                for i in range(1, steps + 1)
            ]
            flags["policy_paused"] = True
            flags["keyboard_locked_after_pause"] = True
            self._write_ik_status(
                "ok",
                cmd=cmd,
                steps=steps,
                gain=gain,
                max_joint_delta=max_joint_delta,
                delta_norm=float(np.linalg.norm(delta[:12])),
            )
            logger.info(
                "[LiveIK] planned ACT-direction continue steps=%d gain=%.2f max_delta=%.3f",
                steps,
                gain,
                max_joint_delta,
            )
            return True
        if cmd in {"nudge", "offset", "cartesian_nudge"}:
            state = np.asarray(observations.get("observation.state"), dtype=np.float32)
            if state.shape[0] < 12:
                self._write_ik_status("error", cmd=cmd, error="missing 12D state")
                return False
            try:
                left_pos = (
                    self.env.left_arm.data.body_link_pos_w[0, -1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                right_pos = (
                    self.env.right_arm.data.body_link_pos_w[0, -1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            except Exception as e:
                self._write_ik_status(
                    "error", cmd=cmd, error=f"could not read gripper poses: {e}"
                )
                return False

            delta_world = np.array(
                [
                    float(command.get("dx", 0.0)),
                    float(command.get("dy", 0.0)),
                    float(command.get("dz", 0.0)),
                ],
                dtype=np.float32,
            )
            norm = float(np.linalg.norm(delta_world))
            max_delta = float(command.get("max_delta", 0.08))
            if norm <= 1e-5:
                self._write_ik_status("error", cmd=cmd, error="zero nudge")
                return False
            if norm > max_delta:
                delta_world = delta_world * (max_delta / norm)
                norm = max_delta

            grip_mode = str(command.get("gripper", "closed")).lower()
            if grip_mode in {"closed", "close", "c"}:
                gripper = GRIPPER_CLOSED
                flags["gripper_override"] = "closed"
            elif grip_mode in {"open", "release", "v"}:
                gripper = GRIPPER_OPEN
                flags["gripper_override"] = "open"
            else:
                gripper = (
                    GRIPPER_CLOSED
                    if flags.get("gripper_override") == "closed"
                    else float(state[LEFT_GRIPPER_IDX])
                )

            steps = max(1, int(command.get("steps", 40)))
            next_cursor = self._append_pair_ik_move(
                state.copy(),
                left_pos + delta_world,
                right_pos + delta_world,
                float(gripper),
                steps,
                "live_nudge",
            )
            if next_cursor is None:
                self._write_ik_status("error", cmd=cmd, error="IK failed")
                return False
            flags["policy_paused"] = True
            flags["keyboard_locked_after_pause"] = True
            self._write_ik_status(
                "ok",
                cmd=cmd,
                steps=steps,
                delta_world=delta_world.tolist(),
                left_start=left_pos.tolist(),
                right_start=right_pos.tolist(),
                left_target=(left_pos + delta_world).tolist(),
                right_target=(right_pos + delta_world).tolist(),
            )
            logger.info(
                "[LiveIK] planned synchronized world nudge dx=%.3f dy=%.3f dz=%.3f steps=%d",
                float(delta_world[0]),
                float(delta_world[1]),
                float(delta_world[2]),
                steps,
            )
            return True
        if cmd != "move":
            self._write_ik_status("error", cmd=cmd, error="unknown command")
            return False

        flags["policy_paused"] = True
        flags["keyboard_locked_after_pause"] = True
        grip_mode = str(command.get("gripper", "keep")).lower()
        if grip_mode in {"closed", "close", "c"}:
            flags["gripper_override"] = "closed"
        elif grip_mode in {"open", "release", "v"}:
            flags["gripper_override"] = "open"
        elif grip_mode in {"act", "clear", "none", "z"}:
            flags["gripper_override"] = None

        arm = str(command.get("arm", "nearest")).lower()
        steps = int(command.get("steps", getattr(self.args, "click_ik_steps", 45)))
        if arm == "both":
            left_pixel = command.get("left_pixel")
            right_pixel = command.get("right_pixel")
            if left_pixel is None or right_pixel is None:
                self._write_ik_status(
                    "error", cmd=cmd, error="both-arm move requires left_pixel and right_pixel"
                )
                return False
            depth = observations.get("observation.top_depth")
            left_world = self._pixel_to_world(int(left_pixel[0]), int(left_pixel[1]), depth)
            right_world = self._pixel_to_world(int(right_pixel[0]), int(right_pixel[1]), depth)
            if not self._valid_cloth_world_target(left_world, "left_pixel"):
                return False
            if not self._valid_cloth_world_target(right_world, "right_pixel"):
                return False
            state = np.asarray(observations.get("observation.state"), dtype=np.float32)
            if state.shape[0] < 12:
                self._write_ik_status("error", cmd=cmd, error="missing 12D state")
                return False
            gripper = (
                GRIPPER_CLOSED
                if flags.get("gripper_override") == "closed"
                else GRIPPER_OPEN
                if flags.get("gripper_override") == "open"
                else float(state[LEFT_GRIPPER_IDX])
            )
            next_cursor = self._append_pair_ik_move(
                state.copy(), left_world, right_world, float(gripper), max(1, steps), "live_both"
            )
            if next_cursor is None:
                self._write_ik_status("error", cmd=cmd, error="IK failed")
                return False
            self._write_ik_status(
                "ok",
                cmd=cmd,
                arm=arm,
                left_world=left_world.tolist(),
                right_world=right_world.tolist(),
                steps=steps,
            )
            logger.info("[LiveIK] planned both-arm move steps=%d", steps)
            return True

        pixel = command.get("pixel")
        if pixel is None:
            self._write_ik_status("error", cmd=cmd, error="move requires pixel")
            return False
        planned = self._plan_trajectory(
            (int(pixel[0]), int(pixel[1])),
            observations,
            flags,
            arm_override=None if arm == "nearest" else arm,
            steps_override=max(1, steps),
        )
        if planned:
            self._write_ik_status("ok", cmd=cmd, arm=arm, pixel=pixel, steps=steps)
            return True
        self._write_ik_status("error", cmd=cmd, error="IK failed")
        return False

    def update_view(self, observations: Dict[str, Any], flags: Dict[str, Any]):
        if not self.enabled or self.cv2 is None or not self.window_enabled:
            return
        rgb = observations.get("observation.images.top_rgb")
        if rgb is None:
            return
        frame = np.asarray(rgb)
        if frame.ndim != 3:
            return
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        display = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
        mode = flags.get("gripper_override") or "ACT"
        paused = "PAUSED" if flags.get("policy_paused") else "ACT-RUN"
        text = f"{paused} grip={mode} | left-click target | C hold V release R resume"
        self.cv2.putText(
            display,
            text,
            (10, 24),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.imshow(self.WINDOW_NAME, display)
        self.cv2.waitKey(1)

    def next_action(
        self, observations: Dict[str, Any], flags: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        if self.trajectory:
            self._maybe_save_visual_debug(observations)
            return self.trajectory.pop(0)
        if self._consume_ik_command(observations, flags):
            if self.trajectory:
                return self.trajectory.pop(0)
            return None
        # Auto-fold (G key) takes priority — plan bimanual move to garment centroid.
        if flags.get("auto_fold_request"):
            flags["auto_fold_request"] = False
            self._plan_auto_fold(observations, flags)
            if self.trajectory:
                return self.trajectory.pop(0)
            return None
        # Visual repair macro: detect garment landmarks, grasp them with IK,
        # carry slowly toward the centroid, then release/place.
        if flags.get("visual_repair_request"):
            flags["visual_repair_request"] = False
            self._plan_visual_repair(observations, flags)
            if self.trajectory:
                return self.trajectory.pop(0)
            return None
        if self.pending_click is None:
            return None
        if not flags.get("policy_paused"):
            self.pending_click = None
            logger.info("[ClickIK] Ignored click; press C or V first to pause physics")
            return None
        click = self.pending_click
        self.pending_click = None
        self._plan_trajectory(click, observations, flags)
        if self.trajectory:
            self._maybe_save_visual_debug(observations)
            return self.trajectory.pop(0)
        return None

    def _pixel_to_world(self, u: int, v: int, depth_mm: np.ndarray) -> Optional[np.ndarray]:
        from scipy.spatial.transform import Rotation as R

        if depth_mm is None:
            return None
        depth = np.asarray(depth_mm)
        if depth.ndim > 2:
            depth = depth.squeeze()
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        x0, x1 = max(0, u - 3), min(w, u + 4)
        y0, y1 = max(0, v - 3), min(h, v + 4)
        patch = depth[y0:y1, x0:x1].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        z_m = float(np.median(valid) / 1000.0)

        fx, fy = 482.0, 482.0
        cx, cy = 320.0, 240.0
        x_cam = (u - cx) * z_m / fx
        y_cam = (v - cy) * z_m / fy
        point_cam = np.array([x_cam, y_cam, z_m], dtype=np.float32)

        quat_wxyz = [0.1650476, -0.9862856, 0.0, 0.0]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        r_usd_to_base = R.from_quat(quat_xyzw).as_matrix()
        r_optical_to_usd = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        r_mix = np.dot(r_usd_to_base, r_optical_to_usd)
        translation_cam_to_robot = np.array([0.225, -0.5, 0.6])
        point_local = np.dot(point_cam, r_mix.T) + translation_cam_to_robot

        t_robot_to_world = np.array([0.23, -0.25, 0.5])
        r_robot_to_world = R.from_euler("z", 180, degrees=True).as_matrix()
        point_world = np.dot(point_local, r_robot_to_world.T) + t_robot_to_world
        point_world[2] += float(getattr(self.args, "click_ik_z_offset", 0.05))
        return point_world.astype(np.float32)

    def _choose_arm(self, target_world: np.ndarray) -> str:
        left_pos = self.env.left_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
        right_pos = self.env.right_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
        left_dist = float(np.linalg.norm(left_pos - target_world))
        right_dist = float(np.linalg.norm(right_pos - target_world))
        return "left" if left_dist <= right_dist else "right"

    def _segment_garment_centroid(self, rgb: np.ndarray) -> Optional[Tuple[int, int]]:
        """Return (u, v) pixel of the garment centroid via HSV segmentation."""
        try:
            from scripts.cv_collar_pol.segment_and_landmark import (
                segment_garment, compute_landmarks,
            )
        except Exception as e:
            logger.warning(f"[AutoFold] segmentation import failed: {e}")
            return None
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # segment_garment expects BGR; env passes RGB. Flip.
        bgr = rgb[..., ::-1].copy()
        try:
            mask = segment_garment(bgr)
            lm = compute_landmarks(mask)
        except Exception as e:
            logger.warning(f"[AutoFold] segmentation failed: {e}")
            return None
        if lm is None:
            return None
        return (int(lm.centroid[0]), int(lm.centroid[1]))

    def _detect_landmarks(self, rgb: np.ndarray):
        """Detect garment landmarks from top RGB. Returns Landmarks or None."""
        try:
            from scripts.cv_collar_pol.segment_and_landmark import (
                segment_garment,
                compute_landmarks,
                detect_grippers,
            )
        except Exception as e:
            logger.warning(f"[VisualRepair] segmentation import failed: {e}")
            return None
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        bgr = rgb[..., ::-1].copy()
        try:
            mask = segment_garment(bgr)
            lm = compute_landmarks(mask)
            if lm is not None:
                lm.left_gripper, lm.right_gripper = detect_grippers(bgr)
            return lm
        except Exception as e:
            logger.warning(f"[VisualRepair] segmentation failed: {e}")
            return None

    def _maybe_save_visual_debug(self, observations: Dict[str, Any]) -> None:
        if not self.visual_repair_debug_active:
            return
        every = int(getattr(self.args, "visual_repair_debug_every", 30))
        if every <= 0:
            return
        self.visual_repair_debug_step += 1
        remaining = len(self.trajectory)
        if (
            self.visual_repair_debug_step != 1
            and self.visual_repair_debug_step % every != 0
            and remaining > 1
        ):
            return
        self._save_visual_debug(
            observations,
            f"repair{self.visual_repair_plan_id:02d}_step{self.visual_repair_debug_step:04d}_rem{remaining:04d}",
        )
        if remaining <= 1:
            self.visual_repair_debug_active = False

    def _save_visual_debug(self, observations: Dict[str, Any], label: str) -> None:
        rgb = observations.get("observation.images.top_rgb")
        if rgb is None:
            return
        try:
            import cv2
            frame = np.asarray(rgb)
            if frame.ndim != 3:
                return
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            lm = self._detect_landmarks(frame)
            if lm is not None:
                points = [
                    ("TL", lm.top_left, (0, 255, 255)),
                    ("TR", lm.top_right, (0, 255, 255)),
                    ("BL", lm.bottom_left, (255, 128, 0)),
                    ("BR", lm.bottom_right, (255, 128, 0)),
                    ("C", lm.centroid, (0, 255, 0)),
                ]
                for name, uv, color in points:
                    u, v = int(uv[0]), int(uv[1])
                    cv2.circle(bgr, (u, v), 5, color, -1)
                    cv2.putText(
                        bgr,
                        name,
                        (u + 6, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            out_dir = Path(getattr(self.args, "visual_repair_debug_dir", "/tmp/lehome_visual_repair_debug"))
            out_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
            out_path = out_dir / f"{int(time.time())}_{safe}.png"
            cv2.imwrite(str(out_path), bgr)
            logger.info("[VisualRepairDebug] saved %s", out_path)
        except Exception as e:
            logger.warning("[VisualRepairDebug] failed to save %s: %s", label, e)

    def _append_pair_ik_move(
        self,
        cursor: np.ndarray,
        left_world: np.ndarray,
        right_world: np.ndarray,
        gripper_value: float,
        steps: int,
        label: str,
    ) -> Optional[np.ndarray]:
        """Append a slow bimanual IK move and return the resulting target state."""
        from lehome.utils import compute_joints_from_world_point

        joints_left = compute_joints_from_world_point(
            self.solver,
            self.env,
            "left",
            left_world.astype(np.float32),
            current_joints=cursor[0:6].copy(),
            state_unit="rad",
            gripper_angle=float(gripper_value),
        )
        joints_right = compute_joints_from_world_point(
            self.solver,
            self.env,
            "right",
            right_world.astype(np.float32),
            current_joints=cursor[6:12].copy(),
            state_unit="rad",
            gripper_angle=float(gripper_value),
        )
        if joints_left is None or joints_right is None:
            logger.warning(
                "[VisualRepair] IK failed in %s: left=%s right=%s",
                label,
                joints_left,
                joints_right,
            )
            return None

        target = cursor.copy()
        target[0:6] = joints_left.astype(np.float32)
        target[6:12] = joints_right.astype(np.float32)
        target[LEFT_GRIPPER_IDX] = gripper_value
        target[RIGHT_GRIPPER_IDX] = gripper_value

        for i in range(1, steps + 1):
            self.trajectory.append(
                (cursor + (target - cursor) * (i / steps)).astype(np.float32)
            )
        return target

    def _append_hold(
        self,
        cursor: np.ndarray,
        gripper_value: float,
        steps: int,
    ) -> np.ndarray:
        target = cursor.copy()
        target[LEFT_GRIPPER_IDX] = gripper_value
        target[RIGHT_GRIPPER_IDX] = gripper_value
        for _ in range(max(1, steps)):
            self.trajectory.append(target.astype(np.float32).copy())
        return target

    def _plan_visual_repair(
        self, observations: Dict[str, Any], flags: Dict[str, Any]
    ) -> None:
        """Plan a slow visual landmark grasp/fold/release macro.

        This is intentionally conservative: it uses public top RGB/depth only,
        makes no check_point calls, and moves slowly to reduce cloth slip.
        """
        rgb = observations.get("observation.images.top_rgb")
        depth = observations.get("observation.top_depth")
        if rgb is None or depth is None:
            logger.warning("[VisualRepair] missing top_rgb or top_depth")
            return
        lm = self._detect_landmarks(np.asarray(rgb))
        if lm is None:
            logger.warning("[VisualRepair] could not detect garment landmarks")
            return

        grasp_mode = str(getattr(self.args, "visual_repair_grasp", "top"))
        if grasp_mode == "bottom":
            left_uv, right_uv = lm.bottom_left, lm.bottom_right
        else:
            left_uv, right_uv = lm.top_left, lm.top_right
        center_uv = lm.centroid
        top_mid_uv = (
            int(0.5 * (lm.top_left[0] + lm.top_right[0])),
            int(0.5 * (lm.top_left[1] + lm.top_right[1])),
        )
        bottom_mid_uv = (
            int(0.5 * (lm.bottom_left[0] + lm.bottom_right[0])),
            int(0.5 * (lm.bottom_left[1] + lm.bottom_right[1])),
        )
        opposite_uv = bottom_mid_uv if grasp_mode == "top" else top_mid_uv

        left_surface = self._pixel_to_world(left_uv[0], left_uv[1], depth)
        right_surface = self._pixel_to_world(right_uv[0], right_uv[1], depth)
        center_surface = self._pixel_to_world(center_uv[0], center_uv[1], depth)
        opposite_surface = self._pixel_to_world(opposite_uv[0], opposite_uv[1], depth)
        if (
            left_surface is None
            or right_surface is None
            or center_surface is None
            or opposite_surface is None
        ):
            logger.warning(
                "[VisualRepair] invalid depth for landmarks: L=%s R=%s C=%s O=%s",
                left_surface,
                right_surface,
                center_surface,
                opposite_surface,
            )
            return

        state = np.asarray(observations.get("observation.state"), dtype=np.float32)
        if state.shape[0] < 12:
            logger.warning("[VisualRepair] expected 12D bimanual state")
            return

        # All z values target gripper_frame_link. _pixel_to_world already adds
        # click_ik_z_offset so jaws land near the cloth surface.
        approach_z = 0.07
        drag_z = 0.006
        press_z = -0.010
        retract_z = 0.12
        steps = max(20, min(int(getattr(self.args, "click_ik_steps", 45)), 45))
        close_hold = 35
        release_hold = 25

        left_approach = left_surface + np.array([0.0, 0.0, approach_z], dtype=np.float32)
        right_approach = right_surface + np.array([0.0, 0.0, approach_z], dtype=np.float32)

        # Phase 1: drag the grasped landmarks near the center at contact height.
        # _pixel_to_world already added the wrist-to-jaw offset, so adding a
        # large z offset here makes the jaws float above the cloth.
        left_gather = center_surface + 0.20 * (left_surface - center_surface)
        right_gather = center_surface + 0.20 * (right_surface - center_surface)
        left_gather[2] = max(center_surface[2], left_surface[2]) + drag_z
        right_gather[2] = max(center_surface[2], right_surface[2]) + drag_z

        # Phase 2: low overfold past the centroid toward the opposite side of
        # the garment (top landmarks -> bottom body; bottom landmarks -> top
        # body). Keep this close to the cloth so friction/contact, not airborne
        # carrying, does the fold.
        over_center = center_surface + 0.75 * (opposite_surface - center_surface)
        left_overfold = over_center + 0.10 * (left_surface - center_surface)
        right_overfold = over_center + 0.10 * (right_surface - center_surface)
        left_overfold[2] = max(center_surface[2], opposite_surface[2]) + drag_z
        right_overfold[2] = max(center_surface[2], opposite_surface[2]) + drag_z

        left_place = left_overfold.copy()
        right_place = right_overfold.copy()
        left_place[2] = opposite_surface[2] + press_z
        right_place[2] = opposite_surface[2] + press_z
        left_retract = left_place + np.array([0.0, 0.0, retract_z], dtype=np.float32)
        right_retract = right_place + np.array([0.0, 0.0, retract_z], dtype=np.float32)

        self.trajectory.clear()
        cursor = state.copy()
        plan = [
            ("approach", left_approach, right_approach, GRIPPER_OPEN, steps),
            ("descend", left_surface, right_surface, GRIPPER_OPEN, max(20, steps // 2)),
            ("close", left_surface, right_surface, GRIPPER_CLOSED, max(15, steps // 3)),
            ("pinch", left_surface, right_surface, GRIPPER_CLOSED, max(15, steps // 2)),
            ("gather_drag", left_gather, right_gather, GRIPPER_CLOSED, steps + 10),
            ("overfold_drag", left_overfold, right_overfold, GRIPPER_CLOSED, steps + 15),
            ("press_place", left_place, right_place, GRIPPER_CLOSED, max(25, steps // 2)),
        ]
        for label, left_world, right_world, gripper, n_steps in plan:
            next_cursor = self._append_pair_ik_move(
                cursor, left_world, right_world, gripper, n_steps, label
            )
            if next_cursor is None:
                self.trajectory.clear()
                return
            cursor = next_cursor
            if label == "close":
                cursor = self._append_hold(cursor, GRIPPER_CLOSED, close_hold)

        cursor = self._append_hold(cursor, GRIPPER_OPEN, release_hold)
        next_cursor = self._append_pair_ik_move(
            cursor, left_retract, right_retract, GRIPPER_OPEN, max(20, steps // 2), "retract"
        )
        if next_cursor is None:
            self.trajectory.clear()
            return

        flags["policy_paused"] = True
        flags["gripper_override"] = None
        self.visual_repair_plan_id += 1
        self.visual_repair_debug_active = True
        self.visual_repair_debug_step = 0
        self.last_visual_repair_plan_len = len(self.trajectory)
        self._save_visual_debug(
            observations,
            f"repair{self.visual_repair_plan_id:02d}_planned",
        )
        logger.info(
            "[VisualRepair] Planned %s-landmark repair: L%s R%s center%s opposite%s frames=%d",
            grasp_mode,
            tuple(map(int, left_uv)),
            tuple(map(int, right_uv)),
            tuple(map(int, center_uv)),
            tuple(map(int, opposite_uv)),
            len(self.trajectory),
        )

    def _plan_auto_fold(
        self, observations: Dict[str, Any], flags: Dict[str, Any]
    ) -> None:
        """Plan a bimanual move: both grippers toward the garment centroid (lifted)."""
        from lehome.utils import compute_joints_from_world_point

        rgb = observations.get("observation.images.top_rgb")
        depth = observations.get("observation.top_depth")
        if rgb is None or depth is None:
            logger.warning("[AutoFold] missing top_rgb or top_depth")
            return
        centroid_uv = self._segment_garment_centroid(np.asarray(rgb))
        if centroid_uv is None:
            logger.warning("[AutoFold] could not detect garment centroid")
            return
        centroid_world = self._pixel_to_world(centroid_uv[0], centroid_uv[1], depth)
        if centroid_world is None:
            logger.warning("[AutoFold] could not map centroid to world (depth invalid)")
            return

        state = np.asarray(observations.get("observation.state"), dtype=np.float32)
        if state.shape[0] < 12:
            logger.warning("[AutoFold] expected 12D bimanual state")
            return

        # Current gripper world positions via env body links
        left_pos = self.env.left_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()
        right_pos = self.env.right_arm.data.body_link_pos_w[0, -1].detach().cpu().numpy()

        lift = float(getattr(self.args, "click_ik_z_offset", 0.05)) + 0.03  # +3cm extra lift
        # Each arm's target = halfway between current and centroid (xy), lifted in z.
        # This brings both grippers gradually toward the body center, dragging cloth.
        def _target(grip_pos, centroid):
            tx = 0.5 * (grip_pos[0] + centroid[0])
            ty = 0.5 * (grip_pos[1] + centroid[1])
            tz = max(grip_pos[2], centroid[2]) + lift
            return np.array([tx, ty, tz], dtype=np.float32)

        target_left_world = _target(left_pos, centroid_world)
        target_right_world = _target(right_pos, centroid_world)

        gripper_l = GRIPPER_CLOSED  # held closed during fold
        gripper_r = GRIPPER_CLOSED

        joints_left = compute_joints_from_world_point(
            self.solver, self.env, "left", target_left_world,
            current_joints=state[0:6].copy(), state_unit="rad",
            gripper_angle=float(gripper_l),
        )
        joints_right = compute_joints_from_world_point(
            self.solver, self.env, "right", target_right_world,
            current_joints=state[6:12].copy(), state_unit="rad",
            gripper_angle=float(gripper_r),
        )
        if joints_left is None or joints_right is None:
            logger.warning(
                "[AutoFold] IK failed: left=%s right=%s",
                joints_left, joints_right,
            )
            return

        target = state.copy()
        target[0:6] = joints_left.astype(np.float32)
        target[6:12] = joints_right.astype(np.float32)
        steps = max(1, int(getattr(self.args, "click_ik_steps", 45)))
        self.trajectory = [
            (state + (target - state) * (i / steps)).astype(np.float32)
            for i in range(1, steps + 1)
        ]
        flags["policy_paused"] = True
        logger.info(
            "[AutoFold] Planned bimanual fold: L→(%.2f,%.2f,%.2f) R→(%.2f,%.2f,%.2f) steps=%d",
            *target_left_world.tolist(), *target_right_world.tolist(), steps,
        )

    def _plan_trajectory(
        self,
        click: Tuple[int, int],
        observations: Dict[str, Any],
        flags: Dict[str, Any],
        arm_override: Optional[str] = None,
        steps_override: Optional[int] = None,
    ) -> bool:
        from lehome.utils import compute_joints_from_world_point

        depth = observations.get("observation.top_depth")
        target_world = self._pixel_to_world(click[0], click[1], depth)
        if not self._valid_cloth_world_target(target_world, "pixel"):
            logger.warning("[ClickIK] Could not map click to depth/world point")
            return False

        state = np.asarray(observations.get("observation.state"), dtype=np.float32)
        if state.shape[0] < 12:
            logger.warning("[ClickIK] Expected 12D bimanual state, got shape=%s", state.shape)
            return False

        arm = arm_override if arm_override in {"left", "right"} else self._choose_arm(target_world)
        sl = slice(0, 6) if arm == "left" else slice(6, 12)
        current_arm = state[sl].copy()
        if flags.get("gripper_override") == "closed":
            gripper = GRIPPER_CLOSED
        elif flags.get("gripper_override") == "open":
            gripper = GRIPPER_OPEN
        else:
            gripper = current_arm[5]

        joints = compute_joints_from_world_point(
            self.solver,
            self.env,
            arm,
            target_world,
            current_joints=current_arm,
            state_unit="rad",
            gripper_angle=float(gripper),
        )
        if joints is None:
            logger.warning("[ClickIK] IK failed for arm=%s target=%s", arm, target_world)
            return False

        target = state.copy()
        target[sl] = joints.astype(np.float32)
        steps = max(1, int(steps_override or getattr(self.args, "click_ik_steps", 45)))
        self.trajectory = [
            (state + (target - state) * (i / steps)).astype(np.float32)
            for i in range(1, steps + 1)
        ]
        flags["policy_paused"] = True
        logger.info(
            "[ClickIK] Planned %s-arm move to world=(%.3f, %.3f, %.3f), steps=%d",
            arm,
            float(target_world[0]),
            float(target_world[1]),
            float(target_world[2]),
            steps,
        )
        return True


def create_dataset_if_needed(
    args: argparse.Namespace,
) -> Tuple[Optional[LeRobotDataset], Optional[Path], Optional[Any], bool]:
    """Create LeRobotDataset if recording is enabled.

    Args:
        args: Command-line arguments containing recording configuration.

    Returns:
        Tuple of (dataset, json_path, solver, is_bi_arm):
            - dataset: LeRobotDataset instance or None if not recording
            - json_path: Path to object initial pose JSON file or None
            - solver: RobotKinematics solver instance or None
            - is_bi_arm: Boolean indicating if dual-arm configuration

    Raises:
        ValueError: If record_ee_pose is enabled but ee_urdf_path is not provided.
        FileNotFoundError: If URDF file is not found.
    """
    if not args.enable_record:
        return None, None, None, False

    action_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    is_bi_arm = ("Bi" in (args.task or "")) or (
        getattr(args, "teleop_device", "") or ""
    ).startswith("bi-")

    if is_bi_arm:
        left_names = [f"left_{n}" for n in action_names]
        right_names = [f"right_{n}" for n in action_names]
        joint_names = left_names + right_names
    else:
        joint_names = action_names

    dim = len(joint_names)
    features: Dict[str, Dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dim,),
            "names": joint_names,
        },
    }

    if not getattr(args, "disable_depth", False):
        features["observation.top_depth"] = {
            "dtype": "uint16",
            "shape": (480, 640),
            "names": ["height", "width"],
            "info": {
                "unit": "millimeters",
                "range_mm": [0, 65535],
                "range_m": [0.0, 65.535],
                "precision_mm": 1,
                "conversion": "depth_meters = uint16_value / 1000.0"
            }
        }

    if is_bi_arm:
        image_keys = ["top_rgb", "left_rgb", "right_rgb"]
    else:
        image_keys = ["top_rgb", "wrist_rgb"]

    for key in image_keys:
        features[f"observation.images.{key}"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    if getattr(args, "record_ee_pose", False):
        if is_bi_arm:
            ee_pose_dim = 16
            ee_pose_names = [
                "left_x",
                "left_y",
                "left_z",
                "left_qx",
                "left_qy",
                "left_qz",
                "left_qw",
                "left_gripper",
                "right_x",
                "right_y",
                "right_z",
                "right_qx",
                "right_qy",
                "right_qz",
                "right_qw",
                "right_gripper",
            ]
        else:
            ee_pose_dim = 8
            ee_pose_names = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]

        features["observation.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_pose_dim,),
            "names": ee_pose_names,
        }
        features["action.ee_pose"] = {
            "dtype": "float32",
            "shape": (ee_pose_dim,),
            "names": ee_pose_names,
        }

    root_path = Path(getattr(args, "dataset_root", "Datasets/record"))

    dataset = LeRobotDataset.create(
        repo_id="abc",
        fps=30,
        root=get_next_experiment_path_with_gap(root_path),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=0,
        features=features,
    )
    json_path = dataset.root / "meta" / "garment_info.json"

    solver = None
    if getattr(args, "record_ee_pose", False):
        if not args.ee_urdf_path:
            raise ValueError("--record_ee_pose requires --ee_urdf_path")

        urdf_path = Path(args.ee_urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        from lehome.utils import RobotKinematics

        if is_bi_arm:
            solver_joint_names = [n.replace("left_", "") for n in joint_names[:5]]
        else:
            solver_joint_names = joint_names[:5]

        solver = RobotKinematics(
            str(urdf_path),
            target_frame_name="gripper_frame_link",
            joint_names=solver_joint_names,
        )
        arm_type = "dual-arm" if is_bi_arm else "single-arm"
        logger.info(f"End-effector pose solver loaded ({arm_type})")

    return dataset, json_path, solver, is_bi_arm


def run_idle_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    count_render: int,
) -> Tuple[Optional[Dict[str, Any]], int]:
    """Run idle phase before recording starts.

    Handles environment preparation, stabilization, and waits for user to press
    S key to start recording.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
        count_render: Current render count.

    Returns:
        Tuple of (object_initial_pose, updated_count_render).
    """
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)

    actions = teleop_interface.advance()
    object_initial_pose = None

    if count_render == 0:
        logger.info("[Idle Phase] Initializing observations...")
        env.initialize_obs()
        count_render += 1

        logger.info("[Idle Phase] Stabilizing garment after initialization...")
        stabilize_garment_after_reset(env, args)
        logger.info("[Idle Phase] Ready for recording")

    if actions is None:
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = (
                    torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
                )
            else:
                maintain_action = torch.zeros(
                    1, len(current_state), dtype=torch.float32, device=env.device
                )
        else:
            action_dim = 12 if "Bi" in args.task else 6
            maintain_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=env.device
            )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)
        object_initial_pose = env.get_all_pose()

    if object_initial_pose is None:
        object_initial_pose = env.get_all_pose()

    return object_initial_pose, count_render


def run_recording_phase(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
    flags: Dict[str, bool],
    dataset: LeRobotDataset,
    json_path: Path,
    initial_object_pose: Optional[Dict[str, Any]],
    ee_solver: Optional[Any] = None,
    is_bi_arm: bool = False,
    assist_policy: Optional[Any] = None,
    click_ik: Optional[ClickIKController] = None,
    scripted_oracle: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run recording phase after S key is pressed and recording is enabled.

    Records episodes until num_episode is reached. Each episode can be marked as
    successful (N key), discarded (D key), or aborted (ESC key).

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
        flags: Status flags dictionary.
        dataset: LeRobotDataset instance.
        json_path: Path to object initial pose JSON file.
        initial_object_pose: Initial object pose dictionary.
        ee_solver: Optional kinematic solver for end-effector pose computation.
        is_bi_arm: Whether using dual-arm configuration.

    Returns:
        Final object initial pose dictionary.
    """
    episode_index = 0
    attempt_index = 0
    object_initial_pose = initial_object_pose
    gripper_override_log_step = 0
    score_probe_steps = _parse_probe_steps(getattr(args, "score_probe_steps", ""))
    early_restart_schedule = _parse_early_restart_schedule(
        getattr(args, "early_restart_schedule", "")
    )

    # Ensure we have a valid initial pose for the first episode
    if object_initial_pose is None:
        object_initial_pose = env.get_all_pose()

    while episode_index < args.num_episode:
        # Check if recording should be aborted
        if flags["abort"]:
            dataset.clear_episode_buffer()
            dataset.finalize()
            logger.warning(f"Recording aborted, completed {episode_index} episodes")
            return object_initial_pose

        flags["success"] = False
        flags["remove"] = False
        flags["restart"] = False
        flags["manual"] = False
        flags["policy_paused"] = False
        flags["gripper_override"] = None
        flags["auto_fold_request"] = False
        flags["visual_repair_request"] = False
        flags["last_motion_state"] = None
        flags["last_motion_delta"] = None
        episode_step = 0
        auto_visual_repair_attempts_used = 0
        last_auto_visual_repair_step = -1
        early_restart_checked = False
        early_restart_schedule_checked: set[int] = set()
        score_probe_seen_steps: set[int] = set()
        auto_save_candidate_step: Optional[int] = None

        # Loop within a single episode
        while not flags["success"]:
            # Check if recording should be aborted
            if flags["abort"]:
                dataset.clear_episode_buffer()
                dataset.finalize()
                logger.warning(f"Recording aborted, completed {episode_index} episodes")
                return object_initial_pose

            oracle_action_np = None
            click_action_np = None
            try:
                dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
                pre_observations = env._get_observations()
                pre_observations.pop("policy", None)
                current_state_for_motion = np.asarray(
                    pre_observations.get("observation.state"),
                    dtype=np.float32,
                )
                if (
                    current_state_for_motion.shape[0] >= 12
                    and not flags.get("policy_paused")
                    and oracle_action_np is None
                    and click_action_np is None
                ):
                    previous_motion_state = flags.get("last_motion_state")
                    if previous_motion_state is not None:
                        previous_motion_state = np.asarray(previous_motion_state, dtype=np.float32)
                        if previous_motion_state.shape[0] >= 12:
                            delta = current_state_for_motion - previous_motion_state
                            # Store actual arm motion only; gripper deltas are controlled separately.
                            delta[LEFT_GRIPPER_IDX] = 0.0
                            delta[RIGHT_GRIPPER_IDX] = 0.0
                            flags["last_motion_delta"] = delta.copy()
                    flags["last_motion_state"] = current_state_for_motion.copy()
                auto_repair_step = int(getattr(args, "auto_visual_repair_step", -1))
                max_auto_repair_attempts = max(
                    1, int(getattr(args, "auto_visual_repair_attempts", 3))
                )
                auto_repair_settle_steps = max(
                    1, int(getattr(args, "auto_visual_repair_settle_steps", 40))
                )
                should_trigger_auto_repair = False
                auto_repair_reason = "initial"
                if (
                    auto_repair_step >= 0
                    and click_ik is not None
                    and auto_visual_repair_attempts_used < max_auto_repair_attempts
                ):
                    if (
                        auto_visual_repair_attempts_used == 0
                        and episode_step >= auto_repair_step
                    ):
                        should_trigger_auto_repair = True
                    elif (
                        auto_visual_repair_attempts_used > 0
                        and not click_ik.is_busy()
                        and last_auto_visual_repair_step >= 0
                    ):
                        previous_plan_len = max(
                            1, int(getattr(click_ik, "last_visual_repair_plan_len", 1))
                        )
                        retry_step = (
                            last_auto_visual_repair_step
                            + previous_plan_len
                            + auto_repair_settle_steps
                        )
                        if episode_step >= retry_step:
                            if bool(env._get_success()):
                                flags["success"] = True
                                logger.info(
                                    "[AutoVisualRepair] Success checker passed after attempt %d; saving.",
                                    auto_visual_repair_attempts_used,
                                )
                                continue
                            should_trigger_auto_repair = True
                            auto_repair_reason = "retry"

                if should_trigger_auto_repair:
                    flags["visual_repair_request"] = True
                    flags["policy_paused"] = True
                    auto_visual_repair_attempts_used += 1
                    last_auto_visual_repair_step = episode_step
                    try:
                        click_ik._save_visual_debug(
                            pre_observations,
                            f"attempt{auto_visual_repair_attempts_used:02d}_{auto_repair_reason}_pre",
                        )
                    except Exception:
                        pass
                    logger.info(
                        "[AutoVisualRepair] Triggering visual repair attempt %d/%d at step %d (%s)",
                        auto_visual_repair_attempts_used,
                        max_auto_repair_attempts,
                        episode_step,
                        auto_repair_reason,
                    )
                elif (
                    auto_repair_step >= 0
                    and click_ik is not None
                    and auto_visual_repair_attempts_used >= max_auto_repair_attempts
                    and not click_ik.is_busy()
                    and last_auto_visual_repair_step >= 0
                ):
                    previous_plan_len = max(
                        1, int(getattr(click_ik, "last_visual_repair_plan_len", 1))
                    )
                    decision_step = (
                        last_auto_visual_repair_step
                        + previous_plan_len
                        + auto_repair_settle_steps
                    )
                    if episode_step >= decision_step:
                        if bool(env._get_success()):
                            flags["success"] = True
                            logger.info(
                                "[AutoVisualRepair] Success checker passed after final attempt; saving."
                            )
                            continue
                        flags["restart"] = True
                        logger.info(
                            "[AutoVisualRepair] Final attempt failed by step %d; restarting episode.",
                            episode_step,
                        )
                if click_ik is not None:
                    click_ik.update_view(pre_observations, flags)

                oracle_action_np = (
                    scripted_oracle.next_action(pre_observations, flags)
                    if scripted_oracle is not None
                    else None
                )
                click_action_np = (
                    click_ik.next_action(pre_observations, flags)
                    if click_ik is not None and oracle_action_np is None
                    else None
                )

                if oracle_action_np is not None:
                    actions = (
                        torch.from_numpy(oracle_action_np).float().unsqueeze(0).to(env.device)
                    )
                elif click_action_np is not None:
                    action_np = apply_assisted_gripper_override(click_action_np, flags)
                    actions = torch.from_numpy(action_np).float().unsqueeze(0).to(env.device)
                elif assist_policy is not None and not flags["manual"]:
                    if flags["policy_paused"]:
                        state = pre_observations.get("observation.state")
                        action_np = np.asarray(state, dtype=np.float32)
                    else:
                        action_np = assist_policy.select_action(pre_observations).astype(np.float32)
                    action_np = apply_assisted_gripper_override(action_np, flags)
                    if flags.get("gripper_override") is not None:
                        gripper_override_log_step += 1
                        if gripper_override_log_step % 30 == 1:
                            state = np.asarray(
                                pre_observations.get("observation.state"),
                                dtype=np.float32,
                            )
                            logger.info(
                                "[AssistGrip] mode=%s state=(%.3f, %.3f) cmd=(%.3f, %.3f)",
                                flags["gripper_override"],
                                float(state[LEFT_GRIPPER_IDX]),
                                float(state[RIGHT_GRIPPER_IDX]),
                                float(action_np[LEFT_GRIPPER_IDX]),
                                float(action_np[RIGHT_GRIPPER_IDX]),
                            )
                    actions = torch.from_numpy(action_np).float().unsqueeze(0).to(env.device)
                else:
                    actions = teleop_interface.advance()
                    if assist_policy is not None and actions is None:
                        state = pre_observations.get("observation.state")
                        actions = torch.from_numpy(state).float().unsqueeze(0).to(env.device)
            except Exception as e:
                logger.error(f"[Recording] Error in teleop interface: {e}")
                traceback.print_exc()
                actions = None

            physics_soft_paused = (
                flags.get("policy_paused")
                and not flags.get("manual")
                and oracle_action_np is None
                and click_action_np is None
                and not flags.get("restart")
                and not flags.get("remove")
                and not flags.get("success")
            )
            if physics_soft_paused:
                # A real pause for cloth physics: do not call env.step().
                # Keep rendering/polling so LiveIK commands can plan the next move.
                force_gripper_joint_state_to_sim(env, flags.get("gripper_override"))
                env.render()
                time.sleep(0.02)
                continue

            if actions is None:
                env.render()
            else:
                actions = apply_auto_grip_timing(actions, args, episode_step)
                env.step(actions)

            if args.log_success or getattr(args, "auto_save_success", False):
                raw_success_result = _raw_garment_success_result(env)
                success = bool(raw_success_result.get("success", False))
                if (
                    episode_step in score_probe_steps
                    and episode_step not in score_probe_seen_steps
                ):
                    score_probe_seen_steps.add(episode_step)
                    metrics = _success_progress_metrics(raw_success_result)
                    _write_score_probe(
                        args,
                        episode_index=episode_index,
                        attempt_index=attempt_index,
                        episode_step=episode_step,
                        outcome="probe",
                        result=raw_success_result,
                    )
                    logger.info(
                        "[ScoreProbe] attempt=%d step=%d passed=%d/%d %s",
                        attempt_index,
                        episode_step,
                        int(metrics["passed_count"]),
                        int(metrics["total_count"]),
                        _format_success_details(raw_success_result),
                    )
                    near_miss_release_ok = True
                    near_miss_release_reason = "release gate disabled"
                    if bool(getattr(args, "auto_save_near_miss_require_release", False)):
                        near_miss_release_ok, near_miss_release_reason = (
                            _gripper_release_gate(env, pre_observations, args)
                        )
                    near_miss_ok = (
                        bool(getattr(args, "auto_save_near_miss", False))
                        and not bool(success)
                        and int(metrics["passed_count"])
                        >= int(getattr(args, "auto_save_near_miss_min_passed", 4))
                        and np.isfinite(float(metrics["worst_close_ratio"]))
                        and float(metrics["worst_close_ratio"])
                        <= float(
                            getattr(
                                args,
                                "auto_save_near_miss_max_worst_close_ratio",
                                1.10,
                            )
                        )
                        and near_miss_release_ok
                    )
                    if near_miss_ok:
                        _write_score_probe(
                            args,
                            episode_index=episode_index,
                            attempt_index=attempt_index,
                            episode_step=episode_step,
                            outcome="near_miss_save_probe",
                            result=raw_success_result,
                        )
                        flags["success"] = True
                        logger.info(
                            "[AutoNearMissSave] Saving probe near-miss at step %d: "
                            "passed=%d/%d worst_close_ratio=%.2f (%s).",
                            episode_step,
                            int(metrics["passed_count"]),
                            int(metrics["total_count"]),
                            float(metrics["worst_close_ratio"]),
                            near_miss_release_reason,
                        )
                for rule in early_restart_schedule:
                    rule_step = int(rule["step"])
                    if (
                        rule_step in early_restart_schedule_checked
                        or episode_step < rule_step
                        or bool(success)
                        or flags.get("success")
                        or flags.get("restart")
                        or flags.get("remove")
                    ):
                        continue
                    early_restart_schedule_checked.add(rule_step)
                    metrics = _success_progress_metrics(raw_success_result)
                    best_close_ratio = float(metrics["best_close_ratio"])
                    passed_count = int(metrics["passed_count"])
                    close_passed_count = int(metrics["close_passed_count"])
                    min_passed = int(rule["min_passed"])
                    min_close_passed = int(rule["min_close_passed"])
                    close_ratio_limit = float(rule["best_close_ratio"])
                    should_restart_stage = (
                        (min_passed >= 0 and passed_count < min_passed)
                        or (
                            min_close_passed >= 0
                            and close_passed_count < min_close_passed
                        )
                        or (
                            np.isfinite(close_ratio_limit)
                            and np.isfinite(best_close_ratio)
                            and best_close_ratio > close_ratio_limit
                        )
                    )
                    logger.info(
                        "[EarlyRestartStage] step=%d passed=%d close_passed=%d "
                        "best_close_ratio=%.2f limits=(min_passed=%d, "
                        "min_close_passed=%d, best_close<=%.2f) decision=%s",
                        episode_step,
                        passed_count,
                        close_passed_count,
                        best_close_ratio,
                        min_passed,
                        min_close_passed,
                        close_ratio_limit,
                        "restart" if should_restart_stage else "continue",
                    )
                    if should_restart_stage:
                        flags["restart"] = True
                        break

                if (
                    int(getattr(args, "early_restart_step", -1)) >= 0
                    and not early_restart_checked
                    and episode_step >= int(getattr(args, "early_restart_step", -1))
                    and not bool(success)
                    and not flags.get("success")
                    and not flags.get("restart")
                    and not flags.get("remove")
                ):
                    early_restart_checked = True
                    progress_result = raw_success_result
                    metrics = _success_progress_metrics(progress_result)
                    best_close_ratio = float(metrics["best_close_ratio"])
                    passed_count = int(metrics["passed_count"])
                    close_ratio_limit = float(
                        getattr(args, "early_restart_close_ratio", 3.0)
                    )
                    min_passed = int(getattr(args, "early_restart_min_passed", 0))
                    should_restart_early = (
                        passed_count < min_passed
                        or (
                            np.isfinite(best_close_ratio)
                            and best_close_ratio > close_ratio_limit
                        )
                    )
                    logger.info(
                        "[EarlyRestart] step=%d passed=%d best_close_ratio=%.2f "
                        "limits=(min_passed=%d, close_ratio<=%.2f) decision=%s",
                        episode_step,
                        passed_count,
                        best_close_ratio,
                        min_passed,
                        close_ratio_limit,
                        "restart" if should_restart_early else "continue",
                    )
                    if should_restart_early:
                        flags["restart"] = True
                if (
                    getattr(args, "auto_save_success", False)
                    and episode_step >= int(getattr(args, "auto_save_min_steps", 120))
                    and success
                    and not flags.get("success")
                ):
                    release_ok, release_reason = _gripper_release_gate(
                        env, pre_observations, args
                    )
                    if release_ok:
                        settle_steps = max(
                            0,
                            int(getattr(args, "auto_save_success_settle_steps", 0)),
                        )
                        if auto_save_candidate_step is None:
                            auto_save_candidate_step = episode_step
                            if settle_steps > 0:
                                logger.info(
                                    "[AutoSaveGate] success+release passed at step %d "
                                    "(%s); waiting %d settle steps before saving.",
                                    episode_step,
                                    release_reason,
                                    settle_steps,
                                )
                        waited_steps = episode_step - auto_save_candidate_step
                        if waited_steps >= settle_steps:
                            _write_score_probe(
                                args,
                                episode_index=episode_index,
                                attempt_index=attempt_index,
                                episode_step=episode_step,
                                outcome="save",
                                result=raw_success_result,
                            )
                            flags["success"] = True
                            logger.info(
                                "[AutoSave] Success checker and release gate held for "
                                "%d settle steps at step %d (%s); saving episode.",
                                waited_steps,
                                episode_step,
                                release_reason,
                            )
                    elif episode_step in score_probe_steps:
                        auto_save_candidate_step = None
                        logger.info(
                            "[AutoSaveGate] step=%d success true but waiting: %s",
                            episode_step,
                            release_reason,
                        )
                else:
                    auto_save_candidate_step = None
                if (
                    int(getattr(args, "auto_restart_fail_steps", -1)) >= 0
                    and episode_step >= int(getattr(args, "auto_restart_fail_steps", -1))
                    and not bool(success)
                    and not flags.get("success")
                    and not flags.get("restart")
                    and not flags.get("remove")
                ):
                    final_result = raw_success_result
                    metrics = _success_progress_metrics(final_result)
                    near_miss_release_ok = True
                    near_miss_release_reason = "release gate disabled"
                    if bool(getattr(args, "auto_save_near_miss_require_release", False)):
                        near_miss_release_ok, near_miss_release_reason = (
                            _gripper_release_gate(env, pre_observations, args)
                        )
                    near_miss_ok = (
                        bool(getattr(args, "auto_save_near_miss", False))
                        and int(metrics["passed_count"])
                        >= int(getattr(args, "auto_save_near_miss_min_passed", 4))
                        and np.isfinite(float(metrics["worst_close_ratio"]))
                        and float(metrics["worst_close_ratio"])
                        <= float(
                            getattr(
                                args,
                                "auto_save_near_miss_max_worst_close_ratio",
                                1.10,
                            )
                        )
                        and near_miss_release_ok
                    )
                    if near_miss_ok:
                        _write_score_probe(
                            args,
                            episode_index=episode_index,
                            attempt_index=attempt_index,
                            episode_step=episode_step,
                            outcome="near_miss_save",
                            result=final_result,
                        )
                        flags["success"] = True
                        logger.info(
                            "[AutoNearMissSave] Saving near-miss at step %d: "
                            "passed=%d/%d worst_close_ratio=%.2f (%s).",
                            episode_step,
                            int(metrics["passed_count"]),
                            int(metrics["total_count"]),
                            float(metrics["worst_close_ratio"]),
                            near_miss_release_reason,
                        )
                        continue
                    _write_score_probe(
                        args,
                        episode_index=episode_index,
                        attempt_index=attempt_index,
                        episode_step=episode_step,
                        outcome="restart",
                        result=final_result,
                    )
                    flags["restart"] = True
                    logger.info(
                        "[AutoRestart] No success by step %d; discarding and retrying.",
                        episode_step,
                    )
                    logger.info(
                        "[AutoRestartDetail] passed=%d/%d best_close_ratio=%.2f %s",
                        int(metrics["passed_count"]),
                        int(metrics["total_count"]),
                        float(metrics["best_close_ratio"]),
                        _format_success_details(final_result),
                    )

            observations = env._get_observations()
            observations.pop("policy", None)
            if (
                getattr(args, "disable_depth", False)
                and "observation.top_depth" in observations
            ):
                observations.pop("observation.top_depth")

            if getattr(args, "enable_pointcloud", False):
                # Converting pointcloud online is time-consuming, please convert offline
                # pointcloud = env._get_workspace_pointcloud(
                #     num_points=4096, use_fps=True
                # )
                print("Converting pointcloud online is time-consuming, please convert offline")
            _, truncated = env._get_dones()
            frame = {**observations, "task": args.task_description}

            if (
                ee_solver is not None
                and "observation.state" in observations
                and "action" in observations
            ):
                from lehome.utils import compute_ee_pose_single_arm

                obs_state = np.array(
                    observations["observation.state"], dtype=np.float32
                )
                action_state = np.array(observations["action"], dtype=np.float32)

                if is_bi_arm:
                    obs_left = compute_ee_pose_single_arm(
                        ee_solver, obs_state[:6], args.ee_state_unit
                    )
                    obs_right = compute_ee_pose_single_arm(
                        ee_solver, obs_state[6:12], args.ee_state_unit
                    )
                    frame["observation.ee_pose"] = np.concatenate(
                        [obs_left, obs_right], axis=0
                    )

                    act_left = compute_ee_pose_single_arm(
                        ee_solver, action_state[:6], args.ee_state_unit
                    )
                    act_right = compute_ee_pose_single_arm(
                        ee_solver, action_state[6:12], args.ee_state_unit
                    )
                    frame["action.ee_pose"] = np.concatenate(
                        [act_left, act_right], axis=0
                    )
                else:
                    frame["observation.ee_pose"] = compute_ee_pose_single_arm(
                        ee_solver, obs_state, args.ee_state_unit
                    )
                    frame["action.ee_pose"] = compute_ee_pose_single_arm(
                        ee_solver, action_state, args.ee_state_unit
                    )

            dataset.add_frame(frame)
            episode_step += 1

            if not flags["success"] and (truncated or flags["remove"] or flags["restart"]):
                dataset.clear_episode_buffer()
                reason = "restart" if flags["restart"] else "discard/timeout"
                logger.info(f"Re-recording episode {episode_index} ({reason})")
                try:
                    env.reset()
                    if assist_policy is not None:
                        assist_policy.reset()
                    if click_ik is not None:
                        click_ik.reset()
                    if scripted_oracle is not None:
                        scripted_oracle.reset()
                    teleop_interface.reset()
                    flags["manual"] = False
                    flags["policy_paused"] = False
                    flags["gripper_override"] = None
                    flags["auto_fold_request"] = False
                    flags["visual_repair_request"] = False
                    flags["keyboard_locked_after_pause"] = False
                    flags["last_motion_state"] = None
                    flags["last_motion_delta"] = None
                    episode_step = 0
                    attempt_index += 1
                    max_attempts = int(getattr(args, "max_attempts_per_episode", -1))
                    if max_attempts > 0 and attempt_index >= max_attempts:
                        dataset.clear_episode_buffer()
                        dataset.finalize()
                        logger.warning(
                            "[Recording] Max attempts reached for episode %d "
                            "(attempts=%d); aborting this recording run.",
                            episode_index,
                            attempt_index,
                        )
                        return object_initial_pose
                    early_restart_checked = False
                    early_restart_schedule_checked = set()
                    score_probe_seen_steps = set()
                    auto_save_candidate_step = None
                    auto_visual_repair_attempts_used = 0
                    last_auto_visual_repair_step = -1
                    stabilize_garment_after_reset(env, args)
                    object_initial_pose = env.get_all_pose()
                except Exception as e:
                    logger.error(
                        f"[Recording] Failed to reset environment during re-recording: {e}"
                    )
                    traceback.print_exc()
                    try:
                        object_initial_pose = env.get_all_pose()
                    except Exception:
                        object_initial_pose = None
                flags["remove"] = False
                flags["restart"] = False
                continue

        save_start_time = time.time()
        logger.info(f"[Recording] Saving episode {episode_index}...")
        try:
            dataset.save_episode()
            save_duration = time.time() - save_start_time
            logger.info(
                f"[Recording] Episode {episode_index} saved (took {save_duration:.1f}s)"
            )
        except Exception as e:
            logger.error(f"[Recording] Failed to save episode {episode_index}: {e}")
            traceback.print_exc()

        garment_name = None
        if hasattr(env, "cfg") and hasattr(env.cfg, "garment_name"):
            garment_name = env.cfg.garment_name

        scale = None
        if hasattr(env, "object") and hasattr(env.object, "init_scale"):
            try:
                scale = env.object.init_scale
            except Exception:
                logger.warning("Failed to get scale from garment object")

        try:
            append_episode_initial_pose(
                json_path,
                episode_index,
                object_initial_pose,
                garment_name=garment_name,
                scale=scale,
            )
        except Exception as e:
            logger.error(
                f"[Recording] Failed to save episode metadata for episode {episode_index}: {e}"
            )
            traceback.print_exc()

        episode_index += 1
        logger.info(
            f"Episode {episode_index - 1} completed, progress: {episode_index}/{args.num_episode}"
        )

        try:
            env.reset()
            if assist_policy is not None:
                assist_policy.reset()
            if click_ik is not None:
                click_ik.reset()
            teleop_interface.reset()
            if scripted_oracle is not None:
                scripted_oracle.reset()
            flags["manual"] = False
            flags["policy_paused"] = False
            flags["gripper_override"] = None
            flags["auto_fold_request"] = False
            flags["visual_repair_request"] = False
            flags["keyboard_locked_after_pause"] = False
            flags["last_motion_state"] = None
            flags["last_motion_delta"] = None
            stabilize_garment_after_reset(env, args)
        except Exception as e:
            logger.error(f"[Recording] Failed to reset environment: {e}")
            traceback.print_exc()

        try:
            object_initial_pose = env.get_all_pose()
        except Exception as e:
            logger.error(f"[Recording] Failed to get initial pose: {e}")
            traceback.print_exc()
            object_initial_pose = None
    dataset.clear_episode_buffer()
    dataset.finalize()
    logger.info(f"All {args.num_episode} episodes recording completed!")
    return object_initial_pose


def run_live_control_without_record(
    env: DirectRLEnv,
    teleop_interface: Any,
    args: argparse.Namespace,
) -> None:
    """Run live teleoperation control without recording.

    Handles the case when S key is pressed but recording is not enabled.
    Performs simple teleoperation control without writing to dataset.

    Args:
        env: Environment instance.
        teleop_interface: Teleoperation interface.
        args: Command-line arguments.
    """
    dynamic_reset_gripper_effort_limit_sim(env, args.teleop_device)
    actions = teleop_interface.advance()

    if actions is None:
        current_obs = env._get_observations()
        if "observation.state" in current_obs:
            current_state = current_obs["observation.state"]
            if isinstance(current_state, np.ndarray):
                maintain_action = (
                    torch.from_numpy(current_state).float().unsqueeze(0).to(env.device)
                )
            else:
                maintain_action = torch.zeros(
                    1, len(current_state), dtype=torch.float32, device=env.device
                )
        else:
            action_dim = 12 if "Bi" in args.task else 6
            maintain_action = torch.zeros(
                1, action_dim, dtype=torch.float32, device=env.device
            )
        env.step(maintain_action)
        env.render()
    else:
        env.step(actions)

    if args.log_success:
        _ = env._get_success()


def record_dataset(args: argparse.Namespace, simulation_app: SimulationApp) -> None:
    """Record dataset."""
    # Get device configuration (default to "cpu" for compatibility)
    device = getattr(args, "device", "cpu")

    env_cfg = parse_env_cfg(
        args.task,
        device=device,
    )
    task_name = args.task

    env_cfg.garment_name = args.garment_name
    env_cfg.garment_version = args.garment_version
    env_cfg.garment_cfg_base_path = args.garment_cfg_base_path
    env_cfg.particle_cfg_path = args.particle_cfg_path

    if args.use_random_seed:
        env_cfg.use_random_seed = True
        logger.info("Using random seed (no fixed seed)")
    else:
        env_cfg.use_random_seed = False
        env_cfg.random_seed = args.seed
        logger.info(f"Using fixed random seed: {args.seed}")

    env: DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
    teleop_interface = create_teleop_interface(env, args)
    flags = register_teleop_callbacks(
        teleop_interface, recording_enabled=args.enable_record, args=args
    )
    flags["unattended_auto"] = bool(getattr(args, "auto_save_success", False)) and (
        bool(getattr(args, "auto_start_record", False))
        or int(getattr(args, "auto_restart_fail_steps", -1)) >= 0
        or int(getattr(args, "auto_visual_repair_step", -1)) >= 0
    )
    teleop_interface.reset()
    assist_policy = create_assist_policy(args)
    click_ik = ClickIKController(env, args) if getattr(args, "enable_click_ik", False) else None
    scripted_oracle = None
    if getattr(args, "scripted_oracle", None):
        try:
            from .oracle_fold import OracleFolder
            cat = args.scripted_oracle if args.scripted_oracle != "auto" else None
            scripted_oracle = OracleFolder(env, args, category=cat)
            logger.info(
                f"[Oracle] Scripted fold oracle enabled (category={scripted_oracle.category})"
            )
        except Exception as e:
            logger.error(f"[Oracle] Failed to initialize; falling back to teleop: {e}")
            traceback.print_exc()
            scripted_oracle = None
    dataset, json_path, ee_solver, is_bi_arm = create_dataset_if_needed(args)
    count_render = 0
    printed_instructions = False
    idle_frame_counter = 0
    object_initial_pose: Optional[Dict[str, Any]] = None

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                if not flags["start"]:
                    pose, count_render = run_idle_phase(
                        env,
                        teleop_interface,
                        args,
                        count_render,
                    )
                    if pose is not None:
                        object_initial_pose = pose

                    if count_render > 0:
                        idle_frame_counter += 1
                        if idle_frame_counter == 100 and not printed_instructions:
                            logger.info("=" * 60)
                            logger.info("🎮 CONTROL INSTRUCTIONS 🎮")
                            logger.info("=" * 60)
                            logger.info(str(teleop_interface))
                            logger.info("=" * 60 + "\n\n")
                            printed_instructions = True
                        # Auto-start when scripted oracle is in use: skip
                        # waiting for human keypress once init/stabilize is done.
                        if (
                            (
                                scripted_oracle is not None
                                or int(getattr(args, "auto_visual_repair_step", -1)) >= 0
                                or bool(getattr(args, "auto_start_record", False))
                            )
                            and not flags["start"]
                            and idle_frame_counter >= 5
                        ):
                            flags["start"] = True
                            logger.info("[Auto] Auto-starting recording phase")
                elif args.enable_record and dataset is not None:
                    object_initial_pose = run_recording_phase(
                        env,
                        teleop_interface,
                        args,
                        flags,
                        dataset,
                        json_path,
                        object_initial_pose,
                        ee_solver,
                        is_bi_arm,
                        assist_policy,
                        click_ik,
                        scripted_oracle,
                    )
                    break
                else:
                    run_live_control_without_record(env, teleop_interface, args)
    except KeyboardInterrupt:
        logger.warning("\n[Ctrl+C] Interrupt signal detected")
        # If Ctrl+C is pressed during recording, clear the current buffer
        if args.enable_record and dataset is not None and flags["start"]:
            logger.info("Clearing current episode buffer...")
            dataset.clear_episode_buffer()
            logger.info("Buffer cleared, dataset remains intact")
            dataset.finalize()
            logger.info("Dataset saved")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    finally:
        if click_ik is not None:
            click_ik.close()
        env.close()
