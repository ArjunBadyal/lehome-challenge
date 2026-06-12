"""Calibrate PolicyStabilizer thresholds from Top-Short demo data.

Extracts three data-driven thresholds:
1. max_joint_delta: 95th percentile of per-step |Δ| per arm joint
2. arm_vel_threshold (release delay): 90th percentile of arm speed at release
3. gripper_empty_threshold: midpoint between "commanded-closed with cloth" and
   "commanded-closed empty" gripper state clusters

Usage:
    python scripts/calibrate_stabilizer.py

Output: prints thresholds to stdout, writes to outputs/stabilizer_thresholds.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "Datasets" / "example" / "top_short_merged"
OUT_PATH = REPO_ROOT / "outputs" / "stabilizer_thresholds.json"

LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11
LEFT_ARM_INDICES = [0, 1, 2, 3, 4]
RIGHT_ARM_INDICES = [6, 7, 8, 9, 10]
JOINT_NAMES = [
    "L_shoulder_pan", "L_shoulder_lift", "L_elbow", "L_wrist_flex", "L_wrist_roll", "L_gripper",
    "R_shoulder_pan", "R_shoulder_lift", "R_elbow", "R_wrist_flex", "R_wrist_roll", "R_gripper",
]
N_FRAMES = 20000  # enough to cover ~60 episodes worth of transitions


def main():
    print(f"[INFO] Loading dataset from {DATASET_ROOT}...", flush=True)
    ds = LeRobotDataset(repo_id="lehome", root=str(DATASET_ROOT))
    limit = min(len(ds), N_FRAMES)
    print(f"[INFO] Scanning {limit} frames", flush=True)

    left_g, right_g = [], []
    cmd_left_g, cmd_right_g = [], []
    joint_deltas = []
    arm_speeds = []
    release_arm_speeds = []  # arm speed at moments when action commands gripper open

    prev_state = None
    prev_action = None
    for i in range(limit):
        f = ds[i]
        state = f["observation.state"].numpy()
        action = f["action"].numpy()
        left_g.append(state[LEFT_GRIPPER_IDX])
        right_g.append(state[RIGHT_GRIPPER_IDX])
        cmd_left_g.append(action[LEFT_GRIPPER_IDX])
        cmd_right_g.append(action[RIGHT_GRIPPER_IDX])

        if prev_state is not None and f["frame_index"].item() > 0:
            delta = np.abs(state[:12] - prev_state[:12])
            joint_deltas.append(delta)
            left_arm_speed = delta[LEFT_ARM_INDICES].max()
            right_arm_speed = delta[RIGHT_ARM_INDICES].max()
            arm_speeds.append((left_arm_speed, right_arm_speed))

            # Release events: commanded transition from closed to open
            if prev_action is not None:
                if (action[LEFT_GRIPPER_IDX] > 0.3) and (prev_action[LEFT_GRIPPER_IDX] < 0.1):
                    release_arm_speeds.append(left_arm_speed)
                if (action[RIGHT_GRIPPER_IDX] > 0.3) and (prev_action[RIGHT_GRIPPER_IDX] < 0.1):
                    release_arm_speeds.append(right_arm_speed)

        prev_state = state
        prev_action = action

    left_g = np.array(left_g)
    right_g = np.array(right_g)
    cmd_left_g = np.array(cmd_left_g)
    cmd_right_g = np.array(cmd_right_g)
    joint_deltas = np.array(joint_deltas)
    arm_speeds = np.array(arm_speeds)
    release_arm_speeds = np.array(release_arm_speeds) if release_arm_speeds else np.array([])

    print(f"[INFO] Collected {len(joint_deltas)} transitions, {len(release_arm_speeds)} release events")

    # === 1. Rate limit: per-joint 95th percentile ===
    print("\n=== 1. Rate limit (per-joint |Δ| 95th percentile) ===")
    per_joint_p95 = {}
    for j in range(12):
        p95 = float(np.percentile(joint_deltas[:, j], 95))
        p99 = float(np.percentile(joint_deltas[:, j], 99))
        per_joint_p95[JOINT_NAMES[j]] = p95
        print(f"  {JOINT_NAMES[j]:20s}: p95={p95:.4f}  p99={p99:.4f}")

    arm_p95 = [per_joint_p95[JOINT_NAMES[j]] for j in LEFT_ARM_INDICES + RIGHT_ARM_INDICES]
    scalar_cap = float(np.max(arm_p95))
    print(f"\n  scalar max over arm joints (p95): {scalar_cap:.4f}")
    # Grippers change faster, so use separate cap
    gripper_p95 = max(per_joint_p95["L_gripper"], per_joint_p95["R_gripper"])
    print(f"  scalar max for grippers (p95):    {gripper_p95:.4f}")

    # === 2. Release-delay threshold ===
    print("\n=== 2. Release-delay threshold (arm speed @ release, 90th percentile) ===")
    if len(release_arm_speeds) > 0:
        release_p90 = float(np.percentile(release_arm_speeds, 90))
        release_p50 = float(np.percentile(release_arm_speeds, 50))
        release_p95 = float(np.percentile(release_arm_speeds, 95))
        print(f"  n={len(release_arm_speeds)}  p50={release_p50:.4f}  p90={release_p90:.4f}  p95={release_p95:.4f}")
    else:
        release_p90 = 0.08
        print(f"  No release events found. Defaulting to 0.08")

    # === 3. Gripper empty vs cloth-grasp threshold ===
    print("\n=== 3. Gripper empty vs cloth-grasp threshold ===")
    # "commanded closed" = action < 0.1; look at stable values after stability check
    # For simplicity, just look at the distribution of states when command is close
    for side, state_arr, cmd_arr in [("L", left_g, cmd_left_g), ("R", right_g, cmd_right_g)]:
        closed_mask = cmd_arr < 0.1
        states_when_closed = state_arr[closed_mask]
        print(f"  {side} gripper state distribution when cmd<0.1 (n={closed_mask.sum()}):")
        if len(states_when_closed):
            for lo, hi in [(0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 0.50)]:
                frac = ((states_when_closed >= lo) & (states_when_closed < hi)).sum() / len(states_when_closed) * 100
                if frac > 1:
                    print(f"    [{lo:.2f}, {hi:.2f}): {frac:.1f}%")
            print(f"    percentiles 5/25/50/75/95: {np.percentile(states_when_closed, [5, 25, 50, 75, 95])}")

    # Use a simple heuristic: cluster at low end = empty; cluster above 0.02 = cloth held
    # If 25th percentile is below 0.02 and 75th above 0.02, the gap is populated → bimodal
    closed_left = left_g[cmd_left_g < 0.1]
    closed_right = right_g[cmd_right_g < 0.1]
    combined = np.concatenate([closed_left, closed_right])

    p25 = np.percentile(combined, 25)
    p50 = np.percentile(combined, 50)
    p75 = np.percentile(combined, 75)

    # Midpoint between low-cluster (p25) and high-cluster (p75)
    empty_threshold = float((p25 + p75) / 2.0)
    # If p25 and p75 are both very close, clusters not separable
    separable = (p75 - p25) > 0.02
    print(f"\n  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f}  midpoint={empty_threshold:.4f}  separable={separable}")

    if not separable:
        print(f"  WARNING: Clusters not separable (gap={p75-p25:.4f} < 0.02). DISABLE grasp retry.")

    # === Final ===
    thresholds = {
        "max_joint_delta_arm": round(scalar_cap, 4),
        "max_joint_delta_gripper": round(gripper_p95, 4),
        "arm_vel_threshold_release": round(release_p90, 4),
        "gripper_empty_threshold": round(empty_threshold, 4) if separable else None,
        "gripper_clusters_separable": bool(separable),
        "per_joint_p95": {k: round(v, 4) for k, v in per_joint_p95.items()},
        "n_frames": int(limit),
        "n_release_events": int(len(release_arm_speeds)),
    }

    print("\n" + "=" * 60)
    print("RECOMMENDED THRESHOLDS:")
    print("=" * 60)
    print(f"  max_joint_delta (arm):     {thresholds['max_joint_delta_arm']}")
    print(f"  max_joint_delta (gripper): {thresholds['max_joint_delta_gripper']}")
    print(f"  arm_vel_threshold:         {thresholds['arm_vel_threshold_release']}")
    if separable:
        print(f"  gripper_empty_threshold:   {thresholds['gripper_empty_threshold']}")
    else:
        print(f"  gripper_empty_threshold:   DISABLE (clusters not separable)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(thresholds, indent=2))
    print(f"\n[INFO] Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
