"""Random search over AI-teleop hyperparameters.

Searches over the v9 parameterization for Top_Short_Unseen_0 recovery:
  trigger_step       ∈ [180, 240]
  gripper_tightness  ∈ [-0.22, -0.12]
  template_length    ∈ [60, 130]
  blend_steps        ∈ [5, 20]
  hold_duration      ∈ [10, 40]

Each trial runs 2 episodes on Top_Short_Unseen_0 with seed=42, takes the
mean return as fitness. Logs all trials to outputs/cv_collar_pol/random_search.jsonl.

Usage:
  python -m scripts.eval_policy.ai_teleop_random_search --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "outputs/cv_collar_pol/random_search.jsonl"


def sample_params(rng: random.Random) -> dict:
    return {
        "trigger_step": rng.randint(180, 240),
        "gripper_tightness": round(rng.uniform(-0.22, -0.12), 3),
        "template_length": rng.choice([60, 80, 100, 120]),
        "blend_steps": rng.choice([5, 10, 15]),
        "hold_duration": rng.choice([10, 20, 30, 40]),
    }


def run_eval(params: dict) -> tuple[float, list[float]]:
    """Run 2-episode eval with these params, return (mean_return, per_ep_returns)."""
    env = os.environ.copy()
    env["AI_TELEOP_GARMENT"] = "Top_Short_Unseen_0"
    env["AI_TELEOP_TRIGGER_STEP"] = str(params["trigger_step"])
    env["AI_TELEOP_GRIPPER_TIGHTNESS"] = str(params["gripper_tightness"])
    env["AI_TELEOP_TEMPLATE_LENGTH"] = str(params["template_length"])
    env["AI_TELEOP_BLEND_STEPS"] = str(params["blend_steps"])
    env["AI_TELEOP_HOLD_DURATION"] = str(params["hold_duration"])

    log_file = REPO_ROOT / "outputs/cv_collar_pol/_rs_eval.log"
    cmd = [
        str(REPO_ROOT / "third_party/IsaacLab/isaaclab.sh"), "-p", "-m", "scripts.eval",
        "--policy_type", "ai_teleop_top_short",
        "--policy_path", "outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model",
        "--dataset_root", "Datasets/example/top_short_merged",
        "--task_description", "Fold a garment with bimanual robot arms",
        "--garment_type", "top_short",
        "--eval_list_override", "outputs/cv_collar_pol/ai_teleop_garment.txt",
        "--max_steps", "600", "--num_episodes", "2", "--seed", "42",
        "--enable_cameras", "--device", "cpu", "--headless",
    ]
    with open(log_file, "wb") as f:
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                cwd=str(REPO_ROOT), timeout=600)

    # Parse returns from log
    text = log_file.read_text(errors="ignore")
    returns = []
    for line in text.splitlines():
        if "Episode " in line and "Return=" in line:
            try:
                ret = float(line.split("Return=")[1].split(",")[0])
                returns.append(ret)
            except Exception:
                pass
    if not returns:
        return -1000.0, []
    return float(np.mean(returns)), returns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_fitness = -1e9
    best_params = None

    print(f"Random search: {args.n_trials} trials")
    print(f"Logging to {LOG_PATH}")

    for trial in range(args.n_trials):
        params = sample_params(rng)
        t0 = time.time()
        fitness, ep_returns = run_eval(params)
        elapsed = time.time() - t0
        improved = fitness > best_fitness
        if improved:
            best_fitness = fitness
            best_params = params
        rec = {
            "trial": trial,
            "params": params,
            "mean_return": fitness,
            "ep_returns": ep_returns,
            "elapsed_s": round(elapsed, 1),
            "best_so_far": round(best_fitness, 2),
            "best_params": best_params,
            "improved": improved,
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"trial {trial:3d}: mean={fitness:7.2f} (best={best_fitness:7.2f}) {'*' if improved else ''} "
              f"params={params} eps={[round(r,1) for r in ep_returns]}  ({elapsed:.0f}s)",
              flush=True)

    print(f"\nBEST: mean_return={best_fitness:.2f}, params={best_params}")
    with open(LOG_PATH.with_suffix(".best.json"), "w") as f:
        json.dump({"best_params": best_params, "best_fitness": best_fitness}, f, indent=2)


if __name__ == "__main__":
    main()
