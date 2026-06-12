"""CEM trajectory optimization for Top-Short collar recovery (offline demo repair).

Pipeline:
  1. Pick a Top-Short Seen garment + seed.
  2. Initial knots = mean of last-100-actions sampled at 6 evenly-spaced points
     from successful Seen_2 / Seen_3 / Seen_8 templates (controllable joints only).
  3. CEM loop:
       - Sample N candidate knot vectors from N(mean, std).
       - For each candidate, write knots to /tmp/cem_knots.npy and run a 1-episode
         eval. Capture final reward + success flag.
       - Keep top-K elites. Update mean = elites.mean(), std = elites.std().
       - If any candidate succeeded, save its full trajectory as a recovery demo.
  4. Repeat for each (garment, seed) combo.

Usage:
  python -m scripts.cem_recovery.driver --garments Top_Short_Seen_0,Top_Short_Seen_2,...

Output: outputs/cem_recovery/recovery_demos.jsonl + per-success trajectory CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs/cem_recovery"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# 12D action: only the 6 controllable joints are searched
LEFT_SHOULDER_LIFT = 1
LEFT_ELBOW = 2
LEFT_WRIST_FLEX = 3
RIGHT_SHOULDER_LIFT = 7
RIGHT_ELBOW = 8
RIGHT_WRIST_FLEX = 9
CONTROL_JOINTS = [LEFT_SHOULDER_LIFT, LEFT_ELBOW, LEFT_WRIST_FLEX,
                  RIGHT_SHOULDER_LIFT, RIGHT_ELBOW, RIGHT_WRIST_FLEX]

N_KNOTS = 6
N_CONTROL = 6
KNOTS_FILE = "/tmp/cem_knots.npy"


def extract_seed_knots(traj_csv: str, n_last: int = 100) -> np.ndarray:
    """Extract N_KNOTS knots as DELTAS from action[-n_last] anchor."""
    rows = []
    with open(traj_csv) as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            rows.append(r)
    actions = np.array([[float(r[f'action_{i}']) for i in range(12)] for r in rows], dtype=np.float32)
    if len(actions) < n_last:
        n_last = len(actions)
    last = actions[-n_last:]
    anchor = last[0]
    idx = np.linspace(0, n_last - 1, N_KNOTS).astype(int)
    abs_knots = last[idx][:, CONTROL_JOINTS]
    anchor_ctrl = anchor[CONTROL_JOINTS]
    delta_knots = abs_knots - anchor_ctrl  # delta from anchor
    return delta_knots


def parse_eval_log(log_path: str) -> tuple[float, list[float], list[bool]]:
    """Return (fitness, per_episode_returns, per_episode_success).

    Fitness = mean_return + 1000 * fraction_succeeded. Strong success bonus so
    CEM converges hard on any successful trajectory.
    """
    text = Path(log_path).read_text(errors="ignore")
    rets, succ = [], []
    for line in text.splitlines():
        if "Episode " in line and "Return=" in line:
            try:
                ret = float(line.split("Return=")[1].split(",")[0])
                rets.append(ret)
                if "Success=True" in line:
                    succ.append(True)
                else:
                    succ.append(False)
            except Exception:
                pass
    if not rets:
        return -1e6, [], []
    mean_ret = float(np.mean(rets))
    succ_frac = float(np.mean([1.0 if s else 0.0 for s in succ]))
    fitness = mean_ret + 1000.0 * succ_frac
    return fitness, rets, succ


def run_candidate(knots: np.ndarray, garment: str, trigger_step: int,
                  horizon: int, seed: int, log_path: str,
                  n_trials: int = 1) -> tuple[float, bool]:
    """Run rollout(s) with given knots, return (best_score, any_success).

    Run n_trials episodes and use the BEST score (max), not mean — we want
    to find the case where physics goes well, since cloth sim is non-deterministic.
    """
    np.save(KNOTS_FILE, knots.astype(np.float32))
    env = os.environ.copy()
    env["CEM_KNOTS_FILE"] = KNOTS_FILE
    env["CEM_TRIGGER_STEP"] = str(trigger_step)
    env["CEM_HORIZON"] = str(horizon)
    garments_txt = OUT_ROOT / f"_garment_{garment}.txt"
    garments_txt.write_text(garment + "\n")

    cmd = [
        str(REPO_ROOT / "third_party/IsaacLab/isaaclab.sh"), "-p", "-m", "scripts.eval",
        "--policy_type", "cem_recovery_top_short",
        "--policy_path", "outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model",
        "--dataset_root", "Datasets/example/top_short_merged",
        "--task_description", "Fold a garment with bimanual robot arms",
        "--garment_type", "top_short",
        "--eval_list_override", str(garments_txt),
        "--max_steps", "600", "--num_episodes", str(n_trials), "--seed", str(seed),
        "--enable_cameras", "--device", "cpu", "--headless",
    ]
    with open(log_path, "wb") as f:
        try:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           cwd=str(REPO_ROOT), timeout=900)
        except subprocess.TimeoutExpired:
            return -1e6, False
    fitness, rets, successes = parse_eval_log(log_path)
    if not rets:
        return -1e6, False
    # Use BEST trial (max return) as candidate score — cloth physics noise
    # makes mean too pessimistic; we care about whether the trajectory CAN succeed.
    succ_frac = float(np.mean([1.0 if s else 0.0 for s in successes]))
    best_ret = float(max(rets))
    score = best_ret + 1000.0 * succ_frac
    return score, any(successes)


def cem_loop(garment: str, seed_knots: list[np.ndarray], trigger_step: int,
             horizon: int, sim_seed: int, n_candidates: int, n_iters: int,
             n_trials_per_cand: int = 3,
             elite_frac: float = 0.25) -> tuple[np.ndarray, list[dict]]:
    """Run CEM for one (garment, sim_seed) combo. Return best knots + log."""
    # Init mean = first seed knot (don't average — averaging incoherent templates
    # produces a worse trajectory than any individual seed).
    mean = seed_knots[0].copy()
    # Very tight initial std for delta-knot perturbations.
    std = np.full_like(mean, 0.02)

    best_knots = mean.copy()
    best_score = -1e9
    success_log = []
    iter_log = []

    rng = np.random.default_rng(42 + sim_seed)

    for it in range(n_iters):
        candidates = []
        # First candidate is always the unperturbed mean (so we always score the
        # current best-known, never lose it to bad perturbations).
        candidates.append(mean.copy())
        for _ in range(n_candidates - 1):
            k = mean + rng.normal(0, 1, size=mean.shape) * std
            candidates.append(k)

        log_path = OUT_ROOT / "logs" / f"cem_{garment}_seed{sim_seed}_iter{it}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        scores = []
        for ci, k in enumerate(candidates):
            t0 = time.time()
            score, succ = run_candidate(k, garment, trigger_step, horizon, sim_seed,
                                          str(log_path), n_trials=n_trials_per_cand)
            elapsed = time.time() - t0
            scores.append(score)
            if succ:
                success_log.append({
                    "garment": garment, "sim_seed": sim_seed, "iter": it,
                    "candidate": ci, "score": score, "knots": k.tolist(),
                })
                # Save knots for this success
                np.save(OUT_ROOT / f"success_{garment}_seed{sim_seed}_it{it}_c{ci}.npy", k)
            print(f"  iter {it} cand {ci:2d}: score={score:7.2f} succ={succ} ({elapsed:.0f}s)",
                  flush=True)
            if score > best_score:
                best_score = score
                best_knots = k.copy()

        scores_arr = np.array(scores)
        n_elite = max(1, int(n_candidates * elite_frac))
        elite_idx = np.argsort(scores_arr)[-n_elite:]
        elites = np.stack([candidates[i] for i in elite_idx])
        new_mean = elites.mean(axis=0)
        new_std = elites.std(axis=0).clip(min=0.01, max=0.05)  # bounded
        iter_log.append({
            "iter": it, "n_candidates": n_candidates, "elite_scores": scores_arr[elite_idx].tolist(),
            "best_score_so_far": best_score, "n_successes_so_far": len(success_log),
        })
        mean, std = new_mean, new_std

    return best_knots, {"successes": success_log, "iter_log": iter_log,
                         "best_score": best_score}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--garments", default="Top_Short_Seen_0,Top_Short_Seen_2,Top_Short_Seen_5,Top_Short_Seen_6")
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--n-iters", type=int, default=3)
    p.add_argument("--trigger-step", type=int, default=220)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--sim-seed", type=int, default=42)
    args = p.parse_args()

    # Load seed knots from successful Seen episode templates
    seed_traj_csvs = [
        REPO_ROOT / "outputs/eval_traj/router_v4_5ep_top_short/Top_Short_Seen_2_episode2.csv",
        REPO_ROOT / "outputs/eval_traj/router_v4_5ep_top_short/Top_Short_Seen_7_episode1.csv",
    ]
    seed_knots = [extract_seed_knots(str(p)) for p in seed_traj_csvs if p.exists()]
    if not seed_knots:
        print("ERROR: no seed templates found", file=sys.stderr)
        sys.exit(1)
    print(f"Seed templates: {len(seed_knots)} (each {seed_knots[0].shape})")

    garments = [g.strip() for g in args.garments.split(",")]
    print(f"CEM over garments: {garments}")
    print(f"Candidates: {args.n_candidates}, iters: {args.n_iters}")
    print(f"Trigger step: {args.trigger_step}, horizon: {args.horizon}")
    print(f"Total rollouts: {len(garments) * args.n_candidates * args.n_iters}")
    print(f"Logging to {OUT_ROOT}")

    all_results = {}
    for g in garments:
        print(f"\n=== {g} (seed={args.sim_seed}) ===")
        t0 = time.time()
        best_knots, info = cem_loop(
            garment=g, seed_knots=seed_knots, trigger_step=args.trigger_step,
            horizon=args.horizon, sim_seed=args.sim_seed,
            n_candidates=args.n_candidates, n_iters=args.n_iters,
        )
        elapsed = time.time() - t0
        np.save(OUT_ROOT / f"best_knots_{g}.npy", best_knots)
        all_results[g] = {**info, "elapsed_s": elapsed}
        print(f"  done {g}: best_score={info['best_score']:.2f}, "
              f"successes={len(info['successes'])}, elapsed={elapsed:.0f}s")
        with open(OUT_ROOT / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    total_succ = sum(len(r['successes']) for r in all_results.values())
    print(f"\n==== CEM DONE ====")
    print(f"Total successes across all (garment, iter): {total_succ}")
    print(f"Per-garment best scores:")
    for g, r in all_results.items():
        print(f"  {g}: best={r['best_score']:.2f}, n_succ={len(r['successes'])}")


if __name__ == "__main__":
    main()
