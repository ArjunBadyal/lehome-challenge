# LeHome Challenge — Approach Notes for Review

## Training log

| Model | Dataset (episodes / frames) | Steps trained | Final loss | Status |
|-------|:---------------------------:|:-------------:|:----------:|--------|
| Narrow ACT (Top-Long) | top_long_merged (250 / 83K) | 20K | 0.152 | complete — used for first residual experiments only |
| Unified ACT (chunk100) | four_types_merged (1000 / 266K) | 30K | 0.171 | complete — fallback policy |
| Unified ACT (chunk10) | four_types_merged (1000 / 266K) | 30K | 0.137 | complete — rejected (eval 5% < chunk100 20%) |
| Top-Short specialist | top_short_merged (250 / 76K) | 40K → resuming 80K | 0.114 @ 42K (still trending down) | **overnight** — 30% eval at 40K |
| Top-Long specialist v2 | top_long_merged (250 / 83K) | 40K → resuming 80K | TBD | **overnight** — 0% eval at 40K |
| Pant-Long specialist | pant_long_merged (250 / 66K) | 40K → resuming 80K | TBD | **overnight** — 0% eval at 40K (partial) |
| Pant-Short specialist | pant_short_merged (250 / 41K) | 40K → resuming 80K | TBD | **overnight** — untested at 40K |
| Garment classifier | first-frame images, 4×250 = 1000 samples | 100 epochs | — | complete — 97.5% val accuracy |

### Hardware

- **ACT imitation training** — single GPU (`cuda:0`), no simulator involved. Each
  40K-step run takes ~2.5 h; 30K-step runs take ~2 h.
- **Eval / simulator** — CPU PhysX cloth (~0.13 s/step) + GPU for ACT inference
  and classifier. A 12-garment × 5-episode category eval takes ~80 min.
- **GPU cloth sim** — attempted, works at the tensor level (~3× throughput) but
  produces incorrect gripper-cloth contact because the robot's custom STL
  collision meshes fall back to CPU-only collision on GPU (documented NVIDIA
  limitation). Fix would require convex-hull collision approximation on the
  gripper prims; parked pending more submission value (see
  `lehome-challenge-gpu/GPU_INVESTIGATION.md`).

### Eval protocol used for all reported success rates

- `scripts.eval` with official defaults:
  `--max_steps 600 --num_episodes 5 --enable_cameras --device cpu --headless`.
- 12 garments per category (10 Seen + 2 Unseen) × 5 eps = 60 episodes.
- Success is the simulator's conjunctive boolean across the category's
  geometric checkpoint thresholds (4 conditions for pants, 5 for tops);
  dense score is not used as an eval metric — only for training and
  diagnostics.
- All numbers in this doc use this protocol unless otherwise stated. Earlier
  numbers quoted with `--num_episodes 1` are marked explicitly.

### Why 40K → 80K for all four

ACT training loss on the unified-dataset runs (1000 episodes) was still
trending down at 30K: chunk100 ended at 0.171, chunk10 at 0.137. Per-category
specialists have 4× less data per optimizer step, so gradient updates are
less redundant and further training is more likely to help rather than
overfit — a single-category dataset of 250 episodes at 40K steps = ~8 epochs,
which is modest by imitation-learning standards. We did not capture the
loss curve at 40K on the specialist runs (training stdout was redirected
through `tail -1` during the sequential-launch script), so we are extending
all four on the assumption that at least Top-Short, which is our only scoring
category, is still improving. We accept ~10 h of overnight compute to resolve
the question.

This document summarises our current approach for the LeHome Garment Folding Challenge
(simulation phase). It is written for an expert reviewer familiar with imitation learning
and RL on manipulation tasks, and is intended to surface the key design decisions and
their tradeoffs. The current leaderboard shows top teams at 57–72% per-category success
rate; our approach is not yet on the board — this is a status and direction document,
not a submission writeup.

## 0. Bottom-line conclusion (2026-04-16)

After following the expert reviewer's first recommendation — "make the policy
challenge-compliant and benchmark ACT alone through the official eval path" — the
one-sentence finding is:

> The challenge-compliant unified imitation policy (ACT trained on the 4-category
> merged dataset) improved some out-of-distribution behaviour but still produced
> **0% success** and **underperformed the narrow Top-Long policy on the Top-Long
> benchmark (259.99 vs 302.47 avg return)**. The remaining gap is therefore not
> demo quantity alone; it is likely **model capacity, representation, and overly
> open-loop control for cloth**.

What this invalidates from earlier sections of this document:
- The v1 flat-residual and v2 hierarchical-residual runs were both trained against
  a model of reality that included privileged task-metric channels. Per the
  reviewer and confirmed against upstream code, `condition_margins` and
  `checkpoint_positions_cm` are **not** in the public observation dict. v2 as
  described below is not submittable and its "beat v1 in mean return" claim was
  against a bad benchmarking setup (`--num_episodes 1`, our own fork's obs
  surface). Those artefacts remain in the repo as learning steps but are not
  the path forward.
- The forward path is: **unified ACT as the sole base**; **residual on public
  observations or ACT encoder features only**; **short-horizon replanning**
  instead of 100-step ACT chunk reuse.

The remaining sections below capture what we did and what we learned; they are
no longer the design.

## 1. Challenge summary

- Bimanual SO-101 robot arms (2 × 6-DoF = 12 joints, absolute joint position commands)
- Four garment categories: Long-Sleeved Tops, Short-Sleeved Tops, Long Pants, Shorts
- Per category: 10 "Seen" garments + 2 "Unseen" garments; score = % of garments
  successfully folded end-to-end within the 60 s episode budget
- Success is binary per garment, defined by geometric checkpoint distances
  (e.g. for Tops: `dist(p0,p4) ≤ 16 cm` AND four other conditions — see
  `source/lehome/lehome/utils/success_checker_chanllege.py`)
- A per-step `dense_score ∈ [0, 1]` is available from `_get_task_metrics()` but is
  **not** the leaderboard metric — you get graded on boolean success only
- Top-8 teams at the end of the sim phase (30 Apr 2026) advance to the real-robot round

Observations per step (12D joint state, 3 × 480×640 RGB cameras, depth) and rewards are
fully available during training. At eval time the env exposes the same observation dict.

## 2. Why a pure-RL policy is not competitive here

We confirmed empirically that a vanilla SAC agent operating on the 12D joint state
alone cannot make progress — it never sees the garment, so even the dense reward is
essentially unobservable to the policy. This matches what you would predict: the garment
state lives in the image/particle-cloud channels, not in the proprioceptive channel.

Any competitive approach therefore has to either:

1. Use the vision channels directly (large policy, rich observations), or
2. Bootstrap from demonstrations that are already vision-conditioned.

The dataset ships with a moderate number of LeRobot-format demos per garment category
(~250 episodes for Long-Sleeved Tops, ~80k frames), so option 2 is cheap to start from.

## 3. Architecture

The approach is **a frozen ACT base policy with a learned hierarchical residual on top**.

```
                ┌──────────────────── frozen ────────────────────┐
  images (3x)   │                                                │
  joint state   ├─► ACT (LeRobot)  ─► action_chunk[100, 12] ─────┤
                └────────────────────────────────────────────────┘
                                              │
                                              ▼
  joint state ─┐                       act_action [12]
  margins (5) ─┼─► [47D obs] ─► SubPolicy_k  ─► delta [12]
  ckpt (18) ───┘          ▲              │
                          │              ▼
                  MetaController    final = clip(act_action + scale * delta)
                  (rule-based)
```

### 3.1 Base: frozen ACT
- Checkpoint trained with `lerobot-train` on the `top_long_merged` dataset
- 20k optimizer steps, loss ~0.15 at stop (continuing to decrease)
- Predicts chunks of 100 actions per invocation; the training/eval loops reuse the
  chunk for 100 env steps before re-querying

**Why ACT and not Diffusion Policy or SmolVLA?**
- Chunking is a natural fit for residual RL — the residual corrects within a chunk
  without fighting the base policy's long-horizon plan
- ACT is fast enough for the residual loop (we re-plan every 100 steps)
- Dataset is LeRobot-native so no conversion overhead

### 3.2 Residual SAC (single sub-policy — our v1)
- `ResidualActor`: 24D → 12D MLP, squashed Gaussian, zero-init mean head so initial
  residual is ≈ 0 (preserves ACT baseline behaviour at start of training)
- Observation: `[joint_state(12), act_action(12)]` = 24D
- Final action: `clip(act_action + scale * delta, joint_low, joint_high)`
- `scale` anneals from 0.1 → 0.15 over 30k steps (chosen conservatively after
  observing that larger residuals degrade ACT; see §5.1)
- Reward: raw env reward (dense score + success bonus + penalties), standard SAC

**Results** (50k steps, 8 Seen garments for training, evaluated on all 12 garments):
- Average return: 252.6
- Per-garment success: **0/12**
- Residual magnitude: 7–11% of ACT magnitude (healthy band)
- Dense score trajectory: ~0.07 → 0.3–0.8 peaks within an episode, but never all
  five conditions simultaneously

So the residual learns *something* useful (dense score rises, generalises to Unseen
within noise of Seen) but does not close the gap to actual folds.

### 3.3 Hierarchical residual (our v2 — the current run)

Our hypothesis from inspecting rollouts: the failure mode is not *direction of force*
but *coordination between arms across phases*. A single residual MLP has to
simultaneously produce "bring sleeves in" corrections AND "collapse front-to-back"
corrections AND "maintain extensiveness" corrections. Those require different arm
synergies, and the flat residual oscillates between them.

So: decompose the task into fold directions, each with its own residual. Select which
residual to use at each step based on which geometric condition is most in need.

**Sub-policies** (4 learned + 1 hardcoded):

| id | name       | targets                                   | sub-reward                             |
|----|------------|-------------------------------------------|----------------------------------------|
| 0  | fold_up    | margin[1] (dist p2–p3, front/back)        | Δmargin[1]                             |
| 1  | fold_down  | margin[1]                                 | Δmargin[1]                             |
| 2  | fold_left  | margin[0], margin[2] (sleeves, sides)     | Δmin(margin[0], margin[2])             |
| 3  | fold_right | margin[0], margin[2]                      | Δmin(margin[0], margin[2])             |
| 4  | no_op      | —                                         | zero residual, ACT runs unmodified     |

Each learned sub-policy is a full SAC agent (actor + twin Q + entropy) over a 47D obs
that adds the 5 condition margins and 18 flattened checkpoint positions to the 24D
base.

**Meta-controller** — rule-based, not learned. At each step:
1. If either secondary "not over-compressed" condition is < −0.5, pick `no_op`
   (the garment is already in a physically dangerous state; let ACT stabilise it)
2. Otherwise, find `argmin(primary_margins[0..2])` — the worst primary condition
3. Map that to a sub-policy using spatial heuristics (which side of the centre line
   the worse checkpoint is on, etc. — see `hierarchical_model.py`)
4. Hold the selection for at least `hold_steps=50` (~0.55 s) before re-evaluating, to
   prevent oscillation

Why rule-based? With 0 % success on v1 and only ~27 episodes of training data per
segment, there is nowhere near enough signal to train a learned meta-controller. A
rule-based selector is deterministic, debuggable, and costs nothing. It can always be
replaced once the sub-policies themselves are strong.

**Replay buffer** — `TaggedReplayBuffer` adds a `sub_policy_id` column. SAC updates
for sub-policy *k* sample only transitions tagged *k*. Buffer sized 400k so each
sub-policy effectively has ~100k independent transitions.

### 3.4 Training loop engineering

- **Garment cycling** happens *out of process* via `train_hierarchical_schedule.py`.
  Each "segment" is a fresh `train_hierarchical_sac.py` invocation on a single
  garment, saving `model.pt` + `replay_buffer.npz` on exit; the scheduler then
  launches the next segment on the next garment, passing those artifacts as
  `--checkpoint`/`--replay_buffer_path`.
  - **Why:** IsaacLab's in-process cloth recreation repeatedly hangs on
    `_cloth_prim_view.initialize()` after swap. We spent time chasing timeouts and
    faulthandlers; the out-of-process scheduler sidesteps the whole failure class.
- **Episode length** set to 10 s during training (vs 60 s at eval) to improve
  gradient signal density. The evaluator uses the full 60 s budget.
- **`terminate_on_success = True`** during training so successful rollouts end
  immediately and a new episode starts, avoiding wasted sim time.
- **`--headless`** saves GPU-rendering overhead we don't need for training.

## 4. Current results and positioning

All numbers below are against the **official eval path**
(`--max_steps 600 --num_episodes 5`, 12 garments per category, `0/60` = perfect).
Earlier reported numbers that used `--num_episodes 1` or non-public episode caps
are excluded as non-comparable.

### 4.1 Long-Sleeved Tops only (the original experimental target)

| Variant                                    | Train data                   | Steps | Mean return | Success |
|--------------------------------------------|------------------------------|-------|-------------|---------|
| Flat SAC on 12D state only                 | —                            | 50k   | ~80         | 0/60    |
| ACT alone, **narrow** (Top-Long only)      | 250 ep / 83k frames          | 20k   | **302.47**  | 0/60    |
| ACT alone, **unified** (all 4 categories)  | 1000 ep / 266k frames        | 30k   | 259.99      | 0/60    |
| v1: flat residual on narrow ACT            | (uses privileged obs)         | 50k   | not re-benched (prior 252 number used `--num_episodes 1` and had a buggier obs surface) | — |
| v2: hierarchical residual on narrow ACT    | (uses privileged obs)         | ~20k  | not submittable — see §0 | — |

Per-garment comparison of the two ACT variants (the comparison that actually matters):

| Garment    | Narrow-20K | Unified-30K | Δ      | Split |
|------------|:----------:|:-----------:|:------:|:------|
| Seen_0     | 303.6      | 230.5       | −73    | train |
| Seen_1     | 359.8      | 306.1       | −54    | train |
| Seen_2     | 325.3      | 276.5       | −49    | train |
| Seen_3     | 313.4      | 319.1       | +6     | train |
| Seen_4     | 322.3      | 285.1       | −37    | train |
| Seen_5     | 326.9      | 194.9       | −132   | train |
| Seen_6     | 275.5      | 241.0       | −34    | train |
| Seen_7     | 227.4      | 183.0       | −44    | train |
| Seen_8     | 320.0      | 291.4       | −29    | val   |
| Seen_9     | 346.1      | 349.6       | +3     | val   |
| Unseen_0   | 253.9      | 313.6       | **+60**| test  |
| Unseen_1   | 255.6      | 128.9       | **−127**| test  |

Reading:
- On training garments the unified policy is clearly worse (−52 pts on mean). Same
  capacity, but spread across 4 categories.
- On validation it is roughly tied.
- On unseen it is **bimodal**: +60 on one and −127 on the other. The diversity
  pays off on some held-out geometries and fails catastrophically on others.
- Neither policy produces a single successful fold at 600 steps. The per-step
  dense-score trajectory shows both reach 0.3–0.8 peaks but never satisfy all
  five conditions simultaneously.

### 4.2 Multi-category results

Not yet measured. The narrow ACT scores 0 on Short Tops / Long Pants / Shorts by
construction (it never saw them). The unified ACT is expected to be the only
variant with non-zero scores on those three categories. Benchmarking in progress;
this section will be updated once all four categories have the full 60-episode
eval done.

### 4.3 Positioning vs leaderboard

- Leaderboard scores a per-category success rate (0 – 100 %) with an overall
  score that is (roughly) the mean across the four categories.
- Top team on Long-Sleeved Tops: 58 %. Our best: 0 %.
- Top-8 cutoff on overall score: 46 %. We are not on the board.
- The relevant metric for ranking is the **overall** score. Any approach that
  scores 0 on three of four categories is structurally non-competitive regardless
  of its Long-Sleeved Tops number. This is the main reason the unified policy is
  the only valid base going forward, even though it is worse on Long-Sleeved Tops
  in isolation.

## 5. Things we deliberately chose not to do, and why

### 5.1 We capped `residual_scale_max` at 0.15 (plan said 0.3)
In the v1 run, scale annealed to 0.3 → residual ratio climbed to 33 % of ACT
magnitude → returns collapsed from 518 to 60. The residual was overriding ACT
rather than correcting it. At 0.15 the residual stays in the 8–11 % band which is
what the literature (e.g. Residual Policy Learning, Silver et al.) suggests is the
useful regime.

### 5.2 Meta-controller is rule-based
Addressed in §3.3. A learned meta-controller over 5 classes with near-zero success
rate as supervision signal would be a waste of parameters right now.

### 5.3 We train on CPU physics, not GPU
A commit landed upstream (`32b5359`) that fixed GPU coord-space mismatches in the
success checker, but cloth recreation on GPU still fails inside IsaacSim's
`_cloth_prim_view` tensor API. On CPU with the out-of-process scheduler, the
pipeline is stable. Moving to GPU cloth is a throughput win (probably 5–10×, more
with `num_envs > 1`) but not a correctness win. Current bottleneck is policy/demo
quality, not throughput.

### 5.4 We have not trained category-specific ACT for Short-Sleeved Tops / Pants / Shorts
Bandwidth constraint. Long-Sleeved Tops is the hardest of the four categories for the
top team (57 % vs 60–72 % on the others), so we chose it to develop the approach
against the hardest case first. The same pipeline transfers to the other three
categories; it just needs per-category ACT checkpoints + per-category residual runs.

## 6. Known limitations we want an expert view on

1. **Reward shaping for the residual.** We add
   `2.0 · Δ(target_margin)` on top of the env reward for learned sub-policies. The
   factor 2.0 was a guess; we have not ablated it. In the v2 training logs, the
   `fold_down` sub-policy ends up with 4–10× fewer transitions than the others, so
   there's real imbalance across sub-policies in terms of how much signal each gets.

2. **Meta-controller brittleness.** Our selection rule reads checkpoint positions
   and picks "fold toward centre" based on x-coordinate signs. For garments whose
   initial orientation differs significantly from the training demos, this might
   pick the wrong direction and fight ACT. We haven't probed this.

3. **No multi-garment within a single ACT pass.** ACT is trained on the full mix of
   250 demo episodes across all 10 Seen garments, so it *should* generalise, but we
   have not empirically verified per-garment imitation quality without the residual.
   A residual that learns to correct one garment well can easily overfit if the ACT
   base is weaker on other garments.

4. **Eval is slow.** Full eval on 12 garments takes ~30 minutes with camera
   rendering every step. We cannot cheaply run eval-every-5k-steps during training,
   so we depend on proxy metrics (return, dense_score, residual ratio) until a
   training run finishes.

5. **We have not recorded new demos.** The provided demos cover one teleop style.
   Domain-randomising initial garment poses and recording more varied demos would
   almost certainly help, but is the most time-expensive improvement.

## 7. What we plan to do next

Rewritten after the unified-ACT experiment (see §0). Priority order:

1. **Benchmark unified ACT 30K on the other three categories** (Short-Sleeved
   Tops, Long Pants, Shorts) at `--max_steps 600 --num_episodes 5`. Fill in
   §4.2. This is the only way to judge whether the Top-Long regression is
   acceptable in exchange for non-zero scores on the other three categories.
   Estimated wall time ~3 hours.

2. **Identify whether the 0% success floor is structural**. Instrument one or
   two rollouts of the unified ACT and log `condition_margins` trajectories
   against the sim-side success thresholds. If the trajectories plateau just
   below threshold (what we suspect from dense-score peaks of 0.3–0.8), the
   bottleneck is last-centimetre closure. If they plateau far from threshold,
   the bottleneck is the high-level plan.

3. **Replace open-loop ACT chunk reuse with short-horizon replanning**. Re-query
   ACT every 5–10 sim steps and use only the first 1–2 actions. For cloth this
   should matter more than any other single change. The residual framework
   supports this trivially — it is just a different chunk_size.

4. **Build a submission-compliant residual** on top of unified ACT that takes
   **only the public observation dict plus ACT encoder features** as input. No
   `condition_margins`, no `checkpoint_positions_cm`. Potential-based shaping
   on the worst primary margin used as training reward only, never as a policy
   input.

5. Cover the other categories with the same stack.

6. Only if we still see 0% success after all of the above: invest in new demo
   recording / GPU cloth simulation.

## 8. File map

Anything with an existing upstream counterpart is labelled accordingly; the rest is
ours.

```
scripts/
├── train_sac.py                        # v0 flat SAC — abandoned
├── train_residual_sac.py               # v1 flat residual on ACT
├── train_hierarchical_sac.py           # v2 hierarchical, single-garment trainer
├── train_hierarchical_schedule.py      # v2 out-of-process multi-garment scheduler
├── rl/
│   ├── sac_model.py                    # SquashedGaussianActor, QNetwork (shared)
│   ├── residual_model.py               # ResidualActor (24D → 12D) for v1
│   ├── hierarchical_model.py           # SubPolicyBank + RuleBasedMetaController (v2)
│   ├── replay_buffer.py                # standard buffer (v0/v1)
│   ├── tagged_replay_buffer.py         # adds sub_policy_id column (v2)
│   ├── act_wrapper.py                  # ACTChunkProvider around LeRobotPolicy
│   └── demo_loader.py                  # parquet → SAC transitions (v0/v1 only)
└── eval_policy/
    ├── residual_policy.py              # v1 eval adapter
    └── hierarchical_residual_policy.py # v2 eval adapter
```

Both eval policies register with the upstream `PolicyRegistry` so the existing
`scripts.eval` entrypoint works unchanged.

## 9. Reviewer questions we'd especially value input on

- Is the rule-based meta-controller fundamentally limiting, or is it a reasonable
  choice until we have success signal to bootstrap a learned one?
- Does the residual scaling scheme (clip to joint limits, scale *applied to tanh*
  delta) interact well with ACT's action distribution, or should we be operating in
  a different action parameterisation (e.g. residual in normalised joint space,
  residual as additive noise on intermediate features)?
- For cloth manipulation specifically, is there literature we should be drawing
  from that we're missing? Most of the residual-RL work we've read is on rigid-body
  tasks.
- Is our choice of the 5 condition-margins + 18 checkpoint coords as the sub-policy
  observation the right summarisation, or would something like a learned cloth
  state embedding from the base ACT's encoder features be better?
