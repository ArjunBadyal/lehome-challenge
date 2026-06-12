# Teleop demo recording — quick start

Goal: collect ~20–40 demos targeting the failure modes blocking 70%, **without leaking on the eval test set**.

## ⚠️ Compliance: do NOT teleop on Unseen garments

`Assets/objects/Challenge_Garment/Release/<Cat>/<Cat>.txt` lists both Seen AND Unseen garments — those Unseen USDs are part of the official evaluation. Recording demos on `Top_Short_Unseen_0` etc. = training on test data. Don't.

Instead, record on **Seen garments that exhibit the same failure mode**. The skill transfers; the test integrity is preserved.

## Failure-mode → Seen-garment mapping

| Eval failure to fix | Recommended Seen garment to demo on | Why |
|---|---|---|
| Top_Short_Unseen_0 (collar never folds) | **Top_Short_Seen_9** (0–20% rate; all conditions fail similarly) | Demonstrating a full clean fold here teaches the missing collar/shoulder-fold step |
| Pant_Long_Unseen_1 (left leg dangles) | **Pant_Long_Seen_3** (d(1,5) right-leg-loose mode, same failure pattern) | Same skill (second-leg correction), different garment |
| Top-Short generic precision (Seen_5/6 ~20–60%) | **Top_Short_Seen_5** (d(0,4) and d(1,5) ~18 cm) | Sleeves don't fully meet — precision-fold demos |

## Launch (one per priority target)

```bash
bash scripts/teleop_helper/launch_top_short_seen_9.sh    # collar-fold skill
bash scripts/teleop_helper/launch_pant_long_seen_3.sh    # second-leg skill
bash scripts/teleop_helper/launch_top_short_seen_5.sh    # precision sleeve fold
```

For unattended attempts that save only successes:

```bash
bash scripts/teleop_helper/auto_visual_repair_top_short_seen_9.sh
bash scripts/teleop_helper/auto_visual_repair_pant_long_seen_3.sh
```

Tune the ACT-prefix length if needed:

```bash
AUTO_REPAIR_STEP=220 NUM_SUCCESS=5 bash scripts/teleop_helper/auto_visual_repair_top_short_seen_9.sh
```

Each launches an assisted recorder: the current ACT specialist runs first, then
you can take over manually around the failure point. Keyboard motion is captured
globally, so the Isaac viewport does not need focus.

The launchers also open a `Click-IK Top Camera` window. This is the reliable
click-to-move UI: click the top-camera image, not the raw Isaac viewport.

## In-sim controls (BiKeyboard)

| Key | Effect |
|-----|--------|
| `S` | Start recording; ACT begins driving the robot |
| `M` | Toggle manual takeover |
| `R` | Resume simulation/policy after manual takeover |
| `P` | Pause/resume simulation physics; cloth stops moving while paused |
| `C` | Force both grippers closed and pause simulation physics for LiveIK/manual placement |
| `V` | Force both grippers open/released and pause simulation physics |
| `Z` | Clear gripper override; ACT controls grippers again |
| `E` | Visual repair macro: segment garment landmarks from the top camera, run slow bimanual IK grasp → fold → release |
| `G` | Auto-fold: after `C` (cloth grasped), plan a bimanual IK move that drags both grippers toward the garment centroid (HSV-segmented from top RGB) — replaces clicking |
| `N` | Mark current episode as **success** and save |
| `D` | Discard current episode |
| `X` | Restart current episode without saving |
| `ESC` | Abort and exit |

Press `S` when ready. Watch ACT run. If the trajectory is good but release is
wrong, use `C` to close/hold and freeze physics. Then either:

- press `E` for **visual repair** if ACT failed to grip or dropped the cloth:
  the system detects top/bottom garment landmarks from the top camera, moves
  both grippers slowly to those points with IK, closes, lifts, folds toward the
  centroid, places, and releases. This is the best option for collecting repair
  demos when the failure is grasp/slip rather than high-level planning.
- press `G` for **auto-fold**: the system segments the garment from the top
  camera, finds its centroid, and runs bimanual IK to drag both closed
  grippers toward that centroid (lifted). Use this once the cloth is grasped
  to fold it onto itself without clicking.
- or click the `Click-IK Top Camera` window where you want the *nearest*
  gripper to move (single-arm move, manual placement).

Press `V` to release/place once the arms are at the desired point. Use `Z` if
you want ACT to control grippers again, or `R` to fully resume simulation.

If the arm trajectory itself is wrong, press `M`, correct the fold with the
keyboard, then press `N` only if the final fold is clean. Press `X` to
immediately reset/retry the current episode. Press `D` for failed attempts.
Movement keys per arm are shown in the on-screen banner.

## After recording

```bash
# Merge teleop demos with existing per-category dataset
bash scripts/teleop_helper/merge_teleop.sh top_short
bash scripts/teleop_helper/merge_teleop.sh pant_long

# Retrain ACT (resumes from current best, +15K aug-enabled steps, ~3h GPU)
bash scripts/teleop_helper/retrain.sh top_short
bash scripts/teleop_helper/retrain.sh pant_long

# Eval each saved checkpoint on 5-ep CPU
bash scripts/teleop_helper/eval_checkpoints.sh top_short
bash scripts/teleop_helper/eval_checkpoints.sh pant_long
```

## Target collection sizes

- Top-Short Seen_9: 5–10 successful demos (clean fold including collar)
- Pant-Long Seen_3: 5–10 successful demos (both legs folded symmetrically)
- Top-Short Seen_5: 3–5 successful demos (precision-fold variation)

10–20 well-targeted demos at the right failure modes is plenty — ACT learns from small numbers of clean demonstrations.
