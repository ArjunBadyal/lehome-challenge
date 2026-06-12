# Submission Methodology Log

> Living document — updated as artifacts are produced. Every checkpoint, classifier,
> stabilizer config, and dataset modification used in the final submission must be
> traceable from this file.

## 1. Submission stack (v7 Pant-Long balanced-weak, 2026-04-30)

**Packaged-policy eval estimate: 63.75% (153/240).** This combines the exact
Docker submission policy artifacts evaluated through the local
`submission_bundle` adapter: rate-limit-only for Top-Short, delayed release +
rate limit for Top-Long/Pant-Short, rate-only winner stabilizer for Pant-Long,
static aug60 for Pant-Short, and the balanced-weak 090250 Pant-Long
checkpoint. The only v6 → v7 artifact change is the Pant-Long specialist.

**Compliance note for routed ACT stack:** the final v7 policy is an ACT/LeRobot
imitation stack, not a VLA stack. The router never reads garment filenames,
asset paths, eval-list entries, simulator-provided garment category labels,
checkpoint particle positions, success metrics, rewards, or condition margins.
Its only routing signal is a learned ResNet classifier applied to public RGB
camera observations. The per-category ACT specialist and stabilizer selection is
therefore based on visual inference from the observation stream, not direct
metadata matching. If the final organizer interpretation forbids all
classifier-selected ACT specialists (not only VLA multi-weight routing), the
compliance fallback is the single unified ACT policy in
`outputs/train/act_all_cats`.

| Cat | Specialist | Source | 5-ep eval |
|---|---|---|---|
| Top-Short | aug 55K | image-aug retrain | 50.00% exact packaged eval |
| Top-Long | aug 90K | image-aug retrain | 68.33% exact packaged eval |
| Pant-Long | **balanced-weak 090250** | aug90 + tiny mixed replay | **55.00% exact packaged eval** |
| Pant-Short | **aug 60K static** | image-aug retrain | **81.67% exact packaged eval** |
| Classifier | v2 (88% val acc) | unchanged | — |
| Stabilizer | rate-limit defaults | unchanged | — |

**Pant-Short v6 change:** the kNN portfolio is disabled in `submission/policy.py`
(`USE_PANTSHORT_PORTFOLIO = False`) and `submission/assemble_policies.sh`
packages `outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model`
as the Pant-Short specialist. Exact eval:
`/tmp/stabilized_router_pant_short_aug60.log` → **81.67%**, versus v5
Pant-Short portfolio `/tmp/router_v5_5ep_pant_short.log` → **73.33%**.

**Pant-Long v7 change:** `submission/assemble_policies.sh` packages
`outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model`.
This was trained from aug90 with a conservative +250-step, LR=1e-6 mixed-replay
update on the clean diverse Pant-Long base plus 10 weak-Seen successful demos
(Seen_8/Seen_9). Exact packaged CPU eval improved Pant-Long from **50.00%**
in v6 to **55.00%** (`/tmp/submission_bundle_pant_long.log`), raising the
full-stack estimate by +3 successes: **150/240 → 153/240**.

Fine-tunes from earlier harvested demos were not shipped: Top-Short
mixed10p_058000 full eval was **46.67%** and Pant-Long mixed10p_090500 full
eval was **55.00%**, both below their direct specialist baselines or not enough
to move the final stack materially.

**Stabilizer v6/v7 config:** `submission/policy.py` now delays gripper release only
for `{top_long, pant_short}`. Top-Short uses rate-limit only. Pant-Long uses the
full-sweep winner rate-only variant (`arm_scale=0.90`, `gripper_scale=1.30`).
The optional Pant-Long one-sided grasp retry is disabled by default
(`LEHOME_PANT_LONG_GRASP_RETRY=0`), so the shipped result is the conservative
rate-only winner mode.

## 1.0 Previous stack (v5 hybrid, kept for reference)

Pant-Short kNN portfolio routed by image embedding (top + left + right RGB
concatenated → 1536-D ResNet-18 features). At episode start: classify category.
If pant_short, find nearest Seen pant_short by image embedding, look up that
Seen's historically best checkpoint from `best_checkpoints.json`, route to it.
Other categories use the v4 single specialist.

## 1.1 Previous best (router v4, kept for reference)

Classifier-routed policy that picks one of 4 garment-category specialists:

| Garment cat | Specialist checkpoint                                       | Source method                  | Standalone 5-ep |
|-------------|-------------------------------------------------------------|---------------------------------|-----------------|
| Top-Short   | `outputs/train/act_top_short_aug/checkpoints/055000`        | **ACT + image aug (40K → 55K)**, swapped 2026-04-25 | 50.00% |
| Top-Long    | `outputs/train/act_top_long_aug/checkpoints/090000`         | **ACT + image aug (80K → 90K)**, swapped 2026-04-26 | 71.67% |
| Pant-Long   | `outputs/train/act_pant_long_aug/checkpoints/090000`        | ACT + image aug (80K → 90K), swapped 2026-04-23 | 58.33% |
| Pant-Short  | `golden_checkpoints/pant_short_45k`                         | ACT base train (45K steps)      | 83.33% |

**Router eval samples** (same submission stack):
- v4 (2026-04-26): **60.42%** (145/240)
- v5 (2026-04-26): **60.83%** (146/240)
- **Mean: 60.62%**, aggregate variance ±0.2pp

| Cat | v5 | v4 | v3 | v2 | v4/v5 mean |
|---|---|---|---|---|---|
| Top-Short | 41.67% | 50.00% | 43.33% | 36.67% | 45.84% |
| Top-Long | 75.00% | 61.67% | 66.67% | 63.33% | 68.34% |
| Pant-Long | 53.33% | 53.33% | 58.33% | 46.67% | 53.33% |
| Pant-Short | 73.33% | 76.67% | 73.33% | 73.33% | 75.00% |
| **Total** | **60.83%** | **60.42%** | 60.42% | 55.00% | **60.62%** |

v3 → v4: only changed Top-Long specialist (golden 80K → aug 90K, which improved standalone by +5pp). Net router result unchanged — the standalone gain washed out in 5-ep router noise. We keep the aug 90K swap because the standalone evidence is stronger than 5-ep router variance.

**Real submission improvement vs v2 baseline (52.92%):** +7.50pp (60.42% - 52.92%). Driven by Top-Short aug 55K swap (+6.66pp) and Pant-Long aug 90K swap (already in v2 → v3 baseline; aug 90K was the change). Top-Long aug 90K is a marginal swap (no router evidence either way; standalone +5pp).

## 2. Per-artifact methodology

### 2.1 Garment classifier
- File: `outputs/classifier/garment_classifier.pt`
- Script: `scripts/train_classifier.py`
- Architecture: ResNet-18 fine-tuned end-to-end (full network, not just head)
- Training data: 8 frames per episode from 4 categories' merged datasets
- Train/val split: leakage-safe by garment ID — Seen_0..Seen_7 train, Seen_8..Seen_9 val
- Augmentation: torchvision (RandomResizedCrop 224, ColorJitter, RandomHorizontalFlip)
- Validation accuracy: 88.12% (mean confidence 0.84). Confusion matrix shows top_short is the hardest class (~55% recall, memorizes textures).
- In-eval routing accuracy: ≥93% per category in 5-ep router runs
- Runtime config: confidence threshold 0 (always trust top-1 prediction). Override: `ROUTER_CONF_THRESHOLD` env var.

### 2.2 Top-Short specialist (golden 40K)
- Source: `outputs/train/act_top_short/checkpoints/040000` → copied to `golden_checkpoints/top_short_40k`
- Training script: `lerobot-train --config_path configs/train_act_top_short.yaml`
- Steps: 40K
- Dataset: `Datasets/example/top_short_merged` (250 episodes, 76066 frames)
- Image augmentation: **disabled** at this checkpoint
- Backbone: ResNet-18 ImageNet pretrained (frozen until end of train, then fine-tune; default ACT preset)
- Chunk size: 100, n_action_steps: 100
- Optimizer: AdamW, lr 1e-5, weight decay 1e-4
- Validation: per-step eval not run during training (LeRobot offline mode)

### 2.3 Top-Long specialist (golden v2 80K)
- Source: `outputs/train/act_top_long_v2/checkpoints/080000` → `golden_checkpoints/top_long_v2_80k`
- Training script: `lerobot-train --config_path configs/train_act_top_long_v2.yaml`
- Steps: 80K
- Dataset: `Datasets/example/top_long_merged`
- Image augmentation: disabled
- Other config: ACT defaults as in §2.2

### 2.4 Pant-Long specialist (balanced-weak 090250) — **swapped in 2026-04-30**
- Final source: `outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250`
- Base source: `outputs/train/act_pant_long_aug/checkpoints/090000`
- Dataset: `Datasets/teleop_merged_balanced_weak/pant_long_merged`, built from
  the clean diverse Pant-Long replay dataset plus 10 weak-Seen successful ACT
  demos on `Pant_Long_Seen_8` and `Pant_Long_Seen_9`.
- Training method: tiny conservative fine-tune, `lr=1e-6`, +250 steps selected
  by packaged CPU eval. Later checkpoints existed (090500/090750/091000) but
  were not selected because small-data Pant-Long fine-tunes repeatedly showed
  holdout collapse risk.
- Exact packaged eval:
  - v6 aug90 + winner stabilizer: **50.00%** (`/tmp/submission_bundle_pant_long_pantstab_winner.log`)
  - v7 balanced-weak 090250 + winner stabilizer: **55.00%** (`/tmp/submission_bundle_pant_long.log`)
  - v7 balanced-weak 090250 raw Pant-Long mode: **55.00%** (`/tmp/submission_bundle_pant_long_pantstab_raw.log`)
- Direct non-packaged diagnostic for 090250 reached **63.33%**
  (`/tmp/full_eval_pant_long_balanced_weak_090250.log`), but the exact packaged
  CPU policy run is the result of record.

### 2.4a Previous Pant-Long specialist (image-aug 90K) — **swapped in 2026-04-23**
- Source: `outputs/train/act_pant_long_aug/checkpoints/090000`
- Training method: **resume from golden 80K** (`golden_checkpoints/pant_long_80k`) **with image augmentation enabled**
- Resume: copied 80K → `outputs/train/act_pant_long_aug/checkpoints/080000`, edited `train_config.json`:
  - `dataset.image_transforms.enable: true`
  - `output_dir: outputs/train/act_pant_long_aug`
  - `steps: 100000` (additional 20K aug steps)
  - `save_freq: 5000`
  - `resume: true`, `checkpoint_path: outputs/train/act_pant_long_aug/checkpoints/last`
- Launched via: `lerobot-train --config_path=outputs/train/act_pant_long_aug/checkpoints/080000/pretrained_model/train_config.json --resume=true`
- Image transforms (LeRobot built-in):
  - ColorJitter brightness/contrast [0.8, 1.2]
  - ColorJitter saturation [0.5, 1.5], hue [-0.05, 0.05]
  - SharpnessJitter [0.5, 1.5]
  - RandomAffine degrees [-5, 5], translate [0.05, 0.05]
  - max 3 transforms per sample, fixed order
- Checkpoint sweep results (5-ep CPU, standalone, no router):
  - 85K: 50.00%
  - **90K: 58.33% ← selected**
  - 95K: 53.33%
  - 100K: 46.67%
- Selection criterion: highest 5-ep success rate; Unseen_0 jumped 40%→80% (validates aug hypothesis). This remained the default Pant-Long specialist through v6.

### 2.5 Pant-Short specialist (image-aug 60K) — **swapped in v6**
- Source: `outputs/train/act_pant_short_aug/checkpoints/060000`
- Resume source: `outputs/train/act_pant_short/checkpoints/045000` / `golden_checkpoints/pant_short_45k`
- Steps: 60K
- Dataset: `Datasets/example/pant_short_merged`
- Image augmentation: enabled during resume
- Earlier decision kept golden 45K because standalone 5-ep gains looked within noise. v6 reversed that after exact routed/stabilized eval:
  - v5 Pant-Short kNN portfolio: 73.33% (`/tmp/router_v5_5ep_pant_short.log`)
  - **Static aug60: 81.67%** (`/tmp/stabilized_router_pant_short_aug60.log`)
- Decision: package static aug60 and disable Pant-Short kNN portfolio.

### 2.6 Top-Short aug (55K selected)
- Source: golden 40K → `outputs/train/act_top_short_aug/checkpoints/040000`
- Training method: same image-aug recipe as Pant-Long (LeRobot `image_transforms.enable=true`, 20K additional steps, save every 5K)
- Setup: train_config.json edited to enable `image_transforms`, `steps: 60000`, `output_dir: outputs/train/act_top_short_aug`, `resume: true`
- Launch: `lerobot-train --config_path=outputs/train/act_top_short_aug/checkpoints/040000/pretrained_model/train_config.json --resume=true`
- Eval results (5-ep CPU, standalone):
  - 45K: 40.00% (-1.67pp vs router baseline)
  - 50K: 46.67% (+5pp; Seen_8 regressed 80→0%, Unseen_0 still 0%)
  - 55K: **50.00%** (+8.33pp; Seen_2/3=80%, Seen_4=100%, **Unseen_0 still 0%**)
  - 60K: not evaluated — Isaac Sim crashed at sim init (USD schema render_settings.rtx error in extscache)
- Per-garment 55K detail: Seen_0=60, Seen_1=40, Seen_2=80, Seen_3=80, Seen_4=100, Seen_5=20, Seen_6=40, Seen_7=60, Seen_8=60, Seen_9=0, Unseen_0=0, Unseen_1=60.
- Best Top-Short specialist by 5-ep CPU: 55K (+5 episodes over golden, beats the +3.33pp/+2-ep noise threshold).
- **Unseen_0 confirmed structural** — image aug alone cannot fix the missing collar fold (d(p[2],p[3]) consistently 24-29 cm across all aug checkpoints). Triggered scripted-collar-recovery work (§7).
- Decision: selected `outputs/train/act_top_short_aug/checkpoints/055000`. Final v7 uses rate-limit-only stabilizer for Top-Short after full eval showed delayed release regressed this category.

## 3. Stabilizer (eval-time wrapper)

- File: `scripts/eval_policy/policy_stabilizer.py`
- Submission policy type: `stabilized_router`
- Final submission config:
  - `rate_limit: on` (per-joint per-step delta cap, calibrated from demo data)
  - `delay_throws: on` for `top_long`, `pant_short`
  - `delay_throws: off` for `top_short`
  - `pant_long`: rate-only winner variant (`arm_scale=0.90`, `gripper_scale=1.30`)
  - `LEHOME_PANT_LONG_GRASP_RETRY=0` by default
  - `release_hold_steps: 0`
  - `rate_limit_scale: 1.00`, `gripper_rate_scale: 1.00`
- Calibration: `scripts/calibrate_stabilizer.py`. Caps from 95th percentile of per-step joint |Δ| in demo trajectories: arm 0.111 rad/step, gripper 0.104 rad/step.
- Numeric env-var overrides supported: `STABILIZER_RATE_LIMIT_SCALE`, `STABILIZER_GRIPPER_RATE_SCALE`, `STABILIZER_VEL_THRESHOLD`, `STABILIZER_RELEASE_HOLD_STEPS`. Currently all default.
- Stabilizer CEM sweep is complete; the winner mode remains the Pant-Long default.

## 4. Codebase changes that affect the submission

| File | Change | Why |
|---|---|---|
| `source/lehome/lehome/utils/success_checker_chanllege.py` | `_build_dedup_mapping` for PhysX cloth vertex dedup | Without it, check_point indices were off after PhysX deduplication; affects success determination consistency between training data and eval |
| `source/lehome/lehome/tasks/bedroom/garment_bi_v2.py` | (a) Reward function rewrite to potential-based shaping; (b) buffers in `__init__` / `_reset_idx` for `_prev_phis`, `_prev_min_phi`, `_prev_gripper_action`; (c) `_get_joint_position_tensor` helper | Reward changes only affect RL training (which we're not using in submission); buffer init still safe at eval time |
| `scripts/eval_policy/router_policy.py` | (a) `parents[2]` repo-root fix; (b) `ROUTER_CONF_THRESHOLD` env var; (c) `ROUTER_GT_CATEGORY` ground-truth log; (d) `GarmentClassifier` supports both v1 (linear-only) and v2 (full ResNet-18) checkpoints | Routing functional on the new classifier format |
| `scripts/eval_policy/policy_stabilizer.py` | Numeric env-var overrides + `release_hold_steps` (default 0 → no behavior change) | Enables stabilizer sweep without affecting current submission run |
| `scripts/utils/parser.py` + `scripts/utils/evaluation.py` | `--eval_list_override` flag for mini-suite eval | Lets sweep target specific garments; default unchanged |
| `scripts/eval.py` | Refuse `--device cuda` | Submission protocol is CPU-only |

## 5. Reproduction recipe (current submission)

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Sanity check final packaged checkpoints exist
ls outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model/model.safetensors
ls outputs/train/act_top_long_aug/checkpoints/090000/pretrained_model/model.safetensors
ls outputs/train/act_pant_long_balanced_weak_tiny/checkpoints/090250/pretrained_model/model.safetensors
ls outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model/model.safetensors
ls outputs/classifier/garment_classifier.pt

# 3. Assemble the exact submission/policies bundle.
cd submission
bash assemble_policies.sh
cd ..

# 4. Run exact packaged-policy eval per category.
for cat in top_short top_long pant_long pant_short; do
  bash scripts/router_experiments/eval_submission_bundle.sh "$cat"
done
```

## 6. Open work / closed paths

All experimental paths are now closed. Final state:

- **v7 packaged static Pant-Short + balanced-weak Pant-Long**: **63.75%** (153/240) — FINAL SUBMISSION STACK.
- **v6 packaged static Pant-Short + category-aware stabilizer**: **62.50%** (150/240) — superseded by v7 Pant-Long swap.
- **Router v3 (Top-Short aug 55K swap)**: 60.42% — DONE.
- **Router v4 (+ Top-Long aug 90K swap)**: 60.42% — DONE. Standalone Top-Long aug 90K was +5pp over golden but didn't lift router due to noise; kept the swap based on standalone evidence.
- **Top-Long aug training (80K → 100K)**: DONE. Best checkpoint 90K = 71.67% standalone.
- **Phase 2B vision wrapper eval**: ABORTED (gate failed, 0 suffix-attributable rescues).
- **Pant-Long balanced-weak tiny fine-tune**: DONE. 090250 exact packaged full 12-garment / 5-ep eval gave **55.00%**, +3 successes over v6 Pant-Long, so it is shipped as the Pant-Long specialist.
- **Pant-Long stabilizer CEM sweep**: DONE. Mini-suite winner (arm=0.90, grip=1.30) gave 75% on hard 4-garment / 2-ep mini-suite. Exact packaged full 12-garment / 5-ep eval on v6 aug90 gave **50.00%**, a small +1 success improvement over raw packaged Pant-Long (48.33%), so winner mode remains the Pant-Long default.

**Submission is locked at v7 = 63.75% from exact packaged-policy full eval logs.** Submission package assembled and checkpoint-copy verified.

## 6.1 Phase 1 (scripted collar-recovery) — ABORTED 2026-04-25

Per the approved plan (internal notes), Phase 1 had a hard 6h gate: produce ≥5 successful scripted-recovery demos on official Top-Short Seen garments. Result: **0 suffix-attributable rescues across 7 ACT-failure episodes** on Seen_5 + Seen_9. All 3 successes (in 10 episodes) terminated before the trigger fired (env steps 246-336, trigger at policy steps 290-350). The +1-episode swings vs ACT-only baseline are within 5-ep eval noise.

Two suffix designs were tried:
- **v1 (`top_short_collar_v1`)** — full open→approach→close→lift→fold-inward→release sequence. Result: destroyed ACT successes by releasing cloth ACT had grasped. On Top_Short_Seen_0 ep0, ACT had folded the collar (d(p[2],p[3]) = 7.64 ✓) by step 312, then v1 suffix released grippers and disturbed the cloth, ending with d(p[2],p[3]) = 23.33 ✗.
- **v2 (`top_short_collar_v2`)** — conservative compress-inward + lift-slightly + settle, keeping anchor gripper state. Result: too gentle — no observable effect on cloth, never rescued an ACT-failed episode.

Root cause: open-loop scripted folds cannot grasp cloth reactively without cloth-state feedback. The challenge protocol explicitly hides cloth-state observations at eval, so we cannot use check_point coordinates online. Without that feedback, blind joint deltas can't correctly grip a specific cloth feature (collar). The user's pre-launch warning ("cloth is reactive and contact-sensitive; a pure open-loop waypoint script may fail often") was correct.

**Procedural mesh-variant pipeline (Phase 5 in the plan) was never reached** — it was gated behind Phase 1 success and is not worth pursuing without a working scripted demo generator.

Files left in the tree but not used in submission:
- `scripts/eval_policy/scripted_collar_recovery_policy.py` (registered as `act_with_collar_recovery`)
- `outputs/scripted_recovery_v1/` and `outputs/scripted_recovery_v2/` (test runs)
- `submission/policies/` / `assemble_policies.sh` etc. unaffected

## 7. Recovery-policy attempts — all aborted

Three scripted/hand-designed correction approaches were tried, all failed:

| Attempt | Description | Result |
|---|---|---|
| Phase 1 v1 (`scripted_collar_recovery_policy.py`, v1) | Generic 8-phase open-loop fold sequence (open→approach→close→lift→fold→release) | Destroyed ACT's good work by releasing already-grasped cloth; 0/2 useful |
| Phase 1 v2 (same file, v2) | Conservative compress-inward + lift, keep anchor gripper state | Too gentle — no observable effect, 0 suffix-attributable rescues across 10 episodes |
| Phase 2 vision wrapper (`vision_collar_recovery_policy.py`) | Closed-loop visual servo: garment silhouette + gripper detection → proportional control toward landmarks | All 7 successes happened before the trigger fired (ACT-only); 0 suffix-attributable rescues |
| AI-teleop v1-v9 (`ai_teleop_policy.py`) | 9 iterations of per-garment offline-designed corrections on Top_Short_Unseen_0. v1-4: hand-tuned deltas / interp to target state. v5-8: replay actions from a successful Seen episode (Seen_2 ep2 / Seen_7 ep1). v9: state-aware trigger (only fire when grippers actually closed). | 0/5 success across all 9 versions. Returns improved monotonically: ACT-only -210 → v6 -108 → **v9 -101 (best mean)** with best individual episode at -58. But final distances show cloth body never actually folds (d(0,4)≈26 vs target 7.2, d(2,3)≈27 vs 11.25, d(1,5)≈29 vs 11.25). Scripted action injection has a hard ceiling without proprioceptive feedback — can move cloth but cannot reliably grasp+lift+fold. |
| AI-teleop random search (`ai_teleop_random_search.py`) | 30-trial random search over (trigger_step ∈ [180,240], gripper_tightness ∈ [-0.22, -0.12], template_length ∈ {60,80,100,120}, blend_steps ∈ {5,10,15}, hold_duration ∈ {10,20,30,40}) on Top_Short_Unseen_0, 2 episodes per trial. ~50 min CPU. | Best mean=-110.81 (similar to v9 manual best -101). Confirms the scripted-action-injection ceiling is ~-100 mean return regardless of parameters. No success transitions found. The cloth body cannot be reliably folded by open-loop scripted actions in this task. |
| CEM trajectory optimization (`scripts/cem_recovery/`) | Delta-knot parameterization (6 knots × 6 controllable joints, linearly interpolated over 100-step horizon, applied as deltas from snapshot state). Seeded from successful Seen_2 ep2 trajectory. Joint-clamping + per-step rate limiting to prevent physics divergence. Best-of-3-trials per candidate to handle cloth physics non-determinism. 6 candidates × 3 iters × 2 garments × 3 trials = 108 episodes. ~1.5 hours CPU. | 0/108 successes. CEM best score "+600" turned out to be max-of-3-trials artifact: 5-episode verification of saved best knots gave 0/5 success, avg return -248. Cloth physics non-determinism (same params → -42 vs -347 across identical runs) breaks black-box optimization. Final answer: scripted/optimized open-loop action injection has a hard ceiling on this task without proprioceptive feedback. |
| Portfolio router (`scripts/eval_policy/portfolio_router_policy.py`) | Per-garment best-checkpoint lookup driven by ResNet-18 image-embedding kNN over Seen garments. 19 candidate checkpoints loaded simultaneously (~3.8GB CPU RAM). At episode start, classify category + find nearest Seen by image embedding + look up that Seen's historically best checkpoint. Submission-compliant (only top_rgb at runtime). Full 4-cat × 12 garments × 5 ep eval. ~1.5h CPU. | **57.92% total — 2.5pp WORSE than v4 60.42%.** Per-cat: Top-Short 38.33% (-7.5pp vs mean), Top-Long 63.33% (-5pp), Pant-Long 51.67% (-1.7pp), Pant-Short 78.33% (+3.3pp). Failure mode: the per-(garment, ckpt) lookup was based on single 5-ep evals, which carry ±20pp per-cell noise. With ~12 cells per category, the "best" ckpt identified for a garment is rarely statistically meaningful. Oracle upper bound (perfect routing) was ~75% but achievable only with precise per-cell measurements requiring 50+ ep per cell — out of compute budget. |

**Common failure mode**: open-loop or weakly-closed-loop scripted actions on cloth cannot reactively grasp specific anatomical features without proprioceptive feedback. The challenge protocol explicitly hides cloth-state observations at eval (only state + RGB + depth), so action design has to work blind to where cloth fabric actually is. Image-based feature detection works for silhouette but not for anatomical features (collar/sleeve seams).

**Relevant code retained for traceability:** `scripts/eval_policy/{scripted,vision,ai_teleop}_*_policy.py`. None used in submission. Only the LeRobot ACT specialists + classifier + rate-limit stabilizer are in the submission stack.

## 8. Vision-guided collar-recovery pipeline (Phase 2 details)

Replaces the open-loop scripted suffix path (§6.1, aborted). Closed-loop visual servo using only public observations: `top_rgb`, `top_depth` (optional), `observation.state`. No `check_point` access at runtime.

### 7.1 Phase 2A — CV proof-of-life (PASSED, 2026-04-25)

Files:
- `scripts/cv_collar_pol/segment_and_landmark.py` — HSV-based garment segmentation (mask out yellow grippers + near-white table; keep largest connected component) + landmark extraction (centroid, bbox, top-left/top-right corners from top 25% band, gripper centroids from yellow blobs)
- `scripts/cv_collar_pol/analyze_videos.py` — runs the CV pipeline over saved failure videos, generates overlay frames + an overlay mp4 + a stability report

Source videos analyzed:
- `outputs/eval_videos/router_v2_5ep_top_short/failure/Top_Short_Unseen_0_episode{0,1,4}_observation_images_top_rgb.mp4`
- `outputs/scripted_recovery_v2/seen5_videos/{failure,success}/Top_Short_Seen_5_episode*_observation_images_top_rgb.mp4`
- `outputs/scripted_recovery_v2/seen9_videos/failure/Top_Short_Seen_9_episode0_observation_images_top_rgb.mp4`

Stability report (late-phase = step 350+):

| Video | Detection valid | Late stable | Median sh-sep px |
|---|---:|---:|---:|
| Unseen_0_ep0 (fail) | 100% | 70% | 252 |
| Unseen_0_ep1 (fail) | 100% | 100% | 118 |
| Unseen_0_ep4 (fail) | 100% | 76% | 162 |
| Seen_5_ep0 (fail) | 100% | 100% | 257 |
| Seen_5_ep2 (fail) | 100% | 100% | 228 |
| Seen_5_ep1 (success) | 100% | 100% | 242 |
| Seen_9_ep0 (fail) | 100% | 100% | 138 |

Gate (≥70% stable on majority of failure videos): **6/6 PASSED**.

Caveats:
- `shoulder_sep_px` measures silhouette top-band width, not the ground-truth `d(p[2], p[3])`. It does not discriminate folded-vs-unfolded reliably (Seen_5 success and failure both show ~240 px). The proof-of-life gate is on stability, not discrimination — landmarks are stable and visually plausible across all sampled videos.
- Detector is silhouette-based (not pattern-based) — works on patterned, plaid, and solid-colored garments, but localizes outer extents rather than anatomical collar/shoulder seams.

Overlay outputs in `outputs/cv_collar_pol/overlays/<video_label>/`.

### 7.2 Phase 2B — Closed-loop visual servo wrapper (in progress)

File: `scripts/eval_policy/vision_collar_recovery_policy.py`, registered as `act_with_vision_collar_recovery`. Sub-policy state machine:

1. ACT runs normally for the first part of the episode.
2. Trigger: stationarity (30 stationary policy steps) OR fixed step 350.
3. Suffix phases (in policy steps after trigger):
   - **approach (0-50)**: each gripper servoed toward the corresponding top-corner of the segmented garment. Proportional control on shoulder_pan / shoulder_lift / wrist_flex; max delta 0.04 rad/step.
   - **pinch (50-70)**: hold position, close grippers.
   - **fold (70-120)**: servo both grippers toward the garment centroid (centerline).
   - **release (120-150)**: open grippers.
   - **settle (150-200)**: hold position.
4. Detection failure → command current state (zero motion).

Action mapping (no IK to image frame): pixel-space error normalized to [-0.5, +0.5] of image dimension, multiplied by per-axis gain (K_SHOULDER_PAN=0.30, K_SHOULDER_LIFT=0.10, K_WRIST_FLEX=0.10), clipped to MAX_DELTA_PER_STEP=0.04 rad. Applied incrementally from previous action.

### 7.3 Phase 2B gate — FAILED 2026-04-25

Wrapper eval (5 ep × 4 garments = 20 episodes):

| Garment | Wrapper | aug 55K baseline | Δ episodes |
|---|---|---|---|
| Seen_5 | 60% (3/5) | 20% | +2 |
| Seen_9 | 40% (2/5) | 0% | +2 |
| Unseen_0 | 0% (0/5) | 0% | 0 |
| Unseen_1 | 40% (2/5) | 60% | -1 |
| **Total** | **35%** (7/20) | 20% (4/20) | +3 ep |

**All 7 wrapper successes had `episode_length < trigger_step`** — ACT terminated successfully before the visual servo suffix could fire. **Zero suffix-attributable rescues.** The +3 episode aggregate vs the ACT-only baseline is plausibly sim variance (we re-ran the same ACT specialist with a different seed effectively — same policy, different physics rollout).

Per plan gate (≥3 suffix-attributable rescues + ≤1 destroyed ACT success), Phase 2B failed. The vision-guided demo-collection path is aborted.

**Why this didn't help**: even though the CV pipeline detects landmarks reliably, ACT either (a) succeeds before the trigger fires (~step 350) or (b) fails irrecoverably. There's no clean "ACT got partway then stalled" intermediate state where servo correction can finish the fold. When ACT fails, the cloth ends up in a configuration the silhouette-based detector cannot decompose into anatomical landmarks — the centroid moves but the "top corners" are still just the outer extents of a crumpled blob.

The fundamental limit is **the silhouette is not the same as the anatomical structure**. Closing this gap would require either a learned segmentation model (collar/sleeve part-segmentation) or richer cloth-state reasoning (depth + mesh inference) — both out of scope at 4 days to deadline.

Submission stays on router v3 60.42%.

## 7. Variant + scripted-demo pipeline (in progress)

Built to address: **Top-Short Unseen_0 fails with d(p[2], p[3]) ≈ 24-29 cm vs 11.25 cm threshold**, every episode, both golden 40K and aug 50K. Failure is structural (collar never folded), not visual. Image augmentation cannot fix it; new training data with collar-fold demonstrations on shape-variant garments is required.

### 7.1 Pipeline stages

1. **Mesh deformation** (`scripts/garment_authoring/deform_garment.py`)
   - Inputs: existing seen-garment USD + JSON (e.g. `Top_Short_Seen_0/TCSC_067_obj_exp.{usd,json}`)
   - Operation: read mesh `points` attribute, identify regions by 3D distance to check_point anchors, apply per-region affine transforms (e.g. widen shoulders by stretching x-axis in p[0]/p[4] neighborhood, raise collar by translating z-axis in p[2]/p[3] neighborhood)
   - Output: new USD with deformed mesh, new JSON with same check_point indices (topology preserved → indices remain anatomically correct), updated `success_distance` thresholds re-derived from the deformed geometry
   - Tooling: `usd-core` (pxr) for read/write, numpy for vertex ops. No sim required.

2. **Category-fit verification** (`scripts/garment_authoring/verify_category.py`)
   - **Critical step.** A variant must still belong to the same category — otherwise it pollutes training and fools the classifier.
   - Hard constraints per category:
     - **Top-Short**: sleeve length ≤ 1.5× torso half-width, sleeve aspect ratio < 1.0 (sleeve span < torso height)
     - **Top-Long**: sleeve length > 1.5× torso half-width
     - **Pant-Short**: leg length ≤ 1.0× hip width
     - **Pant-Long**: leg length > 1.5× hip width
   - Geometric checks (cm, after scale 0.45):
     - Bounding-box aspect ratio matches reference distribution (mean ± 2σ from existing seen garments per category)
     - check_point pairwise distances within reference distribution
     - No self-intersection / degenerate triangles
   - Visual check via classifier:
     - Render the deformed garment in sim once (reset pose)
     - Pass top-down RGB through `outputs/classifier/garment_classifier.pt`
     - Predicted category must match the source category with confidence > 0.7
   - Reject any variant that fails any check; log to `outputs/garment_authoring/rejected.jsonl`.

3. **Procedural fold policy** (`scripts/eval_policy/scripted_fold_policy.py`)
   - `BasePolicy` subclass, registered as `scripted_fold`
   - Hand-coded waypoints in EE space per category: pre-grasp → grasp → lift → fold → place → release
   - For Top-Short specifically: an explicit collar-fold step (two-arm pinch at p[2]/p[3] region, lift, then fold across to opposite shoulder)
   - IK via existing `RobotKinematics` (Pinocchio)
   - Episode-state machine; reads only public observations (state + RGB), not check_point positions

4. **Scripted demo recording** (`scripts/utils/dataset_record.py` mini-patch + `scripts/record_scripted_demos.py`)
   - Add `--scripted_policy <name>` flag to dataset_record.py, falling through to PolicyRegistry instead of teleop
   - Run on each variant garment, save only successful episodes (success determined by existing checker)
   - Target: ~30-50 demos per category, balanced across variants

5. **Dataset merge + retrain**
   - Merge new demos into existing `Datasets/example/{cat}_merged` LeRobot dataset (preserving meta/info.json, garment_info.json updated to include variant IDs)
   - Retrain ACT specialist for the affected category(ies), starting from existing aug or golden checkpoint
   - Save every 5K steps as before

6. **Eval + selection**
   - 5-ep CPU eval per checkpoint
   - Compare to golden + aug 90K
   - Swap only if Unseen Success Rate clearly improves AND Seen does not regress
   - Document in §2.x of this file with full method trace

### 7.2 What gets logged for submission traceability

For every new variant garment authored, write to `outputs/garment_authoring/manifest.jsonl`:
```json
{
  "variant_id": "Top_Short_Variant_0001",
  "source_garment": "Top_Short_Seen_0/TCSC_067_obj_exp",
  "deformations": [
    {"op": "stretch_x", "region": "shoulder", "factor": 1.20, "anchor_idx": [964, 1354]},
    {"op": "translate_z", "region": "collar", "delta": 0.05, "anchor_idx": [8185, 9499]}
  ],
  "category_check": {"classifier_pred": "top_short", "conf": 0.82, "passed": true},
  "geometric_check": {"sleeve_torso_ratio": 0.95, "passed": true},
  "output_usd": "outputs/garment_authoring/Top_Short_Variant_0001/TCSC_var01_obj_exp.usd",
  "output_json": "outputs/garment_authoring/Top_Short_Variant_0001/TCSC_var01_obj_exp.json"
}
```

For every scripted demo recorded:
```json
{
  "demo_id": "scripted_top_short_v0001_ep00",
  "garment": "Top_Short_Variant_0001",
  "policy": "scripted_fold/top_short_v1",
  "success": true,
  "final_distances": {"d_0_4": 5.2, "d_2_3": 9.8, "d_1_5": 11.0},
  "episode_length": 412,
  "dataset_path": "Datasets/scripted/top_short_variant/episode_000.parquet"
}
```

Reproducibility: commit hash of `scripts/garment_authoring/`, `scripts/eval_policy/scripted_fold_policy.py`, and the procedural fold script version (e.g. `top_short_v1`) goes into the per-checkpoint methodology entry in §2.x.

### 7.3 Risk register

- **Variant shapes don't match challenge Unseen** — we don't know what the actual Unseen test garments look like; variant generation is a hedge. Mitigation: cover shape diversity along multiple axes (collar height, sleeve length, shoulder width, garment width).
- **Procedural fold fails on cloth dynamics** — scripted waypoints don't react to cloth state. Mitigation: only keep successful demos; reject failed ones from training.
- **ACT overfits to scripted action patterns** — synthetic demos may have less rich high-frequency behavior than human teleop. Mitigation: mix ratio (e.g. 80% human demos, 20% scripted) rather than full replacement.
- **Time budget vs deadline** — 4-5 days estimated, deadline 2026-04-30. Risk of incomplete pipeline. Fallback: existing 55% submission stays untouched in `golden_checkpoints/` and `outputs/train/act_pant_long_aug/checkpoints/090000`.
