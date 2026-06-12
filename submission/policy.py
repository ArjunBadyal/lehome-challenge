"""LeHome Challenge submission policy.

Stack:
  - Garment category classifier (frozen ResNet-18 + linear head) — routes to a
    per-category ACT specialist at the start of each episode, using only the
    top-down RGB image from the first observation.
  - Per-category ACT specialists (chunk_size=100) for Top-Long, Top-Short,
    Pant-Long, Pant-Short. Final stack uses image-aug specialists for all
    categories, with Pant-Short fixed to aug60 instead of the noisy kNN
    portfolio.
  - Eval-time action stabilizer: per-joint rate limit for Top-Long,
    Top-Short, Pant-Long, and Pant-Short, plus delayed gripper release for
    Top-Long and Pant-Short. Pant-Long uses a rate-only "winner" variant
    (slightly tighter arm motion, looser gripper motion) because the exact
    packaged rerun improved from 48.33% raw to 50.00%.

Submission-compliant:
  - Classifier input is derived from the public observation dict
    (observation.images.top_rgb only).
  - All specialists and the unified fallback are ACT policies trained by
    lerobot-train, using only public demo data.
  - No simulator-internal signals (condition_margins, checkpoint positions,
    task metrics) are read at inference time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from server import BasePolicyServer

# LeRobot/HF metadata loading creates cache lock files even for local datasets.
# Force /tmp by default so the submission container and local smoke tests do
# not depend on a writable home directory. Use LEHOME_HF_HOME only if a caller
# explicitly wants a different writable cache.
_HF_HOME = os.environ.get("LEHOME_HF_HOME", "/tmp/lehome_hf")
os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.environ.get("LEHOME_HF_DATASETS_CACHE", str(Path(_HF_HOME) / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
try:
    import datasets.config as _datasets_config
    _datasets_config.HF_CACHE_HOME = Path(_HF_HOME)
    _datasets_config.HF_DATASETS_CACHE = Path(os.environ["HF_DATASETS_CACHE"])
    _datasets_config.HF_MODULES_CACHE = Path(_HF_HOME) / "modules"
except Exception:
    pass


# ---------------------------------------------------------------------------
# Paths (inside the container)
# ---------------------------------------------------------------------------

POLICIES_ROOT = Path(os.environ.get("POLICIES_ROOT", "/app/policies"))

CLASSIFIER_PATH = POLICIES_ROOT / "classifier" / "garment_classifier.pt"

SPECIALIST_PATHS = {
    "top_long":   POLICIES_ROOT / "specialists" / "top_long"   / "pretrained_model",
    "top_short":  POLICIES_ROOT / "specialists" / "top_short"  / "pretrained_model",
    "pant_long":  POLICIES_ROOT / "specialists" / "pant_long"  / "pretrained_model",
    "pant_short": POLICIES_ROOT / "specialists" / "pant_short" / "pretrained_model",
}
SPECIALIST_DATASET_ROOTS = {
    "top_long":   POLICIES_ROOT / "datasets" / "top_long_merged",
    "top_short":  POLICIES_ROOT / "datasets" / "top_short_merged",
    "pant_long":  POLICIES_ROOT / "datasets" / "pant_long_merged",
    "pant_short": POLICIES_ROOT / "datasets" / "pant_short_merged",
}
UNIFIED_PATH = POLICIES_ROOT / "unified" / "pretrained_model"
UNIFIED_DATASET_ROOT = POLICIES_ROOT / "datasets" / "four_types_merged"

# Pant-Short portfolio support is retained for traceability, but disabled in
# the final stack. The per-garment kNN portfolio was noisy; a static aug60
# specialist evaluated better (81.67% vs 73.33% for v5 portfolio).
USE_PANTSHORT_PORTFOLIO = False
PANTSHORT_PORTFOLIO_PATHS = {
    "aug_50k": POLICIES_ROOT / "portfolio" / "pant_short" / "aug_50k" / "pretrained_model",
    "aug_55k": POLICIES_ROOT / "portfolio" / "pant_short" / "aug_55k" / "pretrained_model",
    "aug_60k": POLICIES_ROOT / "portfolio" / "pant_short" / "aug_60k" / "pretrained_model",
    "aug_65k": POLICIES_ROOT / "portfolio" / "pant_short" / "aug_65k" / "pretrained_model",
}
PORTFOLIO_EMBEDDINGS_PATH = POLICIES_ROOT / "portfolio" / "seen_embeddings_multicam.npz"
PORTFOLIO_BEST_CKPT_PATH = POLICIES_ROOT / "portfolio" / "best_checkpoints.json"

# ---------------------------------------------------------------------------
# Stabilizer thresholds (calibrated from demos — see scripts/calibrate_stabilizer.py)
# ---------------------------------------------------------------------------

MAX_JOINT_DELTA_ARM = 0.111      # rad/step, 95th percentile of demo arm motion
MAX_JOINT_DELTA_GRIPPER = 0.104  # rad/step, 95th percentile of demo gripper motion
PER_JOINT_MAX_DELTA = np.array([
    MAX_JOINT_DELTA_ARM,      # 0  L shoulder pan
    MAX_JOINT_DELTA_ARM,      # 1  L shoulder lift
    MAX_JOINT_DELTA_ARM,      # 2  L elbow
    MAX_JOINT_DELTA_ARM,      # 3  L wrist flex
    MAX_JOINT_DELTA_ARM,      # 4  L wrist roll
    MAX_JOINT_DELTA_GRIPPER,  # 5  L gripper
    MAX_JOINT_DELTA_ARM,      # 6  R shoulder pan
    MAX_JOINT_DELTA_ARM,      # 7  R shoulder lift
    MAX_JOINT_DELTA_ARM,      # 8  R elbow
    MAX_JOINT_DELTA_ARM,      # 9  R wrist flex
    MAX_JOINT_DELTA_ARM,      # 10 R wrist roll
    MAX_JOINT_DELTA_GRIPPER,  # 11 R gripper
], dtype=np.float32)

ACTION_DIM = 12
CATEGORY_ORDER = ["top_long", "top_short", "pant_long", "pant_short"]
LEFT_GRIPPER_IDX = 5
RIGHT_GRIPPER_IDX = 11
LEFT_ARM_INDICES = np.array([0, 1, 2, 3, 4])
RIGHT_ARM_INDICES = np.array([6, 7, 8, 9, 10])
ARM_HIGH_VELOCITY = 0.111
DELAY_THROWS_CATEGORIES = {"top_long", "pant_short"}
NO_STABILIZER_CATEGORIES = {"pant_long"}
PANT_LONG_STABILIZER_MODE = os.environ.get("LEHOME_PANT_LONG_STABILIZER", "winner").lower()
PANT_LONG_GRASP_RETRY = os.environ.get("LEHOME_PANT_LONG_GRASP_RETRY", "0") == "1"
PANT_LONG_BLEND_ALPHA = float(os.environ.get("LEHOME_PANT_LONG_BLEND_ALPHA", "0.0"))
PANT_LONG_BLEND_START_STEP = int(os.environ.get("LEHOME_PANT_LONG_BLEND_START_STEP", "0"))
PANT_LONG_BLEND_END_STEP = int(os.environ.get("LEHOME_PANT_LONG_BLEND_END_STEP", "1000000"))
PANT_LONG_BLEND_MIN_CONF = float(os.environ.get("LEHOME_PANT_LONG_BLEND_MIN_CONF", "0.0"))
PANT_LONG_BLEND_PATH = Path(
    os.environ.get(
        "LEHOME_PANT_LONG_BLEND_PATH",
        str(POLICIES_ROOT / "blend" / "pant_long_candidate" / "pretrained_model"),
    )
)
GRIPPER_OPEN_CMD = 0.50
GRIPPER_CLOSED_CMD = -0.18
GRIPPER_CLOSE_CMD_THRESHOLD = -0.08
GRIPPER_EMPTY_STATE_THRESHOLD = -0.147
GRIPPER_HELD_STATE_THRESHOLD = -0.143
PANT_LONG_WINNER_DELTA = np.array(PER_JOINT_MAX_DELTA, copy=True)
PANT_LONG_WINNER_DELTA[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]] *= 0.90
PANT_LONG_WINNER_DELTA[[LEFT_GRIPPER_IDX, RIGHT_GRIPPER_IDX]] *= 1.30
PANT_LONG_RETRY_MAX_STEP = 190
PANT_LONG_RETRY_OPEN_STEPS = 8
PANT_LONG_RETRY_CLOSE_STEPS = 18
PANT_LONG_RETRY_TOTAL_STEPS = PANT_LONG_RETRY_OPEN_STEPS + PANT_LONG_RETRY_CLOSE_STEPS


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class GarmentClassifier:
    """Garment category classifier.

    Supports both checkpoint formats used during development:
    - v1: ImageNet ResNet-18 backbone + saved Linear head only.
    - v2: full fine-tuned ResNet-18 state dict with `fc` classifier.
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.device = device
        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        self.idx_to_label = ckpt["idx_to_label"]
        state = ckpt["classifier_state_dict"]
        num_classes = len(self.idx_to_label)
        self.IMAGENET_MEAN = self.IMAGENET_MEAN.to(device)
        self.IMAGENET_STD = self.IMAGENET_STD.to(device)

        if "conv1.weight" in state:
            self.format = "v2"
            self.normalize = True
            self.resize_hw = (224, 224)
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(512, num_classes)
            self.model.load_state_dict(state)
            self.model.eval().to(device)
            self.backbone = None
            self.head = None
        else:
            self.format = "v1"
            self.normalize = False
            self.resize_hw = None
            self.model = None
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone.fc = nn.Identity()
            self.backbone.eval().to(device)
            self.head = nn.Linear(512, num_classes)
            self.head.load_state_dict(state)
            self.head.eval().to(device)

    @torch.no_grad()
    def predict(self, top_rgb: np.ndarray) -> tuple[str, float]:
        img = torch.from_numpy(top_rgb).float().permute(2, 0, 1) / 255.0
        img = img.unsqueeze(0).to(self.device)
        if self.resize_hw is not None:
            img = torch.nn.functional.interpolate(
                img, size=self.resize_hw, mode="bilinear", align_corners=False
            )
        if self.normalize:
            img = (img - self.IMAGENET_MEAN) / self.IMAGENET_STD

        if self.format == "v2":
            logits = self.model(img)
        else:
            features = self.backbone(img)
            logits = self.head(features)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return self.idx_to_label[idx.item()], conf.item()


# ---------------------------------------------------------------------------
# LeRobot ACT wrapper
# ---------------------------------------------------------------------------

class ACTWrapper:
    """Loads a LeRobot ACT checkpoint and exposes select_action(observation_dict)."""

    def __init__(self, policy_path: Path, dataset_root: Path, device: torch.device):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.processor.core import TransitionKey

        self.device = device
        self.TransitionKey = TransitionKey

        meta = LeRobotDatasetMetadata(repo_id="lehome", root=str(dataset_root))
        policy_cfg = PreTrainedConfig.from_pretrained(str(policy_path), cli_overrides={})
        policy_cfg.pretrained_path = str(policy_path)
        # Match scripts/eval_policy/lerobot_policy.py: force the checkpoint
        # load path to the requested inference device before make_policy().
        policy_cfg.device = str(self.device)

        self.input_features = set(policy_cfg.input_features.keys()) if hasattr(policy_cfg, "input_features") else None
        if self.input_features:
            self._filter_metadata(meta, self.input_features)

        self.policy = make_policy(policy_cfg, ds_meta=meta)
        self.policy.eval().to(device)

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_path),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        self.action_dim = ACTION_DIM

    @staticmethod
    def _filter_metadata(meta, expected_keys):
        dataset_features = set(meta.features.keys())
        system_features = {"timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"}
        extra = dataset_features - expected_keys - system_features
        for feature in list(extra):
            if feature.startswith("observation."):
                del meta.features[feature]

    def reset(self):
        self.policy.reset()

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        # Filter to policy-expected keys
        if self.input_features:
            observation = {k: v for k, v in observation.items()
                           if not k.startswith("observation.") or k in self.input_features}

        # Build tensor batch
        obs_for_preproc = {}
        for key, value in observation.items():
            if not key.startswith("observation."):
                continue
            if isinstance(value, np.ndarray):
                t = torch.from_numpy(value).float()
                if value.ndim == 3 and value.shape[-1] == 3:  # (H,W,C) → (C,H,W), /255
                    t = t.permute(2, 0, 1).to(self.device) / 255.0
                    obs_for_preproc[key] = t.unsqueeze(0)
                else:
                    obs_for_preproc[key] = t.unsqueeze(0)
            else:
                obs_for_preproc[key] = value

        dummy_action = torch.zeros(1, self.action_dim, dtype=torch.float32, device=self.device)
        transition = {
            self.TransitionKey.OBSERVATION: obs_for_preproc,
            self.TransitionKey.ACTION: dummy_action,
            self.TransitionKey.COMPLEMENTARY_DATA: {"task": "Fold a garment with bimanual robot arms"},
        }

        transformed = self.preprocessor._forward(transition)
        batch_obs = self.preprocessor.to_output(transformed)

        with torch.inference_mode():
            batch_action = self.policy.select_action(batch_obs)
        if self.postprocessor:
            batch_action = self.postprocessor(batch_action)
        return batch_action.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Submission policy
# ---------------------------------------------------------------------------

class PantShortPortfolio:
    """Multi-checkpoint Pant-Short policy with kNN routing.

    Holds the default golden_45k specialist plus aug_50/55/60/65k. At reset,
    knows nothing; at first select_action call, an outer router calls
    `pick_checkpoint(observation)` to choose which specialist to run. Then
    delegates select_action to it for the rest of the episode.
    """

    def __init__(self, default_specialist: ACTWrapper, alt_specialists: Dict[str, ACTWrapper],
                 embeddings_path: Path, best_ckpt_path: Path, device: torch.device,
                 emb_model, emb_preprocess):
        self.default = default_specialist
        self.alts = alt_specialists  # {label: ACTWrapper}
        self.device = device
        self.emb_model = emb_model
        self.emb_preprocess = emb_preprocess
        self.seen_embeddings: Dict[str, np.ndarray] = {}
        self.best_lookup: Dict[str, str] = {}
        if embeddings_path.exists():
            data = np.load(embeddings_path)
            self.seen_embeddings = {k: data[k] for k in data.files
                                     if k.startswith("Pant_Short")}
        if best_ckpt_path.exists():
            import json
            with open(best_ckpt_path) as f:
                self.best_lookup = {k: v for k, v in json.load(f).items()
                                     if k.startswith("Pant_Short")}
        self._active: ACTWrapper | None = None

    def reset(self):
        self.default.reset()
        for s in self.alts.values():
            s.reset()
        self._active = None

    def _embed(self, observation):
        """Top + left + right RGB → 1536-D concat embedding."""
        embs = []
        for view in ["observation.images.top_rgb", "observation.images.left_rgb",
                      "observation.images.right_rgb"]:
            img = observation.get(view)
            if img is None:
                embs.append(np.zeros(512, dtype=np.float32))
                continue
            rgb = np.asarray(img)
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            t = self.emb_preprocess(rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                e = self.emb_model(t).squeeze().cpu().numpy()
            embs.append(e)
        return np.concatenate(embs)

    def _pick(self, observation) -> ACTWrapper:
        if not self.seen_embeddings or not self.best_lookup:
            return self.default
        try:
            emb = self._embed(observation)
        except Exception as e:
            print(f"[PantShortPortfolio] embed failed: {e}, using default", flush=True)
            return self.default
        # Nearest Seen pant_short
        best_garment, best_dist = None, float("inf")
        for g, ge in self.seen_embeddings.items():
            if "Seen" not in g: continue
            d = float(np.linalg.norm(ge - emb))
            if d < best_dist:
                best_dist, best_garment = d, g
        if best_garment is None:
            return self.default
        label = self.best_lookup.get(best_garment, "golden_45k")
        if label == "golden_45k" or label not in self.alts:
            chosen = self.default
            print(f"[PantShortPortfolio] nearest={best_garment} → golden_45k (default)", flush=True)
        else:
            chosen = self.alts[label]
            print(f"[PantShortPortfolio] nearest={best_garment} → {label}", flush=True)
        return chosen

    def select_action(self, observation):
        if self._active is None:
            self._active = self._pick(observation)
        return self._active.select_action(observation)


class LeHomePolicy(BasePolicyServer):
    """Classifier + static per-category specialist router."""

    def __init__(self):
        requested_device = os.environ.get("LEHOME_POLICY_DEVICE", "cpu").lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        print(f"[LeHomePolicy] Using device: {self.device}", flush=True)

        # Classifier
        self.classifier = GarmentClassifier(CLASSIFIER_PATH, self.device)
        print(f"[LeHomePolicy] Loaded classifier (labels: {list(self.classifier.idx_to_label.values())})", flush=True)

        self.specialists: Dict[str, object] = {}
        self.unified = None
        self._active: ACTWrapper | None = None
        self._active_category: str | None = None
        self._active_confidence: float = 0.0
        self._pant_long_blend: ACTWrapper | None = None
        self._classified = False
        self._prev_action: np.ndarray | None = None
        self._prev_state: np.ndarray | None = None
        self._step_count = 0
        self._pant_long_retry_countdown = 0
        self._pant_long_retry_arm: str | None = None
        self._pant_long_retries_used = {"left": 0, "right": 0}
        self._pant_long_empty_counts = {"left": 0, "right": 0}

        # Embedding model for Pant-Short kNN routing (disabled in final stack).
        if USE_PANTSHORT_PORTFOLIO:
            self._build_embedder()

        # Specialists (load only those present on disk)
        for cat in CATEGORY_ORDER:
            ppath = SPECIALIST_PATHS[cat]
            droot = SPECIALIST_DATASET_ROOTS[cat]
            if ppath.exists() and droot.exists():
                try:
                    default = ACTWrapper(ppath, droot, self.device)
                    print(f"[LeHomePolicy] Loaded specialist: {cat}", flush=True)
                    if cat == "pant_short" and USE_PANTSHORT_PORTFOLIO:
                        # Wrap in portfolio with extra checkpoints
                        alts = {}
                        for label, p in PANTSHORT_PORTFOLIO_PATHS.items():
                            if p.exists():
                                try:
                                    alts[label] = ACTWrapper(p, droot, self.device)
                                    print(f"[LeHomePolicy] Loaded portfolio: pant_short/{label}", flush=True)
                                except Exception as e:
                                    print(f"[LeHomePolicy] FAILED portfolio {label}: {e}", flush=True)
                        self.specialists[cat] = PantShortPortfolio(
                            default_specialist=default, alt_specialists=alts,
                            embeddings_path=PORTFOLIO_EMBEDDINGS_PATH,
                            best_ckpt_path=PORTFOLIO_BEST_CKPT_PATH,
                            device=self.device,
                            emb_model=self._emb_model, emb_preprocess=self._emb_preprocess,
                        )
                    else:
                        self.specialists[cat] = default
                except Exception as e:
                    print(f"[LeHomePolicy] FAILED to load specialist {cat}: {e}", flush=True)

        if PANT_LONG_BLEND_ALPHA > 0.0 and PANT_LONG_BLEND_PATH.exists():
            try:
                self._pant_long_blend = ACTWrapper(
                    PANT_LONG_BLEND_PATH,
                    SPECIALIST_DATASET_ROOTS["pant_long"],
                    self.device,
                )
                print(
                    f"[LeHomePolicy] Loaded Pant-Long blend candidate "
                    f"(alpha={PANT_LONG_BLEND_ALPHA:.3f}, start={PANT_LONG_BLEND_START_STEP}, "
                    f"end={PANT_LONG_BLEND_END_STEP}, min_conf={PANT_LONG_BLEND_MIN_CONF:.3f})",
                    flush=True,
                )
            except Exception as e:
                print(f"[LeHomePolicy] FAILED to load Pant-Long blend candidate: {e}", flush=True)

        # Unified fallback
        if UNIFIED_PATH.exists() and UNIFIED_DATASET_ROOT.exists():
            try:
                self.unified = ACTWrapper(UNIFIED_PATH, UNIFIED_DATASET_ROOT, self.device)
                print("[LeHomePolicy] Loaded unified fallback", flush=True)
            except Exception as e:
                print(f"[LeHomePolicy] FAILED to load unified: {e}", flush=True)

        if not self.specialists and self.unified is None:
            raise RuntimeError("No policies loaded — check POLICIES_ROOT and checkpoint paths")

    def _build_embedder(self):
        from torchvision import models, transforms
        ckpt = torch.load(CLASSIFIER_PATH, map_location=self.device, weights_only=False)
        n_classes = len(ckpt["idx_to_label"])
        bb = models.resnet18(weights=None)
        bb.fc = torch.nn.Linear(512, n_classes)
        bb.load_state_dict(ckpt["classifier_state_dict"])
        bb.eval().to(self.device)
        self._emb_model = torch.nn.Sequential(*list(bb.children())[:-1]).eval().to(self.device)
        self._emb_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def reset(self):
        self._classified = False
        self._active = None
        self._active_category = None
        self._active_confidence = 0.0
        self._prev_action = None
        self._prev_state = None
        self._step_count = 0
        self._pant_long_retry_countdown = 0
        self._pant_long_retry_arm = None
        self._pant_long_retries_used = {"left": 0, "right": 0}
        self._pant_long_empty_counts = {"left": 0, "right": 0}
        for p in self.specialists.values():
            p.reset()
        if self._pant_long_blend is not None:
            self._pant_long_blend.reset()
        if self.unified is not None:
            self.unified.reset()

    def _classify_and_route(self, observation: Dict[str, np.ndarray]):
        top_rgb = observation.get("observation.images.top_rgb")
        if top_rgb is not None:
            category, confidence = self.classifier.predict(top_rgb)
            print(f"[LeHomePolicy] Classified: {category} (conf={confidence:.3f})", flush=True)
            # Threshold 0 = always trust top-1 specialist prediction.
            # Matches the router_v4 eval that scored 60.42%. The classifier
            # is at 94% in-eval accuracy on the public dataset; a misrouted
            # specialist still produces sensible behavior because all four
            # ACT specialists were trained on related cloth manipulation.
            if category in self.specialists and confidence >= 0.0:
                self._active = self.specialists[category]
                self._active_category = category
                self._active_confidence = confidence
                self._classified = True
                self._active.reset()
                if category == "pant_long" and self._pant_long_blend is not None:
                    self._pant_long_blend.reset()
                return

        # Fallbacks
        if self.unified is not None:
            self._active = self.unified
            self._active_category = "unified"
            self._active_confidence = 0.0
            print("[LeHomePolicy] Using unified fallback", flush=True)
        elif self.specialists:
            cat = next(iter(self.specialists))
            self._active = self.specialists[cat]
            self._active_category = cat
            self._active_confidence = 0.0
            print(f"[LeHomePolicy] No unified; falling back to specialist: {cat}", flush=True)
        else:
            raise RuntimeError("No policy available for inference")
        self._classified = True
        self._active.reset()

    def _maybe_blend_pant_long(self, base_action: np.ndarray, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if (
            self._active_category != "pant_long"
            or self._pant_long_blend is None
            or PANT_LONG_BLEND_ALPHA <= 0.0
            or self._active_confidence < PANT_LONG_BLEND_MIN_CONF
        ):
            return base_action

        candidate_action = self._pant_long_blend.select_action(observation).astype(np.float32)
        if not (PANT_LONG_BLEND_START_STEP <= self._step_count <= PANT_LONG_BLEND_END_STEP):
            return base_action

        alpha = float(np.clip(PANT_LONG_BLEND_ALPHA, 0.0, 1.0))
        return ((1.0 - alpha) * base_action + alpha * candidate_action).astype(np.float32)

    def _apply_pant_long_grasp_retry(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Retry a one-sided empty grasp before Pant-Long starts folding.

        This is deliberately conservative and opt-in. It uses only public robot
        joint state: when both grippers are commanded closed but one side is at
        the empty hard-stop while the other side stopped early, infer that the
        hard-stop side missed cloth. Retry only that gripper, while keeping the
        successful gripper closed.
        """
        if (
            not PANT_LONG_GRASP_RETRY
            or self._active_category != "pant_long"
            or state.shape[0] < ACTION_DIM
            or self._step_count > PANT_LONG_RETRY_MAX_STEP
        ):
            return action

        if self._pant_long_retry_countdown > 0 and self._pant_long_retry_arm is not None:
            retry = action.copy()
            arm = self._pant_long_retry_arm
            gidx = LEFT_GRIPPER_IDX if arm == "left" else RIGHT_GRIPPER_IDX
            other_gidx = RIGHT_GRIPPER_IDX if arm == "left" else LEFT_GRIPPER_IDX
            retry[other_gidx] = GRIPPER_CLOSED_CMD
            if self._pant_long_retry_countdown > PANT_LONG_RETRY_CLOSE_STEPS:
                retry[gidx] = GRIPPER_OPEN_CMD
            else:
                retry[gidx] = GRIPPER_CLOSED_CMD
            self._pant_long_retry_countdown -= 1
            if self._pant_long_retry_countdown == 0:
                self._pant_long_retry_arm = None
            return retry

        left_cmd_closed = action[LEFT_GRIPPER_IDX] < GRIPPER_CLOSE_CMD_THRESHOLD
        right_cmd_closed = action[RIGHT_GRIPPER_IDX] < GRIPPER_CLOSE_CMD_THRESHOLD
        if not (left_cmd_closed and right_cmd_closed):
            self._pant_long_empty_counts = {"left": 0, "right": 0}
            return action

        left_state = float(state[LEFT_GRIPPER_IDX])
        right_state = float(state[RIGHT_GRIPPER_IDX])
        left_empty = left_state < GRIPPER_EMPTY_STATE_THRESHOLD
        right_empty = right_state < GRIPPER_EMPTY_STATE_THRESHOLD
        left_held = left_state > GRIPPER_HELD_STATE_THRESHOLD
        right_held = right_state > GRIPPER_HELD_STATE_THRESHOLD

        if left_empty and right_held:
            self._pant_long_empty_counts["left"] += 1
        else:
            self._pant_long_empty_counts["left"] = 0
        if right_empty and left_held:
            self._pant_long_empty_counts["right"] += 1
        else:
            self._pant_long_empty_counts["right"] = 0

        for arm in ("left", "right"):
            if self._pant_long_empty_counts[arm] >= 4 and self._pant_long_retries_used[arm] < 1:
                self._pant_long_retries_used[arm] += 1
                self._pant_long_retry_arm = arm
                self._pant_long_retry_countdown = PANT_LONG_RETRY_TOTAL_STEPS
                print(
                    f"[PantLongGraspRetry] retrying {arm} empty grasp at step={self._step_count} "
                    f"state_l={left_state:.3f} state_r={right_state:.3f}",
                    flush=True,
                )
                return self._apply_pant_long_grasp_retry(action, state)

        return action

    def _apply_stabilizer(self, action: np.ndarray, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Category-aware eval-time action stabilizer.

        Top-Short regressed with delayed gripper release, so it uses rate limit
        only. Pant-Long uses a rate-only "winner" variant by default: tighter
        arm deltas and looser gripper deltas, with no delayed release. Other
        categories keep release-delay plus rate limit, matching the strongest
        routed eval evidence.
        """
        state = np.asarray(observation.get("observation.state", []), dtype=np.float32)
        pant_long_mode = PANT_LONG_STABILIZER_MODE if self._active_category == "pant_long" else None
        action = self._apply_pant_long_grasp_retry(action, state)
        if self._active_category in NO_STABILIZER_CATEGORIES and pant_long_mode in (None, "none", "off", "raw"):
            self._prev_action = action.astype(np.float32).copy()
            self._prev_state = state.copy() if state.shape[0] >= ACTION_DIM else None
            self._step_count += 1
            return action.astype(np.float32)

        if (
            self._active_category in DELAY_THROWS_CATEGORIES
            and state.shape[0] >= ACTION_DIM
            and self._prev_state is not None
            and self._prev_action is not None
        ):
            arm_vel_left = np.abs(state[LEFT_ARM_INDICES] - self._prev_state[LEFT_ARM_INDICES]).max()
            arm_vel_right = np.abs(state[RIGHT_ARM_INDICES] - self._prev_state[RIGHT_ARM_INDICES]).max()
            for gidx, prev_g, arm_vel in [
                (LEFT_GRIPPER_IDX, self._prev_action[LEFT_GRIPPER_IDX], arm_vel_left),
                (RIGHT_GRIPPER_IDX, self._prev_action[RIGHT_GRIPPER_IDX], arm_vel_right),
            ]:
                opening = action[gidx] > prev_g + 0.1
                if opening and arm_vel > ARM_HIGH_VELOCITY:
                    action[gidx] = prev_g

        if self._prev_action is None:
            self._prev_action = action.astype(np.float32).copy()
            self._prev_state = state.copy() if state.shape[0] >= ACTION_DIM else None
            self._step_count += 1
            return action
        delta = action - self._prev_action
        max_delta = PANT_LONG_WINNER_DELTA if pant_long_mode == "winner" else PER_JOINT_MAX_DELTA
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        smoothed = (self._prev_action + delta_clipped).astype(np.float32)
        self._prev_action = smoothed
        self._prev_state = state.copy() if state.shape[0] >= ACTION_DIM else None
        self._step_count += 1
        return smoothed

    def infer(self, observation: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if not self._classified:
            self._classify_and_route(observation)

        raw = self._active.select_action(observation).astype(np.float32)
        raw = self._maybe_blend_pant_long(raw, observation)
        stabilized = self._apply_stabilizer(raw, observation)
        return [stabilized]  # single-action chunk


if __name__ == "__main__":
    LeHomePolicy().run()
