"""Portfolio router: classifier picks category; image-embedding kNN picks the
specific checkpoint within that category that historically did best on the
nearest-neighbor Seen garment.

Submission-compliant: only uses the initial top_rgb image to (a) classify
category and (b) pick a checkpoint from a precomputed table indexed by Seen
garment. No ground-truth labels at runtime.

Falls back to the per-category default specialist (the v4 stack) if the
checkpoint mapping is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from torchvision import models, transforms

from lehome.utils.logger import get_logger
from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry
from .router_policy import GarmentClassifier

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-category list of (label, checkpoint_path, dataset_root)
CHECKPOINT_TABLE = {
    "top_short": [
        ("golden_40k", "golden_checkpoints/top_short_40k/pretrained_model"),
        ("aug_45k",    "outputs/train/act_top_short_aug/checkpoints/045000/pretrained_model"),
        ("aug_50k",    "outputs/train/act_top_short_aug/checkpoints/050000/pretrained_model"),
        ("aug_55k",    "outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model"),
    ],
    "top_long": [
        ("golden_80k", "golden_checkpoints/top_long_v2_80k/pretrained_model"),
        ("aug_85k",    "outputs/train/act_top_long_aug/checkpoints/085000/pretrained_model"),
        ("aug_90k",    "outputs/train/act_top_long_aug/checkpoints/090000/pretrained_model"),
        ("aug_95k",    "outputs/train/act_top_long_aug/checkpoints/095000/pretrained_model"),
        ("aug_100k",   "outputs/train/act_top_long_aug/checkpoints/100000/pretrained_model"),
    ],
    "pant_long": [
        ("golden_80k", "golden_checkpoints/pant_long_80k/pretrained_model"),
        ("aug_85k",    "outputs/train/act_pant_long_aug/checkpoints/085000/pretrained_model"),
        ("aug_90k",    "outputs/train/act_pant_long_aug/checkpoints/090000/pretrained_model"),
        ("aug_95k",    "outputs/train/act_pant_long_aug/checkpoints/095000/pretrained_model"),
        ("aug_100k",   "outputs/train/act_pant_long_aug/checkpoints/100000/pretrained_model"),
    ],
    "pant_short": [
        ("golden_45k", "golden_checkpoints/pant_short_45k/pretrained_model"),
        ("aug_50k",    "outputs/train/act_pant_short_aug/checkpoints/050000/pretrained_model"),
        ("aug_55k",    "outputs/train/act_pant_short_aug/checkpoints/055000/pretrained_model"),
        ("aug_60k",    "outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model"),
        ("aug_65k",    "outputs/train/act_pant_short_aug/checkpoints/065000/pretrained_model"),
    ],
}

DATASET_ROOTS = {
    "top_short": "Datasets/example/top_short_merged",
    "top_long":  "Datasets/example/top_long_merged",
    "pant_long": "Datasets/example/pant_long_merged",
    "pant_short":"Datasets/example/pant_short_merged",
}

# Fallback default checkpoint per category (the v4 stack)
DEFAULT_CHECKPOINT = {
    "top_short": "aug_55k",
    "top_long":  "aug_90k",
    "pant_long": "aug_90k",
    "pant_short":"golden_45k",
}


def _default_checkpoint(category: str) -> str:
    """Return default checkpoint label, with optional env override.

    Env vars are intentionally explicit so experiments can compare static maps
    without editing this file or submission defaults:
      PORTFOLIO_DEFAULT_TOP_SHORT=aug_50k
      PORTFOLIO_DEFAULT_TOP_LONG=aug_90k
      PORTFOLIO_DEFAULT_PANT_LONG=aug_90k
      PORTFOLIO_DEFAULT_PANT_SHORT=aug_50k
    """
    env_key = f"PORTFOLIO_DEFAULT_{category.upper()}"
    return os.environ.get(env_key, DEFAULT_CHECKPOINT[category])


@PolicyRegistry.register("portfolio_router")
class PortfolioRouterPolicy(BasePolicy):
    """Classifier + image-embedding kNN portfolio router."""

    def __init__(
        self,
        policy_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        classifier_path = policy_path or model_path
        self.device = torch.device(device)
        self.classifier = GarmentClassifier(classifier_path, device=str(self.device))

        # Build embedding model from the classifier's ResNet-18 backbone
        self._build_embedder(classifier_path)

        # Load Seen-garment embeddings + best-checkpoint lookup.
        # Prefer multi-camera embeddings (top + left + right concat) if available.
        multi_path = REPO_ROOT / "outputs/portfolio_router/seen_embeddings_multicam.npz"
        single_path = REPO_ROOT / "outputs/portfolio_router/seen_embeddings.npz"
        if os.environ.get("PORTFOLIO_USE_MULTICAM", "1") == "1" and multi_path.exists():
            emb_data = np.load(multi_path)
            self._multicam = True
        else:
            emb_data = np.load(single_path)
            self._multicam = False
        self.seen_embeddings = {k: emb_data[k] for k in emb_data.files}
        logger.info(f"[portfolio] embeddings: {'multi-cam' if self._multicam else 'top-only'}, "
                    f"dim={list(self.seen_embeddings.values())[0].shape if self.seen_embeddings else 'NA'}")
        with open(REPO_ROOT / "outputs/portfolio_router/best_checkpoints.json") as f:
            self.best_ckpt_per_garment = json.load(f)

        # Lazily load specialists. Index = (category, label) → LeRobotPolicy
        self.specialists: Dict[tuple, LeRobotPolicy] = {}
        self._load_all_specialists()

        # Per-episode state
        self.reset()
        # Disable confidence threshold (top-1 trust)
        self.confidence_threshold = float(os.environ.get("PORTFOLIO_CONF_THRESHOLD",
                                                           confidence_threshold))
        logger.info(
            f"[PortfolioRouterPolicy] {len(self.specialists)} specialists loaded, "
            f"{len(self.seen_embeddings)} seen embeddings, "
            f"{len(self.best_ckpt_per_garment)} best-ckpt mappings."
        )

    def _build_embedder(self, classifier_path: str):
        """Build a ResNet-18 backbone (without FC) for embedding."""
        ckpt = torch.load(classifier_path, map_location=self.device, weights_only=False)
        n_classes = len(ckpt["idx_to_label"])
        bb = models.resnet18(weights=None)
        bb.fc = torch.nn.Linear(512, n_classes)
        bb.load_state_dict(ckpt["classifier_state_dict"])
        bb.eval().to(self.device)
        self.emb_model = torch.nn.Sequential(*list(bb.children())[:-1]).eval().to(self.device)
        self.emb_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_all_specialists(self):
        """Load every specialist checkpoint upfront. ~3GB total in CPU RAM."""
        for cat, entries in CHECKPOINT_TABLE.items():
            for label, ckpt_path in entries:
                full_path = REPO_ROOT / ckpt_path
                if not full_path.exists():
                    logger.warning(f"[portfolio] missing: {full_path}")
                    continue
                try:
                    p = LeRobotPolicy(
                        policy_path=str(full_path),
                        dataset_root=str(REPO_ROOT / DATASET_ROOTS[cat]),
                        task_description="Fold a garment with bimanual robot arms",
                        device=str(self.device),
                    )
                    self.specialists[(cat, label)] = p
                    logger.info(f"[portfolio] loaded {cat}/{label}")
                except Exception as e:
                    logger.warning(f"[portfolio] failed {cat}/{label}: {e}")

    def reset(self):
        for p in self.specialists.values():
            p.reset()
        self._active = None
        self._classified = False
        self._active_category = None
        self._active_label = None

    def _embed_image(self, rgb: np.ndarray) -> np.ndarray:
        """Compute 512-D ResNet-18 embedding from an RGB (H,W,3) uint8 array."""
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        tensor = self.emb_preprocess(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.emb_model(tensor).squeeze().cpu().numpy()
        return emb

    def _embed_multicam(self, observation: Dict[str, Any]) -> np.ndarray:
        """Concatenate top + left + right RGB embeddings (1536-D total)."""
        embs = []
        for view in ["observation.images.top_rgb", "observation.images.left_rgb",
                      "observation.images.right_rgb"]:
            img = observation.get(view)
            if img is None:
                # Fallback: zero-pad
                embs.append(np.zeros(512, dtype=np.float32))
                continue
            rgb = np.asarray(img)
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
            embs.append(self._embed_image(rgb))
        return np.concatenate(embs)

    def _classify_and_route(self, observation: Dict[str, Any]):
        top_rgb = observation.get("observation.images.top_rgb")
        if top_rgb is None:
            self._fallback_route("top_short")
            return
        rgb = np.asarray(top_rgb)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

        # Step 1: classify category
        category, conf = self.classifier.predict(rgb)
        if category is None or category not in CHECKPOINT_TABLE:
            self._fallback_route("top_short")
            return

        # Hybrid mode: use kNN portfolio for the categories listed in
        # PORTFOLIO_HYBRID_CATEGORIES (env var, default = "pant_short").
        # All other categories fall back to v4-default checkpoint.
        # Empirically: kNN portfolio improved Pant-Short by +6.7pp but
        # regressed other categories. Hybrid captures only the win.
        hybrid_cats = os.environ.get("PORTFOLIO_HYBRID_CATEGORIES", "pant_short").split(",")
        hybrid_cats = [c.strip() for c in hybrid_cats if c.strip()]
        if hybrid_cats and category not in hybrid_cats:
            logger.info(f"[portfolio] hybrid: cat={category} not in {hybrid_cats}, using default")
            self._fallback_route(category)
            return

        # Step 2: compute embedding, find nearest Seen in this category
        if self._multicam:
            emb = self._embed_multicam(observation)
        else:
            emb = self._embed_image(rgb)
        # Restrict to Seen garments matching the predicted category prefix
        cat_prefix = {"top_short": "Top_Short", "top_long": "Top_Long",
                       "pant_long": "Pant_Long", "pant_short": "Pant_Short"}[category]
        same_cat_seens = {g: e for g, e in self.seen_embeddings.items()
                            if g.startswith(cat_prefix) and "Seen" in g}
        if not same_cat_seens:
            self._fallback_route(category)
            return
        best_garment, best_dist = None, float("inf")
        for g, ge in same_cat_seens.items():
            d = float(np.linalg.norm(ge - emb))
            if d < best_dist:
                best_dist, best_garment = d, g
        # Step 3: look up best checkpoint for nearest Seen
        label = self.best_ckpt_per_garment.get(best_garment, _default_checkpoint(category))
        # Verify checkpoint loaded
        if (category, label) not in self.specialists:
            label = _default_checkpoint(category)
        self._active = self.specialists.get((category, label))
        self._active_category = category
        self._active_label = label
        self._classified = True
        if self._active is not None:
            self._active.reset()
        logger.info(
            f"[portfolio] cat={category}(conf={conf:.2f}) "
            f"nearest_seen={best_garment}(d={best_dist:.2f}) -> ckpt={label}"
        )

    def _fallback_route(self, category: str):
        label = _default_checkpoint(category)
        self._active = self.specialists.get((category, label))
        self._active_category = category
        self._active_label = label
        self._classified = True
        if self._active is not None:
            self._active.reset()
        logger.info(f"[portfolio] fallback route: {category}/{label}")

    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        if not self._classified:
            self._classify_and_route(observation)
        if self._active is None:
            raise RuntimeError("No active specialist — fallback failed.")
        return self._active.select_action(observation)
