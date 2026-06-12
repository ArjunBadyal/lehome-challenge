"""Seen-garment identity portfolio router.

This router is intentionally conservative:
  1. A 40-way classifier predicts a specific Seen garment from the public top RGB.
  2. If confidence is high enough, route to that Seen garment's best checkpoint.
  3. Otherwise fall back to a static category checkpoint inferred by the existing
     4-way category classifier.

It does not read simulator garment labels at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from lehome.utils.logger import get_logger
from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .portfolio_router_policy import CHECKPOINT_TABLE, DATASET_ROOTS, _default_checkpoint
from .registry import PolicyRegistry
from .router_policy import GarmentClassifier

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]

PREFIX_TO_CATEGORY = {
    "Top_Short": "top_short",
    "Top_Long": "top_long",
    "Pant_Long": "pant_long",
    "Pant_Short": "pant_short",
}


class SeenGarmentClassifier:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.idx_to_label = {int(k): v for k, v in ckpt["idx_to_label"].items()}
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, len(self.idx_to_label))
        model.load_state_dict(ckpt["classifier_state_dict"])
        model.eval().to(self.device)
        self.model = model
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        logger.info(
            "[SeenRouter] loaded seen classifier val_acc=%s mean_conf=%s",
            ckpt.get("val_accuracy"),
            ckpt.get("mean_top1_confidence"),
        )

    @torch.no_grad()
    def predict(self, top_rgb: np.ndarray) -> tuple[str, float]:
        if top_rgb.dtype != np.uint8:
            top_rgb = (
                (np.clip(top_rgb, 0, 1) * 255).astype(np.uint8)
                if top_rgb.max() <= 1.0
                else top_rgb.astype(np.uint8)
            )
        tensor = self.preprocess(top_rgb).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return self.idx_to_label[int(idx.item())], float(conf.item())


@PolicyRegistry.register("seen_garment_router")
class SeenGarmentRouterPolicy(BasePolicy):
    def __init__(
        self,
        policy_path: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        confidence_threshold: float = 0.70,
        **kwargs,
    ):
        super().__init__()
        seen_ckpt = (
            policy_path
            or model_path
            or str(REPO_ROOT / "outputs/seen_garment_router/seen_garment_classifier.pt")
        )
        cat_ckpt = os.environ.get(
            "SEEN_ROUTER_CATEGORY_CKPT",
            str(REPO_ROOT / "outputs/classifier/garment_classifier.pt"),
        )
        self.device = torch.device(device)
        self.seen_classifier = SeenGarmentClassifier(seen_ckpt, device=str(self.device))
        self.category_classifier = GarmentClassifier(cat_ckpt, device=str(self.device))
        self.conf_threshold = float(
            os.environ.get("SEEN_ROUTER_CONF_THRESHOLD", confidence_threshold)
        )
        self.use_seen_categories = {
            c.strip()
            for c in os.environ.get(
                "SEEN_ROUTER_CATEGORIES", "top_short,top_long,pant_long,pant_short"
            ).split(",")
            if c.strip()
        }
        map_path = Path(
            os.environ.get(
                "SEEN_ROUTER_MAP_PATH",
                str(REPO_ROOT / "outputs/portfolio_router/best_checkpoints.json"),
            )
        )
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        with open(map_path) as f:
            self.best_ckpt_per_garment = json.load(f)
        logger.info("[SeenRouter] map=%s entries=%d", map_path, len(self.best_ckpt_per_garment))
        self.specialists: Dict[tuple[str, str], LeRobotPolicy] = {}
        self.reset()
        logger.info(
            "[SeenRouter] threshold=%.3f categories=%s",
            self.conf_threshold,
            sorted(self.use_seen_categories),
        )

    def reset(self):
        for policy in self.specialists.values():
            policy.reset()
        self._active: Optional[LeRobotPolicy] = None
        self._classified = False
        self._active_category: Optional[str] = None
        self._active_label: Optional[str] = None

    def _label_to_category(self, seen_label: str) -> Optional[str]:
        for prefix, category in PREFIX_TO_CATEGORY.items():
            if seen_label.startswith(prefix + "_Seen_"):
                return category
        return None

    def _load_specialist(self, category: str, label: str) -> Optional[LeRobotPolicy]:
        key = (category, label)
        if key in self.specialists:
            return self.specialists[key]
        entry = next((e for e in CHECKPOINT_TABLE[category] if e[0] == label), None)
        if entry is None:
            logger.warning("[SeenRouter] unknown checkpoint label %s/%s", category, label)
            return None
        _, ckpt_rel = entry
        ckpt = REPO_ROOT / ckpt_rel
        if not ckpt.exists():
            logger.warning("[SeenRouter] missing checkpoint %s", ckpt)
            return None
        policy = LeRobotPolicy(
            policy_path=str(ckpt),
            dataset_root=str(REPO_ROOT / DATASET_ROOTS[category]),
            task_description="Fold a garment with bimanual robot arms",
            device=str(self.device),
        )
        self.specialists[key] = policy
        return policy

    def _fallback_route(self, category: str) -> None:
        label = _default_checkpoint(category)
        self._active = self._load_specialist(category, label)
        self._active_category = category
        self._active_label = label
        if self._active is not None:
            self._active.reset()
        logger.info("[SeenRouter] fallback route: %s/%s", category, label)

    def _route(self, observation: Dict[str, Any]) -> None:
        top_rgb = observation.get("observation.images.top_rgb")
        if top_rgb is None:
            self._fallback_route("top_short")
            self._classified = True
            return
        rgb = np.asarray(top_rgb)
        if rgb.dtype != np.uint8:
            rgb = (
                (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                if rgb.max() <= 1.0
                else rgb.astype(np.uint8)
            )

        seen_label, seen_conf = self.seen_classifier.predict(rgb)
        seen_category = self._label_to_category(seen_label)
        if (
            seen_category in self.use_seen_categories
            and seen_conf >= self.conf_threshold
        ):
            ckpt_label = self.best_ckpt_per_garment.get(
                seen_label, _default_checkpoint(seen_category)
            )
            self._active = self._load_specialist(seen_category, ckpt_label)
            if self._active is not None:
                self._active_category = seen_category
                self._active_label = ckpt_label
                self._active.reset()
                logger.info(
                    "[SeenRouter] seen route: %s conf=%.3f -> %s/%s",
                    seen_label,
                    seen_conf,
                    seen_category,
                    ckpt_label,
                )
                self._classified = True
                return

        category, cat_conf = self.category_classifier.predict(rgb)
        if category not in CHECKPOINT_TABLE:
            category = "top_short"
        logger.info(
            "[SeenRouter] low-conf seen=%s conf=%.3f; category=%s conf=%.3f",
            seen_label,
            seen_conf,
            category,
            cat_conf,
        )
        self._fallback_route(category)
        self._classified = True

    def select_action(self, observation: Dict[str, Any]) -> np.ndarray:
        if not self._classified:
            self._route(observation)
        if self._active is None:
            raise RuntimeError("SeenGarmentRouterPolicy failed to select an active policy")
        return self._active.select_action(observation)
