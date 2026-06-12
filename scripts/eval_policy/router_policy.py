"""Router policy: classifies garment category, delegates to the best specialist.

At episode start, uses a frozen ResNet-18 + linear classifier on the top-down
camera image to determine the garment category. Then delegates all subsequent
select_action() calls to the category-specific specialist policy.

Submission-compliant: uses only public observations for classification.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from .base_policy import BasePolicy
from .lerobot_policy import LeRobotPolicy
from .registry import PolicyRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

CATEGORY_ORDER = ["top_short", "top_long", "pant_long", "pant_short"]

SPECIALIST_CONFIGS = {
    "top_short": {
        "policy_path": "outputs/train/act_top_short_aug/checkpoints/055000/pretrained_model",
        "dataset_root": "Datasets/example/top_short_merged",
    },
    "top_long": {
        "policy_path": "outputs/train/act_top_long_aug/checkpoints/090000/pretrained_model",
        "dataset_root": "Datasets/example/top_long_merged",
    },
    "pant_long": {
        "policy_path": "outputs/train/act_pant_long_aug/checkpoints/090000/pretrained_model",
        "dataset_root": "Datasets/example/pant_long_merged",
    },
    "pant_short": {
        "policy_path": "outputs/train/act_pant_short_aug/checkpoints/060000/pretrained_model",
        "dataset_root": "Datasets/example/pant_short_merged",
    },
}

UNIFIED_CONFIG = {
    "policy_path": "outputs/train/act_all_cats/checkpoints/last/pretrained_model",
    "dataset_root": "Datasets/example/four_types_merged",
}


class GarmentClassifier:
    """Garment-category classifier. Supports both old and new checkpoint formats.

    - Old format (v1): ImageNet-frozen ResNet-18 backbone + trained Linear(512, 4) head.
      `classifier_state_dict` contains only the linear head's weights.
    - New format (v2): Fine-tuned ResNet-18 with `fc` replaced by Linear(512, 4).
      `classifier_state_dict` contains the full model's state_dict.
      Trained with ImageNet normalization and 224×224 resize.
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.idx_to_label = ckpt["idx_to_label"]
        state = ckpt["classifier_state_dict"]
        num_classes = len(self.idx_to_label)

        # Detect format: v2 has "conv1.weight" (full model), v1 has only "weight"/"bias"
        if "conv1.weight" in state:
            self.format = "v2"
            self.normalize = True
            self.resize_hw = (224, 224)
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(512, num_classes)
            model.load_state_dict(state)
            model.eval().to(self.device)
            self.model = model
            print(
                f"[GarmentClassifier] Loaded v2 (fine-tuned ResNet-18) "
                f"val_acc={ckpt.get('val_accuracy', 'n/a')} "
                f"mean_conf={ckpt.get('mean_confidence', 'n/a')}",
                flush=True,
            )
        else:
            self.format = "v1"
            self.normalize = False
            self.resize_hw = None
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Identity()
            backbone.eval().to(self.device)
            head = nn.Linear(512, num_classes)
            head.load_state_dict(state)
            head.eval().to(self.device)
            self.backbone = backbone
            self.head = head
            self.model = None  # sentinel
            print(f"[GarmentClassifier] Loaded v1 (linear probe)", flush=True)

        self.IMAGENET_MEAN = self.IMAGENET_MEAN.to(self.device)
        self.IMAGENET_STD = self.IMAGENET_STD.to(self.device)

    @torch.no_grad()
    def predict(self, top_rgb: np.ndarray) -> tuple[str, float]:
        """Classify garment from top-down RGB image.

        Args:
            top_rgb: (H, W, 3) uint8 numpy array

        Returns:
            (category_name, confidence)
        """
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
        label = self.idx_to_label[idx.item()]
        return label, conf.item()


@PolicyRegistry.register("router")
class RouterPolicy(BasePolicy):
    """Routes garments to category-specific specialist policies.

    On the first observation of each episode, classifies the garment category
    from the top-down camera image, then delegates to the specialist for
    that category. Falls back to the unified ACT if the specialist checkpoint
    doesn't exist or classifier confidence is too low.
    """

    def __init__(self, model_path: str = None, device: str = "cuda",
                 confidence_threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        import os
        self.device = device
        # Allow override via env var (0 = always trust top-1 specialist)
        env_threshold = os.environ.get("ROUTER_CONF_THRESHOLD")
        if env_threshold is not None:
            confidence_threshold = float(env_threshold)
        self.confidence_threshold = confidence_threshold
        print(f"[RouterPolicy] confidence_threshold={self.confidence_threshold}", flush=True)

        # Load classifier
        classifier_path = model_path or str(REPO_ROOT / "outputs" / "classifier" / "garment_classifier.pt")
        self.classifier = GarmentClassifier(classifier_path, device=device)

        # Load specialist policies (only those that exist)
        self.specialists: dict[str, LeRobotPolicy] = {}
        for cat, cfg in SPECIALIST_CONFIGS.items():
            policy_path = str(REPO_ROOT / cfg["policy_path"])
            dataset_root = str(REPO_ROOT / cfg["dataset_root"])
            if os.path.exists(policy_path):
                try:
                    self.specialists[cat] = LeRobotPolicy(
                        policy_path=policy_path,
                        dataset_root=dataset_root,
                        task_description="Fold a garment with bimanual robot arms",
                        device=device,
                    )
                    print(f"[RouterPolicy] Loaded specialist: {cat}", flush=True)
                except Exception as e:
                    print(f"[RouterPolicy] Failed to load specialist {cat}: {e}", flush=True)

        # Unified fallback
        unified_path = str(REPO_ROOT / UNIFIED_CONFIG["policy_path"])
        unified_root = str(REPO_ROOT / UNIFIED_CONFIG["dataset_root"])
        if os.path.exists(unified_path):
            self.unified = LeRobotPolicy(
                policy_path=unified_path,
                dataset_root=unified_root,
                task_description="Fold a garment with bimanual robot arms",
                device=device,
            )
            print("[RouterPolicy] Loaded unified fallback", flush=True)
        else:
            self.unified = None
            print("[RouterPolicy] WARNING: No unified fallback available", flush=True)

        self._active_policy: LeRobotPolicy | None = None
        self._classified = False
        self._active_category: str | None = None

    def reset(self):
        self._classified = False
        self._active_policy = None
        self._active_category = None
        for p in self.specialists.values():
            p.reset()
        if self.unified:
            self.unified.reset()

    def select_action(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        if not self._classified:
            self._classify_and_route(observation)

        return self._active_policy.select_action(observation)

    def _classify_and_route(self, observation: dict[str, np.ndarray]):
        top_rgb = observation.get("observation.images.top_rgb")

        if top_rgb is not None:
            category, confidence = self.classifier.predict(top_rgb)
            # Ground-truth category passed via env var by the eval script so
            # we can audit misclassifications.
            import os
            gt = os.environ.get("ROUTER_GT_CATEGORY", "")
            verdict = "✓" if gt and category == gt else ("✗" if gt else "?")
            print(
                f"[RouterPolicy] Classified: {category} (conf={confidence:.3f}) "
                f"gt={gt or 'unknown'} {verdict}",
                flush=True,
            )

            # Always trust the top-1 specialist prediction. The unified
            # fallback is trained on only ~15K steps and performs worse than
            # even a misrouted specialist in practice.
            if category in self.specialists and (
                self.confidence_threshold <= 0.0 or confidence >= self.confidence_threshold
            ):
                self._active_policy = self.specialists[category]
                self._active_category = category
                self._classified = True
                self._active_policy.reset()
                return

        # Fallback to unified (only reached if confidence check is strict)
        if self.unified:
            self._active_policy = self.unified
            self._active_category = "unified"
            print("[RouterPolicy] Using unified fallback", flush=True)
        elif self.specialists:
            # Last resort: use any available specialist
            cat = next(iter(self.specialists))
            self._active_policy = self.specialists[cat]
            self._active_category = cat
            print(f"[RouterPolicy] No unified, falling back to {cat}", flush=True)
        else:
            raise RuntimeError("No policies available")

        self._classified = True
        self._active_policy.reset()
