"""Train a garment-category classifier from demo dataset frames.

Improvements over v1:
- Samples 8 frames per episode (steps 0..50) instead of only step 0
- Leakage-safe split by garment ID (train Seen_0-7, val Seen_8-9)
- Image augmentation (crop, flip, color-jitter, rotation)
- Unfrozen ResNet-18 fine-tune (not just linear probe)
- Target: val acc >= 99%, mean correct-class softmax conf >= 0.85

The dataset merged order is sequential by garment: episodes 0..24 = Seen_0,
episodes 25..49 = Seen_1, etc. Verified empirically by mean RGB signature.

Output (same format as v1 so GarmentClassifier in router_policy.py loads it):
    outputs/classifier/garment_classifier.pt
        { classifier_state_dict, idx_to_label, label_map, val_accuracy,
          mean_confidence, backbone: "resnet18" }
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

REPO_ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = {
    "top_short": "Datasets/example/top_short_merged",
    "top_long": "Datasets/example/top_long_merged",
    "pant_long": "Datasets/example/pant_long_merged",
    "pant_short": "Datasets/example/pant_short_merged",
}
LABEL_MAP = {name: idx for idx, name in enumerate(CATEGORIES)}
IDX_TO_LABEL = {idx: name for name, idx in LABEL_MAP.items()}

EPISODES_PER_GARMENT = 25
NUM_GARMENTS = 10
# Train on Seen_0..Seen_7 (8 garments → eps 0..199), val on Seen_8, Seen_9 (eps 200..249)
TRAIN_GARMENTS = list(range(8))
VAL_GARMENTS = [8, 9]
FRAMES_PER_EP = 8
# Sample frames between step 0 and step 50 (garment is settling; robot has not
# done anything meaningful yet so the view matches what the classifier sees
# at the start of each eval episode).
FRAME_STEP_RANGE = (0, 50)


def extract_frames_and_labels():
    """Return (images_uint8_NHWC, labels, is_train) as numpy/torch arrays.

    Uses meta/episodes parquet to find `dataset_from_index` for each episode,
    then reads the LeRobotDataset at those exact frame indices. Frames are
    returned as uint8 (H, W, C) in [0, 255] — augmentation handles float and
    normalisation later.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = np.random.default_rng(42)

    images, labels, is_train_flags = [], [], []

    for cat_name, root in CATEGORIES.items():
        print(f"[INFO] {cat_name}: loading {root}", flush=True)
        ds = LeRobotDataset(repo_id="lehome", root=str(REPO_ROOT / root))
        meta_df = pd.read_parquet(
            REPO_ROOT / root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        total_eps = len(meta_df)
        assert total_eps == NUM_GARMENTS * EPISODES_PER_GARMENT, (
            f"{cat_name}: expected {NUM_GARMENTS * EPISODES_PER_GARMENT} episodes, "
            f"got {total_eps}"
        )

        train_count = val_count = 0
        for ep_idx in range(total_eps):
            garment_idx = ep_idx // EPISODES_PER_GARMENT
            if garment_idx in TRAIN_GARMENTS:
                is_train = True
                train_count += 1
            elif garment_idx in VAL_GARMENTS:
                is_train = False
                val_count += 1
            else:
                continue

            ep_meta = meta_df.iloc[ep_idx]
            start = int(ep_meta["dataset_from_index"])
            ep_len = int(ep_meta["length"])
            # Sample FRAMES_PER_EP indices from [0, min(ep_len, FRAME_STEP_RANGE[1]))
            max_off = min(ep_len, FRAME_STEP_RANGE[1])
            offsets = rng.choice(max_off, size=FRAMES_PER_EP, replace=False)

            for off in offsets:
                frame = ds[start + int(off)]
                # frame["observation.images.top_rgb"] is (C, H, W), float [0, 1]
                img = (frame["observation.images.top_rgb"] * 255.0).to(torch.uint8)
                images.append(img)  # keep as (C, H, W) tensor
                labels.append(LABEL_MAP[cat_name])
                is_train_flags.append(is_train)

        print(
            f"  {cat_name}: {train_count} train episodes × {FRAMES_PER_EP} frames = "
            f"{train_count * FRAMES_PER_EP}, "
            f"{val_count} val episodes × {FRAMES_PER_EP} = {val_count * FRAMES_PER_EP}",
            flush=True,
        )

    images_t = torch.stack(images)  # (N, 3, H, W) uint8
    labels_t = torch.tensor(labels, dtype=torch.long)
    is_train_t = torch.tensor(is_train_flags, dtype=torch.bool)
    return images_t, labels_t, is_train_t


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        return img, self.labels[idx]


def build_transforms(train: bool):
    base = [
        T.ToDtype(torch.float32, scale=True),  # uint8 → float [0,1]
    ]
    if train:
        aug = [
            T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            T.RandomRotation(degrees=15, fill=1.0),
        ]
    else:
        aug = [T.Resize(size=(224, 224), antialias=True)]
    norm = [
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return T.Compose(base + aug + norm)


def build_model(num_classes: int = 4) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    conf_sum = 0.0
    per_class_correct = torch.zeros(4, dtype=torch.long)
    per_class_total = torch.zeros(4, dtype=torch.long)
    confusion = torch.zeros(4, 4, dtype=torch.long)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        # Mean confidence of the correct class
        conf_sum += probs[torch.arange(len(labels)), labels].sum().item()
        for c in range(4):
            mask = labels == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (preds[mask] == c).sum().item()
        for t, p in zip(labels.cpu(), preds.cpu()):
            confusion[t, p] += 1
    return {
        "accuracy": correct / max(total, 1),
        "mean_confidence": conf_sum / max(total, 1),
        "per_class_accuracy": (per_class_correct.float() / per_class_total.float().clamp(min=1)).tolist(),
        "confusion": confusion.tolist(),
        "total": total,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)

    images, labels, is_train = extract_frames_and_labels()
    print(
        f"[INFO] Total frames: {len(images)} "
        f"(train: {is_train.sum().item()}, val: {(~is_train).sum().item()})",
        flush=True,
    )
    print(f"[INFO] Image tensor shape: {tuple(images.shape)} dtype={images.dtype}", flush=True)

    train_images = images[is_train]
    train_labels = labels[is_train]
    val_images = images[~is_train]
    val_labels = labels[~is_train]

    train_ds = ImageDataset(train_images, train_labels, build_transforms(train=True))
    val_ds = ImageDataset(val_images, val_labels, build_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    model = build_model(num_classes=4).to(device)
    # Separate LR for backbone vs head
    head_params = list(model.fc.parameters())
    head_ids = set(id(p) for p in head_params)
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_acc = 0.0
    best_metrics = None
    best_state = None

    for epoch in range(1, 31):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * lbls.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == lbls).sum().item()
            train_total += lbls.size(0)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        train_acc = train_correct / train_total
        train_loss /= train_total
        print(
            f"Epoch {epoch:2d}/30  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  val_conf={val_metrics['mean_confidence']:.4f}",
            flush=True,
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    print("\n=== Best validation results ===", flush=True)
    print(f"  Val accuracy     : {best_metrics['accuracy']:.4f}", flush=True)
    print(f"  Mean confidence  : {best_metrics['mean_confidence']:.4f}", flush=True)
    for cat, idx in LABEL_MAP.items():
        print(f"    {cat}: {best_metrics['per_class_accuracy'][idx]:.4f}", flush=True)
    print("  Confusion matrix (rows=gt, cols=pred, order=top_short,top_long,pant_long,pant_short):", flush=True)
    for row in best_metrics["confusion"]:
        print(f"    {row}", flush=True)

    # Save: same format as v1 + extra info
    out_dir = REPO_ROOT / "outputs" / "classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "garment_classifier.pt"
    torch.save(
        {
            "classifier_state_dict": best_state,
            "label_map": LABEL_MAP,
            "idx_to_label": IDX_TO_LABEL,
            "val_accuracy": best_val_acc,
            "mean_confidence": best_metrics["mean_confidence"],
            "per_class_accuracy": best_metrics["per_class_accuracy"],
            "confusion": best_metrics["confusion"],
            "backbone": "resnet18",
            "train_garments": TRAIN_GARMENTS,
            "val_garments": VAL_GARMENTS,
            "frames_per_ep": FRAMES_PER_EP,
            "frame_step_range": FRAME_STEP_RANGE,
        },
        save_path,
    )
    print(f"[INFO] Saved classifier to: {save_path}", flush=True)


if __name__ == "__main__":
    main()
