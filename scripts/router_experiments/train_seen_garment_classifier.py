"""Train a Seen-garment identity classifier for portfolio routing.

The existing category classifier is only 4-way and the embedding kNN router
does not separate instances reliably. This trains a supervised 40-way classifier
for Seen_0..Seen_9 across the four garment categories using only public training
demo images. Unseen garments are deliberately excluded; the runtime router uses
a confidence threshold and falls back to static category defaults when the
classifier is uncertain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2 as T

REPO_ROOT = Path(__file__).resolve().parents[2]

CATEGORIES: Dict[str, Tuple[str, str]] = {
    "Top_Short": ("top_short", "Datasets/example/top_short_merged"),
    "Top_Long": ("top_long", "Datasets/example/top_long_merged"),
    "Pant_Long": ("pant_long", "Datasets/example/pant_long_merged"),
    "Pant_Short": ("pant_short", "Datasets/example/pant_short_merged"),
}

EPISODES_PER_GARMENT = 25
TRAIN_EPISODES_PER_GARMENT = 20
FRAMES_PER_EP = 8
FRAME_STEP_RANGE = (0, 50)
EPOCHS = 25


def label_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    labels: List[str] = []
    for prefix in CATEGORIES:
        labels.extend(f"{prefix}_Seen_{i}" for i in range(10))
    label_to_idx = {name: i for i, name in enumerate(labels)}
    idx_to_label = {i: name for name, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


class ImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.transform(self.images[idx]), self.labels[idx]


def build_transforms(train: bool):
    base = [T.ToDtype(torch.float32, scale=True)]
    if train:
        aug = [
            T.RandomResizedCrop(size=(224, 224), scale=(0.75, 1.0), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
            T.RandomRotation(degrees=12, fill=1.0),
        ]
    else:
        aug = [T.Resize(size=(224, 224), antialias=True)]
    norm = [T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(base + aug + norm)


def extract_frames_and_labels():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rng = torch.Generator().manual_seed(123)
    label_to_idx, idx_to_label = label_maps()
    images, labels, is_train = [], [], []

    for prefix, (_, root_rel) in CATEGORIES.items():
        root = REPO_ROOT / root_rel
        print(f"[INFO] Loading {prefix}: {root}", flush=True)
        ds = LeRobotDataset(repo_id="lehome", root=str(root))
        meta = pd.read_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        expected = EPISODES_PER_GARMENT * 10
        if len(meta) < expected:
            raise RuntimeError(f"{root}: expected at least {expected} episodes, got {len(meta)}")

        for ep_idx in range(expected):
            seen_idx = ep_idx // EPISODES_PER_GARMENT
            ep_in_seen = ep_idx % EPISODES_PER_GARMENT
            train = ep_in_seen < TRAIN_EPISODES_PER_GARMENT
            ep = meta.iloc[ep_idx]
            start = int(ep["dataset_from_index"])
            ep_len = int(ep["length"])
            max_off = min(ep_len, FRAME_STEP_RANGE[1])
            if max_off <= 0:
                continue
            offsets = torch.randperm(max_off, generator=rng)[:FRAMES_PER_EP].tolist()
            label = label_to_idx[f"{prefix}_Seen_{seen_idx}"]

            for off in offsets:
                frame = ds[start + int(off)]
                img = (frame["observation.images.top_rgb"] * 255.0).to(torch.uint8)
                images.append(img)
                labels.append(label)
                is_train.append(train)

    images_t = torch.stack(images)
    labels_t = torch.tensor(labels, dtype=torch.long)
    is_train_t = torch.tensor(is_train, dtype=torch.bool)
    print(
        f"[INFO] Frames: total={len(images_t)} train={int(is_train_t.sum())} "
        f"val={int((~is_train_t).sum())}",
        flush=True,
    )
    return images_t, labels_t, is_train_t, idx_to_label


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int):
    model.eval()
    total = correct = 0
    conf_sum = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        total += lbls.numel()
        correct += (preds == lbls).sum().item()
        conf_sum += conf.sum().item()
        for t, p in zip(lbls.cpu(), preds.cpu()):
            confusion[int(t), int(p)] += 1
    return {
        "accuracy": correct / max(total, 1),
        "mean_top1_confidence": conf_sum / max(total, 1),
        "confusion": confusion.tolist(),
        "total": total,
    }


def main() -> None:
    label_to_idx, idx_to_label = label_maps()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)

    images, labels, is_train, idx_to_label = extract_frames_and_labels()
    train_ds = ImageDataset(images[is_train], labels[is_train], build_transforms(train=True))
    val_ds = ImageDataset(images[~is_train], labels[~is_train], build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(len(label_to_idx)).to(device)
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 7e-5},
            {"params": head_params, "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_acc = -1.0
    best_state = None
    best_metrics = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
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
            loss_sum += float(loss.item()) * lbls.numel()
            correct += (logits.argmax(dim=-1) == lbls).sum().item()
            total += lbls.numel()
        scheduler.step()
        metrics = evaluate(model, val_loader, device, len(label_to_idx))
        train_acc = correct / max(total, 1)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} loss={loss_sum/max(total,1):.4f} "
            f"train_acc={train_acc:.4f} val_acc={metrics['accuracy']:.4f} "
            f"val_conf={metrics['mean_top1_confidence']:.4f}",
            flush=True,
        )
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out_dir = REPO_ROOT / "outputs" / "seen_garment_router"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seen_garment_classifier.pt"
    torch.save(
        {
            "classifier_state_dict": best_state,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "val_accuracy": best_metrics["accuracy"],
            "mean_top1_confidence": best_metrics["mean_top1_confidence"],
            "confusion": best_metrics["confusion"],
            "backbone": "resnet18",
            "train_episodes_per_garment": TRAIN_EPISODES_PER_GARMENT,
            "frames_per_ep": FRAMES_PER_EP,
            "frame_step_range": FRAME_STEP_RANGE,
        },
        out_path,
    )
    print(f"[INFO] Saved {out_path}", flush=True)
    print(f"[INFO] Best val_acc={best_metrics['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
