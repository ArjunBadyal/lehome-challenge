"""Offline experiment: do multi-frame embeddings beat single-frame for the
checkpoint-portfolio router?

For each Seen garment, build two embeddings from saved eval rollout MP4s:
  A) single-frame: ResNet-18 multicam at frame 0  (1536-D)
  B) multi-frame:  ResNet-18 multicam mean over frames 0/25/50  (1536-D)

Run within-category leave-one-out kNN routing using `best_checkpoints.json`
as the supervision label. Report routing accuracy per scheme + a chance
baseline (most-common label).

CPU-only, ~3 min total. No sim.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

REPO_ROOT = Path(__file__).resolve().parents[2]

EVAL_VIDEOS_ROOT = REPO_ROOT / "outputs" / "eval_videos"
CLASSIFIER_PATH = REPO_ROOT / "outputs" / "classifier" / "garment_classifier.pt"
BEST_CKPTS_PATH = REPO_ROOT / "outputs" / "portfolio_router" / "best_checkpoints.json"

# Pick a canonical (always-available) checkpoint dir per category to source
# rollouts from. The choice doesn't matter much for the experiment as long as
# each garment has a rollout in that dir.
CANONICAL_CKPT_DIR = {
    "Top_Short":  "top_short_aug_050k",
    "Top_Long":   "top_long_aug_090k",
    "Pant_Long":  "pant_long_aug_090k",
    "Pant_Short": "pant_short_aug_050k",
}

CATEGORIES = list(CANONICAL_CKPT_DIR.keys())

FRAMES_TO_SAMPLE = [0, 25, 50]

VIEWS = ["top_rgb", "left_rgb", "right_rgb"]


def load_resnet_embedder(device: torch.device):
    """ResNet-18 backbone (final FC removed) loaded from the trained classifier."""
    ckpt = torch.load(str(CLASSIFIER_PATH), map_location=device, weights_only=False)
    n_classes = len(ckpt["idx_to_label"])
    bb = models.resnet18(weights=None)
    bb.fc = nn.Linear(512, n_classes)
    bb.load_state_dict(ckpt["classifier_state_dict"])
    bb.eval().to(device)
    embedder = nn.Sequential(*list(bb.children())[:-1]).eval().to(device)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return embedder, preprocess


def list_garments_in_dir(ckpt_dir: Path) -> List[str]:
    """Return Seen garment names that have an episode-0 top_rgb video here."""
    found = set()
    for sub in ("success", "failure"):
        for p in (ckpt_dir / sub).glob("*_episode0_observation_images_top_rgb.mp4"):
            name = p.name.split("_episode0_observation_images_top_rgb.mp4")[0]
            if "Seen" in name:
                found.add(name)
    return sorted(found)


def find_video(ckpt_dir: Path, garment: str, episode: int, view: str) -> Optional[Path]:
    fname = f"{garment}_episode{episode}_observation_images_{view}.mp4"
    for sub in ("success", "failure"):
        p = ckpt_dir / sub / fname
        if p.exists():
            return p
    return None


def extract_frames_bgr(video_path: Path, frame_indices: List[int]) -> List[Optional[np.ndarray]]:
    """Pull specific frames from an MP4. Returns BGR uint8 arrays or None per slot."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: List[Optional[np.ndarray]] = []
    for idx in frame_indices:
        if idx >= total:
            out.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        out.append(frame if ok else None)
    cap.release()
    return out


def embed_image_bgr(bgr: np.ndarray, embedder, preprocess, device) -> np.ndarray:
    """BGR uint8 → 512-D ResNet-18 embedding."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedder(tensor).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def build_embeddings(
    seen_garments_per_cat: Dict[str, List[str]],
    embedder, preprocess, device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build single-frame and multi-frame embeddings for every garment.
    Returns (single_emb, multi_emb), each garment_name → 1536-D vector."""
    single_emb: Dict[str, np.ndarray] = {}
    multi_emb: Dict[str, np.ndarray] = {}

    for cat, garments in seen_garments_per_cat.items():
        ckpt_dir = EVAL_VIDEOS_ROOT / CANONICAL_CKPT_DIR[cat]
        for garment in garments:
            view_singles, view_multis = [], []
            ok = True
            for view in VIEWS:
                vp = find_video(ckpt_dir, garment, episode=0, view=view)
                if vp is None:
                    ok = False
                    break
                frames = extract_frames_bgr(vp, FRAMES_TO_SAMPLE)
                if frames[0] is None:
                    ok = False
                    break
                view_singles.append(embed_image_bgr(frames[0], embedder, preprocess, device))
                # Multi-frame: average available frames in the slot
                multi_stack = [embed_image_bgr(f, embedder, preprocess, device)
                               for f in frames if f is not None]
                view_multis.append(np.mean(np.stack(multi_stack, axis=0), axis=0))
            if not ok:
                print(f"[skip] {garment}: missing video")
                continue
            single_emb[garment] = np.concatenate(view_singles)
            multi_emb[garment]  = np.concatenate(view_multis)
    return single_emb, multi_emb


def loo_routing_accuracy(
    embeddings: Dict[str, np.ndarray],
    best_ckpts: Dict[str, str],
    cat_prefix: str,
) -> Tuple[float, int, int]:
    """For each garment in cat, leave it out, kNN among others in cat, predict
    its best checkpoint, compare to the labeled one. Returns (accuracy, ncorrect, ntotal)."""
    members = [g for g in embeddings if g.startswith(cat_prefix) and "Seen" in g]
    if len(members) < 2:
        return 0.0, 0, 0
    correct = 0
    total = 0
    for held in members:
        held_emb = embeddings[held]
        held_label = best_ckpts.get(held)
        if held_label is None:
            continue
        # Nearest other member
        best_d, best_g = float("inf"), None
        for g in members:
            if g == held:
                continue
            d = float(np.linalg.norm(embeddings[g] - held_emb))
            if d < best_d:
                best_d, best_g = d, g
        pred_label = best_ckpts.get(best_g)
        if pred_label is None:
            continue
        total += 1
        if pred_label == held_label:
            correct += 1
    return (correct / total if total > 0 else 0.0), correct, total


def chance_accuracy(best_ckpts: Dict[str, str], cat_prefix: str) -> Tuple[float, int, int]:
    """Most-common-label baseline accuracy for the category."""
    labels = [v for k, v in best_ckpts.items()
              if k.startswith(cat_prefix) and "Seen" in k]
    if not labels:
        return 0.0, 0, 0
    most_common, n = Counter(labels).most_common(1)[0]
    return n / len(labels), n, len(labels)


def main():
    device = torch.device("cpu")
    print(f"Loading ResNet-18 embedder from {CLASSIFIER_PATH}")
    embedder, preprocess = load_resnet_embedder(device)

    print(f"Loading best_checkpoints from {BEST_CKPTS_PATH}")
    with open(BEST_CKPTS_PATH) as f:
        best_ckpts = json.load(f)

    seen_garments_per_cat: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        ckpt_dir = EVAL_VIDEOS_ROOT / CANONICAL_CKPT_DIR[cat]
        if not ckpt_dir.exists():
            print(f"[warn] missing {ckpt_dir}; skipping category {cat}")
            seen_garments_per_cat[cat] = []
            continue
        garments = list_garments_in_dir(ckpt_dir)
        # Keep only those that have a label
        garments = [g for g in garments if g in best_ckpts]
        seen_garments_per_cat[cat] = garments
        print(f"[{cat}] {len(garments)} garments with rollouts: {garments}")

    print("\nComputing embeddings ...")
    single_emb, multi_emb = build_embeddings(
        seen_garments_per_cat, embedder, preprocess, device
    )
    print(f"Built {len(single_emb)} single-frame, {len(multi_emb)} multi-frame embeddings\n")

    print("=" * 78)
    print(f"{'Category':<14} {'Chance':<10} {'SingleFrame':<14} {'MultiFrame':<14} {'Δ':<8}")
    print("=" * 78)

    overall = {"chance": [0, 0], "single": [0, 0], "multi": [0, 0]}
    for cat in CATEGORIES:
        prefix = cat
        c_acc, c_n, c_t = chance_accuracy(best_ckpts, prefix)
        s_acc, s_n, s_t = loo_routing_accuracy(single_emb, best_ckpts, prefix)
        m_acc, m_n, m_t = loo_routing_accuracy(multi_emb,  best_ckpts, prefix)
        delta = (m_acc - s_acc) * 100
        print(f"{cat:<14} "
              f"{c_acc*100:5.1f}% ({c_n}/{c_t})  "
              f"{s_acc*100:5.1f}% ({s_n}/{s_t})  "
              f"{m_acc*100:5.1f}% ({m_n}/{m_t})  "
              f"{delta:+5.1f}pp")
        overall["chance"][0] += c_n; overall["chance"][1] += c_t
        overall["single"][0] += s_n; overall["single"][1] += s_t
        overall["multi"][0]  += m_n; overall["multi"][1]  += m_t

    print("-" * 78)
    c_n, c_t = overall["chance"]; s_n, s_t = overall["single"]; m_n, m_t = overall["multi"]
    c_acc = c_n / max(1, c_t); s_acc = s_n / max(1, s_t); m_acc = m_n / max(1, m_t)
    delta = (m_acc - s_acc) * 100
    print(f"{'Aggregate':<14} "
          f"{c_acc*100:5.1f}% ({c_n}/{c_t})  "
          f"{s_acc*100:5.1f}% ({s_n}/{s_t})  "
          f"{m_acc*100:5.1f}% ({m_n}/{m_t})  "
          f"{delta:+5.1f}pp")

    # Save embeddings for downstream use if multi-frame is a winner
    out_dir = REPO_ROOT / "outputs" / "router_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "single_frame_resnet_multicam.npz", **single_emb)
    np.savez(out_dir / "multi_frame_0_25_50_resnet_multicam.npz", **multi_emb)
    print(f"\nWrote embedding caches to {out_dir}/")

    # Decision rule
    if (m_acc - s_acc) >= 0.05:
        print(f"\n[decision] Multi-frame beats single-frame by {delta:+.1f}pp ≥ 5pp.")
        print("           Worth building runtime checkpoint switching at step 50.")
    else:
        print(f"\n[decision] Multi-frame gain {delta:+.1f}pp is below the 5pp bar.")
        print("           Don't build runtime switching; the lift is too small to justify "
              "the integration cost (mid-episode policy reset, action-chunk buffer reset).")


if __name__ == "__main__":
    main()
