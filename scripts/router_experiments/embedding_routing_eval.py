"""Extended router-embedding experiment.

Compares 6 embedding schemes for within-category checkpoint-portfolio routing:
  R1  ResNet-18 multicam,  frame 0,         L2-distance
  R2  ResNet-18 multicam,  avg(0,25,50),    L2-distance
  R3  ResNet-18 multicam,  frame 0,         cosine (L2-normalized)
  R4  ResNet-18 multicam,  avg(0,25,50),    cosine (L2-normalized)
  D1  DINOv2-S/14 multicam, frame 0,        cosine (L2-normalized)
  D2  DINOv2-S/14 multicam, avg(0,25,50),   cosine (L2-normalized)

Within-category leave-one-out kNN, accuracy vs `best_checkpoints.json` labels,
plus chance (most-common-label) baseline. CPU + DINOv2 on GPU if available.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_VIDEOS_ROOT = REPO_ROOT / "outputs" / "eval_videos"
CLASSIFIER_PATH = REPO_ROOT / "outputs" / "classifier" / "garment_classifier.pt"
BEST_CKPTS_PATH = REPO_ROOT / "outputs" / "portfolio_router" / "best_checkpoints.json"

CANONICAL_CKPT_DIR = {
    "Top_Short":  "top_short_aug_050k",
    "Top_Long":   "top_long_aug_090k",
    "Pant_Long":  "pant_long_aug_090k",
    "Pant_Short": "pant_short_aug_050k",
}
CATEGORIES = list(CANONICAL_CKPT_DIR.keys())
FRAMES_TO_SAMPLE = [0, 25, 50]
VIEWS = ["top_rgb", "left_rgb", "right_rgb"]


# --------- model loading ---------

def load_resnet_embedder(device):
    ckpt = torch.load(str(CLASSIFIER_PATH), map_location=device, weights_only=False)
    n = len(ckpt["idx_to_label"])
    bb = tvm.resnet18(weights=None); bb.fc = nn.Linear(512, n)
    bb.load_state_dict(ckpt["classifier_state_dict"])
    bb.eval().to(device)
    emb = nn.Sequential(*list(bb.children())[:-1]).eval().to(device)
    pre = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return emb, pre


def load_dinov2_embedder(device):
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    m.eval().to(device)
    pre = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return m, pre


# --------- video frame helpers ---------

def list_seen_garments(ckpt_dir: Path) -> List[str]:
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


def extract_frames_bgr(vp: Path, idxs: List[int]) -> List[Optional[np.ndarray]]:
    cap = cv2.VideoCapture(str(vp))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in idxs:
        if i >= total:
            out.append(None); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        out.append(fr if ok else None)
    cap.release()
    return out


def embed_one(bgr: np.ndarray, model, pre, device, kind: str) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = pre(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        if kind == "resnet":
            e = model(t).squeeze().cpu().numpy()
        else:
            e = model(t).squeeze().cpu().numpy()  # DINOv2 vits14 returns 384-D CLS
    return e.astype(np.float32)


# --------- build embedding tables ---------

def build_for_model(garments_per_cat, model, pre, device, kind: str):
    """Return (single, multi): garment_name -> 1D vector, concatenated 3 views."""
    single, multi = {}, {}
    for cat, garms in garments_per_cat.items():
        ckpt_dir = EVAL_VIDEOS_ROOT / CANONICAL_CKPT_DIR[cat]
        for g in garms:
            vs1, vsM = [], []
            ok = True
            for view in VIEWS:
                vp = find_video(ckpt_dir, g, 0, view)
                if vp is None: ok = False; break
                frames = extract_frames_bgr(vp, FRAMES_TO_SAMPLE)
                if frames[0] is None: ok = False; break
                vs1.append(embed_one(frames[0], model, pre, device, kind))
                stack = [embed_one(f, model, pre, device, kind) for f in frames if f is not None]
                vsM.append(np.mean(np.stack(stack, axis=0), axis=0))
            if not ok: continue
            single[g] = np.concatenate(vs1)
            multi[g]  = np.concatenate(vsM)
    return single, multi


# --------- routing accuracy ---------

def loo_acc(emb: Dict[str, np.ndarray], labels: Dict[str, str], cat: str,
            metric: str = "l2") -> Tuple[float, int, int]:
    members = [g for g in emb if g.startswith(cat) and "Seen" in g]
    if len(members) < 2: return 0.0, 0, 0
    # Optionally L2-normalize for cosine
    if metric == "cosine":
        norm = {g: emb[g] / (np.linalg.norm(emb[g]) + 1e-9) for g in members}
    else:
        norm = {g: emb[g] for g in members}
    correct, total = 0, 0
    for held in members:
        held_e = norm[held]; held_l = labels.get(held)
        if held_l is None: continue
        best_d, best_g = float("inf"), None
        for g in members:
            if g == held: continue
            if metric == "cosine":
                d = 1.0 - float(np.dot(norm[g], held_e))
            else:
                d = float(np.linalg.norm(norm[g] - held_e))
            if d < best_d:
                best_d, best_g = d, g
        pred = labels.get(best_g)
        if pred is None: continue
        total += 1
        if pred == held_l: correct += 1
    return (correct / total if total > 0 else 0.0), correct, total


def chance_acc(labels: Dict[str, str], cat: str) -> Tuple[float, int, int]:
    ls = [v for k, v in labels.items() if k.startswith(cat) and "Seen" in k]
    if not ls: return 0.0, 0, 0
    _, n = Counter(ls).most_common(1)[0]
    return n / len(ls), n, len(ls)


# --------- main ---------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("Loading ResNet-18 embedder ...")
    resnet, resnet_pre = load_resnet_embedder(device)
    print("Loading DINOv2-S/14 embedder ...")
    dino, dino_pre = load_dinov2_embedder(device)

    with open(BEST_CKPTS_PATH) as f:
        labels = json.load(f)

    garments_per_cat: Dict[str, List[str]] = {}
    for cat in CATEGORIES:
        ckpt_dir = EVAL_VIDEOS_ROOT / CANONICAL_CKPT_DIR[cat]
        if not ckpt_dir.exists():
            print(f"[warn] missing {ckpt_dir}"); garments_per_cat[cat] = []; continue
        gs = list_seen_garments(ckpt_dir)
        gs = [g for g in gs if g in labels]
        garments_per_cat[cat] = gs
        print(f"[{cat}] {len(gs)} garments")

    print("\nBuilding ResNet embeddings ...")
    R_single, R_multi = build_for_model(garments_per_cat, resnet, resnet_pre, device, "resnet")
    print(f"  built {len(R_single)} single + {len(R_multi)} multi")

    print("Building DINOv2 embeddings ...")
    D_single, D_multi = build_for_model(garments_per_cat, dino, dino_pre, device, "dinov2")
    print(f"  built {len(D_single)} single + {len(D_multi)} multi\n")

    schemes: List[Tuple[str, Dict[str, np.ndarray], str]] = [
        ("ResNet/single/L2 ", R_single, "l2"),
        ("ResNet/multi /L2 ", R_multi,  "l2"),
        ("ResNet/single/cos", R_single, "cosine"),
        ("ResNet/multi /cos", R_multi,  "cosine"),
        ("DINOv2/single/cos", D_single, "cosine"),
        ("DINOv2/multi /cos", D_multi,  "cosine"),
    ]

    print("=" * 110)
    header = f"{'Category':<14} {'Chance':<14}"
    for name, _, _ in schemes:
        header += f" {name:<19}"
    print(header)
    print("=" * 110)

    agg = {"chance": [0, 0]}
    for name, _, _ in schemes:
        agg[name] = [0, 0]

    for cat in CATEGORIES:
        c_acc, c_n, c_t = chance_acc(labels, cat)
        row = f"{cat:<14} {c_acc*100:5.1f}% ({c_n}/{c_t})  "
        agg["chance"][0] += c_n; agg["chance"][1] += c_t
        for name, table, metric in schemes:
            acc, n, t = loo_acc(table, labels, cat, metric)
            row += f" {acc*100:5.1f}% ({n}/{t})    "
            agg[name][0] += n; agg[name][1] += t
        print(row)

    print("-" * 110)
    cn, ct = agg["chance"]; ch = cn / max(1, ct)
    line = f"{'Aggregate':<14} {ch*100:5.1f}% ({cn}/{ct})  "
    for name, _, _ in schemes:
        n, t = agg[name]; line += f" {n/max(1,t)*100:5.1f}% ({n}/{t})    "
    print(line)

    # Decide
    print()
    print(f"Chance baseline (most-common-label per category): {ch*100:.1f}%")
    best_scheme = max(schemes, key=lambda s: agg[s[0]][0] / max(1, agg[s[0]][1]))
    n, t = agg[best_scheme[0]]
    best_acc = n / max(1, t)
    print(f"Best embedding scheme: {best_scheme[0].strip():>20s} = {best_acc*100:.1f}% ({n}/{t})")
    if best_acc > ch + 0.05:
        print(f"=> Embedding routing beats chance by {(best_acc-ch)*100:+.1f}pp; "
              f"worth swapping in.")
    else:
        print(f"=> Best scheme is {(best_acc-ch)*100:+.1f}pp vs chance — kNN routing "
              f"isn't reliable enough. Use a static 'most-common-best per category' map "
              f"instead.")


if __name__ == "__main__":
    main()
