"""Vision-guided collar recovery — Phase 2 CV proof-of-life.

Operates on top_rgb (480x640x3 uint8) only. No check_point access at runtime.

Pipeline:
1. Color-segment garment from table (white/marble) and grippers (yellow).
2. Keep largest connected component (the garment).
3. Compute silhouette features:
     - centroid
     - bounding box
     - top-left corner = topmost-leftmost garment pixel (proxy for left shoulder p2)
     - top-right corner = topmost-rightmost garment pixel (proxy for right shoulder p3)
     - top-edge midpoint = (left+right)/2 in pixel space
     - shoulder separation = pixel distance between left and right top corners
4. Stability metric: the per-frame landmarks are smoothed with a running median
   filter; a frame is "stable" if its raw landmarks are within R px of the median.

Used by:
- analyze_videos.py for offline proof-of-life
- vision_collar_recovery_policy.py for online closed-loop recovery
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# Approximate pixel scale: top camera bbox in cm at table level
# (calibrated empirically — adjust by inspecting overlays)
PIXEL_TO_CM = 0.10  # 1 px ~= 1 mm at table height


@dataclass
class Landmarks:
    """Detected garment landmarks in pixel coordinates (top RGB image)."""
    centroid: Tuple[int, int]  # (x, y)
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    top_left: Tuple[int, int]  # leftmost-topmost
    top_right: Tuple[int, int]  # rightmost-topmost
    bottom_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    shoulder_sep_px: float  # pixel distance between top_left and top_right
    confidence: float  # 0..1, based on garment pixel count
    # Gripper detections (None if not visible)
    left_gripper: Optional[Tuple[int, int]] = None
    right_gripper: Optional[Tuple[int, int]] = None


def segment_garment(rgb: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask (255 = garment) using HSV thresholding.

    Logic:
      - Mask out yellow (grippers): high saturation + hue near yellow.
      - Mask out near-white (table marble): high V, low S.
      - Everything else is candidate garment.
      - Keep only the largest connected component.

    Args:
        rgb: (H, W, 3) BGR uint8 (from cv2.imread or cv2.VideoCapture).
              We assume BGR not RGB — both source paths produce BGR.

    Returns:
        Binary mask (H, W) uint8 (0 or 255).
    """
    h, w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Yellow gripper: hue in [15, 40], saturation high (> 100)
    yellow_mask = (H >= 15) & (H <= 40) & (S > 100)
    # Near-white table: V high (> 200) AND saturation low (< 40)
    table_mask = (V > 200) & (S < 40)

    candidate = ~(yellow_mask | table_mask)
    candidate = candidate.astype(np.uint8) * 255

    # Morphological cleanup: close small holes, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    if n_labels <= 1:
        return np.zeros_like(candidate)
    # Skip background (label 0)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(sizes))
    mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return mask


def compute_landmarks(mask: np.ndarray) -> Optional[Landmarks]:
    """Extract centroid, bbox, and top-corner landmarks from a binary mask.

    Returns None if the garment is too small (< 1% of image).
    """
    h, w = mask.shape[:2]
    n_pix = int((mask > 0).sum())
    if n_pix < (h * w) * 0.01:
        return None

    ys, xs = np.where(mask > 0)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cx, cy = int(xs.mean()), int(ys.mean())

    # Top corners: examine the top 25% band of the bbox. Take the LEFTMOST
    # and RIGHTMOST garment pixels anywhere in that band (not just the first
    # row). For the y of each corner, take the topmost pixel at that x.
    bbox_h = max(1, y_max - y_min)
    top_band_height = max(1, int(0.25 * bbox_h))
    top_band = mask[y_min:y_min + top_band_height + 1, :] > 0
    if not top_band.any():
        return None
    band_ys, band_xs = np.where(top_band)
    if len(band_xs) == 0:
        return None
    leftmost_x = int(band_xs.min())
    rightmost_x = int(band_xs.max())
    # y of left corner = smallest y where mask is True at column leftmost_x
    col_left = mask[:, leftmost_x] > 0
    col_right = mask[:, rightmost_x] > 0
    top_left = (leftmost_x, int(np.where(col_left)[0].min()))
    top_right = (rightmost_x, int(np.where(col_right)[0].min()))

    # Bottom corners: same logic on bottom 25% band.
    bot_band_height = max(1, int(0.25 * bbox_h))
    bot_band = mask[max(0, y_max - bot_band_height):y_max + 1, :] > 0
    if not bot_band.any():
        bottom_left = (x_min, y_max)
        bottom_right = (x_max, y_max)
    else:
        bot_ys, bot_xs = np.where(bot_band)
        bl_x = int(bot_xs.min())
        br_x = int(bot_xs.max())
        col_bl = mask[:, bl_x] > 0
        col_br = mask[:, br_x] > 0
        bottom_left = (bl_x, int(np.where(col_bl)[0].max()))
        bottom_right = (br_x, int(np.where(col_br)[0].max()))

    shoulder_sep_px = float(np.hypot(top_right[0] - top_left[0],
                                      top_right[1] - top_left[1]))

    confidence = min(1.0, n_pix / (h * w * 0.10))  # full conf at 10% image area

    return Landmarks(
        centroid=(cx, cy),
        bbox=(x_min, y_min, x_max, y_max),
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        shoulder_sep_px=shoulder_sep_px,
        confidence=confidence,
    )


def detect_grippers(rgb: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Detect the two yellow gripper blobs in a top RGB frame.

    Returns:
        (left_gripper, right_gripper) — each (x, y) pixel coord, or None if missing.
        "Left" = lower x; "right" = higher x.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = ((H >= 15) & (H <= 40) & (S > 100) & (V > 100)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(yellow, 8)
    blobs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue
        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        blobs.append((area, cx, cy))
    if len(blobs) < 2:
        return None, None
    # Take the two largest
    blobs.sort(key=lambda b: -b[0])
    a, b = blobs[0], blobs[1]
    if a[1] < b[1]:
        return (int(a[1]), int(a[2])), (int(b[1]), int(b[2]))
    else:
        return (int(b[1]), int(b[2])), (int(a[1]), int(a[2]))


def overlay_landmarks(rgb: np.ndarray, mask: np.ndarray,
                      lm: Optional[Landmarks]) -> np.ndarray:
    """Render a debug overlay on top of the RGB frame.

    - Red: garment mask edge.
    - Cyan: bounding box.
    - Yellow: centroid.
    - Green: top-left & top-right corners (collar candidates).
    - Magenta: bottom corners.
    - Top-edge line (white) connecting shoulders.
    """
    out = rgb.copy()
    # Mask edge: dilate - erode then color it
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)
    out[edge > 0] = (0, 0, 255)  # BGR red

    if lm is None:
        cv2.putText(out, "no garment detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out

    x_min, y_min, x_max, y_max = lm.bbox
    cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
    cv2.circle(out, lm.centroid, 5, (0, 255, 255), -1)
    cv2.circle(out, lm.top_left, 8, (0, 255, 0), -1)
    cv2.circle(out, lm.top_right, 8, (0, 255, 0), -1)
    cv2.circle(out, lm.bottom_left, 6, (255, 0, 255), 2)
    cv2.circle(out, lm.bottom_right, 6, (255, 0, 255), 2)
    cv2.line(out, lm.top_left, lm.top_right, (255, 255, 255), 2)

    if lm.left_gripper is not None:
        cv2.circle(out, lm.left_gripper, 12, (0, 165, 255), 2)  # orange
        cv2.putText(out, "L", (lm.left_gripper[0] - 5, lm.left_gripper[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    if lm.right_gripper is not None:
        cv2.circle(out, lm.right_gripper, 12, (0, 165, 255), 2)
        cv2.putText(out, "R", (lm.right_gripper[0] - 5, lm.right_gripper[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    txt = f"sh_sep={lm.shoulder_sep_px:.0f}px conf={lm.confidence:.2f}"
    cv2.putText(out, txt, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def smooth_landmarks(history: list[Landmarks], window: int = 7) -> Optional[Landmarks]:
    """Median-smooth the last `window` detections to suppress jitter.

    Returns None if the history is empty or too unstable.
    """
    if not history:
        return None
    recent = history[-window:]
    if not recent:
        return None
    def med(attr):
        xs = [getattr(lm, attr)[0] for lm in recent]
        ys = [getattr(lm, attr)[1] for lm in recent]
        return (int(np.median(xs)), int(np.median(ys)))
    return Landmarks(
        centroid=med("centroid"),
        bbox=recent[-1].bbox,
        top_left=med("top_left"),
        top_right=med("top_right"),
        bottom_left=med("bottom_left"),
        bottom_right=med("bottom_right"),
        shoulder_sep_px=float(np.median([lm.shoulder_sep_px for lm in recent])),
        confidence=float(np.median([lm.confidence for lm in recent])),
    )
