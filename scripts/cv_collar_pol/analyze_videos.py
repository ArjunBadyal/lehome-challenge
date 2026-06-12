"""Run the CV pipeline over saved failure videos to assess proof-of-life.

Reads top_rgb mp4 files, runs segmentation + landmark detection per frame,
saves overlay frames, and reports stability metrics:
  - % frames with valid detection
  - Median shoulder separation in late phase (steps 350+)
  - Stability: % of late-phase frames within R px of the median landmark

Output:
  outputs/cv_collar_pol/overlays/<video_name>/frame_*.png  (sparse overlays)
  outputs/cv_collar_pol/overlays/<video_name>/overlay.mp4  (full overlaid video)
  outputs/cv_collar_pol/reports/<video_name>.json          (stats)

Usage:
  python -m scripts.cv_collar_pol.analyze_videos
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.cv_collar_pol.segment_and_landmark import (
    segment_garment,
    compute_landmarks,
    detect_grippers,
    overlay_landmarks,
    smooth_landmarks,
)


VIDEOS = [
    # (label, video_path, expected_pattern)
    ("Unseen_0_ep0", "outputs/eval_videos/router_v2_5ep_top_short/failure/Top_Short_Unseen_0_episode0_observation_images_top_rgb.mp4"),
    ("Unseen_0_ep1", "outputs/eval_videos/router_v2_5ep_top_short/failure/Top_Short_Unseen_0_episode1_observation_images_top_rgb.mp4"),
    ("Unseen_0_ep4", "outputs/eval_videos/router_v2_5ep_top_short/failure/Top_Short_Unseen_0_episode4_observation_images_top_rgb.mp4"),
    ("Seen_5_ep0_fail",   "outputs/scripted_recovery_v2/seen5_videos/failure/Top_Short_Seen_5_episode0_observation_images_top_rgb.mp4"),
    ("Seen_5_ep2_fail",   "outputs/scripted_recovery_v2/seen5_videos/failure/Top_Short_Seen_5_episode2_observation_images_top_rgb.mp4"),
    ("Seen_5_ep1_succ", "outputs/scripted_recovery_v2/seen5_videos/success/Top_Short_Seen_5_episode1_observation_images_top_rgb.mp4"),
    ("Seen_9_ep0_fail",   "outputs/scripted_recovery_v2/seen9_videos/failure/Top_Short_Seen_9_episode0_observation_images_top_rgb.mp4"),
]

LATE_PHASE_START = 350  # frames from this index onward count as "late phase"
STABILITY_R_PX = 30      # max deviation from median landmark to count as stable
SAMPLE_OVERLAY_EVERY = 50  # save every Nth frame as a static overlay


def analyze_one(label: str, video_path: str, out_root: str = "outputs/cv_collar_pol") -> dict:
    if not os.path.exists(video_path):
        print(f"[{label}] missing: {video_path}")
        return {"label": label, "error": "missing"}

    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    overlay_dir = Path(out_root) / "overlays" / label
    overlay_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_writer = cv2.VideoWriter(
        str(overlay_dir / "overlay.mp4"), fourcc, fps, (w, h)
    )

    history: list = []
    landmarks_per_frame = []
    detect_count = 0
    valid_late_count = 0
    late_phase_count = 0
    sep_late = []
    centroid_xs_late = []
    centroid_ys_late = []
    top_left_xs_late = []
    top_right_xs_late = []

    for idx in range(nframes):
        ret, frame = cap.read()
        if not ret:
            break
        mask = segment_garment(frame)
        lm = compute_landmarks(mask)
        if lm is not None:
            lg, rg = detect_grippers(frame)
            lm.left_gripper = lg
            lm.right_gripper = rg
            history.append(lm)
            detect_count += 1
        smoothed = smooth_landmarks(history, window=7) if lm is not None else None
        if smoothed is not None and lm is not None:
            smoothed.left_gripper = lm.left_gripper
            smoothed.right_gripper = lm.right_gripper
        landmarks_per_frame.append(smoothed)

        ovl = overlay_landmarks(frame, mask, smoothed)
        cv2.putText(ovl, f"frame {idx}/{nframes}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        overlay_writer.write(ovl)

        if idx % SAMPLE_OVERLAY_EVERY == 0 or idx == nframes - 1:
            cv2.imwrite(str(overlay_dir / f"frame_{idx:04d}.png"), ovl)

        if idx >= LATE_PHASE_START:
            late_phase_count += 1
            if smoothed is not None:
                valid_late_count += 1
                sep_late.append(smoothed.shoulder_sep_px)
                centroid_xs_late.append(smoothed.centroid[0])
                centroid_ys_late.append(smoothed.centroid[1])
                top_left_xs_late.append(smoothed.top_left[0])
                top_right_xs_late.append(smoothed.top_right[0])

    cap.release()
    overlay_writer.release()

    # Stability calculations on late phase
    if valid_late_count > 0:
        med_cx = np.median(centroid_xs_late)
        med_cy = np.median(centroid_ys_late)
        med_tlx = np.median(top_left_xs_late)
        med_trx = np.median(top_right_xs_late)
        n_stable = sum(
            1 for cx, cy, tlx, trx in zip(centroid_xs_late, centroid_ys_late,
                                            top_left_xs_late, top_right_xs_late)
            if abs(cx - med_cx) < STABILITY_R_PX
            and abs(cy - med_cy) < STABILITY_R_PX
            and abs(tlx - med_tlx) < STABILITY_R_PX
            and abs(trx - med_trx) < STABILITY_R_PX
        )
        stable_frac = n_stable / valid_late_count
        med_sep = float(np.median(sep_late))
    else:
        stable_frac = 0.0
        med_sep = 0.0

    valid_frac = detect_count / nframes if nframes > 0 else 0.0
    late_valid_frac = valid_late_count / late_phase_count if late_phase_count > 0 else 0.0

    report = {
        "label": label,
        "video_path": video_path,
        "nframes": nframes,
        "valid_detection_frac": round(valid_frac, 3),
        "late_phase_valid_frac": round(late_valid_frac, 3),
        "late_phase_stable_frac": round(stable_frac, 3),
        "late_phase_median_shoulder_sep_px": round(med_sep, 1),
        "overlay_dir": str(overlay_dir),
    }
    print(f"[{label}] frames={nframes}, valid={valid_frac:.1%}, "
          f"late_valid={late_valid_frac:.1%}, late_stable={stable_frac:.1%}, "
          f"med_sep_px={med_sep:.0f}")
    return report


def main():
    out_root = "outputs/cv_collar_pol"
    Path(f"{out_root}/reports").mkdir(parents=True, exist_ok=True)
    reports = []
    for label, path in VIDEOS:
        rep = analyze_one(label, path, out_root)
        reports.append(rep)
        with open(f"{out_root}/reports/{label}.json", "w") as f:
            json.dump(rep, f, indent=2)
    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'video':30s} {'valid%':>6s} {'late_valid%':>11s} {'late_stable%':>12s} {'med_sep_px':>10s}")
    for r in reports:
        if "error" in r:
            print(f"{r['label']:30s}  MISSING")
            continue
        print(f"{r['label']:30s} {r['valid_detection_frac']:6.1%} "
              f"{r['late_phase_valid_frac']:11.1%} {r['late_phase_stable_frac']:12.1%} "
              f"{r['late_phase_median_shoulder_sep_px']:10.0f}")
    with open(f"{out_root}/reports/summary.json", "w") as f:
        json.dump(reports, f, indent=2)
    # Phase 2 gate: late_stable_frac must be ≥ 0.70 on at least 2/3 failure videos
    threshold = 0.70
    failure_reports = [r for r in reports
                       if "error" not in r
                       and r["label"] in ("Unseen_0_ep0", "Unseen_0_ep1", "Unseen_0_ep4",
                                          "Seen_5_ep0_fail", "Seen_5_ep2_fail",
                                          "Seen_9_ep0_fail")]
    n_pass = sum(1 for r in failure_reports if r["late_phase_stable_frac"] >= threshold)
    print(f"\nGate: {n_pass}/{len(failure_reports)} failure videos passed "
          f"late-phase stability ≥ {threshold:.0%}")
    if n_pass >= max(1, len(failure_reports) * 0.7):
        print("PROOF-OF-LIFE: PASS")
    else:
        print("PROOF-OF-LIFE: FAIL")


if __name__ == "__main__":
    main()
