#!/usr/bin/env python3
"""Send a live IK command to dataset_record.py.

The recorder polls /tmp/lehome_ik_command.json while the sim keeps stepping.
Use the Isaac toolbar pause only for visual inspection; for controllable pauses
use P/C/V or `pause` here so the Python loop can still consume commands.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="/tmp/lehome_ik_command.json",
        help="Command file polled by the recorder.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("restart")
    sub.add_parser("snapshot")

    grip = sub.add_parser("grip")
    grip.add_argument("mode", choices=["closed", "open", "act"])

    move = sub.add_parser("move")
    move.add_argument("--arm", choices=["nearest", "left", "right", "both"], default="nearest")
    move.add_argument("--pixel", nargs=2, type=int, metavar=("U", "V"))
    move.add_argument("--left-pixel", nargs=2, type=int, metavar=("U", "V"))
    move.add_argument("--right-pixel", nargs=2, type=int, metavar=("U", "V"))
    move.add_argument("--gripper", choices=["keep", "closed", "open", "act"], default="keep")
    move.add_argument("--steps", type=int, default=70)

    cont = sub.add_parser("continue")
    cont.add_argument("--steps", type=int, default=30)
    cont.add_argument("--gain", type=float, default=4.0)
    cont.add_argument("--max-joint-delta", type=float, default=0.12)
    cont.add_argument("--gripper", choices=["closed", "open", "keep"], default="closed")

    ext = sub.add_parser("extend")
    ext.add_argument("--steps", type=int, default=30)
    ext.add_argument("--gain", type=float, default=4.0)
    ext.add_argument("--max-joint-delta", type=float, default=0.12)
    ext.add_argument("--gripper", choices=["closed", "open", "keep"], default="closed")

    nudge = sub.add_parser("nudge")
    nudge.add_argument("--dx", type=float, default=0.0, help="World-frame x delta in meters")
    nudge.add_argument("--dy", type=float, default=0.0, help="World-frame y delta in meters")
    nudge.add_argument("--dz", type=float, default=0.0, help="World-frame z delta in meters")
    nudge.add_argument("--steps", type=int, default=40)
    nudge.add_argument("--max-delta", type=float, default=0.08)
    nudge.add_argument("--gripper", choices=["closed", "open", "keep"], default="closed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, object] = {"cmd": args.cmd, "sent_at": time.time()}

    if args.cmd == "grip":
        payload["mode"] = args.mode
    elif args.cmd == "move":
        payload.update({"arm": args.arm, "gripper": args.gripper, "steps": args.steps})
        if args.arm == "both":
            if args.left_pixel is None or args.right_pixel is None:
                raise SystemExit("both-arm move requires --left-pixel U V and --right-pixel U V")
            payload["left_pixel"] = args.left_pixel
            payload["right_pixel"] = args.right_pixel
        else:
            if args.pixel is None:
                raise SystemExit("single-arm move requires --pixel U V")
            payload["pixel"] = args.pixel
    elif args.cmd in {"continue", "extend"}:
        payload.update(
            {
                "steps": args.steps,
                "gain": args.gain,
                "max_joint_delta": args.max_joint_delta,
                "gripper": args.gripper,
            }
        )
    elif args.cmd == "nudge":
        payload.update(
            {
                "dx": args.dx,
                "dy": args.dy,
                "dz": args.dz,
                "steps": args.steps,
                "max_delta": args.max_delta,
                "gripper": args.gripper,
            }
        )

    path = Path(args.file)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    print(f"sent {payload}")


if __name__ == "__main__":
    main()
