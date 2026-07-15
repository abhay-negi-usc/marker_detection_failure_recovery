#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import sys

import cv2
import numpy as np

def natural_key(s: str):
    """
    Split a string into a list of text and integer chunks for natural sorting.
    e.g., "frame10a" -> ["frame", 10, "a"]
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def list_pngs(input_dir: Path, recursive: bool = False):
    if recursive:
        files = sorted((p for p in input_dir.rglob("*.png")), key=lambda p: natural_key(p.stem))
    else:
        files = sorted((p for p in input_dir.glob("*.png")), key=lambda p: natural_key(p.stem))
    return files

def main():
    ap = argparse.ArgumentParser(
        description="Turn a directory of PNGs into a video (natural filename order)."
    )
    ap.add_argument("--input_dir", default="/media/rp/Elements1/abhay_ws/marker_detection_failure_recovery/marker_data_collection/july12/glare/combined_1x2_visualizations", type=Path, help="Directory containing PNG frames")
    ap.add_argument("--output", default="/media/rp/Elements1/abhay_ws/marker_detection_failure_recovery/marker_data_collection/july12/glare/combined_1x2_visualizations/glare.mp4", type=Path, help="Output video file (e.g., out.mp4)")
    ap.add_argument("--fps", type=float, default=5.0, help="Frames per second (default: 30)")
    ap.add_argument("--recursive", action="store_true", help="Search PNGs recursively")
    ap.add_argument("--codec", default="mp4v",
                    help="FourCC codec (e.g., mp4v, avc1, H264, XVID). Default: mp4v")
    ap.add_argument("--size", default=None,
                    help="Force output size WxH (e.g., 1920x1080). If omitted, use first frame size.")
    ap.add_argument("--preserve_alpha", action="store_true",
                    help="Keep alpha channel if codec/container supports it (many do not).")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging")
    args = ap.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"ERROR: input_dir '{args.input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    pngs = list_pngs(args.input_dir, args.recursive)
    if not pngs:
        print("ERROR: No PNG files found.", file=sys.stderr)
        sys.exit(1)

    # Determine frame size
    first = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        print(f"ERROR: Could not read first image: {pngs[0]}", file=sys.stderr)
        sys.exit(1)

    if args.size:
        try:
            w_str, h_str = args.size.lower().split("x")
            out_w, out_h = int(w_str), int(h_str)
        except Exception:
            print("ERROR: --size must be WxH, e.g., 1920x1080", file=sys.stderr)
            sys.exit(1)
    else:
        out_h, out_w = first.shape[:2]

    # Decide channel handling
    has_alpha = (first.shape[2] == 4) if (len(first.shape) == 3 and first.shape[2] in (3,4)) else False
    keep_alpha = args.preserve_alpha and has_alpha

    # Open VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    # Note: Most common containers/codecs do not support alpha. If keep_alpha=True, you may
    # need a codec/container that supports it (e.g., .mov with qtrle or ProRes 4444 via ffmpeg).
    # OpenCV VideoWriter generally writes 3-channel BGR; alpha often isn’t preserved.
    # We’ll still allow the flag but will warn and drop alpha if unsupported.
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (out_w, out_h), isColor=True)
    if not writer.isOpened():
        print("ERROR: Failed to open VideoWriter. Try a different --codec or file extension.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Found {len(pngs)} frames.")
        print(f"Output: {args.output} @ {args.fps} fps, size {out_w}x{out_h}, codec {args.codec}")
        if keep_alpha:
            print("Note: Alpha requested, but OpenCV VideoWriter typically drops alpha. Proceeding with RGB.")

    # Process frames
    written = 0
    for i, path in enumerate(pngs, 1):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"WARNING: Skipping unreadable image: {path}", file=sys.stderr)
            continue

        # Ensure 3-channel BGR for writer
        if len(img.shape) == 2:
            # grayscale -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # BGRA -> BGR (drop alpha)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        if (img.shape[1], img.shape[0]) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(img)
        written += 1
        if not args.quiet and i % 50 == 0:
            print(f"Wrote {i}/{len(pngs)} frames...")

    writer.release()

    if written == 0:
        print("ERROR: Wrote 0 frames. Nothing produced.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Done. Wrote {written} frames to '{args.output}'.")

if __name__ == "__main__":
    main()
