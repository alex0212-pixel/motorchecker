"""Generate a labeling template CSV from an image directory.

Usage (local or RunPod):

  source .venv/bin/activate
  python runpod/make_label_template.py \
    --image_dir data/motor_checker \
    --out runpod/labels.csv

Then open `runpod/labels.csv` in Excel/Google Sheet and fill:
- has_sticker: true/false
- color: 초록색/노란색/빨간색 (or empty if no sticker)
- number: digits only (or empty)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"No images found under: {image_dir}")

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "has_sticker", "color", "number"])
        for p in images:
            # store as project-relative path when possible
            rel = p.as_posix()
            w.writerow([rel, "", "", ""])

    print(f"[OK] Wrote template: {out} ({len(images)} rows)")


if __name__ == "__main__":
    main()

