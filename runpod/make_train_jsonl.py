"""Convert a labels CSV into the JSONL format expected by runpod/train_qwen2vl_lora.py.

The training JSONL schema:
  {"image": "/workspace/...", "question": "...", "answer": "{...json...}"}

We generate a *single-step* supervision by default:
  question: "스티커가 있나요? ... JSON으로만 답해"  (forces structured output)
  answer:   {"has_sticker": bool, "number": str|null, "color": str|null}

Usage:
  source .venv/bin/activate
  python runpod/make_train_jsonl.py \
    --labels_csv runpod/labels.csv \
    --out train.jsonl \
    --image_root /workspace

Notes:
- In RunPod, mount your repo under /workspace.
- If your CSV contains relative image paths like data/motor_checker/xxx.jpg,
  this script will convert them to /workspace/data/motor_checker/xxx.jpg.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_QUESTION = (
    "스티커가 있나요? 있다면 번호(number: 숫자만)와 색(color: 초록색/노란색/빨간색)을 "
    "다음 JSON 형식으로만 답하세요: {\"has_sticker\":true/false,\"number\":string|null,\"color\":string|null}"
)


def _to_bool(v: str) -> bool:
    v = (v or "").strip().lower()
    return v in {"1", "true", "t", "yes", "y", "o"}


def _norm_color(v: str) -> str | None:
    v = (v or "").strip()
    if not v:
        return None
    # allow common abbreviations
    mapping = {
        "초": "초록색",
        "초록": "초록색",
        "green": "초록색",
        "노": "노란색",
        "노랑": "노란색",
        "yellow": "노란색",
        "빨": "빨간색",
        "빨강": "빨간색",
        "red": "빨간색",
    }
    return mapping.get(v.lower(), v)


def _norm_number(v: str) -> str | None:
    v = (v or "").strip()
    if not v:
        return None
    # keep digits only
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out", required=True)
    # default to current working directory so it works for /workspace/<repo>
    ap.add_argument("--image_root", default=None)
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    args = ap.parse_args()

    if args.image_root is None:
        # Use current working directory by default
        args.image_root = str(Path.cwd())

    labels_csv = Path(args.labels_csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with labels_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise SystemExit(f"No rows in {labels_csv}")

    n_ok = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            img = (r.get("image") or "").strip()
            if not img:
                continue
            has = _to_bool(r.get("has_sticker", ""))
            color = _norm_color(r.get("color", "")) if has else None
            number = _norm_number(r.get("number", "")) if has else None

            # Make absolute path for RunPod container
            img_path = img
            if not img_path.startswith("/"):
                img_path = str(Path(args.image_root) / img_path)

            answer_obj = {
                "has_sticker": bool(has),
                "number": number,
                "color": color,
            }

            ex = {
                "image": img_path,
                "question": args.question,
                "answer": json.dumps(answer_obj, ensure_ascii=False),
            }
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"[OK] wrote {n_ok} examples to {out}")


if __name__ == "__main__":
    main()
