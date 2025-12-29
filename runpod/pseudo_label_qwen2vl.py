"""Pseudo-label images using a base VLM (Qwen2-VL) to reduce manual labeling.

This script generates a CSV with columns:
  image,has_sticker,color,number,raw

You then open the CSV and only fix wrong/uncertain rows.

Usage (RunPod):
  source .venv/bin/activate
  python runpod/pseudo_label_qwen2vl.py \
    --image_dir data/motor_checker \
    --out runpod/labels.auto.csv \
    --model Qwen/Qwen2-VL-7B-Instruct

Env:
  BATCH (default 1)
  MAX_NEW_TOKENS (default 128)

Note:
  This is best-effort. Always manually verify labels before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


def parse_json_from_text(text: str) -> dict | None:
    # strip code block if present
    t = text.strip()
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0].strip()
    elif "```" in t:
        t = t.split("```")[1].split("```")[0].strip()

    # find last JSON object
    s = t.find("{")
    e = t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(t[s : e + 1])
    except Exception:
        return None


def norm_color(v: str | None) -> str | None:
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    # normalize common variants
    v_low = v.lower()
    if "초" in v or "green" in v_low:
        return "초록색"
    if "노" in v or "yellow" in v_low:
        return "노란색"
    if "빨" in v or "red" in v_low:
        return "빨간색"
    return v


def norm_number(v: str | None) -> str | None:
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    digits = "".join(ch for ch in v if ch.isdigit())
    return digits or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])
    if not images:
        raise SystemExit(f"No images found under: {image_dir}")

    from transformers import Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))

    question = (
        "이 이미지에서 스티커가 있나요? 있으면 스티커의 색(color: 초록색/노란색/빨간색)과 "
        "손글씨 숫자(number: 숫자만)를 읽어주세요. 다음 JSON으로만 답하세요: "
        "{\"has_sticker\":true/false,\"color\":string|null,\"number\":string|null}"
    )

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "has_sticker", "color", "number", "raw"])

        for p in images:
            img = Image.open(p).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1)

            decoded = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            obj = parse_json_from_text(decoded)

            has = False
            color = None
            number = None
            if obj is not None:
                has = bool(obj.get("has_sticker", False))
                color = norm_color(obj.get("color")) if has else None
                number = norm_number(obj.get("number")) if has else None

            rel = p.as_posix()
            w.writerow([rel, str(has).lower(), color or "", number or "", decoded.replace("\n", " ")])

    print(f"[OK] wrote pseudo labels: {out} ({len(images)} rows)")


if __name__ == "__main__":
    main()

