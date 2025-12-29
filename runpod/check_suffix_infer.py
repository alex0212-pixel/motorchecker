"""Run inference on specific motor_checker images (by filename suffix).

Purpose:
- Quickly verify that images ending with specific numbers (e.g. 116/148/227/249/336)
  are handled correctly, especially the `has_sticker` gate.

Usage (RunPod):
  cd /workspace/motorchecker
  source .venv/bin/activate

  # pick GPU
  export CUDA_VISIBLE_DEVICES=1

  # base model + adapter
  export MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
  export ADAPTER_DIR=outputs/users/alex0212/qwen2vl-lora

  python runpod/check_suffix_infer.py --suffixes 116 148 227 249 336

Notes:
- It prints RAW output and a JSON parse attempt.
- If `--adapter_dir` points to a HF Hub repo id, PeftModel can also load it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel


DEFAULT_QUESTION = (
    "스티커가 있나요? 아래 JSON 형식으로만 답하세요: "
    '{"has_sticker":true/false,"number":string|null,"color":"초록색"|"노란색"|"빨간색"|null}'
)


def _parse_json(decoded: str):
    text_out = decoded
    if "```json" in text_out:
        text_out = text_out.split("```json")[1].split("```")[0].strip()
    elif "```" in text_out:
        text_out = text_out.split("```")[1].split("```")[0].strip()
    s = text_out.find("{")
    e = text_out.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text_out[s : e + 1])
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/motor_checker")
    ap.add_argument("--suffixes", nargs="+", required=True, help="e.g. 116 148 227")
    ap.add_argument("--question", default=os.environ.get("QUESTION", DEFAULT_QUESTION))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct"))
    ap.add_argument("--adapter_dir", default=os.environ.get("ADAPTER_DIR", "outputs/users/anon/qwen2vl-lora"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    suffixes = [str(s) for s in args.suffixes]

    # match ..._000116.jpg style
    pat = re.compile(r"_0+({})\\.jpg$".format("|".join(map(re.escape, suffixes))))
    images = sorted([p for p in data_dir.glob("*.jpg") if pat.search(p.name)])

    if not images:
        raise SystemExit(f"No matching images under {data_dir} for suffixes={suffixes}")

    from transformers import Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.adapter_dir if os.path.isdir(args.adapter_dir) else args.model_name)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    print(f"MODEL_NAME={args.model_name}")
    print(f"ADAPTER_DIR={args.adapter_dir}")
    print(f"QUESTION={args.question}")
    print("=" * 80)

    for p in images:
        img = Image.open(p).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": args.question},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[img], return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, temperature=0.1)

        decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
        obj = _parse_json(decoded)

        print(f"\n--- {p.name} ---")
        print("RAW:")
        print(decoded)
        print("PARSE:")
        if obj is None:
            print("(failed)")
        else:
            print(json.dumps(obj, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

