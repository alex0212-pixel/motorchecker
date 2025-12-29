"""Debug why a certain image is predicted as yellow when it looks green.

This tool runs multiple inference passes with controlled settings to see if
the output is unstable.

Usage (RunPod):
  cd /workspace/motorchecker
  source .venv/bin/activate
  export CUDA_VISIBLE_DEVICES=1
  export MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
  export ADAPTER_DIR=outputs/users/alex0212/qwen2vl-lora

  python runpod/debug_color_mismatch.py --image data/motor_checker/20240817_000102.jpg

Tips:
- If outputs change across runs, reduce temperature or enforce JSON.
- If consistently wrong, labels for similar samples may be wrong or too few.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel


DEFAULT_QUESTION = (
    "스티커가 있나요? 스티커가 있으면 색(color: 초록색/노란색/빨간색)과 번호(number: 숫자만)를 읽고, "
    "없으면 has_sticker=false로 답하세요. 반드시 아래 JSON 형식으로만 답하세요: "
    '{"has_sticker":true/false,"color":"초록색"|"노란색"|"빨간색"|null,"number":string|null}'
)


def _parse_json(decoded: str):
    t = decoded
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0].strip()
    elif "```" in t:
        t = t.split("```")[1].split("```")[0].strip()
    s = t.find("{")
    e = t.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(t[s : e + 1])
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default=os.environ.get("QUESTION", DEFAULT_QUESTION))
    ap.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct"))
    ap.add_argument("--adapter_dir", default=os.environ.get("ADAPTER_DIR", "outputs/users/anon/qwen2vl-lora"))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from transformers import Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.adapter_dir if os.path.isdir(args.adapter_dir) else args.model_name)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    img = Image.open(args.image).convert("RGB")
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

    print(f"MODEL_NAME={args.model_name}")
    print(f"ADAPTER_DIR={args.adapter_dir}")
    print(f"IMAGE={args.image}")
    print(f"temperature={args.temperature}")
    print("=" * 80)

    for i in range(args.repeat):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=args.temperature,
                do_sample=(args.temperature > 0),
            )
        decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
        obj = _parse_json(decoded)
        print(f"\nRUN {i+1}/{args.repeat}")
        print(decoded)
        print("PARSE:")
        print(json.dumps(obj, ensure_ascii=False, indent=2) if obj else "(failed)")


if __name__ == "__main__":
    main()

