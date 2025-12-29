import os
import json
import torch
from PIL import Image

from transformers import AutoProcessor
from peft import PeftModel


def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
    adapter_dir = os.environ.get("ADAPTER_DIR", "outputs/qwen2vl-lora")
    test_image = os.environ.get("TEST_IMAGE")
    question = os.environ.get("QUESTION", "스티커가 있나요?")

    if not test_image:
        raise SystemExit("TEST_IMAGE env가 비어 있습니다")

    # Qwen2-VL 모델 클래스
    from transformers import Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(adapter_dir if os.path.isdir(adapter_dir) else model_name)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    img = Image.open(test_image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
        )

    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    print("\n=== RAW OUTPUT ===")
    print(decoded)

    # JSON만 뽑아보는 시도(강제는 아니지만 확인용)
    print("\n=== PARSE TRY ===")
    try:
        # 응답에 코드블럭이 섞였을 때 제거
        text_out = decoded
        if "```json" in text_out:
            text_out = text_out.split("```json")[1].split("```")[0].strip()
        elif "```" in text_out:
            text_out = text_out.split("```")[1].split("```")[0].strip()
        # 마지막에 JSON 객체가 있다고 가정하고 탐색
        start = text_out.find("{")
        end = text_out.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text_out[start : end + 1])
            print(json.dumps(obj, ensure_ascii=False, indent=2))
        else:
            print("No JSON object found")
    except Exception as e:
        print("JSON parse failed:", e)


if __name__ == "__main__":
    main()

