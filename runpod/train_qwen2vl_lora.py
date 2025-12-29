"""Qwen2-VL-7B-Instruct LoRA/QLoRA SFT (최소 동작 뼈대)

웹터미널에서 실행한 과정을 '파일로 남기기' 위한 목적의 스크립트입니다.
데이터/모델 버전에 따라 VLM 입력 키가 달라질 수 있으니,
실제 데이터 스키마가 확정되면 collator/labels 마스킹을 더 정교하게 조정하는 걸 추천합니다.

환경변수로 제어:
- MODEL_NAME
- TRAIN_JSONL
- OUTPUT_DIR
- USE_QLORA (true/false)
- NUM_EPOCHS, LR, BATCH_SIZE, GRAD_ACCUM, SAVE_STEPS, LOGGING_STEPS
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    # When running via `accelerate launch runpod/train_qwen2vl_lora.py`,
    # python may not include repo root in sys.path depending on working dir.
    from runpod.augment import AugmentConfig, augment_pil
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from runpod.augment import AugmentConfig, augment_pil


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_messages(image_path: str, question: str, answer: str):
    # 2단계 게이트까지 하고 싶다면 question/answer를 그 스키마로 구성하면 됨
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def preprocess_example(processor: Any, ex: Dict[str, Any]) -> Dict[str, Any]:
    image_path = ex["image"]
    question = ex.get("question", "")
    answer = ex.get("answer", "")

    messages = build_messages(image_path, question, answer)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image = load_image(image_path)
    # on-the-fly augmentation to make model robust to rotation/warp
    if env_bool("AUGMENT", True):
        cfg = AugmentConfig(
            enabled=True,
            p=float(os.environ.get("AUG_P", "0.85")),
            max_rotate=int(os.environ.get("AUG_MAX_ROTATE", "180")),
        )
        image = augment_pil(image, cfg)

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    labels = inputs["input_ids"].clone()
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    inputs["labels"] = labels

    return {k: v[0] for k, v in inputs.items()}


@dataclass
class Collator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]):
        # tokenizer pad로 input_ids/attention_mask/labels 정렬
        to_pad = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "labels": [f["labels"] for f in features],
        }
        padded = self.processor.tokenizer.pad(to_pad, padding=True, return_tensors="pt")
        batch = dict(padded)

        # 나머지 텐서 키(비전 입력)는 stack
        for k in features[0].keys():
            if k in batch:
                continue
            if torch.is_tensor(features[0][k]):
                batch[k] = torch.stack([f[k] for f in features])
        return batch


def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")
    train_jsonl = os.environ.get("TRAIN_JSONL", "/workspace/train.jsonl")
    output_dir = os.environ.get("OUTPUT_DIR", "outputs/qwen2vl-lora")

    use_qlora = env_bool("USE_QLORA", True)

    # augmentation control
    _ = env_bool("AUGMENT", True)

    num_epochs = float(os.environ.get("NUM_EPOCHS", "1"))
    lr = float(os.environ.get("LR", "2e-4"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "16"))
    save_steps = int(os.environ.get("SAVE_STEPS", "200"))
    logging_steps = int(os.environ.get("LOGGING_STEPS", "10"))

    processor = AutoProcessor.from_pretrained(model_name)

    quant_cfg = None
    if use_qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    from transformers import Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=quant_cfg,
        device_map="auto",
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=train_jsonl, split="train")
    ds = ds.map(lambda ex: preprocess_example(processor, ex), remove_columns=ds.column_names)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=Collator(processor),
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
