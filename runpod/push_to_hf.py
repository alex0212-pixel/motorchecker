"""Push Qwen2-VL LoRA adapter (or merged model) to Hugging Face Hub.

Two common strategies
1) Adapter-only (recommended): upload only LoRA adapter weights + processor.
   - Pros: small, fast to upload
   - Cons: need base model at inference time

2) Merged full model: merge LoRA into base then upload full weights.
   - Pros: single artifact for inference
   - Cons: huge (7B), needs much more storage and time

Environment variables:
  HF_TOKEN        (required)
  HF_REPO_ID      (required) e.g. "yourname/motor-sticker-qwen2vl-lora"
  ADAPTER_DIR     default: outputs/qwen2vl-lora
  BASE_MODEL      default: Qwen/Qwen2-VL-7B-Instruct
  MODE            "adapter" | "merged" (default: adapter)

Usage:
  source .venv/bin/activate
  export HF_TOKEN=...
  export HF_REPO_ID=yourname/motor-sticker-qwen2vl-lora
  python runpod/push_to_hf.py
"""

from __future__ import annotations

import os
import shutil
import tempfile


def _require(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing env: {name}")
    return v


def push_adapter_only(repo_id: str, token: str, adapter_dir: str, base_model: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"[OK] repo ready: {repo_id}")

    # HF Hub validates model card metadata for adapter repos.
    # If README.md has an empty `base_model:` field, upload fails (400 validate-yaml).
    # We always create a minimal README.md with a valid base_model.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.abspath(tmp)
        print(f"[INFO] preparing upload dir: {tmp}")
        shutil.copytree(adapter_dir, tmp, dirs_exist_ok=True)

        readme_path = os.path.join(tmp, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                "---\n"
                f"base_model: {base_model}\n"
                "library_name: peft\n"
                "tags:\n"
                "  - peft\n"
                "  - lora\n"
                "  - qwen2-vl\n"
                "---\n\n"
                "# MotorChecker Qwen2-VL LoRA Adapter\n\n"
                "This repository contains a LoRA adapter trained for motor sticker detection (has_sticker/color/number).\n\n"
                "## How to use\n\n"
                "Load the base model + this adapter using PEFT (PeftModel).\n"
            )

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=tmp,
            path_in_repo=".",
            commit_message="upload lora adapter",
        )
    print(f"[OK] uploaded adapter dir: {adapter_dir}")


def push_merged_model(repo_id: str, token: str, base_model: str, adapter_dir: str):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoProcessor
    from peft import PeftModel
    from transformers import Qwen2VLForConditionalGeneration

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"[OK] repo ready: {repo_id}")

    # Load base + adapter, merge, then save
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()

    # Processor: prefer adapter_dir if it has processor files
    processor = AutoProcessor.from_pretrained(adapter_dir if os.path.isdir(adapter_dir) else base_model)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"[INFO] saving merged model to: {tmp}")
        model.save_pretrained(tmp, safe_serialization=True)
        processor.save_pretrained(tmp)
        # keep adapter config too for reference
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            src = os.path.join(adapter_dir, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(tmp, name))

        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=tmp,
            path_in_repo=".",
            commit_message="upload merged model",
        )

    print("[OK] uploaded merged model")


def main():
    token = _require("HF_TOKEN")
    repo_id = _require("HF_REPO_ID")
    adapter_dir = os.environ.get("ADAPTER_DIR", "outputs/qwen2vl-lora")
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    mode = os.environ.get("MODE", "adapter").strip().lower()

    if mode not in {"adapter", "merged"}:
        raise SystemExit("MODE must be 'adapter' or 'merged'")

    if mode == "adapter":
        push_adapter_only(repo_id=repo_id, token=token, adapter_dir=adapter_dir, base_model=base_model)
    else:
        push_merged_model(repo_id=repo_id, token=token, base_model=base_model, adapter_dir=adapter_dir)


if __name__ == "__main__":
    main()
