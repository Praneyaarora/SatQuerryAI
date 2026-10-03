"""SatQuery InternVL3.5-1B Epoch-2 LoRA — Minimal Inference Example.

This script demonstrates how to:
1. Load the official base model (OpenGVLab/InternVL3_5-1B-Instruct).
2. Attach the fine-tuned SatQuery Epoch-2 PEFT/LoRA adapter.
3. Perform multimodal remote-sensing spatial grounding / VQA inference.

Coordinate Convention:
  Bounding boxes are output as [ymin, xmin, ymax, xmax] normalized to [0, 1].
"""

import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int = 448) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 12, image_size: int = 448):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    columns = target_width // image_size
    for i in range(blocks):
        box = (
            (i % columns) * image_size,
            (i // columns) * image_size,
            ((i % columns) + 1) * image_size,
            ((i // columns) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image_tensor(image: Image.Image, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


def main():
    base_model_id = os.getenv("BASE_MODEL", "OpenGVLab/InternVL3_5-1B-Instruct")
    adapter_id = os.getenv("ADAPTER_ID", "Praneyaarora/satquery-internvl35-1b-epoch2-lora")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"1. Loading Base Model: {base_model_id} on {device} ({dtype}) ...")
    model = AutoModel.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"2. Attaching SatQuery Epoch-2 LoRA Adapter: {adapter_id} ...")
    if hasattr(model, "language_model"):
        model.language_model = PeftModel.from_pretrained(model.language_model, adapter_id)
    else:
        model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    print("[OK] Model & Adapter loaded successfully.")

    # 3. Create or load a sample image
    image_path = os.getenv("IMAGE_PATH")
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        # Create a synthetic test RGB swatch if no image path provided
        image = Image.new("RGB", (512, 512), color=(45, 90, 60))

    pixel_values = load_image_tensor(image).to(device).to(dtype)
    prompt = "<image>\nLocate and describe any water bodies or vegetation in this satellite image."
    generation_config = {"max_new_tokens": 512, "do_sample": False}

    print(f"\n3. Running Inference:\nPrompt: {prompt}\n")
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    print("--- RAW MODEL OUTPUT ---")
    print(response)
    print("------------------------")


if __name__ == "__main__":
    main()
