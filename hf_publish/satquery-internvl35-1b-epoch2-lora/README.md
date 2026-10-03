---
language:
- en
license: other
base_model: OpenGVLab/InternVL3_5-1B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
- remote-sensing
- earth-observation
- spatial-grounding
- satellite-imagery
- peft
- lora
- internvl
- satquery
datasets:
- custom
metrics:
- iou
- accuracy
---

# SatQuery InternVL3.5-1B Epoch-2 LoRA

Official LoRA / PEFT adapter for **SatQuery Stage-2** fine-tuning of `OpenGVLab/InternVL3_5-1B-Instruct` for remote-sensing visual understanding, spatial grounding, and satellite image interpretation.

> **Note**: This repository contains **ONLY the LoRA / PEFT adapter weights** (`adapter_model.safetensors` and `adapter_config.json`). The base model weights are not hosted here and must be loaded from [OpenGVLab/InternVL3_5-1B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Instruct).

---

## 1. Overview

SatQuery is an agentic Earth Observation intelligence system that combines vision-language models with deterministic geospatial analysis tools (e.g. optical/SAR fetchers, vegetation indices, affine projections, terrain analysis, and change detection).

This adapter specializes the compact **InternVL3.5-1B-Instruct** model for:
- Precise natural-language spatial grounding and tiny-object localization in satellite rasters.
- Grounded visual question answering (VQA) over Earth observation scenes.
- Detailed remote-sensing scene captioning and visual change description.

---

## 2. Base Model

| Attribute | Specification |
| :--- | :--- |
| **Model ID** | [`OpenGVLab/InternVL3_5-1B-Instruct`](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Instruct) |
| **Total Base Parameters** | ~1.06 Billion (~1,070,990,336 parameters) |
| **LLM Backbone** | Qwen-based Causal LM |
| **Vision Encoder** | InternVision (~300M parameters, **frozen**) |
| **Projector** | MLP Cross-modal Projector (**frozen**) |

---

## 3. Fine-Tuning Method & LoRA Configuration

Fine-tuning was conducted strictly using **Parameter-Efficient Fine-Tuning (PEFT / LoRA)** on the language model projection layers while keeping the vision encoder and cross-modal projector frozen.

| Parameter | Verified Value |
| :--- | :--- |
| **PEFT Type** | LoRA (`LORA`) |
| **LoRA Rank ($r$)** | `16` |
| **LoRA Alpha ($\alpha$)** | `32` |
| **LoRA Dropout** | `0.05` |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Trainable Parameters** | `10,092,544` |
| **Total Parameters** | `1,070,990,336` |
| **Trainable Parameter Fraction** | `~0.9424%` |
| **Precision** | BF16 (`bfloat16`) |
| **Gradient Checkpointing** | Enabled (`true`) |
| **Micro-batch Size** | `1` |
| **Gradient Accumulation Steps** | `8` |
| **Effective Batch Size** | `8` |

---

## 4. Training Dataset (SatQuery 15K SFT Corpus)

The Stage-2 Supervised Fine-Tuning (SFT) corpus consists of 15,000 curated, deduplicated remote-sensing multimodal instruction pairs:

| Subset | Sample Count | Percentage | Primary Task |
| :--- | :---: | :---: | :--- |
| **Spatial Grounding** | 12,000 | 80% | Natural-language bounding-box localization & coordinates |
| **Remote Sensing VQA** | 2,250 | 15% | Multi-turn reasoning on land cover, infrastructure & water |
| **Scene Captioning** | 750 | 5% | Dense technical Earth observation descriptions |
| **Total Corpus** | **15,000** | **100%** | **11,326 unique remote sensing images** |

* **Data Integrity**: 0 duplicate records, 0 synthetic oversampling, 0 evaluation leakage.
* **Coordinate Convention**: All bounding boxes were normalized and formatted strictly as **`[ymin, xmin, ymax, xmax]`** with values in $[0, 1]$ (top-left origin).

---

## 5. Training Objectives & Architectural Principle

### SFT Objectives:
- Enhance natural-language spatial grounding and tiny-feature localization in aerial/satellite scenes.
- Improve structured coordinate output adherence without requiring external grounding heads.
- Strengthen multi-turn satellite visual conversation capabilities.

### Architectural Principle in SatQuery:
> **The VLM is designed for semantic visual interpretation, NOT for deterministic geospatial mathematics.**  
> In the SatQuery architecture, the VLM handles visual feature extraction and rough image-space grounding. Exact operations (NDVI calculation, CRS reprojecting, affine matrix transformations, polygon area calculations, and meteorological data fetching) are delegated to deterministic, mathematically grounded tools.

---

## 6. Training Run Details

- **Run ID**: `internvl35_satquery_stage2_2epoch_20260927_191016`
- **Epochs Completed**: `2`
- **Cumulative Sample Exposures**: `30,000`
- **Optimizer Steps**: `3,750`
- **Total Runtime**: `~5h 42m 23s` (Epoch-2 run: `2h 47m 58s`)
- **Throughput**: `~1.46 - 1.49 samples/sec`
- **Peak PyTorch VRAM**: `2.63 GB`
- **Hardware Stability**: 0 OOM errors, 0 NaNs/Infs, 0 sample failures.
- **Loss Progression**:
  - Epoch 1 Mean Loss: `~0.8655`
  - Epoch 2 Mean Loss: `~0.5537`
  - Final Step Loss: `0.4784` (moving average: `0.5331`)

---

## 7. Benchmark Results

Evaluation was performed on a frozen benchmark suite comprising **2,300 samples per model (6,900 total evaluations across 0 failures)** on standard Earth observation benchmarks: **VRSBench**, **RSVQA-HR**, and **SECOND**.

### Multi-Model Benchmark Comparison

| Metric | Clean Zero-Shot InternVL3.5-1B | SatQuery Epoch-2 (This Checkpoint) | SatQuery Epoch-3 |
| :--- | :---: | :---: | :---: |
| **VRSBench Overall** | 30.28% | **43.55%** | 42.69% |
| **Grounding Mean IoU** | 0.0434 | **0.4786** | 0.4799 |
| **Grounding Median IoU** | 0.0000 | **0.5262** | 0.5330 |
| **Grounding IoU@0.25** | 8.00% | **75.20%** | 75.60% |
| **Grounding IoU@0.50** | 0.80% | **54.40%** | 54.80% |
| **Grounding IoU@0.75** | 0.00% | **21.60%** | 20.80% |
| **Grounding Coord MAE** | 0.3970 | **0.0599** | 0.0598 |
| **Grounding Format Validity** | 58.40% | **100.00%** | 100.00% |
| **VRSBench Caption Score** | 29.57% | **40.72%** | 40.77% |
| **VRSBench VQA Accuracy** | **43.60%** | 42.80% | 41.00% |
| **SECOND Change Detection** | 3.33% | **24.00%** | 25.00% |
| **RSVQA-HR Accuracy** | **44.30%** | 35.10% | 33.70% |

### Latency Profile

| Metric | Clean Zero-Shot | SatQuery Epoch-2 | SatQuery Epoch-3 |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | 1.7727s | 1.5438s | 1.4147s |
| **P95 Latency** | 10.4799s | 6.0242s | 5.8252s |

---

## 8. Results Interpretation & Deployment Selection

**Epoch-2** was selected as the primary SatQuery deployment candidate based on multi-objective trade-offs:
1. **Dramatic Grounding Leap**: Mean IoU increased by $+1002\%$ ($0.0434 \rightarrow 0.4786$) and format validity reached $100\%$.
2. **Optimal Balance vs Epoch-3**: While Epoch-3 produced marginal additional grounding gains ($+0.0013$ Mean IoU), it showed lower aggregate VRSBench accuracy ($42.69\%$ vs $43.55\%$) and further degradation on VQA benchmarks ($41.00\%$ vs $42.80\%$).
3. **Diminishing Returns**: Results are consistent with diminishing returns from additional epochs, where over-specialization toward bounding box generation slightly trades off general descriptive open-ended VQA reasoning.

---

## 9. Important Limitations

- **Specialization Trade-off**: The adapter specializes the model toward remote-sensing grounding. A modest degradation is observed on general high-resolution VQA (RSVQA-HR: $44.30\% \rightarrow 35.10\%$, VRSBench VQA: $43.60\% \rightarrow 42.80\%$).
- **Approximate Visual Estimates**: Bounding box coordinates represent visual estimation in image space, not sub-meter geodetic measurements. In production, coordinates should be projected through the GeoTIFF's native affine matrix.
- **Resolution & Aspect Ratio**: Best performance is obtained with imagery preprocessed using InternVL dynamic tiling (standard input size $448 \times 448$).

---

## 10. Exact Coordinate Convention

All spatial grounding outputs follow this convention:
$$\text{bbox} = [y_{\min}, x_{\min}, y_{\max}, x_{\max}]$$
Where:
- $y_{\min}, x_{\min}$ is the top-left coordinate, normalized to $[0.0, 1.0]$.
- $y_{\max}, x_{\max}$ is the bottom-right coordinate, normalized to $[0.0, 1.0]$.
- Origin $(0.0, 0.0)$ is at the **top-left corner** of the image.

---

## 11. Quickstart & Inference

### Installation
```bash
pip install torch torchvision transformers peft pillow
```

### Python Example
```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

# 1. Load base InternVL model
base_model_id = "OpenGVLab/InternVL3_5-1B-Instruct"
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

# 2. Attach the SatQuery Epoch-2 LoRA adapter
adapter_id = "Praneyaarora/satquery-internvl35-1b-epoch2-lora"
if hasattr(model, "language_model"):
    model.language_model = PeftModel.from_pretrained(model.language_model, adapter_id)
else:
    model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

# 3. Load image & run inference
# (Use the standard InternVL load_image_tensor / dynamic_preprocess pipeline)
# image = Image.open("sample_satellite.png").convert("RGB")
# pixel_values = load_image_tensor(image).to(device).to(dtype)
# prompt = "<image>\nLocate the airport runway in this satellite image."
# response = model.chat(tokenizer, pixel_values, prompt, {"max_new_tokens": 512})
# print(response)
```

For full preprocessing functions, see [`inference/inference_example.py`](inference/inference_example.py).

---

## 12. SatQuery Integration

To use this Hugging Face adapter directly in the SatQuery environment:

### In `local_model_server/.env`:
```env
LOCAL_VLM_MODEL_ID=internvl-1b
LOCAL_VLM_HF_REPO=OpenGVLab/InternVL3_5-1B-Instruct
LOCAL_VLM_LORA_PATH=Praneyaarora/satquery-internvl35-1b-epoch2-lora
LOCAL_VLM_DEVICE=auto
LOCAL_VLM_PORT=8080
```

The server will automatically download the adapter from Hugging Face on initial startup, cache it in `HF_HOME`, and serve it over `POST /infer`.

---

## 13. License & Compliance

- **Base Model License**: This adapter builds upon `OpenGVLab/InternVL3_5-1B-Instruct`, which is distributed under the Apache 2.0 / OpenGVLab community license.
- **Adapter License**: Released for research, education, and geospatial application development in accordance with the base model and dataset terms of use.

---

## 14. Citation

```bibtex
@misc{satquery2026internvl35,
  title={SatQuery: Agentic Remote-Sensing Intelligence and Spatial Grounding with Fine-Tuned Vision-Language Models},
  author={Arora, Praneya and SatQuery Team},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/Praneyaarora/satquery-internvl35-1b-epoch2-lora}}
}
```
