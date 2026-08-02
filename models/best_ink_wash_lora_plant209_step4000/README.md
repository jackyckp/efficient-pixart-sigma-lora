---
language:
- en
- zh
license: apache-2.0
tags:
- text-to-image
- diffusers
- pixart-sigma
- lora
- Chinese-ink-wash
- sumi-e
- shuimo-hua
base_model: PixArt-alpha/PixArt-Sigma-XL-2-512-MS
instance_prompt: traditional Chinese ink wash painting style, shuimo hua
widget:
- text: "A solitary pine tree standing on a misty mountain cliff, traditional Chinese ink wash painting style, shuimo hua"
---

# 🖌️ PixArt-Sigma Traditional Chinese Ink Wash (Shuimo Hua) LoRA

This repository contains the fine-tuned **LoRA (Low-Rank Adaptation)** adapter for **PixArt-Sigma-XL-2-512-MS**, specialized in generating authentic **Traditional Chinese Ink Wash Paintings (水墨畫 / Sumi-e)**.

---

## 🏆 Benchmark & Model Provenance

- **Base Architecture**: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`
- **LoRA Rank ($r$)**: `16` | **Alpha ($\alpha$)**: `16` | **File Size**: `52.5 MB`
- **Optimal Step Checkpoint**: `4,000` steps (`plant209` dataset)
- **Benchmark Performance (30-Prompt Validation Suite)**:
  - 🥇 **Lowest CMMD Score**: **`0.001229`** (Closest distribution match to real Chinese ink wash masterworks)
  - 🥇 **Highest Avg CLIPScore**: **`0.3602`** (Across Landscapes, Flora & Fauna, Minimalist, and Architecture categories)

---

## 🏷️ Trigger Word

To activate the traditional ink wash style and proper negative space (留白) control, always include the full trigger phrase in your prompt:

```text
traditional Chinese ink wash painting style, shuimo hua
```

---

## 🛠️ How To Use

### 1. Using Hugging Face `diffusers` (Python)

```python
import torch
from diffusers import PixArtSigmaPipeline

# 1. Load base PixArt-Sigma pipeline
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
    torch_dtype=torch.float16
).to("cuda")

# 2. Load the LoRA adapter weights
pipe.load_lora_weights("./models/best_ink_wash_lora_plant209_step4000")

# 3. Define prompt with trigger word
prompt = "A solitary pine tree standing on a misty mountain cliff, traditional Chinese ink wash painting style, shuimo hua"

# 4. Generate image
image = pipe(
    prompt=prompt,
    guidance_scale=1.5,
    num_inference_steps=20,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

# 5. Save output
image.save("ink_wash_output.png")
```

---

### 2. Using the Repository Command-Line Interface

If you have cloned the project repository:

```bash
conda activate pixart311

python scripts/inference/generate_with_prompt.py \
  --prompt "Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua" \
  --adapter models/best_ink_wash_lora_plant209_step4000 \
  --output my_ink_wash.png \
  --guidance-scale 1.5 \
  --seed 42
```

---

## ⚙️ Recommended Sampling Parameters

| Parameter | Recommended Value | Reason / Effect |
| :--- | :--- | :--- |
| **`guidance_scale`** | **`1.5`** (Range: `1.2` – `3.0`) | `1.5` delivers smooth ink wash gradients, soft paper texture, and clean negative space (留白) without contrast blowouts. |
| **`num_inference_steps`** | **`20`** (Range: `20` – `30`) | Fast generation (~0.7s per image on modern GPU). |
| **Resolution** | **`512 x 512`** | Trained at native 512x512 latent dimensions. |

---

## 🎨 Recommended Prompt Categories & Examples

1. **🏔️ Landscapes (山水)**:
   > *"Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua"*
2. **🪶 Flora & Fauna (花鳥)**:
   > *"Ink wash bamboo in the wind, wet brush technique, delicate leaves, subtle grey tones, traditional Chinese ink wash painting style, shuimo hua"*
3. **⛵ Minimalist Composition (留白)**:
   > *"A single small boat on a vast calm lake, minimalist composition, wide white space, traditional Chinese ink wash painting style, shuimo hua"*
4. **🏯 Architecture & Figures (亭台)**:
   > *"Ancient wooden pavilion surrounded by swirling mountain fog, traditional Chinese ink wash painting style, shuimo hua"*