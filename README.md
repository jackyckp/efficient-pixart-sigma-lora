# efficient-pixart-sigma-lora
Domain adaptation and efficient inference sampling benchmarks for PixArt-Sigma (DiT) using LoRA. Explores resource-constrained fine-tuning and optimal sampling configurations for specialized text-to-image generation.

Traning on: GPU NVIDIA GeForce RTX 4070 12GB

## 📋 Project Execution Pipeline

```
[💾 Phase 1: Data] ──> [⚙️ Phase 2: Train Matrix] ──> [🔮 Phase 3: Grid Inference] ──> [📊 Phase 4: Evaluation]
  - 📥 Collect & Clean    - 📐 3 Ranks (4, 8, 16)       - ⏱️ 4 Steps (5, 10, 20, 50)    - 🤖 Quantitative (CLIP)
  - 🏷️ Auto-Captioning    - 📈 3 Data Scales            - 🎯 3 Guidance Scales          - 👥 Qualitative (Human)
  - ✂️ Split Subsets      - 💾 9 LoRA Weights Total     - ✍️ 3 Prompt Complexities      - 🗺️ Pareto Frontier Plot

```

```mermaid
flowchart LR
    %% Phase 1
    subgraph P1 [💾 Phase 1: Data Preprocessing]
        direction TD
        A[📥 Collect Master Dataset <br/> 300 Clean Images] --> B[✂️ Split into Nested Subsets <br/> 50 / 100 / 300 images]
        B --> C[🏷️ Automated Captioning <br/> Run BLIP-2 / LLaVA]
        C --> D[📦 Output: Prepared Image + Text Pairs]
    end

    %% Phase 2
    subgraph P2 [⚙️ Phase 2: Multi-Config Training]
        direction TD
        E[🤖 Initialize Base Model <br/> PixArt-Sigma DiT] --> F[📐 Configure Matrix Parameters <br/> Ranks: 4, 8, 16 x Data: 50, 100, 300]
        F --> G[🚀 Execute Training Loop <br/> run_train_matrix.sh]
        G --> H[💾 Output: 9 Saved LoRA Weights <br/> .safetensors files]
    end

    %% Phase 3
    subgraph P3 [🔮 Phase 3: Automated Grid Inference]
        direction TD
        I[✍️ Prepare 3 Test Prompts <br/> Simple / Combo / Complex] --> J[🔄 Configure Sampling Loops <br/> Steps: 5,10,20,50 x Guidance: 3,5,7.5]
        J --> K[💻 Batch Generation Script <br/> generate_grid.py]
        K --> L[🖼️ Output: 324 Target Images <br/> Structured Output Folder]
    end

    %% Phase 4
    subgraph P4 [📊 Phase 4: Evaluation & Analysis]
        direction TD
        M[🧪 Run Dual Assessments <br/> Latency, CLIP, & Human Metrics] --> N[📊 Consolidate Experimental Data]
        N --> O[🗺️ Map Performance <br/> Plot the Pareto Frontier]
    end

    %% Macro Flow: Horizontal Subgraph-to-Subgraph links
    P1 --> P2
    P2 --> P3
    P3 --> P4

    %% Formatting Style
    style P1 fill:#f9f9f9,stroke:#333,stroke-width:1px
    style P2 fill:#f5f7ff,stroke:#333,stroke-width:1px
    style P3 fill:#f5fff5,stroke:#333,stroke-width:1px
    style P4 fill:#fff5f5,stroke:#333,stroke-width:1px
```

---

## Phase 1: Data Architecture & Preprocessing

Before touching any GPU code, you must build your dataset subsets to guarantee perfect experimental control.

### 1. Data Splitting

Create a single master folder of 300 high-quality images of your chosen style (e.g., Tech Line Art). Ensure the subsets are nested so that the smaller sets are exact components of the larger ones:

* **Dataset A (Micro):** First 50 images.
* **Dataset B (Medium):** First 100 images (includes Dataset A).
* **Dataset C (Full):** All 300 images (includes Dataset B).

### 2. Automated Captioning

Do not caption manually. Use a Python script with **BLIP-2** or **LLaVA** to generate a matching `.txt` file for every single image.

* *File structure requirement:*

```bash
dataset_300/
├── 001.png
├── 001.txt  # Contains: "A technical line art blueprint of a mechanical gear, white background..."
├── 002.png
└── 002.txt

```

---

## Phase 2: Environment & Multi-Configuration Training

Since you need to train **9 distinct LoRA models** ($3 \text{ Ranks} \times 3 \text{ Data Scales}$), your best approach is writing a simple bash script to loop through the training matrix sequentially.

### 1. Core Stack

* **Framework:** Hugging Face `diffusers` + PyTorch.
* **Base Model:** `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (or the 512 variant if VRAM is tight).
* **Script base:** Modify the standard `train_text_to_image_lora.py` from Hugging Face's example repository to support PixArt-Sigma.

### 2. The Training Loop Automation Script (`run_train_matrix.sh`)

Instead of running commands manually 9 times, use this automated script layout:

```bash
#!/bin/bash
# Hyperparameter Arrays
RANKS=(4 8 16)
DATA_DIRS=("dataset_50" "dataset_100" "dataset_300")

for rank in "${RANKS[@]}"; do
  for data_dir in "${DATA_DIRS[@]}"; do
    echo "Running Training: Rank=$rank, Data=$data_dir"
    
    python train_text_to_image_lora.py \
      --pretrained_model_name_or_path="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS" \
      --train_data_dir="./data/$data_dir" \
      --rank=$rank \
      --output_dir="./outputs/lora_r${rank}_${data_dir}" \
      --resolution=1024 \
      --train_batch_size=4 \
      --max_train_steps=1000 \
      --checkpointing_steps=500 \
      --learning_rate=1e-4 \
      --seed=42
  done
done

```

---

## Phase 3: Automated Grid Inference (Sampling Phase)

Once training finishes, you will have 9 `.safetensors` files. Now you must evaluate them against the remaining variables: **4 Step configurations**, **3 Guidance Scales**, and **3 Prompts**.

> ⚠️ **Warning:** $9 \text{ models} \times 4 \text{ steps} \times 3 \text{ guidance scales} \times 3 \text{ prompts} = 324 \text{ generated images}$. **Do not do this manually.**

### 1. Setup Test Prompts

Prepare 3 specific prompt templates of escalating complexity:

* `PROMPT_SIMPLE`: "A car, [your style tag]."
* `PROMPT_COMBO`: "A sports car driving through a city street, [your style tag]."
* `PROMPT_COMPLEX`: "A futuristic aerodynamic sports car speeding down a neon-lit cyberpunk alleyway, intricate details, flawless [your style tag]."

### 2. Automated Evaluation Script (`generate_grid.py`)

Write an inference script that automatically loops through your parameters and names files systematically:

```python
import os
import itertools
from diffusers import PixArtSigmaPipeline
import torch

# Configuration Matrix
ranks = [4, 8, 16]
datasets = ["dataset_50", "dataset_100", "dataset_300"]
steps_list = [5, 10, 20, 50]
guidance_list = [3.0, 5.0, 7.5]
prompts = {"simple": "...", "combo": "...", "complex": "..."}

# Load Base Pipeline
pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", torch_dtype=torch.float16).to("cuda")

# Nested Grid Generation Loop
for r, d in itertools.product(ranks, datasets):
    lora_path = f"./outputs/lora_r{r}_{d}"
    pipe.load_lora_weights(lora_path)
    
    for steps, g_scale, p_name in itertools.product(steps_list, guidance_list, prompts.keys()):
        # Set deterministic seed for fair comparison
        generator = torch.Generator("cuda").manual_seed(42)
        
        image = pipe(
            prompts[p_name], 
            num_inference_steps=steps, 
            guidance_scale=g_scale,
            generator=generator
        ).images[0]
        
        # Save file with completely trackable metadata in the name
        filename = f"r{r}_{d}_step{steps}_g{g_scale}_{p_name}.png"
        image.save(os.path.join("./inference_results", filename))

```

---

## Phase 4: Metrics Collection & Analysis

With your 324 images sorted, finalize your study by mapping out the metrics.

### 1. Quantitative (Code-Driven)

* **Latency Tracking:** In your `generate_grid.py` script, wrap your `pipe()` call with `time.time()` to log exactly how many milliseconds each inference combination takes. Save these directly to a CSV file.
* **CLIPScore / ImageReward:** Write a fast batch script to load your generated images alongside their input text prompts to compute automated text-alignment scores.

### 2. Qualitative (Human Blind Test)

* Pick a subset of the images (e.g., focusing only on the `complex` prompt).
* Create a simple shared spreadsheet for your team. Grade images from 1 to 5 on two clear elements:
* *Style Alignment:* Did it actually look like tech line art/ink wash, or did it bleed back into a generic photo?
* *Structural Integrity:* Are the lines clean, or did the architecture or text turn into chaotic gibberish?

### 3. Deliverable Presentation (The Pareto Frontier)

Plot a 2D scatter plot where:

* **X-axis:** Inference Time (Latency in seconds).
* **Y-axis:** Quality Score (CLIPScore or Human Rating).

Your goal in your final presentation is to draw a line connecting the top-leftmost points. This line represents your **Pareto Frontier**—showing your class exactly where the optimal "quality-speed sweet spots" live when deploying a fine-tuned DiT model with constrained resources.
