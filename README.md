# efficient-pixart-sigma-lora

Domain adaptation and efficient inference sampling benchmarks for PixArt-Sigma (DiT) using LoRA. Explores resource-constrained fine-tuning and optimal sampling configurations for specialized text-to-image generation.

Training setup: GPU NVIDIA GeForce RTX 4070 12GB

## Current local-latent workflow

The first official toy-data smoke test remains available at `notebooks/smoke/pixart_sigma_lora_smoke_test_official.ipynb`. The local workflow now reads the validated clean latent ZIP directly, so it does not extract images or rerun the VAE.

Current contract:

- Python 3.11.2 and PixArt-Sigma 512.
- Canonical dataset size: 260 image-caption pairs; experiments use deterministic nested 50 / 100 / 260 subsets.
- Canonical latent cache: `[260, 4, 64, 64]`, float16, clean scaled latents, fingerprint `b9d3c2d1d404`.
- The validated T5 prompt cache is available locally under `data/features/` and remains Git-ignored because it is about 642 MB.

Before training on a new checkout, obtain these two Git-ignored files from the project shared storage:

```text
data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt
data/features/validation_summary.json
```

Validate all assets that are available now without downloading a model:

```powershell
python scripts/training/train_local_latent_lora.py --validate-assets-only
```

Run the local smoke test with the validated prompt cache:

```powershell
python scripts/training/train_local_latent_lora.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --num-images 50 `
  --rank 8 `
  --max-train-steps 100 `
  --output-dir outputs/local_smoke/r8_n50_steps100
```

This performs 100 optimizer updates, saves a PEFT adapter, reloads it onto a fresh base transformer, and generates one 512 x 512 image with the first selected cached training embedding. See `data/README.md` for the prompt-cache schema and `notebooks/training/pixart_local_latent_smoke.ipynb` for the thin notebook entry point.

Generate from a new prompt after training:

```powershell
python scripts/inference/generate_with_prompt.py `
  --prompt "A majestic black stallion galloping across an open plain, Chinese ink wash painting style, Sumi-e" `
  --adapter outputs/local_smoke/r8_n50_steps100/lora_adapter `
  --output outputs/unseen_galloping_stallion.png `
  --seed 456 `
  --num-inference-steps 20 `
  --guidance-scale 1.0
```

The inference command encodes the supplied text with T5, releases T5 memory, fresh-loads the base transformer plus LoRA adapter, writes a 512 x 512 PNG and adjacent JSON metadata, and rejects an exact training-caption match unless `--allow-seen-prompt` is supplied.

## Data Source

This project uses a 260-image ink-wash corpus collected from Tappu via [scripts/data/download_tappu.py](scripts/data/download_tappu.py). Canonical inputs are tracked as validated ZIP archives under `data/archives/`; extracted folders remain ignored.

- [source image-caption archive](data/archives/ink.zip)
- [clean 512px latent bundle](data/archives/clean_latents_512.zip)
- [asset contracts and validation notes](data/README.md)

The source archive contains category subfolders; every JPG has a same-name `.txt` caption.

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
        A[📥 Collect Master Dataset <br/> 260 Clean Images] --> B[✂️ Split into Nested Subsets <br/> 50 / 100 / 260 images]
        B --> C[🏷️ Automated Captioning <br/> Run BLIP-2 / LLaVA]
        C --> D[📦 Output: Prepared Image + Text Pairs]
    end

    %% Phase 2
    subgraph P2 [⚙️ Phase 2: Multi-Config Training]
        direction TD
        E[🤖 Initialize Base Model <br/> PixArt-Sigma DiT] --> F[📐 Configure Matrix Parameters <br/> Ranks: 4, 8, 16 x Data: 50, 100, 260]
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

Before touching any GPU code, you must build a local dataset in a deterministic structure so that training and evaluation are reproducible.

### 1. Collecting the Source Data

Run [scripts/data/download_tappu.py](scripts/data/download_tappu.py) from the project root to scrape the Tappu gallery and populate the local dataset folders:

```bash
python scripts/data/download_tappu.py
```

The script downloads images and translated captions into the following structure:

```bash
data/ink/
├── animal/
│   ├── 100.jpg
│   ├── 100.txt
│   └── ...
├── plant/
│   ├── 200.jpg
│   ├── 200.txt
│   └── ...
└── others/
    ├── no_num_1001.jpg
    ├── no_num_1001.txt
    └── ...
```

These local image folders are then used directly by the notebook workflow and the captioning script.

### 2. Automated Captioning

Use [scripts/data/auto_caption.py](scripts/data/auto_caption.py) to generate a matching `.txt` file for every image in an extracted local dataset folder.

```bash
python scripts/data/auto_caption.py --dir ./data/ink --model florence-2 --trigger "traditional Chinese ink wash painting style, shuimo hua"
```

The notebook is configured to point at the local dataset under [data/ink](data/ink), so the captioning step can be run directly on that folder.

---

## Phase 2: Environment & Multi-Configuration Training

Since you need to train **9 distinct LoRA models** ($3 \text{ Ranks} \times 3 \text{ Data Scales}$), your best approach is writing a simple bash script to loop through the training matrix sequentially.

### 1. Core Stack

- **Framework:** Hugging Face `diffusers` + PyTorch.
- **Base Model:** `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`.
- **Training entry point:** `scripts/training/train_local_latent_lora.py`, using the official PixArt objective with precomputed local features.

### 2. The Training Loop Automation Script (`run_train_matrix.sh`)

Instead of running commands manually 9 times, use this automated script layout:

```bash
#!/bin/bash
# Hyperparameter Arrays
RANKS=(4 8 16)
DATA_SIZES=(50 100 260)

for rank in "${RANKS[@]}"; do
  for num_images in "${DATA_SIZES[@]}"; do
    echo "Running Training: Rank=$rank, Data=$num_images"
    
    python scripts/training/train_local_latent_lora.py \
      --latent-bundle="data/archives/clean_latents_512.zip" \
      --prompt-cache="data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt" \
      --num-images=$num_images \
      --rank=$rank \
      --output-dir="./outputs/lora_r${rank}_n${num_images}" \
      --max-train-steps=1000 \
      --learning-rate=1e-5 \
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

- `PROMPT_SIMPLE`: "A car, [your style tag]."
- `PROMPT_COMBO`: "A sports car driving through a city street, [your style tag]."
- `PROMPT_COMPLEX`: "A futuristic aerodynamic sports car speeding down a neon-lit cyberpunk alleyway, intricate details, flawless [your style tag]."

### 2. Automated Evaluation Script (`generate_grid.py`)

Write an inference script that automatically loops through your parameters and names files systematically:

```python
import os
import itertools
from diffusers import PixArtSigmaPipeline
import torch

# Configuration Matrix
ranks = [4, 8, 16]
dataset_sizes = [50, 100, 260]
steps_list = [5, 10, 20, 50]
guidance_list = [3.0, 5.0, 7.5]
prompts = {"simple": "...", "combo": "...", "complex": "..."}

# Load Base Pipeline
pipe = PixArtSigmaPipeline.from_pretrained("PixArt-alpha/PixArt-Sigma-XL-2-512-MS", torch_dtype=torch.float16).to("cuda")

# Nested Grid Generation Loop
for r, d in itertools.product(ranks, dataset_sizes):
    lora_path = f"./outputs/lora_r{r}_n{d}"
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

- **Latency Tracking:** In your `generate_grid.py` script, wrap your `pipe()` call with `time.time()` to log exactly how many milliseconds each inference combination takes. Save these directly to a CSV file.
- **CLIPScore / ImageReward:** Write a fast batch script to load your generated images alongside their input text prompts to compute automated text-alignment scores.

### 2. Qualitative (Human Blind Test)

- Pick a subset of the images (e.g., focusing only on the `complex` prompt).
- Create a simple shared spreadsheet for your team. Grade images from 1 to 5 on two clear elements:
- *Style Alignment:* Did it actually look like tech line art/ink wash, or did it bleed back into a generic photo?
- *Structural Integrity:* Are the lines clean, or did the architecture or text turn into chaotic gibberish?

### 3. Deliverable Presentation (The Pareto Frontier)

Plot a 2D scatter plot where:

- **X-axis:** Inference Time (Latency in seconds).
- **Y-axis:** Quality Score (CLIPScore or Human Rating).

Your goal in your final presentation is to draw a line connecting the top-leftmost points. This line represents your **Pareto Frontier**—showing your class exactly where the optimal "quality-speed sweet spots" live when deploying a fine-tuned DiT model with constrained resources.
