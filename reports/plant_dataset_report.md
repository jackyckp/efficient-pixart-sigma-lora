# Plant-Only Dataset Subset Fine-Tuning Report

This report summarizes the addition and fine-tuning of a specialized **Plant-Only Dataset Subset** ($N=209$ images), removing multi-domain noise (animals, web images, and uncategorized items).

---

## 🌿 Dataset Composition & Filtering

The dataset was filtered directly from the canonical latent bundle (`clean_latents_512.zip`):

| Subset Name | Category Included | Image Count ($N$) | Percentage of Master Dataset |
| :--- | :--- | :--- | :--- |
| **`plant209`** | `plant/*` | **209 images** | **80.4%** |
| Master Dataset | All (`plant`, `animal`, `web`, `others`) | 260 images | 100.0% |

---

## ⚙️ Training Execution Details

- **LoRA Rank ($r$)**: 8
- **LoRA Alpha ($\alpha$)**: 8
- **Max Steps**: `1,000` steps (~4.8 epochs on 209 plant images)
- **Training Duration**: 485.35 seconds (~8.0 minutes on RTX 4070)
- **Checkpoint Saved**: `outputs/plant_dataset/r8_plant209_steps1000/lora_adapter`

---

## 🖼️ Sample Generation Output

- **Prompt**: `"A graceful bamboo stalk with delicate leaves, traditional Chinese ink wash painting style, sumi-e style"`
- **Seed**: `42`
- **Steps**: `20`
- **Guidance Scale**: `3.5`

![Plant-Only LoRA Model Output - Bamboo Sumi-e Style](./images/bamboo_plant_lora.png)

---

## 💻 Script Commands

### 1. Plant-Only Training Command
```powershell
python scripts/training/train_local_latent_lora.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --num-images 209 `
  --plant-only `
  --rank 8 `
  --max-train-steps 1000 `
  --output-dir outputs/plant_dataset/r8_plant209_steps1000
```

### 2. Inference Command
```powershell
python scripts/inference/generate_with_prompt.py `
  --prompt "A graceful bamboo stalk with delicate leaves, traditional Chinese ink wash painting style, sumi-e style" `
  --adapter outputs/plant_dataset/r8_plant209_steps1000/lora_adapter `
  --output outputs/plant_dataset/bamboo_plant_lora.png `
  --seed 42 `
  --num-inference-steps 20 `
  --guidance-scale 3.5
```
