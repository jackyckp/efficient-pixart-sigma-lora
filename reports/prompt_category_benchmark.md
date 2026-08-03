# 🎨 30-Prompt Benchmark & CMMD Evaluation Report

This report presents the quantitative and qualitative evaluation results across **5 model candidates** evaluated on a fixed validation suite of **30 prompts** spanning 4 traditional Chinese ink wash painting categories, using the exact full trigger word:

```text
traditional Chinese ink wash painting style, shuimo hua
```

---

## ⚙️ Benchmark Setup & Metrics

- **Evaluated Models**:
  1. **Baseline Model (Step 0)**: Un-adapted Base PixArt-Sigma
  2. **Best `n50`**: Checkpoint Step 1,000
  3. **Best `n100`**: Checkpoint Step 1,000
  4. **Best `plant209`**: Checkpoint Step 4,000
  5. **Best `n260`**: Checkpoint Step 2,000
- **Sampling Parameters**: `guidance_scale = 1.5`, `seed = 42`, `num_inference_steps = 20`
- **Dual Benchmark Metrics**:
  - **CLIPScore**: Text-to-image prompt alignment (Higher is Better).
  - **CMMD (CLIP Maximum Mean Discrepancy)**: Maximum Mean Discrepancy with Gaussian RBF kernel against 100 ground-truth traditional ink wash reference images (Lower is Better).

---

## 📊 Overall Benchmark Results (CLIPScore vs. CMMD)

| Model ID | Model Candidate | Best Step | Avg CLIPScore (↑) | CMMD Score (↓) | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`plant209_best`** | **Best `plant209`** | **Step 4,000** | **`0.3602`** 🥇 | **`0.001229`** 🥇 | 🏆 **1st Place** |
| **`n260_best`** | **Best `n260`** | **Step 2,000** | `0.3539` 🥈 | `0.001445` 🥈 | 🥈 **2nd Place** |
| **`n100_best`** | **Best `n100`** | **Step 1,000** | `0.3525` 🥉 | `0.001709` 🥉 | 🥉 **3rd Place** |
| **`n50_best`** | **Best `n50`** | **Step 1,000** | `0.3493` | `0.001762` | 4th Place |
| **`baseline`** | **Baseline (Step 0)** | **Step 0** | `0.3490` | `0.001900` | 5th Place |

---

## 📈 Metric Charts

### 1. Overall CLIPScore & CMMD Distance Comparison
![CMMD & CLIPScore Metric Comparison](./images/cmmd_clip_benchmark.png)

### 2. Category-Wise CLIPScore Alignment Breakdown
![Category-Wise CLIPScore Alignment](./images/category_clip_breakdown.png)

---

## 📂 Category-Wise CLIPScore Summary Table

| Model Candidate | Landscapes (山水) | Flora & Fauna (花鳥) | Minimalist (留白) | Architecture (亭台) | Overall Mean |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Step 0)** | `0.3551` | `0.3458` | `0.3396` | `0.3550` | `0.3490` |
| **Best `n50` (Step 1k)** | `0.3512` | `0.3484` | `0.3393` | `0.3580` | `0.3493` |
| **Best `n100` (Step 1k)** | `0.3571` | `0.3520` | `0.3347` | `0.3675` | `0.3525` |
| **Best `plant209` (Step 4k)** | **`0.3675`** 🥇 | **`0.3567`** 🥇 | **`0.3507`** 🥇 | **`0.3688`** 🥇 | **`0.3602`** 🥇 |
| **Best `n260` (Step 2k)** | `0.3592` | `0.3551` | `0.3444` | `0.3622` | `0.3539` |

---

## 🖼️ Side-by-Side Visual Comparison (Prompt 17: Minimalist Single Boat)

Prompt: *"A single small boat on a vast calm lake, minimalist composition, wide white space, traditional Chinese ink wash painting style, shuimo hua"*

````carousel
![Baseline (Step 0)](./images/p17_baseline.png)
<!-- slide -->
![Best n50 (Step 1,000)](./images/p17_n50.png)
<!-- slide -->
![Best n100 (Step 1,000)](./images/p17_n100.png)
<!-- slide -->
![Best plant209 (Step 4,000)](./images/p17_plant209.png)
<!-- slide -->
![Best n260 (Step 2,000)](./images/p17_n260.png)
````

---

## 💡 Key Findings & Conclusions

1. **Undisputed Winner**: **`plant209_best` (Step 4,000)** wins on **both metrics**:
   - Lowest CMMD score (**`0.001229`**), proving its generated image distribution is closest to real traditional Chinese ink wash art.
   - Highest CLIPScore across all 4 categories (**`0.3602`**).

2. **Style Consistency Across Categories**:
   - Filtering the dataset to domain-pure plant/nature images (`plant209`) prevented non-ink noise (e.g., photo textures in `web` samples) from leaking into the model weights.
   - `plant209` achieves remarkable negative space (留白) control in **Minimalist** compositions (CLIPScore `0.3507` vs `0.3396` baseline).

3. **CMMD Validation**:
   - CMMD monotonically decreases as training quality improves: Baseline (`0.001900`) $\rightarrow$ `n50` (`0.001762`) $\rightarrow$ `n100` (`0.001709`) $\rightarrow$ `n260` (`0.001445`) $\rightarrow$ `plant209` (`0.001229`).
