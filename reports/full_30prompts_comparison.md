# 🖼️ Full 30-Prompt Multi-Step Comparison: Baseline vs Step 4k vs Step 7k vs Step 10k

This document presents a 4-column side-by-side visual comparison for all **30 validation prompts** across 4 traditional Chinese ink wash painting categories:

1. **Baseline Model (Step 0)**: Base PixArt-Sigma (`--no-adapter`)
2. **Step 4,000 Model**: Rank 4 candidate (`plant209` Step 4,000 / Optimal Ink Bleed & Diffusion)
3. **Step 7,000 Model**: Rank 1 candidate (`plant209` Step 7,000 / Highest Joint Selection Score `+0.7970`)
4. **Step 10,000 Model**: Rank 6 candidate (`plant209` Step 10,000 / High Step Saturation)

- **Full Trigger Word**: `traditional Chinese ink wash painting style, shuimo hua`
- **Sampling**: `guidance_scale = 1.5`, `seed = 42`, `num_inference_steps = 20`
- **Local Comparison Folder**: [outputs/comparison_30prompts/](file:///C:/dev/efficient-pixart-sigma-lora/outputs/comparison_30prompts)


---
## 🎨 Category: Landscapes (山水)

### Prompt 01

> **Prompt**: *"Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P01](./images/p01_baseline.png) | ![Step 4k P01](./images/p01_best.png) | ![Step 7k P01](./images/p01_step7000.png) | ![Step 10k P01](./images/p01_step10000.png) |

### Prompt 02

> **Prompt**: *"Winding river flowing through steep mountain gorges, distant waterfall, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P02](./images/p02_baseline.png) | ![Step 4k P02](./images/p02_best.png) | ![Step 7k P02](./images/p02_step7000.png) | ![Step 10k P02](./images/p02_step10000.png) |

### Prompt 03

> **Prompt**: *"Cascading waterfall plunging into a misty ravine, jagged rock formations, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P03](./images/p03_baseline.png) | ![Step 4k P03](./images/p03_best.png) | ![Step 7k P03](./images/p03_step7000.png) | ![Step 10k P03](./images/p03_step10000.png) |

### Prompt 04

> **Prompt**: *"Snow-covered mountain range in winter, bare trees, frozen lake, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P04](./images/p04_baseline.png) | ![Step 4k P04](./images/p04_best.png) | ![Step 7k P04](./images/p04_step7000.png) | ![Step 10k P04](./images/p04_step10000.png) |

### Prompt 05

> **Prompt**: *"Autumn mountains with sparse foliage, winding stone path leading to a ridge, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P05](./images/p05_baseline.png) | ![Step 4k P05](./images/p05_best.png) | ![Step 7k P05](./images/p05_step7000.png) | ![Step 10k P05](./images/p05_step10000.png) |

### Prompt 06

> **Prompt**: *"Sunrise over sea of clouds and mountain spires, high contrast black ink brushwork, white space, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P06](./images/p06_baseline.png) | ![Step 4k P06](./images/p06_best.png) | ![Step 7k P06](./images/p06_step7000.png) | ![Step 10k P06](./images/p06_step10000.png) |

### Prompt 07

> **Prompt**: *"Quiet lake reflecting towering mountain shadows, serene water surface, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P07](./images/p07_baseline.png) | ![Step 4k P07](./images/p07_best.png) | ![Step 7k P07](./images/p07_step7000.png) | ![Step 10k P07](./images/p07_step10000.png) |

### Prompt 08

> **Prompt**: *"Storm clouds gathering above rugged cliffside pines, dynamic black ink splash technique, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P08](./images/p08_baseline.png) | ![Step 4k P08](./images/p08_best.png) | ![Step 7k P08](./images/p08_step7000.png) | ![Step 10k P08](./images/p08_step10000.png) |


---
## 🎨 Category: Flora & Fauna (花鳥)

### Prompt 09

> **Prompt**: *"Ink wash bamboo in the wind, wet brush technique, delicate leaves, subtle grey tones, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P09](./images/p09_baseline.png) | ![Step 4k P09](./images/p09_best.png) | ![Step 7k P09](./images/p09_step7000.png) | ![Step 10k P09](./images/p09_step10000.png) |

### Prompt 10

> **Prompt**: *"A pair of flying cranes soaring above misty clouds, elegant brushstrokes, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P10](./images/p10_baseline.png) | ![Step 4k P10](./images/p10_best.png) | ![Step 7k P10](./images/p10_step7000.png) | ![Step 10k P10](./images/p10_step10000.png) |

### Prompt 11

> **Prompt**: *"Blooming plum blossoms on a gnarled branch, delicate ink wash gradients, soft grey background, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P11](./images/p11_baseline.png) | ![Step 4k P11](./images/p11_best.png) | ![Step 7k P11](./images/p11_step7000.png) | ![Step 10k P11](./images/p11_step10000.png) |

### Prompt 12

> **Prompt**: *"Solitary eagle perched on an ancient pine branch, sharp gaze, bold black ink brushwork, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P12](./images/p12_baseline.png) | ![Step 4k P12](./images/p12_best.png) | ![Step 7k P12](./images/p12_step7000.png) | ![Step 10k P12](./images/p12_step10000.png) |

### Prompt 13

> **Prompt**: *"Lotus flowers blooming in a quiet pond, large wet ink leaves, dragonfly hovering, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P13](./images/p13_baseline.png) | ![Step 4k P13](./images/p13_best.png) | ![Step 7k P13](./images/p13_step7000.png) | ![Step 10k P13](./images/p13_step10000.png) |

### Prompt 14

> **Prompt**: *"A wild horse galloping across an open plain, dynamic ink wash style, fluid brush lines, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P14](./images/p14_baseline.png) | ![Step 4k P14](./images/p14_best.png) | ![Step 7k P14](./images/p14_step7000.png) | ![Step 10k P14](./images/p14_step10000.png) |

### Prompt 15

> **Prompt**: *"Wild orchids clinging to a mossy cliff, graceful curved leaves, minimalist ink wash style, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P15](./images/p15_baseline.png) | ![Step 4k P15](./images/p15_best.png) | ![Step 7k P15](./images/p15_step7000.png) | ![Step 10k P15](./images/p15_step10000.png) |

### Prompt 16

> **Prompt**: *"Koi fish swimming in clear water, soft ink wash ripples, transparent ink gradients, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P16](./images/p16_baseline.png) | ![Step 4k P16](./images/p16_best.png) | ![Step 7k P16](./images/p16_step7000.png) | ![Step 10k P16](./images/p16_step10000.png) |


---
## 🎨 Category: Minimalist Composition (意境/留白)

### Prompt 17

> **Prompt**: *"A single small boat on a vast calm lake, minimalist composition, wide white space, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P17](./images/p17_baseline.png) | ![Step 4k P17](./images/p17_best.png) | ![Step 7k P17](./images/p17_step7000.png) | ![Step 10k P17](./images/p17_step10000.png) |

### Prompt 18

> **Prompt**: *"Solitary fisherman sitting on a riverbank with a fishing rod, vast empty background, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P18](./images/p18_baseline.png) | ![Step 4k P18](./images/p18_best.png) | ![Step 7k P18](./images/p18_step7000.png) | ![Step 10k P18](./images/p18_step10000.png) |

### Prompt 19

> **Prompt**: *"Single bamboo stalk in the corner of a blank paper canvas, elegant white space composition, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P19](./images/p19_baseline.png) | ![Step 4k P19](./images/p19_best.png) | ![Step 7k P19](./images/p19_step7000.png) | ![Step 10k P19](./images/p19_step10000.png) |

### Prompt 20

> **Prompt**: *"A lone pine tree silhouette against a faint crescent moon, subtle grey wash, high negative space, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P20](./images/p20_baseline.png) | ![Step 4k P20](./images/p20_best.png) | ![Step 7k P20](./images/p20_step7000.png) | ![Step 10k P20](./images/p20_step10000.png) |

### Prompt 21

> **Prompt**: *"Faint outline of a distant mountain peak in heavy fog, minimalist ink wash composition, wide white space, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P21](./images/p21_baseline.png) | ![Step 4k P21](./images/p21_best.png) | ![Step 7k P21](./images/p21_step7000.png) | ![Step 10k P21](./images/p21_step10000.png) |

### Prompt 22

> **Prompt**: *"A single falling leaf landing on still water, delicate ink ripple lines, minimalist composition, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P22](./images/p22_baseline.png) | ![Step 4k P22](./images/p22_best.png) | ![Step 7k P22](./images/p22_step7000.png) | ![Step 10k P22](./images/p22_step10000.png) |

### Prompt 23

> **Prompt**: *"Distant flight of birds vanishing into empty mist, minimalist composition, wide negative space, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P23](./images/p23_baseline.png) | ![Step 4k P23](./images/p23_best.png) | ![Step 7k P23](./images/p23_step7000.png) | ![Step 10k P23](./images/p23_step10000.png) |


---
## 🎨 Category: Architecture & Figures (人物/亭台)

### Prompt 24

> **Prompt**: *"Ancient wooden pavilion surrounded by swirling mountain fog, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P24](./images/p24_baseline.png) | ![Step 4k P24](./images/p24_best.png) | ![Step 7k P24](./images/p24_step7000.png) | ![Step 10k P24](./images/p24_step10000.png) |

### Prompt 25

> **Prompt**: *"Ancient scholar walking along a winding stone path, traditional robes, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P25](./images/p25_baseline.png) | ![Step 4k P25](./images/p25_best.png) | ![Step 7k P25](./images/p25_step7000.png) | ![Step 10k P25](./images/p25_step10000.png) |

### Prompt 26

> **Prompt**: *"Secluded stone temple tucked in a deep pine forest, mist rising, detailed architecture, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P26](./images/p26_baseline.png) | ![Step 4k P26](./images/p26_best.png) | ![Step 7k P26](./images/p26_step7000.png) | ![Step 10k P26](./images/p26_step10000.png) |

### Prompt 27

> **Prompt**: *"Traditional thatched cottage near a bamboo grove, flowing stream, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P27](./images/p27_baseline.png) | ![Step 4k P27](./images/p27_best.png) | ![Step 7k P27](./images/p27_step7000.png) | ![Step 10k P27](./images/p27_step10000.png) |

### Prompt 28

> **Prompt**: *"Ancient stone bridge spanning a misty river, small pavilion on a cliff, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P28](./images/p28_baseline.png) | ![Step 4k P28](./images/p28_best.png) | ![Step 7k P28](./images/p28_step7000.png) | ![Step 10k P28](./images/p28_step10000.png) |

### Prompt 29

> **Prompt**: *"Old scholar sitting inside a pavilion reading a book, mountain view, detailed ink wash technique, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P29](./images/p29_baseline.png) | ![Step 4k P29](./images/p29_best.png) | ![Step 7k P29](./images/p29_step7000.png) | ![Step 10k P29](./images/p29_step10000.png) |

### Prompt 30

> **Prompt**: *"Winding mountain staircase leading to a cloud-wrapped pagoda, traditional Chinese ink wash painting style, shuimo hua"*

| Baseline (Step 0) | Step 4,000 (Optimal Ink Bleed) | Step 7,000 (Top Joint Score) | Step 10,000 (Late Checkpoint) |
| :---: | :---: | :---: | :---: |
| ![Baseline P30](./images/p30_baseline.png) | ![Step 4k P30](./images/p30_best.png) | ![Step 7k P30](./images/p30_step7000.png) | ![Step 10k P30](./images/p30_step10000.png) |

