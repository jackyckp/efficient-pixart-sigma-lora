# efficient-pixart-sigma-lora

## 1. The Main Idea of the Project

We will adapt the pretrained PixArt-Sigma Diffusion Transformer to a specialized visual domain, such as technical line drawings, architectural sketches, or traditional ink-wash paintings. LoRA will enable parameter-efficient text-to-image fine-tuning under limited computational resources.
We will investigate how LoRA rank, training-set size, sampling steps, guidance scale, and LoRA strength affect visual quality, prompt alignment, style consistency, and inference latency.
Possible advanced objectives include:
Exploring two-step or four-step inference using existing accelerated schedulers or PixArt-compatible distilled checkpoints.
Evaluating whether domain-specific LoRA remains effective during low-step generation.
Studying prompt and LoRA-weight interpolation to control style strength.

## 2. The Key Outputs of the Project

A curated domain-specific image-caption dataset.
Trained PixArt-Sigma LoRA weights.
A reproducible fine-tuning and inference pipeline.
Images generated under different training and sampling configurations.
Quantitative results for prompt alignment, image quality, and inference latency.
Structured human evaluation of style consistency and visual quality.
Ablation studies on LoRA rank, data size, sampling steps, and guidance scale.
Analysis of failure cases, limitations, and quality-efficiency trade-offs.

## 3. Course Components Included

Diffusion Models: Sampling Data Distributions via Langevin Dynamics (Part II)
PixArt-Sigma is a diffusion-based generative model. The project examines its forward noising and reverse denoising processes, probabilistic sampling, and the effect of reducing reverse-process sampling steps.
Conditional Generative Models (Part III)
Image generation is conditioned on text prompts and domain-specific LoRA parameters. We will analyze how prompt content, guidance scale, and LoRA strength affect the generated outputs.
Data Generation by Sampling from a Data Distribution (Part I)
The project treats images as samples from an underlying domain-specific data distribution. LoRA fine-tuning adapts the pretrained model toward this distribution, while inference generates new samples from it.

## 4. Requirements and Preparation Plan

- Dataset: We will collect approximately 100–500 images from one specialized visual domain. Potential sources include Ink Wash Paintings (<https://www.kaggle.com/datasets/jianwenzhao/ink-wash/data?select=testB>), and hand-drawn sketches from Google QuickDraw on GitHub (<https://github.com/googlecreativelab/quickdraw-dataset>). We will verify the license of every image and retain its source and attribution information.

- Data preparation: Images will be filtered for quality and relevance, deduplicated, and resized to 512×512 (reduce memory requirements). Initial captions may be produced using BLIP (<https://huggingface.co/Salesforce/blip-image-captioning-base> ), followed by manual correction and the addition of a consistent domain trigger phrase. We will also prepare a fixed set of unseen evaluation prompts.

- Models and code: We will use the official PixArt-Sigma repository (<https://github.com/PixArt-alpha/PixArt-sigma>), the 512×512 checkpoint, and the official LoRA training instructions (<https://github.com/PixArt-alpha/PixArt-sigma/blob/master/asset/docs/pixart_lora.md>). The provided toy dataset (<https://huggingface.co/datasets/PixArt-alpha/pixart-sigma-toy-dataset>) will first be used to validate the training pipeline.

- Evaluation: Prompt alignment and perceptual quality will be evaluated using CLIP (<https://github.com/openai/CLIP>), ImageReward, inference latency, and structured human evaluation. The same prompts and random seeds will be used across experimental settings.
