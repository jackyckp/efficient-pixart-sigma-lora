# Selected 5-Step Student Adapter

This directory contains the deployable PEFT LoRA adapter selected at Student optimizer step 8,000.

| Property | Value |
|---|---|
| Base model | `PixArt-alpha/PixArt-Sigma-XL-2-512-MS` |
| Inference steps | 5 |
| Guidance scale | 1.0 |
| LoRA rank | 4 |
| Student updates | 8,000 |
| SHA-256 | `13413F5322CE4EB340209853CE6F7C5E0BFBAA6B2D756797FCFDD1E78FF85C7E` |

The adapter must be loaded into the same PixArt-Sigma Transformer architecture used during training. It is not a standalone diffusion model and does not include the base model, tokenizer, T5 encoder, VAE, or scheduler.

