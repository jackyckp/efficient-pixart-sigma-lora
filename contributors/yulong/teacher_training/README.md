# Style Teacher Training

`train_pixart_sigma_lora_local.py` trains a PEFT LoRA against precomputed VAE latents and T5 embeddings. The PixArt-Sigma base Transformer remains frozen; only LoRA parameters are optimized.

The LoRA target modules include attention projections and feed-forward/projection layers:

- `to_q`, `to_k`, `to_v`, `to_out.0`
- `ff.net.0.proj`, `ff.net.2`
- `proj_in`, `proj_out`, `proj`, `linear`, `linear_1`, `linear_2`

## Teacher run

```powershell
python .\teacher_training\train_pixart_sigma_lora_local.py `
  --max-train-steps 10000 `
  --checkpointing-steps 1000 `
  --learning-rate 1e-5 `
  --rank 16 `
  --alpha 16 `
  --batch-size 1 `
  --gradient-accumulation-steps 1 `
  --caption-dropout 0 `
  --warmup-steps 0 `
  --seed 42
```

The selected reference for distillation is checkpoint `step_004000`. Its metadata is preserved in `selected_teacher_metadata/`; its model weight is omitted to avoid duplicating another 52.6 MiB adapter in Git history.

Selected Teacher adapter SHA-256:

```text
43E08B370E0AF99AE58C24D78649B380FA5B4154B486D1D58A9CA4FA60509EA3
```

