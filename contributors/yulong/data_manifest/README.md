# Training Data and Cache Manifest

The local experiment used the full aligned ink-wash dataset rather than a smaller subset.

| Item | Value |
|---|---|
| Aligned samples | 260 |
| Training samples | 260 |
| Resolution | 512 x 512 |
| Dataset/cache fingerprint | `b9d3c2d1d404` |
| Image-latent cache | `image_latents_n260_res512_b9d3c2d1d404.pt` |
| T5 text cache | `t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` |
| Text length | 300 tokens |
| Text dtype | FP16 |

The paired dataset follows the repository's ink-wash corpus and caption workflow. Training captions use the project style trigger where applicable:

```text
traditional Chinese ink wash painting style, sumi-e style
```

The `.pt` caches are not committed because they are large derived files. The filename fingerprint is retained so that image latents and text embeddings from different dataset versions cannot be mixed accidentally.

