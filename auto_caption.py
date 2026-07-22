#!/usr/bin/env python3
"""
auto_caption.py — Automated Image Captioning for PixArt-Sigma LoRA Datasets
============================================================================
Loops through an image folder (or multiple dataset subfolders), feeds each
image to a vision-language model, and saves a matching .txt caption file
alongside every image.

Supported models
----------------
  florence-2  microsoft/Florence-2-large                     (~3 GB VRAM, excellent detail)
  blip2       Salesforce/blip2-opt-2.7b                      (~5 GB VRAM float16, fast, supports text hints)
  joycaption  fancyfeast/llama-joycaption-alpha-two-hf       (~5 GB VRAM with 4-bit NF4 quantization, richest descriptions)

Usage examples
--------------
  # Caption a single folder with Florence-2 (default)
  python auto_caption.py --dir data/dataset_300

  # Caption all dataset subfolders inside data/
  python auto_caption.py --dir data --all-subsets --model florence-2

  # Use JoyCaption, append trigger word, overwrite existing captions
  python auto_caption.py --dir data/dataset_web --model joycaption --overwrite

  # Custom trigger phrase (ink-wash project default)
  python auto_caption.py --dir data/dataset_web --trigger "traditional Chinese ink wash painting style"

  # Refine mode — load existing .txt as reference hint in the prompt to improve captions
  python auto_caption.py --dir data/dataset_web --model joycaption --refine --overwrite

  # Dry run — print what would be done without writing anything
  python auto_caption.py --dir data/dataset_web --dry-run

Notes & Features
----------------
  - PixArt-Sigma responds best to natural, descriptive text, not comma-tag lists.
    Florence-2 produces this style natively.
  - JoyCaption produces longer, richer captions; loaded dynamically using 4-bit NF4
    quantization via BitsAndBytesConfig (~5 GB VRAM).
  - Kaggle Auto-Download: If target directory has no images, automatically attempts to
    download `jianwenzhao/ink-wash` (trainB subset) via kagglehub into the folder.
  - Trigger phrase is appended AFTER the model caption so the natural description comes
    first and the style anchor comes last.
  - Existing .txt files are skipped by default; use --overwrite to regenerate.
  - Captions are saved as UTF-8 text with no trailing newline.
  - --refine mode reads the current .txt for each image and passes it as a
    reference hint into the model prompt so the model can improve or extend it.
    Images with no .txt are captioned from scratch. Combine with --overwrite
    to replace the old captions with the refined ones.
"""

import argparse
import os

import ssl
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    del os.environ["SSL_CERT_FILE"]

import sys
import time
from pathlib import Path

# Force UTF-8 encoding for stdout on Windows to support emojis and box-drawing characters
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

INK_WASH_TRIGGER = "Chinese ink wash painting style, Sumi-e"

MODEL_CONFIGS = {
    "florence-2": {
        "model_id": "microsoft/Florence-2-large",
        "backend": "florence-2",
        "description": "Florence-2-large — ~3 GB VRAM, excellent detail",
    },
    "blip2": {
        "model_id": "Salesforce/blip2-opt-2.7b",
        "backend": "blip2",
        "description": "BLIP-2 OPT-2.7B — ~5 GB VRAM float16, fast, supports text hints",
    },
    "joycaption": {
        "model_id": "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        "backend": "joycaption",
        "description": "JoyCaption Alpha Two — richest descriptions, ~12 GB VRAM",
    },
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def scan_images(folder: Path) -> list[Path]:
    """Return a sorted list of supported image paths in *folder*."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def print_section(title: str) -> None:
    width = 60
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------


def load_florence2(model_id: str, device: str):
    """Load Florence-2 processor + model pair."""
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedTokenizerBase

    # CRITICAL: Monkey-patch a known bug in transformers 4.43+ where Florence-2 crashes
    # because it looks for `forced_bos_token_id` before it is initialized in PretrainedConfig.
    if not hasattr(transformers.PretrainedConfig, "forced_bos_token_id"):
        transformers.PretrainedConfig.forced_bos_token_id = None

    # CRITICAL: Fix for Florence-2 + newer transformers: RobertaTokenizer.__getattr__ raises
    # AttributeError for `additional_special_tokens` during processor __init__.
    # We wrap __getattr__ on the base class so any missing attribute named
    # `additional_special_tokens` returns [] instead of crashing.
    _orig_getattr = PreTrainedTokenizerBase.__getattr__
    def _patched_getattr(self, name):
        if name == "additional_special_tokens":
            return []
        return _orig_getattr(self, name)
    PreTrainedTokenizerBase.__getattr__ = _patched_getattr

    print(f"  Loading Florence-2 model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",  # Florence-2 custom class lacks _supports_sdpa
    ).to(device)
    # Re-tie weights that newer transformers no longer auto-ties (fixes garbage captions).
    # Florence-2 uses a BART LM backbone: encoder/decoder embed_tokens and lm_head all
    # share the same tensor as language_model.model.shared, but only shared is in the
    # checkpoint — the others must be bound manually.
    shared = model.language_model.model.shared
    model.language_model.model.encoder.embed_tokens = shared
    model.language_model.model.decoder.embed_tokens = shared
    model.language_model.lm_head.weight = shared.weight
    model.eval()
    return processor, model


def load_joycaption(model_id: str, device: str):
    """Load a JoyCaption / LLaMA-based vision model.

    Both processor and model are loaded from the same official repo (*model_id*,
    i.e. fancyfeast/llama-joycaption-alpha-two-hf-llava) with on-the-fly NF4
    quantization via BitsAndBytesConfig.

    Why NOT load from the pre-quantized John6666 NF4 repo
    -------------------------------------------------------
    The pre-quantized repo was saved with an older bitsandbytes version.
    The current bitsandbytes registers a global __torch_function__ hook that
    incorrectly intercepts the SigLIP vision-encoder's
    MultiheadAttentionPoolingHead out_proj matmul, producing:
        RuntimeError: mat1 and mat2 shapes cannot be
        multiplied (1x1152 and 1x331776)
    On-the-fly quantization with the *current* bitsandbytes avoids this
    incompatibility entirely and uses roughly the same VRAM (~5 GB).
    """
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    print(f"  Loading JoyCaption processor from: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"  Loading JoyCaption model with on-the-fly NF4 quantization: {model_id}")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    return processor, model


def load_blip2(model_id: str, device: str):
    """Load BLIP-2 processor + model (Salesforce/blip2-opt-2.7b).

    BLIP-2 is a lightweight, fast captioner that also accepts free-form text
    prompts (VQA-style).  This makes it the only model other than JoyCaption
    that can use the --refine reference hint.  The 2.7B OPT backbone fits
    comfortably in ~5 GB VRAM when loaded as float16.
    """
    import torch
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    print(f"  Loading BLIP-2 processor from: {model_id}")
    processor = Blip2Processor.from_pretrained(model_id)

    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  Loading BLIP-2 model ({dtype}): {model_id}")
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Caption generators
# ---------------------------------------------------------------------------


def _pad_to_square(image):
    """Pad a PIL image to a square with white background (Florence-2 DaViT requires square input)."""
    from PIL import Image as PILImage
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = PILImage.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def _caption_florence2(processor, model, image, device: str, max_new_tokens: int = 1024,
                        reference_text: str = "") -> str:
    """Generate a detailed caption with Florence-2.

    NOTE: Florence-2 strictly requires the task token (e.g. <MORE_DETAILED_CAPTION>)
    to be the *only* token in the text prompt — adding any extra text causes:
        ValueError: Task token <MORE_DETAILED_CAPTION> should be the only token in the text.
    For this reason, *reference_text* is intentionally ignored here.  The refine
    mode still works (images are re-captioned with --overwrite), but the prior
    caption cannot be fed back as a hint the way it is for JoyCaption.
    If you need reference-aware refinement, use --model joycaption --refine.
    """
    import torch

    # Florence-2's DaViT vision encoder only supports square feature maps
    image = _pad_to_square(image)

    dtype = torch.float16 if device == "cuda" else torch.float32
    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=3,
            use_cache=False,  # avoids EncoderDecoderCache subscript error in newer transformers
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Post-processing key must match the base task token
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(image.width, image.height)
    )
    return parsed_answer[prompt].strip()


def _caption_joycaption(processor, model, image, device: str, max_new_tokens: int = 150,
                         reference_text: str = "") -> str:
    """
    JoyCaption uses a chat-style prompt. We ask for a descriptive, natural
    caption suitable for image generation training.
    When *reference_text* is provided the existing caption is injected into the
    system prompt so the model can refine and extend it.
    """
    import torch

    if reference_text:
        system_prompt = (
            "You are a helpful image captioner for AI training data. "
            "Write a single fluent English sentence describing the image content, "
            "artistic style, mood, and composition. "
            "Be specific and descriptive. Do not use bullet points or lists. "
            f"An existing caption is provided as a reference — you may improve, "
            f"correct, or expand it: [{reference_text.strip()}]"
        )
        user_prompt = (
            "Based on the image and the reference caption above, write an improved, "
            "more detailed training caption. The image is drawn in Chinese ink wash painting style."
        )
    else:
        system_prompt = (
            "You are a helpful image captioner for AI training data. "
            "Write a single fluent English sentence describing the image content, "
            "artistic style, mood, and composition. "
            "Be specific and descriptive. Do not use bullet points or lists."
        )
        user_prompt = (
            "Describe this image in detail for use as a generative AI training caption. "
            "The image drawn by chinese ink wash painting style."
        )

    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Strip the prompt tokens from the generated output
    generated = ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Core processing logic
# ---------------------------------------------------------------------------

def _caption_blip2(processor, model, image, device: str, max_new_tokens: int = 150,
                    reference_text: str = "") -> str:
    """Generate a caption with BLIP-2.

    BLIP-2 accepts optional free-form text prompts, so refine mode is fully
    supported.  When *reference_text* is provided the existing caption is
    embedded as a VQA-style question so the model can improve it.
    """
    import torch

    dtype = torch.float16 if device == "cuda" else torch.float32

    if reference_text:
        # VQA-style prompt: feed the old caption as context
        prompt = (
            f"Question: This image was previously described as: \"{reference_text.strip()}\". "
            "Provide a more detailed and accurate description of this Chinese ink wash painting. "
            "Answer:"
        )
    else:
        prompt = (
            "Question: Describe this image in detail for use as a generative AI training caption. "
            "The image is a Chinese ink wash painting. Answer:"
        )

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=5,
            repetition_penalty=1.5,
        )

    # Decode only the newly generated tokens (strip the prompt)
    generated = generated_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


CAPTION_DISPATCH = {
    "florence-2": _caption_florence2,
    "blip2": _caption_blip2,
    "joycaption": _caption_joycaption,
}


def caption_folder(
    folder: Path,
    processor,
    model,
    backend: str,
    device: str,
    trigger: str,
    overwrite: bool,
    max_new_tokens: int,
    dry_run: bool,
    refine: bool = False,
) -> dict:
    """
    Caption every image in *folder*.

    When *refine* is True the existing .txt sidecar (if present) is read and
    passed to the caption function as a reference so the model can improve it.
    Images with no sidecar are captioned normally.

    Returns a summary dict: {total, captioned, skipped, errors}.
    """
    images = scan_images(folder)
    if not images:
        print(f"  ⚠️  No images found in {folder}")
        return {"total": 0, "captioned": 0, "skipped": 0, "errors": 0}

    caption_fn = CAPTION_DISPATCH[backend]
    stats = {"total": len(images), "captioned": 0, "skipped": 0, "errors": 0}

    from PIL import Image as PILImage

    for idx, img_path in enumerate(images, 1):
        txt_path = img_path.with_suffix(".txt")

        # Skip if caption already exists and --overwrite not set
        # Exception: in refine mode we always (re-)process when --overwrite is set.
        if txt_path.exists() and txt_path.stat().st_size > 0 and not overwrite:
            stats["skipped"] += 1
            continue

        prefix = f"  [{idx:>4}/{len(images)}]"

        # ── Load existing caption as reference (refine mode) ──────
        reference_text = ""
        if refine and txt_path.exists() and txt_path.stat().st_size > 0:
            try:
                reference_text = txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                reference_text = ""

        if dry_run:
            mode_tag = "[REFINE]" if (refine and reference_text) else "[CAPTION]"
            print(f"{prefix} [DRY-RUN] {mode_tag} Would caption → {txt_path.name}")
            if reference_text:
                print(f"          Ref: {reference_text[:100]}{'…' if len(reference_text) > 100 else ''}")
            stats["captioned"] += 1
            continue

        try:
            t0 = time.perf_counter()
            image = PILImage.open(img_path).convert("RGB")
            caption = caption_fn(processor, model, image, device, max_new_tokens,
                                 reference_text=reference_text)

            if not caption:
                print(f"{prefix} ⚠️  Empty caption from model, skipping {img_path.name}")
                stats["skipped"] += 1
                continue

            # Append trigger phrase if provided
            if trigger:
                caption = f"{caption}, {trigger}"

            txt_path.write_text(caption, encoding="utf-8")
            elapsed = time.perf_counter() - t0

            mode_tag = "[REFINE]" if (refine and reference_text) else "[CAPTION]"
            print(f"{prefix} {mode_tag} {img_path.name}  ({elapsed:.1f}s)")
            if refine and reference_text:
                print(f"          Ref:     {reference_text[:100]}{'…' if len(reference_text) > 100 else ''}")
            print(f"          Caption: {caption[:110]}{'…' if len(caption) > 110 else ''}")
            stats["captioned"] += 1

        except Exception as exc:
            print(f"{prefix} ❌ Error on {img_path.name}: {exc}")
            stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated image captioning for PixArt-Sigma LoRA datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Target
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("./data/dataset_300"),
        metavar="PATH",
        help="Path to an image folder (default: ./data/dataset_300).",
    )
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help=(
            "If --dir points to a parent folder (e.g. ./data), "
            "caption all dataset_* subfolders inside it."
        ),
    )

    # Model
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="florence-2",
        help="Vision-language model to use for captioning (default: florence-2).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=75,
        metavar="N",
        help="Maximum new tokens to generate per caption (default: 75).",
    )

    # Trigger phrase
    parser.add_argument(
        "--trigger",
        type=str,
        default=INK_WASH_TRIGGER,
        metavar="TEXT",
        help=(
            f"Trigger phrase appended after the caption. "
            f"Default: '{INK_WASH_TRIGGER}'. "
            "Set to empty string '' to disable."
        ),
    )

    # Behaviour flags
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate captions even when a .txt file already exists.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help=(
            "Refine mode: load the existing .txt caption for each image and inject it "
            "into the model prompt as a reference so the model can improve or extend it. "
            "Images without a .txt are captioned from scratch. "
            "Combine with --overwrite to replace the old captions."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Force a specific device (default: cuda).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Resolve device ────────────────────────────────────────────
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print_section("🖌️  Auto-Caption — PixArt-Sigma LoRA Dataset")
    print(f"  Device        : {device}")
    if device == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    cfg = MODEL_CONFIGS[args.model]
    print(f"  Model         : {args.model}  —  {cfg['description']}")
    print(f"  Trigger phrase: '{args.trigger}'" if args.trigger else "  Trigger phrase: (none)")
    print(f"  Refine mode   : {args.refine}")
    print(f"  Overwrite     : {args.overwrite}")
    print(f"  Dry-run       : {args.dry_run}")

    # ── Collect target folders ────────────────────────────────────
    root = args.dir.resolve()

    if not args.all_subsets:
        has_images = False
        if root.exists() and root.is_dir():
            has_images = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in root.iterdir() if p.is_file())
            
        if not has_images:
            print(f"\n  Images not found in {root}. Attempting to download from Kaggle...")
            try:
                import kagglehub
                import shutil

                print("  Downloading 'jianwenzhao/ink-wash' dataset...")
                download_path = kagglehub.dataset_download("jianwenzhao/ink-wash")
                trainB_path = Path(download_path) / "trainB"

                if trainB_path.exists():
                    root.mkdir(parents=True, exist_ok=True)
                    copied_count = 0
                    for file_path in trainB_path.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                            shutil.copy2(file_path, root / file_path.name)
                            copied_count += 1
                    print(f"  ✅ Successfully copied {copied_count} images to {root}")
                else:
                    print(f"  ⚠️  'trainB' directory not found in {download_path}")

            except ImportError:
                print("  ⚠️  'kagglehub' not installed. Please install it to auto-download datasets.")
                print("      pip install kagglehub")
            except Exception as e:
                print(f"  ⚠️  Error downloading dataset: {e}")

    if args.all_subsets:
        if not root.is_dir():
            print(f"\n❌  Directory not found: {root}")
            sys.exit(1)
        target_folders = sorted(
            d for d in root.iterdir()
            if d.is_dir() and d.name.startswith("dataset_")
        )
        if not target_folders:
            print(f"\n❌  No 'dataset_*' subfolders found under {root}")
            sys.exit(1)
    else:
        if not root.is_dir():
            print(f"\n❌  Directory not found: {root}")
            sys.exit(1)
        target_folders = [root]

    print(f"\n  Target folders ({len(target_folders)}):")
    for f in target_folders:
        imgs = scan_images(f)
        print(f"    {f}  ({len(imgs)} images)")

    if args.dry_run:
        print("\n  ⚡ DRY-RUN mode — no files will be written.")

    # ── Load model (once, shared across all folders) ──────────────
    if not args.dry_run:
        print_section("Loading model weights")
        backend = cfg["backend"]
        if backend == "florence-2":
            processor, model = load_florence2(cfg["model_id"], device)
        elif backend == "blip2":
            processor, model = load_blip2(cfg["model_id"], device)
        elif backend == "joycaption":
            processor, model = load_joycaption(cfg["model_id"], device)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        print("  ✅ Model ready.")
    else:
        processor = model = None
        backend = cfg["backend"]

    # ── Process each folder ───────────────────────────────────────
    grand_total = {"total": 0, "captioned": 0, "skipped": 0, "errors": 0}
    wall_start  = time.perf_counter()

    for folder in target_folders:
        print_section(f"Captioning: {folder.name}")
        stats = caption_folder(
            folder         = folder,
            processor      = processor,
            model          = model,
            backend        = backend,
            device         = device,
            trigger        = args.trigger,
            overwrite      = args.overwrite,
            max_new_tokens = args.max_tokens,
            dry_run        = args.dry_run,
            refine         = args.refine,
        )
        for k in grand_total:
            grand_total[k] += stats[k]
        print(
            f"\n  Folder summary: {stats['captioned']} captioned, "
            f"{stats['skipped']} skipped, {stats['errors']} errors  "
            f"(out of {stats['total']})"
        )

    # ── Free GPU memory ───────────────────────────────────────────
    if not args.dry_run and device == "cuda":
        del model
        torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start
    print_section("✅  Batch captioning complete!")
    print(f"  Total images  : {grand_total['total']}")
    print(f"  Captioned     : {grand_total['captioned']}")
    print(f"  Skipped       : {grand_total['skipped']}")
    print(f"  Errors        : {grand_total['errors']}")
    print(f"  Elapsed       : {elapsed:.1f}s")
    if grand_total["captioned"] > 0:
        avg = elapsed / grand_total["captioned"]
        print(f"  Avg per image : {avg:.1f}s")
    print()

    if grand_total["errors"] > 0:
        sys.exit(1)  # Non-zero exit so CI pipelines can catch failures


if __name__ == "__main__":
    main()
