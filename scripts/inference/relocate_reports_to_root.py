#!/usr/bin/env python3
"""Relocate all Markdown report files and image assets to the root 'reports/' directory with clean relative image paths for GitHub upload."""

import os
import re
import shutil
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parents[2]
    src_reports_dir = root_dir / "outputs" / "reports"
    dst_reports_dir = root_dir / "reports"
    dst_images_dir = dst_reports_dir / "images"

    dst_reports_dir.mkdir(parents=True, exist_ok=True)
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Relocating reports & images to root folder: {dst_reports_dir}...")

    # 1. Copy all PNG files from outputs/reports/images/ to reports/images/
    src_images_dir = src_reports_dir / "images"
    if src_images_dir.exists():
        png_files = list(src_images_dir.glob("*.png"))
        for png in png_files:
            shutil.copy(png, dst_images_dir / png.name)
        print(f"Copied {len(png_files)} PNG image assets to reports/images/")

    # Also copy any images directly in outputs/reports/
    root_pngs = list(src_reports_dir.glob("*.png"))
    for png in root_pngs:
        shutil.copy(png, dst_images_dir / png.name)

    # 2. Copy all MD files from outputs/reports/ to reports/ and update relative paths
    md_files = list(src_reports_dir.glob("*.md"))
    for md in md_files:
        content = md.read_text(encoding="utf-8")

        # Replace any absolute file:/// URIs or outputs/ URIs to clean ./images/<filename>.png
        pattern = r"file:///[A-Za-z]:/[^)]+/([^/\s]+\.png)"
        updated_content = re.sub(pattern, r"./images/\1", content)

        # Replace outputs/reports/images/ with ./images/
        updated_content = re.sub(r"\./outputs/reports/images/", r"./images/", updated_content)
        updated_content = re.sub(r"outputs/reports/images/", r"./images/", updated_content)

        dst_md = dst_reports_dir / md.name
        dst_md.write_text(updated_content, encoding="utf-8")
        print(f"Relocated GitHub-ready report: {dst_md.name}")

    print("\nAll reports and images successfully relocated to root 'reports/' folder!")


if __name__ == "__main__":
    main()
