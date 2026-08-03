#!/usr/bin/env python3
"""Move all report markdown files and images from .gemini artifact directory to outputs/reports/ and convert all image paths to relative paths."""

import os
import re
import shutil
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parents[2]
    artifact_dir = Path(r"C:\Users\hklov\.gemini\antigravity-cli\brain\b70f3ed3-d14e-4144-af3b-784222775bc9")
    reports_dir = root_dir / "outputs" / "reports"
    images_dir = reports_dir / "images"

    reports_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Relocating reports & images from {artifact_dir} to {reports_dir}...")

    # Copy all PNG files to outputs/reports/images/
    png_files = list(artifact_dir.glob("*.png"))
    for png in png_files:
        shutil.copy(png, images_dir / png.name)
    print(f"Copied {len(png_files)} PNG image assets to outputs/reports/images/")

    # Copy all MD files to outputs/reports/ and update image paths to relative paths
    md_files = list(artifact_dir.glob("*.md"))
    for md in md_files:
        content = md.read_text(encoding="utf-8")

        # Regex replace any file:///.../filename.png or absolute path to ./images/filename.png
        # Handles both markdown images ![alt](path) and standard file:/// paths
        pattern = r"file:///[A-Za-z]:/[^)]+/([^/\s]+\.png)"
        updated_content = re.sub(pattern, r"./images/\1", content)

        # Also replace Windows backslash absolute paths if any exist
        win_pattern = r"[A-Za-z]:\\[^\n)]+\\([^\\\n)]+\.png)"
        updated_content = re.sub(win_pattern, r"./images/\1", updated_content)

        dst_md = reports_dir / md.name
        dst_md.write_text(updated_content, encoding="utf-8")
        print(f"Converted and saved relative-path report: {dst_md.name}")

    print("\nAll report markdown files and images successfully relocated with relative paths!")


if __name__ == "__main__":
    main()
