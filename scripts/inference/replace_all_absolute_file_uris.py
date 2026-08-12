#!/usr/bin/env python3
"""Convert all absolute file:/// URIs across workspace Markdown reports and Jupyter Notebooks into local relative paths."""

import os
import re
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parents[2]

    # Search all .md files in workspace
    md_files = list(root_dir.glob("**/*.md"))
    ipynb_files = list(root_dir.glob("**/*.ipynb"))

    print(f"Scanning {len(md_files)} Markdown files and {len(ipynb_files)} Jupyter Notebooks...")

    # Pattern for gemini artifact dir file:/// URIs
    gemini_pattern = r"file:///[A-Za-z]:/[^)]*?\.gemini/antigravity-cli/brain/[a-f0-9\-]+/([^/\s\)\"\']+\.png)"
    # Pattern for dev repo file:/// URIs
    dev_pattern = r"file:///[A-Za-z]:/dev/efficient-pixart-sigma-lora/([^\s\)\"\']+)"

    updated_md_count = 0
    updated_nb_count = 0

    # 1. Update Markdown files
    for md in md_files:
        if ".conda" in str(md) or ".git" in str(md):
            continue
        try:
            content = md.read_text(encoding="utf-8")
            orig = content

            # If inside outputs/reports/, images are in ./images/
            if "outputs\\reports" in str(md) or "outputs/reports" in str(md):
                content = re.sub(gemini_pattern, r"./images/\1", content)
                content = re.sub(dev_pattern, r"../../\1", content)
            else:
                content = re.sub(gemini_pattern, r"./outputs/reports/images/\1", content)
                content = re.sub(dev_pattern, r"./\1", content)

            if content != orig:
                md.write_text(content, encoding="utf-8")
                updated_md_count += 1
                print(f"Updated local relative paths in: {md.relative_to(root_dir)}")
        except Exception as e:
            print(f"Error processing {md}: {e}")

    # 2. Update Jupyter Notebook files
    for nb in ipynb_files:
        if ".conda" in str(nb) or ".ipynb_checkpoints" in str(nb):
            continue
        try:
            content = nb.read_text(encoding="utf-8")
            orig = content

            # Relative path from notebook folder (notebooks/training/ or notebooks/evaluation/) to outputs/reports/images/
            content = re.sub(gemini_pattern, r"../../outputs/reports/images/\1", content)
            content = re.sub(dev_pattern, r"../../\1", content)

            if content != orig:
                nb.write_text(content, encoding="utf-8")
                updated_nb_count += 1
                print(f"Updated local relative paths in notebook: {nb.relative_to(root_dir)}")
        except Exception as e:
            print(f"Error processing notebook {nb}: {e}")

    print(f"\nCompleted! Updated {updated_md_count} Markdown files and {updated_nb_count} Notebooks to local relative paths.")


if __name__ == "__main__":
    main()
