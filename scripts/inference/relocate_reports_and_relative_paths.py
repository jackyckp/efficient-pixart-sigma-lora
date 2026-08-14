import argparse
import os
import re
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Relocate report markdown files and convert image paths to relative paths.")
    parser.add_argument("--source-dir", type=Path, default=None, help="Source directory containing artifacts.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory for reports.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    artifact_dir = args.source_dir if args.source_dir else (root_dir / "outputs" / "reports")
    reports_dir = args.output_dir if args.output_dir else (root_dir / "outputs" / "reports")
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
