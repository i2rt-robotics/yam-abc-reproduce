#!/usr/bin/env python3
"""
Download robot asset files for sim_eval.

Assets are not committed to the repo (see .gitignore). Run this script once
before launching any sim_eval environment.

Usage:
    uv run python sim_eval/scripts/download_assets.py
"""

import argparse
import shutil
import sys
from pathlib import Path

HF_REPO_ID = "TreeePlanter/molmoact2-sim-eval-assets"

ASSETS_DIR = Path(__file__).parent.parent / "assets"


def download(repo_id: str, force: bool = False) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not found. Run: uv sync")

    if ASSETS_DIR.exists() and not force:
        print(f"Assets already present at {ASSETS_DIR}  (pass --force to re-download)")
        return

    if ASSETS_DIR.exists() and force:
        shutil.rmtree(ASSETS_DIR)

    print(f"Downloading assets from {repo_id} → {ASSETS_DIR} ...")
    local = snapshot_download(repo_id=repo_id, repo_type="dataset")
    src = Path(local) / "assets"
    if not src.exists():
        sys.exit(
            f"Downloaded snapshot does not contain an 'assets/' subdirectory.\n"
            f"Check that {repo_id} is the correct HuggingFace dataset repo."
        )
    shutil.copytree(src, ASSETS_DIR)
    print(f"Done. Assets installed at {ASSETS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help="HuggingFace dataset repo containing the assets/ directory "
             f"(default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-download even if assets/ already exists",
    )
    args = parser.parse_args()
    download(args.repo_id, args.force)
