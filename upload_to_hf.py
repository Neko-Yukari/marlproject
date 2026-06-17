#!/usr/bin/env python3
"""
Upload the MARL edge-computing project to a Hugging Face Hub repository.

Usage:
    python upload_to_hf.py --token YOUR_HF_TOKEN --repo_id yourname/marlproject

This script uploads the source code (excluding results/, *.pt, *.pptx, etc. as
defined in .gitignore) to Hugging Face Hub.  Large checkpoints and generated
plots are NOT uploaded by default.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def load_gitignore(root: Path):
    gitignore = root / ".gitignore"
    patterns = []
    if gitignore.exists():
        for line in gitignore.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns


def is_ignored(rel_path: Path, patterns):
    """Very basic gitignore matching; sufficient for this script."""
    s = str(rel_path).replace("\\", "/")
    parts = s.split("/")
    for pat in patterns:
        # Negation
        neg = pat.startswith("!")
        if neg:
            pat = pat[1:]
        # Directory pattern
        dir_pat = pat.endswith("/")
        if dir_pat:
            pat = pat[:-1]
        # Match exact or suffix
        if s == pat or s.endswith("/" + pat):
            return not neg
        # Match any path component (e.g., __pycache__)
        if pat in parts:
            return not neg
        # Glob-like *.ext
        if pat.startswith("*.") and any(p.endswith(pat[1:]) for p in parts):
            return not neg
    return False


def collect_files(root: Path):
    patterns = load_gitignore(root)
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        if is_ignored(rel, patterns):
            continue
        # Additional safety: skip hidden files
        if any(part.startswith(".") for part in rel.parts):
            continue
        files.append(rel)
    return files


def ensure_repo(repo_id: str, token: str, private: bool = False):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type="space" if private else "model")
    except Exception:
        api.create_repo(repo_id=repo_id, token=token, repo_type="model", private=private)
        print(f"Created new HF repo: {repo_id}")


def upload_folder(repo_id: str, token: str, folder: Path):
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Uploaded {folder} to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload MARL project to Hugging Face")
    parser.add_argument("--token", required=True, help="Hugging Face access token")
    parser.add_argument("--repo_id", required=True, help="Target repo, e.g. yourname/marlproject")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "pyyaml"])
        import huggingface_hub

    root = get_project_root()
    print(f"Project root: {root}")
    files = collect_files(root)
    print(f"Collected {len(files)} files to upload")
    for f in files[:20]:
        print("  -", f)
    if len(files) > 20:
        print(f"  ... and {len(files)-20} more")

    ensure_repo(args.repo_id, args.token, args.private)
    upload_folder(args.repo_id, args.token, root)


if __name__ == "__main__":
    main()
