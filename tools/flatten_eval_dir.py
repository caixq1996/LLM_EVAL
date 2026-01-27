#!/usr/bin/env python3
"""
Flatten eval result directory layout.

Moves run directories like:
  <base>/<wrapper>/global_step_XXX/<run_dir>
to:
  <base>/<run_dir>

Existing destination directories are merged (non-overwriting).
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def merge_dir(src: Path, dest: Path) -> None:
    """Move contents from src into dest, skipping existing files."""
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        dest_root = dest / rel
        dest_root.mkdir(parents=True, exist_ok=True)
        for fname in files:
            src_file = Path(root) / fname
            dest_file = dest_root / fname
            if dest_file.exists():
                continue
            shutil.move(str(src_file), str(dest_file))
    shutil.rmtree(src, ignore_errors=True)


def flatten(base_dir: Path) -> tuple[int, int]:
    moved = 0
    skipped = 0

    for wrapper in sorted(base_dir.iterdir()):
        if not wrapper.is_dir():
            continue
        if "_global_step_" not in wrapper.name:
            continue
        # Already flattened? Skip if g1/g2 present directly.
        if (wrapper / "g1").exists() or (wrapper / "g2").exists():
            continue

        step_dirs = [d for d in wrapper.iterdir() if d.is_dir() and d.name.startswith("global_step_")]
        if not step_dirs:
            continue

        for step_dir in step_dirs:
            for child in sorted(step_dir.iterdir()):
                if not child.is_dir():
                    continue
                dest = base_dir / child.name
                if dest.exists():
                    merge_dir(child, dest)
                else:
                    shutil.move(str(child), str(dest))
                moved += 1
            # Clean empty step_dir if possible
            try:
                step_dir.rmdir()
            except OSError:
                pass
        # Clean empty wrapper if possible
        try:
            wrapper.rmdir()
        except OSError:
            pass

    return moved, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir", type=Path, help="Base eval_results directory to flatten")
    args = ap.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise SystemExit(f"[ERROR] base_dir not found or not a directory: {base_dir}")

    moved, skipped = flatten(base_dir)
    print(f"[INFO] Done. Moved {moved} directories. Skipped {skipped}.")


if __name__ == "__main__":
    main()
