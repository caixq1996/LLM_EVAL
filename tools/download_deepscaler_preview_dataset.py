#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, list_repo_files

DEFAULT_REPO_ID = "agentica-org/DeepScaleR-Preview-Dataset"


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "deepscaler"


def _write_jsonl(items: Iterable[object], out_path: Path, add_idx: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            if add_idx and isinstance(item, dict) and "idx" not in item:
                item = {"idx": idx, **item}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _select_repo_files(repo_id: str, filename: str) -> list[str]:
    if filename:
        return [filename]
    files = list_repo_files(repo_id, repo_type="dataset")
    jsonl_files = sorted([f for f in files if f.endswith(".jsonl")])
    if jsonl_files:
        return jsonl_files
    json_files = sorted([f for f in files if f.endswith(".json")])
    if not json_files:
        raise SystemExit(f"No .json or .jsonl files found in {repo_id}.")
    return json_files


def _resolve_output_name(
    source_name: str, output_name: str, split: str, single_source: bool
) -> str:
    if output_name:
        if not single_source:
            raise SystemExit("--output-name can only be used with a single source file.")
        return output_name
    if source_name.endswith(".jsonl"):
        return f"{split}.jsonl" if single_source else Path(source_name).name
    return f"{split}.jsonl" if single_source else Path(source_name).with_suffix(".jsonl").name


def _download_jsonl_file(
    repo_id: str,
    filename: str,
    out_path: Path,
    cache_dir: str | None,
    overwrite: bool,
) -> None:
    if out_path.exists() and not overwrite:
        print(f"[Skip] {out_path} exists. Use --overwrite to replace it.")
        return
    cache_path = hf_hub_download(
        repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir or None,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, out_path)
    print(f"[OK] Downloaded {filename} -> {out_path}")


def _download_json_file(
    repo_id: str,
    filename: str,
    out_dir: Path,
    output_name: str,
    cache_dir: str | None,
    overwrite: bool,
    add_idx: bool,
) -> None:
    cache_path = hf_hub_download(
        repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir or None,
    )
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        out_path = out_dir / output_name
        if out_path.exists() and not overwrite:
            print(f"[Skip] {out_path} exists. Use --overwrite to replace it.")
            return
        _write_jsonl(data, out_path, add_idx=add_idx)
        print(f"[OK] Converted {filename} -> {out_path}")
        return

    if isinstance(data, dict):
        for key, value in data.items():
            if not isinstance(value, list):
                continue
            out_path = out_dir / f"{key}.jsonl"
            if out_path.exists() and not overwrite:
                print(f"[Skip] {out_path} exists. Use --overwrite to replace it.")
                continue
            _write_jsonl(value, out_path, add_idx=add_idx)
            print(f"[OK] Converted {filename}:{key} -> {out_path}")
        return

    raise SystemExit(f"Unsupported JSON structure in {filename}.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download DeepScaleR Preview dataset and write JSONL files."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default="train", help="Used for single-file outputs.")
    parser.add_argument(
        "--output-dir",
        default=str(_default_output_dir()),
        help="Output directory for JSONL files.",
    )
    parser.add_argument(
        "--output-name",
        default="",
        help="Output filename for single-file outputs (e.g., train.jsonl).",
    )
    parser.add_argument(
        "--filename",
        default="",
        help="Specific repo filename to download (defaults to auto-detect).",
    )
    parser.add_argument("--cache-dir", default="", help="Optional HF cache dir.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-idx", action="store_true", help="Do not add idx field.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    add_idx = not args.no_idx
    source_files = _select_repo_files(args.repo_id, args.filename)
    single_source = len(source_files) == 1

    for source_file in source_files:
        output_name = _resolve_output_name(
            source_file, args.output_name, args.split, single_source
        )
        if source_file.endswith(".jsonl"):
            _download_jsonl_file(
                args.repo_id,
                source_file,
                out_dir / output_name,
                args.cache_dir,
                args.overwrite,
            )
        elif source_file.endswith(".json"):
            _download_json_file(
                args.repo_id,
                source_file,
                out_dir,
                output_name,
                args.cache_dir,
                args.overwrite,
                add_idx,
            )
        else:
            print(f"[Skip] Unsupported file type: {source_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
