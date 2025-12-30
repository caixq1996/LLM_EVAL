#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-part outputs from vi_curl_passk_kept_dropped.py.

Expected part files:
  vi_curl_passk__<run_name>__<tag>__part<i>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--num_parts", type=int, required=True)
    ap.add_argument("--delete_parts", action="store_true", default=False)
    return ap.parse_args()


def _merge(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    config = parts[0].get("config", {})
    row_by_step: Dict[int, Dict[str, Any]] = {}
    for payload in parts:
        steps = payload.get("steps", [])
        rows = payload.get("rows", [])
        if len(steps) != len(rows):
            raise RuntimeError("Invalid part payload: len(steps) != len(rows)")
        for s, r in zip(steps, rows):
            row_by_step[int(s)] = r
    steps_sorted = sorted(row_by_step.keys())
    return {
        "run_name": parts[0].get("run_name"),
        "tag": parts[0].get("tag"),
        "num_parts": int(parts[0].get("num_parts", len(parts))),
        "config": config,
        "steps": steps_sorted,
        "rows": [row_by_step[s] for s in steps_sorted],
    }


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    parts: List[Dict[str, Any]] = []
    part_paths: List[Path] = []
    for i in range(int(args.num_parts)):
        p = out_dir / f"vi_curl_passk__{args.run_name}__{args.tag}__part{i}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing part file: {p}")
        part_paths.append(p)
        parts.append(_read_json(p))

    merged = _merge(parts)
    merged_json = out_dir / f"vi_curl_passk__{args.run_name}__{args.tag}.json"
    merged_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    # Plot via helper in the compute script.
    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from tools.vi_curl_passk_kept_dropped import _plot_passk  # type: ignore

    cfg = merged.get("config", {})
    ks = cfg.get("target_ks") or []
    if not ks:
        ks = [1]
    title = f"{args.run_name} kept vs dropped pass@k"
    merged_png = out_dir / f"vi_curl_passk__{args.run_name}__{args.tag}.png"
    _plot_passk(steps=merged["steps"], rows=merged["rows"], ks=ks, out_path=merged_png, title=title)

    print(f"[OK] merged json: {merged_json}")
    print(f"[OK] merged plot: {merged_png}")

    if args.delete_parts:
        for p in part_paths:
            try:
                p.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    main()
