#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-part outputs produced by vf_curl_grad_variance.py (step-sharded runs).

Expected part files:
  vf_curl_grad_variance__<run_name>__<tag>__part<i>.json

Output:
  vf_curl_grad_variance__<run_name>__<tag>.json
  vf_curl_grad_variance__<run_name>__<tag>__*.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--run_name", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--num_parts", type=int, required=True)
    ap.add_argument("--delete_parts", action="store_true", default=False)
    ap.add_argument("--plot_baseline_full_only", action="store_true", default=False,
                    help="Only plot the vs-baseline-full curve (solid line) when baseline is available.")
    return ap.parse_args()


def _merge_payloads(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    title = parts[0].get("title", "")
    config = parts[0].get("config", {})

    row_by_step: Dict[int, Dict[str, Any]] = {}
    base_by_step: Dict[int, Dict[str, Any]] = {}
    have_baseline = False

    for payload in parts:
        steps = payload.get("steps", [])
        rows = payload.get("rows", [])
        if len(steps) != len(rows):
            raise RuntimeError("Invalid part payload: len(steps) != len(rows)")
        for s, r in zip(steps, rows):
            row_by_step[int(s)] = r

        base_rows = payload.get("baseline_rows", None)
        if base_rows is not None:
            have_baseline = True
            if len(base_rows) != len(steps):
                raise RuntimeError("Invalid part payload: len(baseline_rows) != len(steps)")
            for s, br in zip(steps, base_rows):
                base_by_step[int(s)] = br

    merged_steps = sorted(row_by_step.keys())
    merged_rows = [row_by_step[s] for s in merged_steps]

    merged_baseline: Optional[List[Dict[str, Any]]] = None
    if have_baseline:
        merged_baseline = [base_by_step.get(s, row_by_step[s]) for s in merged_steps]

    return {
        "analysis_id": f"merge:{parts[0].get('tag') or ''}",
        "tag": parts[0].get("tag"),
        "num_parts": int(parts[0].get("num_parts", len(parts))),
        "config": config,
        "title": title,
        "steps": merged_steps,
        "rows": merged_rows,
        "baseline_rows": merged_baseline,
        "merged_from": [p.get("_src", "") for p in parts],
    }


def main() -> None:
    args = parse_args()
    if args.num_parts < 1:
        raise ValueError("--num_parts must be >= 1")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    part_paths: List[Path] = []
    parts: List[Dict[str, Any]] = []
    for i in range(int(args.num_parts)):
        p = out_dir / f"vf_curl_grad_variance__{args.run_name}__{args.tag}__part{i}.json"
        part_paths.append(p)
        if not p.exists():
            raise FileNotFoundError(f"Missing part file: {p}")
        payload = _read_json(p)
        payload["_src"] = str(p)
        parts.append(payload)

    merged = _merge_payloads(parts)

    merged_json = out_dir / f"vf_curl_grad_variance__{args.run_name}__{args.tag}.json"
    merged_base = out_dir / f"vf_curl_grad_variance__{args.run_name}__{args.tag}"
    merged_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    # Plot via the same helper to keep style consistent.
    eval_root = Path(__file__).resolve().parents[1]
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    from tools.vi_curl_plot.vf_curl_grad_variance import plot_results  # type: ignore

    saved = plot_results(
        steps=merged["steps"],
        rows=merged["rows"],
        baseline_rows=merged.get("baseline_rows"),
        out_path=merged_base,
        title=merged.get("title", ""),
        baseline_full_only=bool(args.plot_baseline_full_only),
    )

    print(f"[OK] merged json: {merged_json}")
    for p in saved:
        print(f"[OK] merged plot: {p}")

    # If pass@k fields are present, emit passk-only json/png using the same rows.
    rows = merged.get("rows", [])
    has_passk = any(isinstance(r, dict) and "pass_at_k_kept" in r for r in rows)
    if has_passk:
        passk_json = out_dir / f"vi_curl_passk__{args.run_name}__{args.tag}.json"
        passk_base = out_dir / f"vi_curl_passk__{args.run_name}__{args.tag}"
        passk_payload = {
            "run_name": args.run_name,
            "tag": args.tag,
            "num_parts": int(args.num_parts),
            "config": merged.get("config", {}),
            "steps": merged.get("steps", []),
            "rows": rows,
        }
        passk_json.write_text(json.dumps(passk_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        ks_raw = passk_payload.get("config", {}).get("passk_ks", "")
        ks: List[int] = []
        if isinstance(ks_raw, str):
            for part in ks_raw.split(","):
                part = part.strip()
                if part.isdigit():
                    v = int(part)
                    if v > 0:
                        ks.append(v)
        elif isinstance(ks_raw, (list, tuple)):
            ks = [int(k) for k in ks_raw if int(k) > 0]
        if not ks and rows:
            sample = rows[0].get("pass_at_k_kept", {})
            if isinstance(sample, dict):
                for k in sample.keys():
                    if str(k).isdigit():
                        ks.append(int(k))
        ks = sorted(set(ks)) or [1]

        from tools.vi_curl_plot.vi_curl_passk_kept_dropped import _plot_passk  # type: ignore

        saved = _plot_passk(
            steps=passk_payload["steps"],
            rows=rows,
            ks=ks,
            out_path=passk_base,
            title=f"{args.run_name} kept vs dropped pass@k",
        )
        print(f"[OK] passk json: {passk_json}")
        for p in saved:
            print(f"[OK] passk plot: {p}")

    if args.delete_parts:
        for p in part_paths:
            try:
                p.unlink()
            except Exception:
                pass
            base = p.with_suffix("")
            for pdf in base.parent.glob(f"{base.name}__*.pdf"):
                try:
                    pdf.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
