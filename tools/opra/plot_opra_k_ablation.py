#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_rows(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "k": int(row["k"]),
                    "lora_rank": int(row["lora_rank"]) if row.get("lora_rank") not in (None, "", "None") else None,
                    "k_over_r": float(row["k_over_r"]) if row.get("k_over_r") not in (None, "", "None") else None,
                    "run": row.get("run", ""),
                    "step": int(row.get("step", 0)),
                    "performance": float(row["performance"]),
                    "knowledge": float(row["knowledge"]),
                })
        return rows
    data = json.loads(path.read_text())
    return data.get("rows", [])


def _plot_k_curve(rows: List[Dict], key: str, title: str, out_path: Path) -> None:
    def group_label(val: int | None) -> str:
        return f"r={val}" if val is not None else "r=unknown"

    groups: Dict[int | None, List[Dict]] = {}
    for row in rows:
        groups.setdefault(row.get("lora_rank"), []).append(row)

    plt.figure(figsize=(7.2, 4.6))
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.cm.tab10.colors
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for idx, (lora_rank, group_rows) in enumerate(sorted(groups.items(), key=lambda x: (x[0] is None, x[0] or 0))):
        group_rows = sorted(group_rows, key=lambda r: r["k"])
        xs = [r["k"] for r in group_rows]
        ys = [r[key] for r in group_rows]
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]
        plt.plot(
            xs,
            ys,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            color=color,
            label=group_label(lora_rank),
        )
    plt.title(title)
    plt.xlabel("Principal rank k")
    plt.ylabel(key.replace("_", " ").title())
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def _plot_tradeoff(rows: List[Dict], title: str, out_path: Path) -> None:
    def group_label(val: int | None) -> str:
        return f"r={val}" if val is not None else "r=unknown"

    groups: Dict[int | None, List[Dict]] = {}
    for row in rows:
        groups.setdefault(row.get("lora_rank"), []).append(row)

    plt.figure(figsize=(7.0, 4.6))
    plt.style.use("seaborn-v0_8-whitegrid")
    color_cycle = plt.cm.tab10.colors
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for idx, (lora_rank, group_rows) in enumerate(sorted(groups.items(), key=lambda x: (x[0] is None, x[0] or 0))):
        color = color_cycle[idx % len(color_cycle)]
        marker = markers[idx % len(markers)]
        xs = [r["performance"] for r in group_rows]
        ys = [r["knowledge"] for r in group_rows]
        ks = [r["k"] for r in group_rows]
        plt.scatter(xs, ys, s=70, marker=marker, color=color, label=group_label(lora_rank), edgecolor="white", linewidth=0.6)
        for x, y, k in zip(xs, ys, ks):
            plt.text(x, y, f"k={k}", fontsize=8, ha="left", va="bottom", color=color)
    plt.title(title)
    plt.xlabel("Performance (Pass@1)")
    plt.ylabel("Knowledge (Acc)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--title", default="OPRA k Ablation")
    ap.add_argument("--format", default="png", choices=["png", "pdf"])
    args = ap.parse_args()

    rows = _load_rows(args.input)
    if not rows:
        raise SystemExit("[ERROR] No rows found in input")

    out_dir = args.out_dir
    fmt = args.format
    _plot_k_curve(rows, "performance", f"{args.title}: Performance", out_dir / f"k_vs_performance.{fmt}")
    _plot_k_curve(rows, "knowledge", f"{args.title}: Knowledge", out_dir / f"k_vs_knowledge.{fmt}")
    _plot_tradeoff(rows, f"{args.title}: Trade-off", out_dir / f"performance_vs_knowledge.{fmt}")


if __name__ == "__main__":
    main()
