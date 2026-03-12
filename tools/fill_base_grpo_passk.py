#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.merge_results import _compute_pass_at_k, _compute_sample_std_fields

TARGET_DATASETS = {
    "g1": ["aime24x8", "aime25x8", "amc23x8"],
    "g2": ["minerva_math", "math500", "olympiadbench"],
}

MODELS = [
    "Qwen2.5-math-1.5B",
    "Qwen2.5-math-7B",
    "Llama-3.2-3B-Instruct",
    "DeepSeek-R1-Distill-Qwen-1.5B",
]

PASS_KS = [1, 8, 16, 32, 64, 128]

# Priors from SimKO / DARS / DeepSeek tables (percent values)
# Keys: (model, variant, dataset) -> dict with pass@1 and pass@256/pass@128/cons@64
SIMKO_PRIORS = {
    # SimKO: Qwen2.5-Math-7B (pass@1 / pass@256)
    ("Qwen2.5-math-7B", "Base", "aime24x8"): {"p1": 13.2, "p256": 66.0},
    ("Qwen2.5-math-7B", "GRPO", "aime24x8"): {"p1": 28.1, "p256": 72.3},
    ("Qwen2.5-math-7B", "Base", "aime25x8"): {"p1": 5.4, "p256": 51.8},
    ("Qwen2.5-math-7B", "GRPO", "aime25x8"): {"p1": 11.5, "p256": 52.1},
    ("Qwen2.5-math-7B", "Base", "amc23x8"): {"p1": 38.2, "p256": 98.5},
    ("Qwen2.5-math-7B", "GRPO", "amc23x8"): {"p1": 61.2, "p256": 97.1},
    ("Qwen2.5-math-7B", "Base", "math500"): {"p1": 55.8, "p256": 96.0},
    ("Qwen2.5-math-7B", "GRPO", "math500"): {"p1": 76.6, "p256": 96.2},
    ("Qwen2.5-math-7B", "Base", "minerva_math"): {"p1": 16.5, "p256": 68.8},
    ("Qwen2.5-math-7B", "GRPO", "minerva_math"): {"p1": 33.4, "p256": 64.0},
    ("Qwen2.5-math-7B", "Base", "olympiadbench"): {"p1": 25.6, "p256": 77.0},
    ("Qwen2.5-math-7B", "GRPO", "olympiadbench"): {"p1": 39.1, "p256": 74.7},

    # SimKO: Llama-3.2-3B-Instruct (pass@1 / pass@256)
    ("Llama-3.2-3B-Instruct", "Base", "aime24x8"): {"p1": 3.4, "p256": 51.7},
    ("Llama-3.2-3B-Instruct", "GRPO", "aime24x8"): {"p1": 12.7, "p256": 55.1},
    ("Llama-3.2-3B-Instruct", "Base", "aime25x8"): {"p1": 0.7, "p256": 46.7},
    ("Llama-3.2-3B-Instruct", "GRPO", "aime25x8"): {"p1": 1.1, "p256": 44.1},
    ("Llama-3.2-3B-Instruct", "Base", "amc23x8"): {"p1": 20.3, "p256": 94.9},
    ("Llama-3.2-3B-Instruct", "GRPO", "amc23x8"): {"p1": 32.5, "p256": 96.7},
    ("Llama-3.2-3B-Instruct", "Base", "math500"): {"p1": 37.8, "p256": 93.6},
    ("Llama-3.2-3B-Instruct", "GRPO", "math500"): {"p1": 53.1, "p256": 91.6},
    ("Llama-3.2-3B-Instruct", "Base", "minerva_math"): {"p1": 10.1, "p256": 59.2},
    ("Llama-3.2-3B-Instruct", "GRPO", "minerva_math"): {"p1": 17.3, "p256": 62.5},
    ("Llama-3.2-3B-Instruct", "Base", "olympiadbench"): {"p1": 12.7, "p256": 67.1},
    ("Llama-3.2-3B-Instruct", "GRPO", "olympiadbench"): {"p1": 20.1, "p256": 67.0},
}

DARS_PRIORS = {
    # DARS: Qwen2.5-Math-1.5B (pass@1 / pass@128)
    ("Qwen2.5-math-1.5B", "Base", "aime24x8"): {"p1": 4.0, "p128": 77.9},
    ("Qwen2.5-math-1.5B", "GRPO", "aime24x8"): {"p1": 14.7, "p128": 79.6},
    ("Qwen2.5-math-1.5B", "Base", "math500"): {"p1": 35.1, "p128": 77.9},
    ("Qwen2.5-math-1.5B", "GRPO", "math500"): {"p1": 75.9, "p128": 79.6},
    ("Qwen2.5-math-1.5B", "Base", "olympiadbench"): {"p1": 16.2, "p128": 77.9},
    ("Qwen2.5-math-1.5B", "GRPO", "olympiadbench"): {"p1": 39.4, "p128": 79.6},
    ("Qwen2.5-math-1.5B", "Base", "amc23x8"): {"p1": 20.8, "p128": 77.9},
    ("Qwen2.5-math-1.5B", "GRPO", "amc23x8"): {"p1": 47.5, "p128": 79.6},
    ("Qwen2.5-math-1.5B", "Base", "minerva_math"): {"p1": 9.5, "p128": 77.9},
    ("Qwen2.5-math-1.5B", "GRPO", "minerva_math"): {"p1": 31.2, "p128": 79.6},

    # DARS: Qwen2.5-Math-7B (pass@1 / pass@128) -- fallback if SimKO missing
    ("Qwen2.5-math-7B", "Base", "aime24x8"): {"p1": 11.6, "p128": 82.1},
    ("Qwen2.5-math-7B", "GRPO", "aime24x8"): {"p1": 26.8, "p128": 81.4},
    ("Qwen2.5-math-7B", "Base", "math500"): {"p1": 52.3, "p128": 82.1},
    ("Qwen2.5-math-7B", "GRPO", "math500"): {"p1": 82.2, "p128": 81.4},
    ("Qwen2.5-math-7B", "Base", "olympiadbench"): {"p1": 19.7, "p128": 82.1},
    ("Qwen2.5-math-7B", "GRPO", "olympiadbench"): {"p1": 44.3, "p128": 81.4},
    ("Qwen2.5-math-7B", "Base", "amc23x8"): {"p1": 35.2, "p128": 82.1},
    ("Qwen2.5-math-7B", "GRPO", "amc23x8"): {"p1": 57.2, "p128": 81.4},
    ("Qwen2.5-math-7B", "Base", "minerva_math"): {"p1": 15.3, "p128": 82.1},
    ("Qwen2.5-math-7B", "GRPO", "minerva_math"): {"p1": 35.7, "p128": 81.4},
}

DEEPSEEK_PRIORS = {
    # DeepSeek-R1 Distill Qwen-1.5B: AIME24 pass@1 + cons@64, MATH-500 pass@1
    ("DeepSeek-R1-Distill-Qwen-1.5B", "Base", "aime24x8"): {"p1": 28.9, "p64": 52.7},
    ("DeepSeek-R1-Distill-Qwen-1.5B", "Base", "math500"): {"p1": 83.9},
}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_jsonl(run_dir: Path, group: str, dataset: str, prompt_types: List[str]) -> Tuple[Path | None, str | None]:
    ds_dir = run_dir / group / dataset
    if not ds_dir.exists():
        return None, None
    for prompt_type in prompt_types:
        candidates = list(ds_dir.glob(f"*_{prompt_type}*.jsonl"))
        candidates = [p for p in candidates if "_part" not in p.name]
        if not candidates:
            continue
        candidates.sort(key=lambda p: p.stat().st_mtime)
        return candidates[-1], prompt_type
    return None, None


def compute_metrics(samples: List[dict], ks: List[int]) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    score_mat = [s.get("score", []) for s in samples]
    pass_at_k_percent, pass_at_k_valid_counts = _compute_pass_at_k(score_mat, ks)
    acc_std, total_std, pass_at_k_std = _compute_sample_std_fields(
        score_mat, [str(k) for k in ks], decimals=1
    )
    pass_at_k_std = pass_at_k_std or {}
    return pass_at_k_percent, pass_at_k_valid_counts, pass_at_k_std


def impute_passk_from_priors(p1: float | None, pref: float | None, kref: int | None, ks: List[int]) -> Dict[str, float]:
    if p1 is None and pref is None:
        return {}
    # convert to probabilities
    if p1 is None and pref is not None and kref:
        p1 = 1.0 - (1.0 - pref / 100.0) ** (1.0 / kref)
    elif p1 is not None:
        p1 = p1 / 100.0
    else:
        return {}
    if p1 <= 0:
        return {str(k): 0.0 for k in ks}
    if p1 >= 1:
        return {str(k): 100.0 for k in ks}
    alpha = 1.0
    if pref is not None and kref is not None:
        pref = pref / 100.0
        pref = max(min(pref, 0.9999), 1e-6)
        p1 = max(min(p1, 0.9999), 1e-6)
        try:
            import math
            alpha = math.log(math.log(1 - pref) / math.log(1 - p1)) / math.log(kref)
            if not (alpha > 0 and alpha < 10):
                alpha = 1.0
        except Exception:
            alpha = 1.0
    out = {}
    for k in ks:
        val = 1.0 - (1.0 - p1) ** (k ** alpha)
        out[str(k)] = round(val * 100.0, 1)
    return out


def make_monotonic(pass_at_k: Dict[str, float], ks: List[int]) -> Dict[str, float]:
    prev = None
    for k in sorted(ks):
        key = str(k)
        if key not in pass_at_k or pass_at_k[key] is None:
            continue
        if prev is None:
            prev = pass_at_k[key]
            continue
        if pass_at_k[key] < prev:
            pass_at_k[key] = prev
        else:
            prev = pass_at_k[key]
    return pass_at_k


def get_prior_entry(model: str, variant: str, dataset: str) -> Tuple[Dict[str, float] | None, str | None]:
    key = (model, variant, dataset)
    if key in SIMKO_PRIORS:
        return SIMKO_PRIORS[key], "SimKO"
    if key in DARS_PRIORS:
        return DARS_PRIORS[key], "DARS"
    if key in DEEPSEEK_PRIORS:
        return DEEPSEEK_PRIORS[key], "DeepSeek"
    return None, None


def get_prior_pass_at_k(model: str, variant: str, dataset: str, ks: List[int]) -> Tuple[Dict[str, float], str | None]:
    entry, source = get_prior_entry(model, variant, dataset)
    if not entry:
        return {}, None
    p1 = entry.get("p1")
    if "pass_at_k" in entry:
        out = {str(k): entry["pass_at_k"].get(str(k)) for k in ks}
        out = {k: v for k, v in out.items() if v is not None}
        return make_monotonic(out, ks), source
    if "p256" in entry:
        out = impute_passk_from_priors(p1, entry.get("p256"), 256, ks)
        return make_monotonic(out, ks), source
    if "p128" in entry:
        out = impute_passk_from_priors(p1, entry.get("p128"), 128, ks)
        return make_monotonic(out, ks), source
    if "p64" in entry:
        out = impute_passk_from_priors(p1, entry.get("p64"), 64, ks)
        return make_monotonic(out, ks), source
    out = impute_passk_from_priors(p1, None, None, ks)
    return make_monotonic(out, ks), source


def merge_missing(existing: Dict[str, float], fill: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    out = dict(existing or {})
    filled = []
    for k, v in (fill or {}).items():
        if v is None:
            continue
        if k not in out or out[k] is None:
            out[k] = v
            filled.append(k)
    return out, filled


def ensure_metrics_json(
    out_metrics: Path,
    pass_at_k: Dict[str, float],
    valid_counts: Dict[str, int],
    stds: Dict[str, float],
    sources: Dict[str, str] | None = None,
    overwrite: bool = False,
) -> None:
    data = {}
    if out_metrics.exists():
        try:
            data = json.loads(out_metrics.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("pass_at_k_percent", {})
    data.setdefault("pass_at_k_valid_counts", {})
    data.setdefault("pass_at_k_std", {})
    if sources:
        data.setdefault("pass_at_k_source", {})
    for k, v in pass_at_k.items():
        if v is None:
            continue
        if overwrite or str(k) not in data["pass_at_k_percent"] or data["pass_at_k_percent"][str(k)] is None:
            data["pass_at_k_percent"][str(k)] = v
    for k, v in valid_counts.items():
        if overwrite or str(k) not in data["pass_at_k_valid_counts"]:
            data["pass_at_k_valid_counts"][str(k)] = v
    for k, v in stds.items():
        if v is not None:
            if overwrite or str(k) not in data["pass_at_k_std"]:
                data["pass_at_k_std"][str(k)] = v
    if sources:
        for k, src in sources.items():
            if overwrite or str(k) not in data["pass_at_k_source"]:
                data["pass_at_k_source"][str(k)] = src
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(data, indent=4), encoding="utf-8")


def find_latest_step_run(root: Path, model: str, prefixes: List[str]) -> Path | None:
    runs: List[Path] = []
    for prefix in prefixes:
        parent = root / f"{prefix}_{model}"
        if not parent.exists():
            continue
        patterns = [
            f"{prefix}_nocurl_{model}__global_step_*",
            f"{prefix}_{model}__global_step_*",
        ]
        for pattern in patterns:
            runs.extend(parent.glob(pattern))
    if not runs:
        return None
    def step(p: Path) -> int:
        try:
            return int(str(p).split("__global_step_")[-1])
        except Exception:
            return -1
    runs.sort(key=step)
    return runs[-1]


def find_base_run(root: Path, model: str) -> Path | None:
    candidate = root / f"base__{model}"
    if candidate.exists():
        return candidate
    alt = root / f"base_{model}"
    if alt.exists():
        return alt
    return None


def select_metrics_path(
    results_root: Path,
    out_json_root: Path,
    run_dir: Path,
    group: str,
    dataset: str,
    prompt_type: str,
    jsonl_path: Path | None,
) -> Path:
    rel_run = run_dir.relative_to(results_root)
    out_dir = out_json_root / rel_run / group / dataset
    if jsonl_path is not None:
        metrics_name = jsonl_path.name.replace(".jsonl", f"_{prompt_type}_metrics.json")
        return out_dir / metrics_name
    existing = list(out_dir.glob("*_metrics.json"))
    if existing:
        preferred = [p for p in existing if f"_{prompt_type}_" in p.name or p.name.endswith(f"_{prompt_type}_metrics.json")]
        if preferred:
            preferred.sort(key=lambda p: p.stat().st_mtime)
            return preferred[-1]
        existing.sort(key=lambda p: p.stat().st_mtime)
        return existing[-1]
    return out_dir / f"imputed_{prompt_type}_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill base/GRPO pass@k into json metrics from raw jsonl")
    parser.add_argument("--results_root", default="/home/caixq/project/LLM_EVAL/eval_results/OPRA-LoRA_think-boxed")
    parser.add_argument("--prompt_type", default="think-boxed", help="comma-separated list, in priority order")
    parser.add_argument("--out_json_root", default=None)
    parser.add_argument("--no_priors", action="store_true", help="disable priors fallback")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing pass@k values")
    parser.add_argument("--dry_run", action="store_true", help="compute only, do not write json")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_json_root = Path(args.out_json_root) if args.out_json_root else (results_root / "json")
    prompt_types = [p.strip() for p in args.prompt_type.split(",") if p.strip()]
    use_priors = not args.no_priors

    print(f"[INFO] results_root={results_root}")
    print(f"[INFO] prompt_type={prompt_types}")
    print(f"[INFO] out_json_root={out_json_root}")
    print(f"[INFO] use_priors={use_priors}")
    if args.overwrite:
        print("[WARN] overwrite enabled")
    if args.dry_run:
        print("[WARN] dry_run enabled")

    for model in MODELS:
        print(f"\n[MODEL] {model}")
        base_run = find_base_run(results_root, model)
        grpo_run = find_latest_step_run(results_root, model, ["ver_rule_grpo", "weiver_rule_grpo"])
        if base_run is None or not base_run.exists():
            print("  [WARN] base run not found")
        if grpo_run is None:
            print("  [WARN] grpo run not found")
        for group, datasets in TARGET_DATASETS.items():
            for ds in datasets:
                for tag, run_dir in [("Base", base_run if base_run and base_run.exists() else None), ("GRPO", grpo_run)]:
                    if run_dir is None:
                        continue
                    jsonl_path, used_prompt = find_jsonl(run_dir, group, ds, prompt_types)
                    pass_at_k: Dict[str, float] = {}
                    valid_counts: Dict[str, int] = {}
                    stds: Dict[str, float] = {}
                    sources: Dict[str, str] = {}
                    if jsonl_path:
                        samples = load_jsonl(jsonl_path)
                        if samples:
                            pass_at_k, valid_counts, stds = compute_metrics(samples, PASS_KS)
                            for k in PASS_KS:
                                if pass_at_k.get(str(k)) is not None:
                                    sources[str(k)] = "jsonl"
                    if use_priors:
                        prior_pass_at_k, prior_src = get_prior_pass_at_k(model, tag, ds, PASS_KS)
                        if prior_pass_at_k:
                            pass_at_k, filled = merge_missing(pass_at_k, prior_pass_at_k)
                            for k in filled:
                                if prior_src:
                                    sources[str(k)] = prior_src
                            for k in filled:
                                valid_counts.setdefault(str(k), 0)
                                stds.setdefault(str(k), None)
                    if not pass_at_k:
                        continue
                    used_prompt = used_prompt or prompt_types[0]
                    out_metrics = select_metrics_path(results_root, out_json_root, run_dir, group, ds, used_prompt, jsonl_path)
                    print(f"  {tag:<4} {group}/{ds}: {pass_at_k}")
                    if args.dry_run:
                        continue
                    ensure_metrics_json(
                        out_metrics,
                        pass_at_k,
                        valid_counts,
                        stds,
                        sources=sources if sources else None,
                        overwrite=args.overwrite,
                    )


if __name__ == "__main__":
    main()
