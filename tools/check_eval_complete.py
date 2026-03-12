#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path


def _read_env_list(text: str) -> str:
    lines = text.splitlines()
    env_lines = []
    capturing = False
    for line in lines:
        if line.startswith("env_list:"):
            capturing = True
            env_lines.append(line.split("env_list:", 1)[1].strip())
            continue
        if capturing:
            if re.match(r"^[A-Za-z_]+:", line):
                break
            env_lines.append(line.strip())
    return " ".join(env_lines).strip()


def _parse_env_list(env_list: str) -> dict:
    if not env_list:
        return {}
    parts = re.split(r",(?=[A-Za-z_][A-Za-z0-9_]*=)", env_list)
    env = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _split_list(val: str) -> list:
    return [v.strip() for v in (val or "").split(",") if v.strip()]


def _adapter_meta(path: str):
    p = Path(path)
    if p.name == "actor" and p.parent.name.startswith("global_step_"):
        step_name = p.parent.name
        run_name = p.parent.parent.name
    else:
        step_name = p.name
        run_name = p.parent.name
    safe_run_name = run_name.replace(".", "_").replace("-", "_")
    run_tag = f"{run_name}__{step_name}"
    return safe_run_name, run_tag


def _has_pass128(metrics_path: Path) -> bool:
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    passk = data.get("pass_at_k_percent")
    if not isinstance(passk, dict):
        return False
    return "128" in {str(k) for k in passk.keys()}


def _find_jsonl(out_dir: Path) -> bool:
    if not out_dir.exists():
        return False
    files = sorted(out_dir.glob("*.jsonl"))
    if not files:
        return False
    non_part = [p for p in files if "_part" not in p.name]
    return bool(non_part or files)


def _find_metrics(out_dir: Path) -> Path | None:
    if not out_dir.exists():
        return None
    metrics = [p for p in out_dir.glob("*_metrics.json") if "_part" not in p.name]
    if metrics:
        return sorted(metrics)[0]
    return None


def check_complete(env_text: str) -> tuple[bool, str]:
    env_list = _read_env_list(env_text)
    env = _parse_env_list(env_list)
    adapters_raw = env.get("LORA_ADAPTERS", "")
    out_root = env.get("OUT_ROOT", "")
    if not adapters_raw or not out_root:
        return False, "missing LORA_ADAPTERS or OUT_ROOT"

    group1 = _split_list(env.get("EVAL_GROUP1_DATASETS", ""))
    group2 = _split_list(env.get("EVAL_GROUP2_DATASETS", ""))
    datasets = _split_list(env.get("EVAL_DATASETS", "")) or (group1 + group2)
    if not datasets:
        return False, "missing datasets"

    out_root_path = Path(out_root)
    adapters = [a.strip() for a in adapters_raw.split("|") if a.strip()]
    if not adapters:
        return False, "no adapters"

    g1 = set(group1)
    g2 = set(group2)

    for adapter in adapters:
        safe_run, run_tag = _adapter_meta(adapter)
        for ds in datasets:
            if ds in g1:
                g = 1
            elif ds in g2:
                g = 2
            else:
                g = 1
            out_dir = out_root_path / safe_run / run_tag / f"g{g}" / ds
            if not _find_jsonl(out_dir):
                return False, f"missing jsonl: {out_dir}"
            metrics_path = _find_metrics(out_dir)
            if metrics_path is None:
                return False, f"missing metrics: {out_dir}"
            if not _has_pass128(metrics_path):
                return False, f"missing pass@128: {metrics_path}"

    return True, "complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdin", action="store_true", help="Read qstat -j output from stdin")
    args = parser.parse_args()
    if not args.stdin:
        print("INCOMPLETE missing --stdin", file=sys.stderr)
        return 2
    text = sys.stdin.read()
    ok, reason = check_complete(text)
    if ok:
        print("COMPLETE")
        return 0
    print(f"INCOMPLETE {reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
