import json
import sys

import numpy as np


DEFAULT_JSON = "/home/caixq/project/LLM_EVAL/eval_log/vi_curl/grad_variance/vf_curl_grad_variance__vf_majority_vote_curl_Qwen2.5-math-1.5B__auto_6e7f288e08e4.json"
p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSON

with open(p, "r", encoding="utf-8") as f:
    d = json.load(f)

rows = d["rows"]
baseline_rows = d.get("baseline_rows")
base = baseline_rows or rows


def arr(key, src):
    return np.array([float(r.get(key, float("nan"))) for r in src], dtype=np.float64)


steps = np.array([int(r["step"]) for r in rows], dtype=np.int64)
beta = arr("beta_target", rows)
beta = np.where(np.isnan(beta), 1.0, beta)
beta = np.clip(beta, 1e-6, 1.0)

sigma = arr("sigma_kept", rows)
vprob = arr("vprob_kept", rows)
base_sigma_full = arr("sigma_full", base)
base_vprob_full = arr("vprob_full", base)

sigma_num_ratio_full = sigma / np.clip(base_sigma_full, 1e-12, None)
vprob_num_ratio_full = vprob / np.clip(base_vprob_full, 1e-12, None)

proxy = (sigma + vprob) / beta
base_total_full = base_sigma_full + base_vprob_full
proxy_ratio_full = proxy / np.clip(base_total_full, 1e-12, None)

sigma_num_ratio_kept = None
vprob_num_ratio_kept = None
proxy_ratio_kept = None
if baseline_rows is not None:
    base_sigma_kept = arr("sigma_kept", baseline_rows)
    base_vprob_kept = arr("vprob_kept", baseline_rows)
    sigma_num_ratio_kept = sigma / np.clip(base_sigma_kept, 1e-12, None)
    vprob_num_ratio_kept = vprob / np.clip(base_vprob_kept, 1e-12, None)
    base_proxy = (base_sigma_kept + base_vprob_kept) / beta
    proxy_ratio_kept = proxy / np.clip(base_proxy, 1e-12, None)

print("json:", p)
print("steps:", steps.tolist())
print("sigma_num_ratio_full: min/max =", float(np.nanmin(sigma_num_ratio_full)), float(np.nanmax(sigma_num_ratio_full)))
print("vprob_num_ratio_full: min/max =", float(np.nanmin(vprob_num_ratio_full)), float(np.nanmax(vprob_num_ratio_full)))
print("variance_proxy_ratio_full: min/max =", float(np.nanmin(proxy_ratio_full)), float(np.nanmax(proxy_ratio_full)))
if sigma_num_ratio_kept is not None:
    print("sigma_num_ratio_kept: min/max =", float(np.nanmin(sigma_num_ratio_kept)), float(np.nanmax(sigma_num_ratio_kept)))
if vprob_num_ratio_kept is not None:
    print("vprob_num_ratio_kept: min/max =", float(np.nanmin(vprob_num_ratio_kept)), float(np.nanmax(vprob_num_ratio_kept)))
if proxy_ratio_kept is not None:
    print("variance_proxy_ratio_kept: min/max =", float(np.nanmin(proxy_ratio_kept)), float(np.nanmax(proxy_ratio_kept)))
