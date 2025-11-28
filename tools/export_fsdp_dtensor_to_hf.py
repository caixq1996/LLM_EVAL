import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
try:
    from torch.distributed._tensor.api import DTensor, Shard, Replicate
except Exception:
    from torch.distributed.tensor import DTensor, Shard, Replicate


def _rank_id(p):
    m = re.search(r'rank_(\d+)', p.name)
    return int(m.group(1)) if m else -1


def _world_size_from_name(p):
    m = re.search(r'world_size_(\d+)', p.name)
    return int(m.group(1)) if m else -1


def _load_all_rank_states(shard_files):
    sd_list = []
    for f in shard_files:
        sd = torch.load(str(f), map_location='cpu')
        if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
            sd = sd['model']
        sd_list.append(sd)
    return sd_list


def _is_dtensor(x):
    return isinstance(x, DTensor)


def _assemble_param(key, shards):
    v0 = shards[0]
    if _is_dtensor(v0):
        placements = getattr(v0, 'placements', [])
        shard_dim = None
        for pl in placements:
            if isinstance(pl, Shard):
                shard_dim = int(getattr(pl, 'dim', 0))
                break
        if shard_dim is None:
            return v0.to_local().clone()
        locals_ = []
        for v in shards:
            if not _is_dtensor(v):
                locals_.append(v)
            else:
                locals_.append(v.to_local())
        try:
            full = torch.cat(locals_, dim=shard_dim).contiguous()
        except Exception as e:
            raise RuntimeError(
                f'Concatenate shards failed on key={key}, dim={shard_dim}, '
                f'shapes={[tuple(t.shape) for t in locals_]}'
            ) from e
        return full
    else:
        return v0


def export_one_step_to_hf(step_dir, base_model_dir, export_root):
    actor = step_dir / 'actor'
    out_dir = export_root / step_dir.parent.name / step_dir.name
    meta_path = out_dir / 'export_meta.json'
    if out_dir.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('source') == str(step_dir) and meta.get('base_model') == str(base_model_dir):
                return out_dir
        except Exception:
            pass  # Re-export if metadata missing/corrupted

    shard_candidates = sorted(actor.glob('model_world_size_*_rank_*.pt'), key=_rank_id)
    if not shard_candidates:
        raise FileNotFoundError(f'No model shard files under {actor}')
    shards_by_ws: Dict[int, List[Path]] = {}
    for f in shard_candidates:
        ws = _world_size_from_name(f)
        shards_by_ws.setdefault(ws, []).append(f)

    # 优先选择 world_size > 1 的最大值；没有的话退回到 world_size == 1
    ws_choices = sorted([ws for ws in shards_by_ws.keys() if ws and ws > 1], reverse=True)
    if not ws_choices:
        ws_choices = sorted([ws for ws in shards_by_ws.keys() if ws > 0], reverse=True)
    if not ws_choices:
        raise RuntimeError(f'Unable to determine world_size from shard files under {actor}')
    ws = ws_choices[0]
    shard_files = sorted(shards_by_ws[ws], key=_rank_id)

    if ws > 0 and len(shard_files) != ws:
        raise RuntimeError(f'world_size={ws} but found {len(shard_files)} shard files (candidates={len(shard_candidates)})')
    sd_list = _load_all_rank_states(shard_files)
    keys = list(sd_list[0].keys())
    full_sd = {}
    for k in keys:
        shard_vals = [sd[k] for sd in sd_list]
        full_sd[k] = _assemble_param(k, shard_vals)

    cfg = AutoConfig.from_pretrained(str(base_model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    missing, unexpected = model.load_state_dict(full_sd, strict=False)
    if missing or unexpected:
        print(f'[WARN] load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}')
        if missing:
            print('  missing (first 10):', missing[:10])
        if unexpected:
            print('  unexpected (first 10):', unexpected[:10])

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)

    tok_src = actor / 'huggingface'
    try:
        tok = AutoTokenizer.from_pretrained(
            str(tok_src if tok_src.exists() else base_model_dir),
            trust_remote_code=True, use_fast=True
        )
        tok.save_pretrained(str(out_dir))
        jinja = tok_src / 'chat_template.jinja'
        if jinja.exists():
            shutil.copy2(jinja, out_dir / 'chat_template.jinja')
    except Exception as e:
        print(f'[WARN] tokenizer export failed, fallback to base tokenizer. err={e}')
        tok = AutoTokenizer.from_pretrained(str(base_model_dir), trust_remote_code=True, use_fast=True)
        tok.save_pretrained(str(out_dir))

    meta = {
        'source': str(step_dir),
        'base_model': str(base_model_dir),
        'world_size': ws,
        'num_shards': len(shard_files),
    }
    (out_dir / 'export_meta.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    return out_dir


def _step_num_from_dir(p: Path) -> int:
    m = re.search(r'global_step_(\d+)', p.name)
    return int(m.group(1)) if m else -1


def list_step_dirs(run_dir: Path, only_latest: bool = True) -> List[Path]:
    """Return global_step_* directories under run_dir, optionally limited to the latest."""
    step_dirs = [p for p in run_dir.glob('global_step_*') if p.is_dir()]
    if not step_dirs:
        return []

    if only_latest:
        step_dirs = [max(step_dirs, key=_step_num_from_dir)]
    else:
        step_dirs = sorted(step_dirs, key=_step_num_from_dir)
    return step_dirs


def export_all_steps_under_run(run_dir, base_model_dir, export_root, only_latest: bool = True):
    """
    只导出并返回需要评测的 step 目录。默认只取最大的 global_step_*。
    """
    step_dirs = list_step_dirs(run_dir, only_latest=only_latest)
    if not step_dirs:
        return []

    out_dirs = []
    for step in step_dirs:
        try:
            out_dir = export_one_step_to_hf(step, base_model_dir, export_root)
            out_dirs.append(out_dir)
            print(f'[OK] Exported: {out_dir}')
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f'[WARN] Export failed for {step}: {e}')
    return out_dirs
