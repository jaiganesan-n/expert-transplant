"""Evaluate a hybrid: Gemma (A), Gemma with the transplanted layers' FFNs zeroed
(ablated, control C4) and the hybrid, on the same held-out sequences, so gap recovery
comes out of one call. Routing health is recorded during the hybrid pass."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..data import heldout_path
from ..models import surgery
from .metrics import summarize_ppl
from .perplexity import lm_loss_on_array
from .routing import RoutingCollector

_CACHE: dict = {}


def _cache_path(data_cfg: dict) -> Path:
    # Gemma's and the ablated model's losses depend only on the held-out set, the layer set
    # and the precision, so they are computed once and shared by every run.
    return Path(data_cfg["heldout_dir"]) / "eval_cache.json"


def _load_disk_cache(data_cfg: dict) -> dict:
    p = _cache_path(data_cfg)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_disk_cache(data_cfg: dict, cache: dict) -> None:
    p = _cache_path(data_cfg)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    tmp.replace(p)


def _rows_for(tokens: Optional[int], seq_len: int) -> Optional[int]:
    return None if tokens is None else max(1, int(tokens) // seq_len)


def evaluate(hybrid, cfg: dict, device, sets: Optional[list[str]] = None, max_tokens: Optional[int] = None, routing: bool = True, log=print) -> dict:
    ecfg, data_cfg = cfg["eval"], cfg["data"]
    sets = sets or list(ecfg.get("sets", ["mix"]))
    seq_len = int(data_cfg["seq_len"])
    bs = int(ecfg.get("batch_size", 4))
    model = hybrid.model
    was_training = model.training
    model.eval()
    results: dict = {"sets": {}, "routing": {}}
    layers_key = tuple(sorted(surgery.slots(model)))
    for k, name in enumerate(sets):
        path = heldout_path(data_cfg, name)
        if not path.exists():
            log(f"[eval] skipping {name}: {path} not built")
            continue
        arr = np.load(path)
        rows = _rows_for(max_tokens, seq_len)
        tag = f"v2|{cfg['pair']['a']['model_id']}|{hybrid.policy.describe()['a_dtype'] if hybrid.policy else ''}|{device.type}"
        key_a = f"{path.resolve()}|{rows}|a|{tag}"
        key_abl = f"{path.resolve()}|{rows}|ablated|{list(layers_key)}|{tag}"
        for key, which in ((key_a, "original"), (key_abl, "zero")):
            if key not in _CACHE:
                disk = _load_disk_cache(data_cfg)
                if key in disk:
                    _CACHE[key] = disk[key]
                else:
                    with surgery.mode(model, which):
                        _CACHE[key] = lm_loss_on_array(model, arr, bs, device, rows)
                    disk[key] = _CACHE[key]
                    _save_disk_cache(data_cfg, disk)
        collector = RoutingCollector(model) if (routing and k == 0) else None
        with surgery.mode(model, "transplant"), surgery.record_routing(model, on=collector is not None):
            hyb = lm_loss_on_array(model, arr, bs, device, rows, on_batch=collector)
        acc = {"a": _CACHE[key_a].get("accuracy"), "ablated": _CACHE[key_abl].get("accuracy"), "hybrid": hyb["accuracy"]}
        res = summarize_ppl(_CACHE[key_a]["ce"], _CACHE[key_abl]["ce"], hyb["ce"], acc)
        res["tokens"] = hyb["tokens"]
        results["sets"][name] = res
        if collector is not None:
            results["routing"] = {int(i): v for i, v in collector.health().items()}
        log(f"[eval] {name}: ppl A {res['ppl_a']:.3f} | ablated {res['ppl_ablated']:.3f} | hybrid {res['ppl_hybrid']:.3f} "
            f"| GR {res['gr']:.3f} | GR_CE {res['gr_ce']:.3f} | acc A {acc['a']:.3f} hybrid {acc['hybrid']:.3f}")
    if results["routing"]:
        ent = [v["normalized_entropy"] for v in results["routing"].values()]
        results["min_routing_entropy"] = min(ent)
        if min(ent) < float(ecfg.get("entropy_alarm", 0.7)):
            log(f"[eval] ROUTING ALARM: normalized entropy {min(ent):.3f} < {ecfg.get('entropy_alarm', 0.7)}")
    if was_training:
        surgery.train_mode(model)
    return results


def clear_cache() -> None:
    _CACHE.clear()
