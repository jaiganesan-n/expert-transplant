"""Reference numbers: Gemma (C1), Gemma with the transplanted layers' FFNs zeroed (C4) and
MobileMoE (C2, in bits per byte since its tokenizer differs). No training involved."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data import heldout_path
from ..data.heldout import read_jsonl_texts
from ..models import surgery
from ..models.layer_map import parse_layers
from ..models.load import load_model_a, load_model_b_full, load_tokenizer, n_layers_of
from .metrics import summarize_ppl
from .perplexity import bits_per_byte, lm_loss_on_array


class _Zero(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def run_baselines(cfg: dict, device, policy, layer_sets: list, out_dir: Path, log=print) -> dict:
    data_cfg, ecfg = cfg["data"], cfg["eval"]
    seq_len, bs = int(data_cfg["seq_len"]), int(ecfg.get("batch_size", 4))
    model_a = load_model_a(cfg["pair"]["a"], policy.a_dtype, device, cfg.get("precision", {}).get("attn_implementation", "sdpa"))
    tok_a = load_tokenizer(cfg["pair"]["a"], "a")
    n_a = n_layers_of(model_a)
    out: dict = {"a": {}, "ablated": {}, "bpb": {}}
    arrays = {}
    for name in ecfg.get("sets", ["mix"]):
        p = heldout_path(data_cfg, name)
        if p.exists():
            arrays[name] = np.load(p)
    rows = None if ecfg.get("final_tokens") is None else int(ecfg["final_tokens"]) // seq_len
    with torch.no_grad():
        for name, arr in arrays.items():
            out["a"][name] = lm_loss_on_array(model_a, arr, bs, device, rows)
            log(f"[baselines] Gemma {name}: ppl {out['a'][name]['ppl']:.3f}")
        layers = surgery.decoder_layers(model_a)
        for spec in layer_sets:
            idx = parse_layers(spec, n_a)
            saved = {i: layers[i].mlp for i in idx}
            for i in idx:
                layers[i].mlp = _Zero()
            key = str(spec)
            out["ablated"][key] = {}
            for name, arr in arrays.items():
                abl = lm_loss_on_array(model_a, arr, bs, device, rows)
                out["ablated"][key][name] = dict(abl, **summarize_ppl(out["a"][name]["ce"], abl["ce"], abl["ce"]))
                log(f"[baselines] Gemma, FFN zeroed at {key}, {name}: ppl {abl['ppl']:.3f}")
            for i in idx:
                layers[i].mlp = saved[i]
        texts = read_jsonl_texts(Path(data_cfg["heldout_dir"]) / "val_mix.jsonl", ecfg.get("bpb_docs", 500))
        out["bpb"]["a"] = bits_per_byte(model_a, tok_a, texts, seq_len, device, use_logits=False)
        log(f"[baselines] Gemma bits/byte: {out['bpb']['a']['bpb']:.4f}")
        del model_a
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model_b = load_model_b_full(cfg["pair"]["b"], policy.b_dtype, device)
        tok_b = load_tokenizer(cfg["pair"]["b"], "b")
        out["bpb"]["b"] = bits_per_byte(model_b, tok_b, texts, seq_len, device, use_logits=True)
        log(f"[baselines] MobileMoE bits/byte: {out['bpb']['b']['bpb']:.4f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baselines.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out
