"""Ridge initialization of W_in and W_out, from streamed statistics (activations are never stored).

W = (X^T X + lambda I)^{-1} X^T Y, with X^T X and X^T Y accumulated in fp64.

* W_in (pass 1): X = Gemma's N_A(h) at layer i, Y = MobileMoE's own MoE-block input at
  layer j, on positions where both tokenizers end a token at the same character.
  Y is RMS-normalized per token by default (``ridge.target: rms_normalized``): B's
  norm N_B discards scale anyway, and this stops a few huge-norm tokens dominating.
* W_out (pass 2): X = MoE_B(N_B(W_in x)) with W_in fixed, Y = Gemma's FFN_A(x). Both
  come from Gemma's own forward pass, so no alignment is needed.

lambda is chosen per layer on a held-out tenth of the calibration batches.
The first position of every sequence (attention-sink states) is left out.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch

from ..data.heldout import read_jsonl_texts
from ..models.align import encode_aligned
from ..models.capture import capture_ffn_io, capture_module_inputs, run_until
from ..models.layer_map import map_layers, parse_layers
from ..models.load import (
    b_architecture,
    hidden_of,
    load_model_a,
    load_model_b_full,
    load_tokenizer,
    n_layers_of,
)
from ..models.moe_source import extract_blocks, randomize_like


class RidgeStats:
    def __init__(self, d_in: int, d_out: int, device: torch.device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)  # noqa: E731
        self.xtx = {"train": z(d_in, d_in), "val": z(d_in, d_in)}
        self.xty = {"train": z(d_in, d_out), "val": z(d_in, d_out)}
        self.yty = {"train": z(()), "val": z(())}
        self.n = {"train": 0, "val": 0}
        self.device = device

    def add(self, x: torch.Tensor, y: torch.Tensor, split: str = "train") -> None:
        if x.numel() == 0:
            return
        x = x.detach().reshape(-1, x.shape[-1]).float()
        y = y.detach().reshape(-1, y.shape[-1]).float()
        if x.device.type == "mps" and self.device.type == "cpu":
            # MPS has no fp64: form this batch's products on the GPU in fp32 (fast), then
            # add them to the fp64 running sums on the CPU.
            self.xtx[split] += (x.T @ x).cpu().double()
            self.xty[split] += (x.T @ y).cpu().double()
            self.yty[split] += (y * y).sum().cpu().double()
        else:
            x, y = x.to(self.device).double(), y.to(self.device).double()
            self.xtx[split] += x.T @ x
            self.xty[split] += x.T @ y
            self.yty[split] += (y * y).sum()
        self.n[split] += x.shape[0]

    def _solve(self, lam_rel: float, split: str = "train") -> torch.Tensor:
        xtx = self.xtx[split]
        d = xtx.shape[0]
        scale = torch.trace(xtx) / d
        return torch.linalg.solve(xtx + lam_rel * scale * torch.eye(d, dtype=xtx.dtype, device=xtx.device), self.xty[split])

    def val_nmse(self, w: torch.Tensor) -> float:
        """||Y - XW||^2 / ||Y||^2 on the held-out split, from its statistics alone."""
        if self.n["val"] == 0:
            return float("nan")
        err = self.yty["val"] - 2 * torch.sum(w * self.xty["val"]) + torch.sum(w * (self.xtx["val"] @ w))
        return float(err / self.yty["val"].clamp_min(1e-12))

    def solve(self, lambdas: list[float]) -> tuple[torch.Tensor, dict]:
        scores = {}
        for lam in lambdas:
            scores[lam] = self.val_nmse(self._solve(lam))
        valid = [(s, lam) for lam, s in scores.items() if not math.isnan(s)]
        score, lam = min(valid) if valid else (float("nan"), lambdas[len(lambdas) // 2])
        scores = {str(k): v for k, v in scores.items()}
        # Refit on train + val with the chosen lambda.
        for k in ("xtx", "xty", "yty"):
            getattr(self, k)["all"] = getattr(self, k)["train"] + getattr(self, k)["val"]
        w = self._solve(lam, "all")
        return w, {"lambda": lam, "val_nmse": score, "val_nmse_by_lambda": scores, "n_train": self.n["train"], "n_val": self.n["val"]}


def _stats_device(device: torch.device) -> torch.device:
    return torch.device("cpu") if device.type == "mps" else device


def _rms_normalize(y: torch.Tensor) -> torch.Tensor:
    yf = y.float()
    return yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)


def _batches(texts: list[str], size: int):
    for k in range(0, len(texts), size):
        yield k // size, texts[k : k + size]


def ridge_init(cfg: dict, device: torch.device, policy, out_path: Optional[Path] = None, log=print) -> dict:
    """Returns {A layer: {"w_in": (d_B, d_A), "w_out": (d_A, d_B)}} and writes it to ``out_path``."""
    pair, rcfg, data_cfg = cfg["pair"], cfg["ridge"], cfg["data"]
    inner_kind = cfg["transplant"].get("inner", "moe_b")
    seq_len = int(data_cfg["seq_len"])
    bs = int(rcfg.get("batch_size", 8))
    lambdas = [float(x) for x in rcfg.get("lambdas", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])]
    val_every = int(rcfg.get("val_every", 10))
    texts = read_jsonl_texts(Path(data_cfg["heldout_dir"]) / "calib.jsonl", rcfg.get("max_docs"))
    tok_a = load_tokenizer(pair["a"], "a")
    model_a = load_model_a(pair["a"], policy.a_dtype, device, cfg.get("precision", {}).get("attn_implementation", "sdpa"))
    n_a, d_a = n_layers_of(model_a), hidden_of(model_a)
    n_b, d_b = b_architecture(pair["b"])
    a_layers = parse_layers(pair["layers"], n_a)
    layer_map = map_layers(a_layers, n_a, n_b, pair.get("layer_map", "relative_depth"))
    sdev = _stats_device(device)
    report: dict = {"layers": {}, "inner": inner_kind}
    result: dict[int, dict[str, torch.Tensor]] = {}

    if inner_kind == "a_ffn_bottleneck":
        # Control C6: start the bottleneck as the projection onto x's top-d_B principal directions.
        stats = {i: RidgeStats(d_a, 1, sdev) for i in a_layers}
        for _, chunk in _batches(texts, bs):
            enc = tok_a(chunk, truncation=True, max_length=seq_len, padding=True, return_tensors="pt")
            ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            keep = mask.bool().clone()
            keep[:, 0] = False
            with torch.no_grad(), capture_ffn_io(model_a, a_layers, want_output=False, stop_after=max(a_layers)) as st:
                run_until(model_a, input_ids=ids, attention_mask=mask)
            for i in a_layers:
                x = st[i]["x"][keep]
                stats[i].add(x, torch.zeros(x.shape[0], 1, device=x.device))
        for i in a_layers:
            evals, evecs = torch.linalg.eigh(stats[i].xtx["train"])
            v = evecs[:, -d_b:].flip(-1)  # (d_A, d_B), top principal directions first
            result[i] = {"w_in": v.T.float().cpu(), "w_out": v.float().cpu()}
            report["layers"][i] = {"explained": float(evals[-d_b:].sum() / evals.sum())}
        _save(result, report, out_path)
        return result

    # ---- pass 1: W_in from aligned positions --------------------------------------
    tok_b = load_tokenizer(pair["b"], "b")
    model_b = load_model_b_full(pair["b"], policy.b_dtype, device)
    b_layers = sorted(set(layer_map.values()))
    blocks = extract_blocks(model_b, pair["b"], b_layers)
    norms = {j: blocks[j].norm for j in b_layers}
    stats_in = {i: RidgeStats(d_a, d_b, sdev) for i in a_layers}
    target = rcfg.get("target", "rms_normalized")
    n_pairs = n_tok_a = n_tok_b = 0
    for k, chunk in _batches(texts, bs):
        ab = encode_aligned(tok_a, tok_b, chunk, seq_len, skip_first=True)
        split = "val" if k % val_every == val_every - 1 else "train"
        with torch.no_grad():
            with capture_ffn_io(model_a, a_layers, want_output=False, stop_after=max(a_layers)) as sa:
                run_until(model_a, input_ids=ab.ids_a.to(device), attention_mask=ab.mask_a.to(device))
            with capture_module_inputs(norms, stop_after=max(b_layers)) as sb:
                run_until(model_b, input_ids=ab.ids_b.to(device), attention_mask=ab.mask_b.to(device))
        rows, pa, pb = ab.index_tensors(device)
        for i, j in layer_map.items():
            x = sa[i]["x"][rows, pa]
            y = sb[j][rows, pb]
            if target == "rms_normalized":
                y = _rms_normalize(y)
            stats_in[i].add(x, y, split)
        n_pairs += ab.n_pairs
        n_tok_a += ab.n_tokens_a
        n_tok_b += ab.n_tokens_b
    report["aligned_share_of_a_tokens"] = n_pairs / max(n_tok_a, 1)
    report["aligned_share_of_b_tokens"] = n_pairs / max(n_tok_b, 1)
    log(f"[ridge] aligned positions: {n_pairs} ({report['aligned_share_of_a_tokens']:.1%} of Gemma tokens)")
    w_in = {}
    for i in a_layers:
        w, info = stats_in[i].solve(lambdas)
        w_in[i] = w.T.float()  # (d_B, d_A)
        report["layers"][i] = {"b_layer": layer_map[i], "w_in": info}
        log(f"[ridge] layer {i}: W_in lambda={info['lambda']:g} val_nmse={info['val_nmse']:.4f}")
    del stats_in, model_b

    if inner_kind == "random_moe_b":
        blocks = {j: randomize_like(b, seed=int(cfg.get("run", {}).get("seed", 0))) for j, b in blocks.items()}
    for b in blocks.values():
        b.to(device=device, dtype=policy.b_dtype)
        b.shared_enabled = bool(cfg["transplant"].get("shared_expert", True))  # ablation AB6

    # ---- pass 2: W_out from Gemma's own forward pass --------------------------------
    stats_out = {i: RidgeStats(d_b, d_a, sdev) for i in a_layers}
    for k, chunk in _batches(texts, bs):
        enc = tok_a(chunk, truncation=True, max_length=seq_len, padding=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        keep = mask.bool().clone()
        keep[:, 0] = False
        split = "val" if k % val_every == val_every - 1 else "train"
        with torch.no_grad():
            with capture_ffn_io(model_a, a_layers, want_output=True, stop_after=max(a_layers)) as sa:
                run_until(model_a, input_ids=ids, attention_mask=mask)
            for i, j in layer_map.items():
                x = sa[i]["x"]
                u = (x.to(policy.proj_dtype) @ w_in[i].to(x.device, policy.proj_dtype).T).to(policy.b_dtype)
                m = blocks[j](u)
                stats_out[i].add(m[keep], sa[i]["y"][keep], split)
    for i in a_layers:
        w, info = stats_out[i].solve(lambdas)
        result[i] = {"w_in": w_in[i].cpu(), "w_out": w.T.float().cpu()}  # w_out: (d_A, d_B)
        report["layers"][i]["w_out"] = info
        log(f"[ridge] layer {i}: W_out lambda={info['lambda']:g} val_nmse={info['val_nmse']:.4f}")
    _save(result, report, out_path)
    return result


def _save(result: dict, report: dict, out_path: Optional[Path]) -> None:
    if out_path is None:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({int(i): v for i, v in result.items()}, out_path)
    with open(out_path.with_suffix(".json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)


def load_init(path: str | Path) -> dict:
    return {int(k): v for k, v in torch.load(path, map_location="cpu").items()}
