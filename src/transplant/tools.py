"""M0 checks: module names, fp16 overflow, tokenizer alignment, data availability, memory and speed."""
from __future__ import annotations

import inspect as pyinspect
import json
import time
from pathlib import Path

import torch

from .data.sources import DocSource, SourceSpec
from .models.align import encode_aligned
from .models.load import load_model_a, load_model_b_full, load_tokenizer
from .models.moe_source import describe_layer


def inspect_models(cfg: dict, out_dir: Path, log=print) -> dict:
    """Print what the code finds inside Gemma and MobileMoE; closes plan assumptions A2, A3, A4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    a = load_model_a(cfg["pair"]["a"], torch.float32, torch.device("cpu"), "eager")
    report["a"] = {
        "class": type(a).__name__,
        "num_hidden_layers": a.config.num_hidden_layers,
        "hidden_size": a.config.hidden_size,
        "intermediate_size": getattr(a.config, "intermediate_size", None),
        "vocab_size": a.config.vocab_size,
        "final_logit_softcapping": getattr(a.config, "final_logit_softcapping", None),
        "layer0_children": [n for n, _ in a.model.layers[0].named_children()],
    }
    (out_dir / "a_layer0.txt").write_text(str(a.model.layers[0]))
    del a
    b = load_model_b_full(cfg["pair"]["b"], torch.float32 if cfg["pair"]["b"]["model_id"].startswith("tiny") else torch.bfloat16, "cpu")
    report["b"] = describe_layer(b, cfg["pair"]["b"], 0)
    report["b"]["config"] = {k: v for k, v in vars(getattr(b, "config", object())).items() if isinstance(v, (int, float, str, bool, list))} if hasattr(b, "config") else {}
    from .models.moe_source import find_layers, resolve_attr

    _, layers = find_layers(b, cfg["pair"]["b"].get("layers_path", "auto"))
    moe = getattr(layers[0], report["b"]["moe_attr"])
    (out_dir / "b_layer0.txt").write_text(str(layers[0]))
    try:
        (out_dir / "b_moe_forward.py").write_text(pyinspect.getsource(type(moe)))  # read it for A4
    except (OSError, TypeError):
        pass
    with open(out_dir / "m0_inspect.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    log(json.dumps(report, indent=2, default=str))
    log(f"[inspect] wrote {out_dir}/m0_inspect.json, a_layer0.txt, b_layer0.txt, b_moe_forward.py")
    log("[inspect] pin moe_attr / norm_attr / router_attr / shared_attr in configs/pairs/primary.yaml")
    return report


@torch.no_grad()
def overflow_check(cfg: dict, device, n_seqs: int, out_dir: Path, log=print) -> dict:
    """Run Gemma in fp16 and fp32 on the same held-out sequences; record peak |activation|
    per module and logit drift. Modules above fp16's max (65,504) must stay fp32 (A19)."""
    import numpy as np

    from .data import heldout_path

    arr = np.load(heldout_path(cfg["data"], "mix"))[:n_seqs]
    ids = torch.from_numpy(arr.astype("int64"))
    report: dict = {"fp16_max": 65504.0, "modules_over_fp16": {}}
    logits = {}
    for dt in (torch.float32, torch.float16):
        model = load_model_a(cfg["pair"]["a"], dt, device, "eager")
        peaks: dict[str, float] = {}
        hooks = []
        for name, mod in model.named_modules():
            if name.count(".") > 4:
                continue

            def hook(m, i, o, name=name):
                t = o[0] if isinstance(o, tuple) else o
                if torch.is_tensor(t) and t.is_floating_point():
                    v = float(t.detach().abs().float().nan_to_num(posinf=float("inf")).max())
                    peaks[name] = max(peaks.get(name, 0.0), v)

            hooks.append(mod.register_forward_hook(hook))
        outs = []
        for k in range(0, ids.shape[0], 2):
            out = model(input_ids=ids[k : k + 2].to(device), use_cache=False, logits_to_keep=64)
            outs.append(out.logits.float().cpu())
        for h in hooks:
            h.remove()
        logits[dt] = torch.cat(outs)
        key = str(dt).replace("torch.", "")
        report[f"peak_{key}"] = dict(sorted(peaks.items(), key=lambda kv: -kv[1])[:40])
        if dt == torch.float32:
            report["modules_over_fp16"] = {k: v for k, v in peaks.items() if v > 65504.0 * 0.5}
        del model
    diff = (logits[torch.float32] - logits[torch.float16]).abs()
    report["fp16_logits_nonfinite"] = int((~torch.isfinite(logits[torch.float16])).sum())
    report["fp16_vs_fp32_max_abs_logit_diff"] = float(diff.nan_to_num(nan=float("inf")).max())
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "overflow_check.json", "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"[overflow] non-finite fp16 logits: {report['fp16_logits_nonfinite']}; "
        f"modules within 2x of fp16 max (fp32 run): {len(report['modules_over_fp16'])}")
    return report


def align_stats(cfg: dict, n_chars: int, out_dir: Path, log=print) -> dict:
    """Share of token positions where Gemma and MobileMoE tokenizers end a token at the same character (A20)."""
    from .data.heldout import read_jsonl_texts

    tok_a, tok_b = load_tokenizer(cfg["pair"]["a"], "a"), load_tokenizer(cfg["pair"]["b"], "b")
    texts, total = [], 0
    for t in read_jsonl_texts(Path(cfg["data"]["heldout_dir"]) / "val_mix.jsonl"):
        texts.append(t)
        total += len(t)
        if total >= n_chars:
            break
    pairs = ta = tb = 0
    for k in range(0, len(texts), 8):
        ab = encode_aligned(tok_a, tok_b, texts[k : k + 8], max_len=100_000, skip_first=False)
        pairs += ab.n_pairs
        ta += ab.n_tokens_a
        tb += ab.n_tokens_b
    report = {"chars": total, "tokens_a": ta, "tokens_b": tb, "aligned": pairs,
              "share_of_a": pairs / max(ta, 1), "share_of_b": pairs / max(tb, 1), "a20_holds": pairs / max(ta, 1) >= 0.5}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "align_stats.json", "w") as fh:
        json.dump(report, fh, indent=2)
    log(f"[align] {report}")
    return report


def data_check(cfg: dict, n_docs: int, include_optional: bool, log=print) -> list[dict]:
    """Open every data source in streaming mode and read a few documents (availability, fields, speed)."""
    tok = load_tokenizer(cfg["pair"]["a"], "a")
    rows = []
    for d in cfg["data"]["sources"]:
        spec = SourceSpec.from_cfg(d)
        if spec.weight <= 0 and not include_optional:
            continue
        t0 = time.time()
        row = {"name": spec.name, "weight": spec.weight, "ok": False}
        try:
            it = DocSource(spec).open(0)
            toks = chars = 0
            for _ in range(n_docs):
                text = next(it)
                chars += len(text)
                toks += len(tok(text, add_special_tokens=False)["input_ids"])
            row.update(ok=True, docs=n_docs, gemma_tokens_per_doc=toks / n_docs, chars_per_token=chars / max(toks, 1),
                       seconds=round(time.time() - t0, 1))
        except Exception as exc:  # report and continue
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rows.append(row)
        log(f"[data] {json.dumps(row)}")
    return rows


def memory_check(cfg: dict, micro_steps: int = 2, log=print) -> dict:
    """Memory and speed of real training steps, with the run's own settings.

    Builds the hybrid (all transplanted layers), then runs ``micro_steps`` local-matching
    micro-batches and ``micro_steps`` end-to-end micro-batches, each followed by an AdamW
    step, exactly as training does. Token ids are random: memory and speed do not depend
    on the text. Run it in a fresh process (``transplant memcheck``) so nothing loaded
    earlier inflates the numbers.
    """
    from .models import surgery
    from .models.capture import capture_ffn_io, run_until
    from .models.load import build_hybrid
    from .train.losses import chunked_ce_kl, nmse, shift_for_lm
    from .train.optim import make_optimizer
    from .train.run import setup
    from .utils.memory import empty_cache, live_gb, peak_gb, peak_live_gb, recommended_max_gb, reset_peak, track_peak

    device, policy = setup(cfg)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    t_load = time.time()
    hybrid = build_hybrid(cfg, device, policy)
    empty_cache(device)
    model = hybrid.model
    slots = surgery.slots(model)
    reset_peak(device)
    res: dict = {"device": str(device), "transplanted_layers": len(slots), "load_s": round(time.time() - t_load, 1),
                 "weights_gb": round(live_gb(device), 2)}
    log(f"[memcheck] {len(slots)} layers on {device}; weights {res['weights_gb']:.2f} GB")
    opt = make_optimizer(surgery.trainable_parameters(model), 1e-5, 0.0, cfg["optim"])
    seq = int(cfg["data"]["seq_len"])
    vocab = int(model.config.vocab_size)
    gen = torch.Generator().manual_seed(0)

    def batch(mb: int) -> torch.Tensor:
        ids = torch.randint(10, vocab, (mb, seq), generator=gen)
        ids[:, 0] = hybrid.tokenizer.bos_token_id
        return ids.to(device)

    # Local matching: one Gemma pass without gradients feeds every layer, then each layer's backward.
    mb = int(cfg["local"]["micro_batch"])
    model.eval()
    seconds = []
    for _ in range(micro_steps):
        ids = batch(mb)
        sync()
        t = time.time()
        with torch.no_grad(), surgery.mode(model, "original"):
            with capture_ffn_io(model, slots, want_output=True, stop_after=max(slots)) as st:
                run_until(model.model, input_ids=ids, use_cache=False)
        for i, slot in slots.items():
            nmse(slot.transplant(st[i]["x"]), st[i]["y"]).backward()
            track_peak(device)
        del st
        opt.step()
        opt.zero_grad(set_to_none=True)
        track_peak(device)
        sync()
        seconds.append(time.time() - t)
    res["local"] = {"micro_batch": mb, "seconds_per_micro_batch": round(seconds[-1], 2),
                    "tokens_per_s": round(mb * seq / seconds[-1]), "peak_gb": round(peak_gb(device), 2),
                    "peak_live_gb": round(peak_live_gb(device), 2)}
    log(f"[memcheck] local matching: {res['local']}")
    empty_cache(device)  # training also frees cached blocks between the phases
    reset_peak(device)

    # End to end: teacher pass (no grad), hybrid pass with checkpointing, chunked CE + KL, backward.
    ec = cfg["e2e"]
    mb, beta, chunk = int(ec["micro_batch"]), float(ec.get("beta_kl", 1.0)), int(ec.get("loss_chunk", 256))
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if ec.get("grad_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    surgery.train_mode(model)
    seconds = []
    for _ in range(micro_steps):
        ids = batch(mb)
        sync()
        t = time.time()
        with torch.no_grad(), surgery.mode(model, "original"):
            h_t = model.model(input_ids=ids, use_cache=False).last_hidden_state[:, :-1]
        with surgery.mode(model, "transplant"):
            h_s = model.model(input_ids=ids, use_cache=False).last_hidden_state
        track_peak(device)
        hs, labels = shift_for_lm(h_s, ids)
        ce, kl, n = chunked_ce_kl(hs, labels, model.lm_head.weight, h_teacher=h_t, beta=beta, chunk=chunk,
                                  softcap=softcap)
        track_peak(device)
        ((ce + beta * kl) / n).backward()
        track_peak(device)
        del h_t, h_s, hs, ce, kl
        opt.step()
        opt.zero_grad(set_to_none=True)
        track_peak(device)
        sync()
        seconds.append(time.time() - t)
    res["e2e"] = {"micro_batch": mb, "loss_chunk": chunk, "seconds_per_micro_batch": round(seconds[-1], 2),
                  "tokens_per_s": round(mb * seq / seconds[-1]), "peak_gb": round(peak_gb(device), 2),
                  "peak_live_gb": round(peak_live_gb(device), 2)}
    log(f"[memcheck] end to end: {res['e2e']}")
    # driver memory: everything Metal holds, cached blocks included (what macOS must find room for)
    res["peak_gb"] = max(res["local"]["peak_gb"], res["e2e"]["peak_gb"])
    res["peak_live_gb"] = max(res["local"]["peak_live_gb"], res["e2e"]["peak_live_gb"])
    rec = recommended_max_gb(device)
    res["recommended_max_gb"] = round(rec, 2) if rec else None
    return res
