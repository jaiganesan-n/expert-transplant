#!/usr/bin/env python3
"""Check Stage 5's MoE blocks: MobileMoE-S's architecture, no pretrained value, frozen, the same in every process.

    .venv/bin/python scripts/verify_stage5_random.py --config configs/base.yaml --config configs/stages/stage5_random_15m.yaml
    .venv/bin/python scripts/verify_stage5_random.py runs/stage5_random_15m     # a run: before, during or after training

The hybrid is built as training, `transplant eval` and `transplant ask` build it (``build_hybrid`` on the
config; for a run folder its config.yaml and, once it exists, projections.pt). Then:

  1. Architecture: every block has the module classes, tensor names, shapes and dtypes of MobileMoE-S's
     block at the same layer (a separate, fresh load of the donor on the CPU) and the same forward.
  2. No pretrained value: no tensor equals the pretrained one; each weight matrix is uncorrelated with it
     (|r| < 5 / sqrt(n)); the selection bias is 0; the norm gain is one constant, the pretrained gain's RMS.
  3. Scale: each matrix (the gate and up halves of the fused tensor apart) has the pretrained mean and std.
  4. Frozen: no block tensor has requires_grad; only W_in / W_out train.
  5. The same blocks everywhere: blocks redrawn on the CPU from the fresh donor equal the hybrid's bit for
     bit, and their fingerprint equals the one in the run's run_info.json (training) and ridge init.json.
  6. The start: the run's ridge init holds Stage 4's W_in bit for bit (ridge.w_in_from) and its own W_out.
  7. What runs: on one prompt, every transplanted layer runs W_in -> random block -> W_out and never
     Gemma's FFN; the routers pick experts; the first and last slot are recomputed by hand.

Nothing is written. Exit code 0 when every check passes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from transplant.config import load_config  # noqa: E402
from transplant.init.ridge import load_init  # noqa: E402
from transplant.models import surgery  # noqa: E402
from transplant.models.load import build_hybrid, load_model_b_full, random_moe_options  # noqa: E402
from transplant.models.moe_source import _stats, blocks_fingerprint, extract_blocks, randomize_like  # noqa: E402
from transplant.train.run import setup  # noqa: E402

RESULTS: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""), flush=True)


def note(text: str) -> None:
    print(f"[info] {text}", flush=True)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


def halves(name: str, t: torch.Tensor, full: bool) -> list[tuple[str, torch.Tensor]]:
    """The parts whose scale randomize_like matches: the gate and up halves of a fused tensor apart (full)."""
    if full and name.endswith("gate_up_proj") and t.shape[-1] % 2 == 0:
        f = t.shape[-1] // 2
        return [(f"{name}[gate]", t[..., :f]), (f"{name}[up]", t[..., f:])]
    return [(name, t)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, nargs="?", help="a Stage 5 run folder (config.yaml; projections.pt once trained)")
    ap.add_argument("--config", action="append", default=[], help="config files instead of a run (repeatable)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. device=cpu (repeatable)")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args(argv)
    if not args.run and not args.config:
        ap.error("give a run folder or --config files")

    run = args.run
    cfg = load_config([run / "config.yaml"] if run else args.config, args.set)
    if cfg["transplant"].get("inner") != "random_moe_b":
        print(f"transplant.inner is {cfg['transplant'].get('inner')!r}, not 'random_moe_b': not a Stage 5 setup")
        return 2
    run_info = json.loads((run / "run_info.json").read_text()) if run and (run / "run_info.json").exists() else {}
    opts = random_moe_options(cfg)
    device, policy = setup(cfg)
    b_cfg = cfg["pair"]["b"]
    print(f"== {run or ' + '.join(args.config)} on {device}: A {cfg['pair']['a']['model_id']}, "
          f"random blocks shaped like {b_cfg['model_id']} ({opts}) ==", flush=True)

    hybrid = build_hybrid(cfg, device, policy)
    proj_file = run / "projections.pt" if run else None
    if proj_file is not None and proj_file.exists():
        from transplant.train.checkpoint import load_projections

        load_projections(proj_file, hybrid.model)
        note(f"trained projections loaded from {proj_file}")
    else:
        note("no projections.pt: the projections keep build_hybrid's init (the checks below concern the blocks)")
    model = hybrid.model
    model.eval()
    slots = surgery.slots(model)
    layers = surgery.decoder_layers(model)
    kept = [i for i in range(len(layers)) if i not in slots]
    needed = sorted(set(hybrid.layer_map.values()))
    note(f"transplanted A layers {sorted(slots)}; Gemma's FFN kept in {kept}")

    ref_model = load_model_b_full(b_cfg, policy.b_dtype, "cpu")  # a separate load of the pretrained donor
    ref_blocks = extract_blocks(ref_model, b_cfg, needed)

    # 1. Architecture
    print("\n-- 1. architecture --", flush=True)
    bad = []
    for j in needed:
        mine, ref = hybrid.blocks[j], ref_blocks[j]
        cls_m = [(n, type(m).__name__) for n, m in mine.named_modules()]
        cls_r = [(n, type(m).__name__) for n, m in ref.named_modules()]
        sd_m = {n: (tuple(t.shape), t.dtype) for n, t in mine.state_dict().items()}
        sd_r = {n: (tuple(t.shape), t.dtype) for n, t in ref.state_dict().items()}
        if cls_m != cls_r or sd_m != sd_r:
            bad.append(f"B{j}: modules equal {cls_m == cls_r}, tensors equal {sd_m == sd_r}")
    j0 = needed[0]
    n_par = sum(t.numel() for t in hybrid.blocks[j0].state_dict().values())
    report("every random block has MobileMoE-S's modules, tensor names, shapes and dtypes", not bad,
           f"{len(needed)} blocks; block {j0}: {type(hybrid.blocks[j0].moe).__name__}, {n_par:,} parameters"
           + (f"; differences: {bad[:4]}" if bad else ""))

    # 2 and 3. No pretrained value; the scale of each matrix
    print("\n-- 2. no pretrained value, 3. scale --", flush=True)
    equal, correlated, bias_bad, gain_bad, scale_bad = [], [], [], [], []
    worst_r, worst_std = 0.0, 0.0
    for j in needed:
        mine = {n: t.detach().cpu() for n, t in hybrid.blocks[j].state_dict().items()}
        ref = {n: t.detach().cpu() for n, t in ref_blocks[j].state_dict().items()}
        bias_param = hybrid.blocks[j].routing_bias()
        named = list(hybrid.blocks[j].named_parameters()) + list(hybrid.blocks[j].named_buffers())
        bias_name = next((n for n, p in named if bias_param is not None and p is bias_param), None)
        for name, t in mine.items():
            r_t = ref[name]
            if torch.equal(t, r_t) and name != bias_name:  # the bias is 0 by design (checked below)
                equal.append(f"B{j}.{name}")
            if name == bias_name:
                if opts["full"] and bool(t.any()):
                    bias_bad.append(f"B{j}.{name}")
                continue
            if name.startswith("norm.") and t.ndim == 1:
                want = torch.full(t.shape, math.sqrt(float(r_t.double().pow(2).mean())), dtype=torch.float64).to(t.dtype)
                if opts["full"] and not torch.equal(t, want):
                    gain_bad.append(f"B{j}.{name}")
                continue
            if t.ndim < 2:
                continue
            r = abs(corr(t, r_t))
            worst_r = max(worst_r, r * math.sqrt(t.numel()))
            if r >= 5 / math.sqrt(t.numel()):
                correlated.append(f"B{j}.{name}: r={r:.4f}")
            for (part, a), (_, b) in zip(halves(name, t, opts["full"]), halves(name, r_t, opts["full"])):
                (mm, sm), (mr, sr) = _stats(a), _stats(b)
                worst_std = max(worst_std, abs(sm / sr - 1))
                # 5 standard errors of a sample std (plus bf16 rounding), and of a sample mean
                if abs(sm / sr - 1) > 5 / math.sqrt(2 * a.numel()) + 0.002 or abs(mm - mr) > 5 * sr / math.sqrt(a.numel()) + 1e-6:
                    scale_bad.append(f"B{j}.{part}: mean {mm:+.2e} vs {mr:+.2e}, std {sm:.5f} vs {sr:.5f}")
    report("no tensor of any block equals the pretrained one (the selection bias aside)", not equal,
           f"equal: {equal[:5]}" if equal else "")
    report("every weight matrix is uncorrelated with the pretrained one (|r| < 5/sqrt(n))", not correlated,
           f"largest |r| * sqrt(n) = {worst_r:.2f}" + (f"; correlated: {correlated[:4]}" if correlated else ""))
    report("selection bias is 0 and the norm gain is one constant, the pretrained gain's RMS (full)",
           opts["full"] and not bias_bad and not gain_bad,
           "transplant.random_moe.full is false: C5 as first written keeps both" if not opts["full"] else
           (f"bias not 0: {bias_bad[:3]}; gain not flat: {gain_bad[:3]}" if bias_bad or gain_bad else ""))
    report("each matrix has the pretrained mean and std (gate and up apart)", not scale_bad,
           f"largest relative std difference {worst_std:.2%}" + (f"; off: {scale_bad[:4]}" if scale_bad else ""))

    # 4. Frozen
    print("\n-- 4. frozen --", flush=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    block_grad = [f"B{j}.{n}" for j in needed for n, p in hybrid.blocks[j].named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = run_info.get("trainable_params")
    report("no block tensor trains; only W_in / W_out do",
           not block_grad and bool(trainable) and all(".w_in." in n or ".w_out." in n for n in trainable)
           and (expected is None or count == expected),
           f"{len(trainable)} trainable tensors, {count:,} parameters" + (f" (run_info.json: {expected:,})" if expected else "")
           + (f"; block tensors with requires_grad: {block_grad[:4]}" if block_grad else ""))

    # 5. The same blocks in every process
    print("\n-- 5. the same blocks everywhere --", flush=True)
    diffs, by_hand = [], {}
    ends = (hybrid.layer_map[min(slots)], hybrid.layer_map[max(slots)])
    for j in needed:
        again = randomize_like(ref_blocks[j], **opts)
        mine = hybrid.blocks[j].state_dict()
        for name, t in again.state_dict().items():
            if not torch.equal(t, mine[name].detach().cpu()):
                diffs.append(f"B{j}.{name}")
        if j in ends:
            by_hand[j] = again
    fp = blocks_fingerprint(hybrid.blocks)
    report("blocks redrawn on the CPU from a fresh donor == the hybrid's blocks (bit for bit)", not diffs,
           f"{len(needed)} blocks; fingerprint {fp}" + (f"; different: {diffs[:4]}" if diffs else ""))
    del ref_model, ref_blocks
    trained_fp = (run_info.get("random_moe") or {}).get("fingerprint")
    if trained_fp:
        report("fingerprint == the one recorded when the run trained (run_info.json)", trained_fp == fp,
               f"run_info.json {trained_fp}, now {fp}")
    init_path = Path(cfg["transplant"]["init_path"]) if cfg["transplant"].get("init_path") else (run / "init.pt" if run else None)
    init_json = init_path.with_suffix(".json") if init_path else None
    init_rep = json.loads(init_json.read_text()) if init_json and init_json.exists() else {}
    init_fp = (init_rep.get("random_moe") or {}).get("fingerprint")
    if init_fp:
        report("fingerprint == the one ridge init fitted W_out through (init.json)", init_fp == fp,
               f"{init_json} {init_fp}, now {fp}")
    if not trained_fp and not init_fp:
        note("no run_info.json or ridge init.json with a fingerprint yet; nothing to compare it with")

    # 6. The start
    src = cfg["ridge"].get("w_in_from")
    if init_path and init_path.exists() and src and Path(src).exists():
        print("\n-- 6. the start: W_in --", flush=True)
        init, other = load_init(init_path), load_init(src)
        same_in = [i for i in slots if torch.equal(init[i]["w_in"], other[i]["w_in"])]
        diff_out = [i for i in slots if not torch.equal(init[i]["w_out"], other[i]["w_out"])]
        report("the ridge init holds Stage 4's W_in bit for bit, and its own W_out", len(same_in) == len(diff_out) == len(slots),
               f"{init_path}: W_in equal to {src} in {len(same_in)} of {len(slots)} layers; W_out different in "
               f"{len(diff_out)} of {len(slots)}")
    else:
        note(f"ridge init {init_path} or W_in source {src} not there yet; check 6 skipped")

    # 7. What runs on a prompt
    print("\n-- 7. what runs at inference --", flush=True)
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def hook(i: int, part: str):
        def fn(mod, inp, out):
            counts[i][part] += 1
            if part == "slot":
                captured[i] = (inp[0].detach(), out.detach())
        return fn

    for i, layer in enumerate(layers):
        m = layer.mlp
        if i in slots:
            for part, mod in (("slot", m), ("w_in", m.w_in), ("moe", m.inner), ("w_out", m.w_out), ("gemma_ffn", m.ffn_a)):
                handles.append(mod.register_forward_hook(hook(i, part)))
        else:
            handles.append(m.register_forward_hook(hook(i, "gemma_ffn_kept")))
    tok = hybrid.tokenizer
    ids = torch.tensor([tok(args.prompt, add_special_tokens=True)["input_ids"]], device=device)
    with torch.no_grad(), surgery.mode(model, "transplant"), surgery.record_routing(model):
        logits = model(input_ids=ids).logits
        picks = {i: s.inner.selected_experts(s.inner.router_calls[-1]) for i, s in slots.items()
                 if getattr(s.inner, "router_calls", None)}
    for h in handles:
        h.remove()
    wrong = [f"A{i}: {dict(counts[i])}" for i in slots
             if not (counts[i]["w_in"] == counts[i]["moe"] == counts[i]["w_out"] == 1 and counts[i]["gemma_ffn"] == 0)]
    wrong += [f"kept A{i}" for i in kept if counts[i]["gemma_ffn_kept"] != 1]
    report("transplanted layers ran W_in -> random block -> W_out, never Gemma's FFN; kept layers ran Gemma's FFN",
           not wrong and bool(torch.isfinite(logits).all()),
           f"{ids.shape[1]} tokens; logits finite: {bool(torch.isfinite(logits).all())}" + (f"; problems: {wrong[:4]}" if wrong else ""))
    report("routers picked experts in every transplanted layer", len(picks) == len(slots),
           "; ".join(f"A{i}: {len(set(p.flatten().tolist()))} distinct experts" for i, p in list(sorted(picks.items()))[:4]) + " ...")
    worst, lines = 0.0, []
    for i in (min(slots), max(slots)):
        x, y = captured[i]
        s = slots[i]
        block = by_hand[hybrid.layer_map[i]].to(device)
        w_in = nn.Linear(s.w_in.in_features, s.w_in.out_features, bias=False).to(device, s.proj_dtype)
        w_out = nn.Linear(s.w_out.in_features, s.w_out.out_features, bias=False).to(device, s.proj_dtype)
        with torch.no_grad():
            w_in.weight.copy_(s.w_in.weight)
            w_out.weight.copy_(s.w_out.weight)
            y_ref = w_out(block(w_in(x.to(s.proj_dtype)).to(s.b_dtype)).to(s.proj_dtype)).to(x.dtype)
        y, y_ref = y.cpu().double(), y_ref.cpu().double()  # to the CPU first: MPS has no fp64
        r = float((y - y_ref).norm() / y_ref.norm().clamp_min(1e-30))
        worst = max(worst, r)
        lines.append(f"A{i}<-B{hybrid.layer_map[i]}: {r:.2e}")
    tol = 1e-2 if x.dtype in (torch.bfloat16, torch.float16) else 1e-4
    report("slot output == W_out(random block(W_in x)) with the block redrawn on the CPU", worst < tol, "; ".join(lines))
    note(f"next token after {args.prompt!r}: {tok.decode([logits[0, -1].float().argmax().item()])!r}")

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n== {len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed ==")
    for n in failed:
        print(f"   FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
