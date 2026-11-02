"""Check what runs at inference for a finished run: its trained projections and the donor's untouched MoE.

    .venv/bin/python scripts/verify_stage4_inference.py runs/stage4_spread_15m
    .venv/bin/python scripts/verify_stage4_inference.py runs/stage4_spread_15m --device cpu --no-hub

The hybrid is built exactly as `transplant ask` and `transplant eval` build it (``build_hybrid`` on the
run's config.yaml, then ``load_projections(projections.pt)``). Then:

  1. Projections: every W_in / W_out in the model equals projections.pt bit for bit and has moved away
     from the ridge init it started from.
  2. Trainable flags: only W_in / W_out have requires_grad; nothing in the MoE blocks or in Gemma does.
  3. Donor MoE: each transplanted block (pre-MoE norm, router, routed experts, shared expert) equals a
     separate, fresh load of the donor bit for bit; no MoE weight was missing from the checkpoint (it would
     have been initialised at random); the block's tensors equal the raw tensors in the checkpoint files;
     and, with the Hub reachable, the cached weight files' SHA-256 equal the Hub's at the recorded revision.
  4. What runs: on one prompt, forward hooks count W_in, MoE, shared-expert and W_out calls in every
     transplanted layer (Gemma's own FFN must not run there) and Gemma's FFN in the kept layers; the router
     picks are recorded; the first and last slot outputs are recomputed by hand as
     W_out(MoE_ref(N_ref(W_in x))) from the fresh donor block and the projections.pt matrices.

Nothing is written. Exit code 0 when every check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from transplant.models.load import build_hybrid, cast_keep_rope_fp32, is_tiny, load_model_b_full  # noqa: E402
from transplant.models.moe_source import extract_blocks, resolve_attr, use_fast_forward  # noqa: E402
from transplant.train.run import setup  # noqa: E402

RESULTS: list[tuple[str, bool]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""), flush=True)


def note(text: str) -> None:
    print(f"[info] {text}", flush=True)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.cpu().double(), b.cpu().double()  # to the CPU first: MPS has no fp64
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_donor(b_cfg: dict, dtype: torch.dtype):
    """A separate load of the donor on the CPU, the same way the hybrid's donor is loaded, plus the
    loader's report of weights missing from the checkpoint (None for the tiny stand-ins)."""
    if is_tiny(b_cfg):
        return load_model_b_full(b_cfg, dtype, "cpu"), None
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(
        b_cfg["model_id"], revision=b_cfg.get("revision"), torch_dtype=dtype,
        trust_remote_code=bool(b_cfg.get("trust_remote_code", True)), output_loading_info=True,
    )
    model.config.use_cache = False
    kind = b_cfg.get("moe_forward", "bucketed")
    if kind != "native":
        use_fast_forward(model, b_cfg, kind)
    cast_keep_rope_fp32(model, dtype, "cpu")
    model.requires_grad_(False)
    model.eval()
    return model, info


def block_prefixes(b_cfg: dict, block, j: int) -> dict[str, str]:
    """Checkpoint key prefix of the block's two parts: norm -> <layers>.<j>.<norm_attr>, moe -> ...<moe_attr>."""
    layers_path = b_cfg.get("layers_path", "model.layers")
    norm_attr = b_cfg.get("norm_attr") or "post_attention_layernorm"
    moe_attr = b_cfg.get("moe_attr") or "feed_forward"
    return {"norm.": f"{layers_path}.{j}.{norm_attr}.", "moe.": f"{layers_path}.{j}.{moe_attr}."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path, help="finished run folder (config.yaml, projections.pt, run_info.json)")
    ap.add_argument("--device", default=None, help="override the run's device (mps, cpu, cuda)")
    ap.add_argument("--no-hub", action="store_true", help="skip the SHA-256 check against the Hugging Face Hub")
    ap.add_argument("--prompt", default="The capital of France is")
    args = ap.parse_args(argv)

    run = args.run
    overrides = [f"device={args.device}"] if args.device else []
    cfg = load_config([run / "config.yaml"], overrides)
    run_info = json.loads((run / "run_info.json").read_text()) if (run / "run_info.json").exists() else {}
    device, policy = setup(cfg)
    b_cfg = cfg["pair"]["b"]
    print(f"== {run} on {device}: A {cfg['pair']['a']['model_id']}, B {b_cfg['model_id']} ==", flush=True)

    # The hybrid, as `transplant ask` / `transplant eval` build it.
    hybrid = build_hybrid(cfg, device, policy)
    from transplant.train.checkpoint import load_projections

    load_projections(run / "projections.pt", hybrid.model)
    model = hybrid.model
    model.eval()
    layers = surgery.decoder_layers(model)
    slots = surgery.slots(model)
    kept = [i for i in range(len(layers)) if i not in slots]
    note(f"transplanted A layers {sorted(slots)} (A <- B {dict(sorted(hybrid.layer_map.items()))}); "
         f"Gemma's FFN kept in {kept}")

    # 1. Projections
    print("\n-- 1. projections --", flush=True)
    from_file = torch.load(run / "projections.pt", map_location="cpu")
    in_model = surgery.projection_state(model)
    keys_ok = set(from_file) == set(in_model)
    exact = keys_ok and all(torch.equal(in_model[k], from_file[k].to(in_model[k].dtype)) for k in from_file)
    n_proj = sum(t.numel() for t in from_file.values())
    report("W_in / W_out in the model == projections.pt (bit for bit)", exact,
           f"{len(from_file)} tensors, {n_proj:,} parameters; keys match: {keys_ok}")
    init_path = cfg["transplant"].get("init_path")
    cands = [Path(init_path), REPO / init_path] if init_path else []
    cands.append(run / "init.pt")
    init_file = next((p for p in cands if p.exists()), None)
    init = load_init(init_file) if init_file else None
    if init is not None:
        dists = []
        for k, t in from_file.items():
            i, which = int(k.split(".")[0]), k.split(".")[1]
            if i in init and which in init[i]:
                dists.append(rel(t, init[i][which]))
        report("trained projections differ from the ridge init they started from", bool(dists) and min(dists) > 1e-4,
               f"{init_file}: ||P - P_init|| / ||P_init|| min {min(dists):.4f}, median "
               f"{sorted(dists)[len(dists) // 2]:.4f}, max {max(dists):.4f} over {len(dists)} matrices")
    else:
        note("no ridge init file found; skipped the comparison with the init")

    # 2. Trainable flags
    print("\n-- 2. trainable flags --", flush=True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    only_proj = all((".w_in." in n or ".w_out." in n) for n in trainable)
    moe_grad = [n for i, s in slots.items() for n, p in s.inner.named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = run_info.get("trainable_params")
    report("only W_in / W_out are trainable; no MoE parameter is", only_proj and not moe_grad and bool(trainable),
           f"{len(trainable)} trainable tensors, {count:,} parameters"
           + (f" (run_info.json: {expected:,})" if expected else "") + f"; MoE tensors with requires_grad: {len(moe_grad)}")

    # 3. Donor MoE
    print("\n-- 3. donor MoE blocks --", flush=True)
    ref_model, load_info = fresh_donor(b_cfg, policy.b_dtype)
    needed = sorted(set(hybrid.layer_map.values()))
    ref_blocks = extract_blocks(ref_model, b_cfg, needed)
    n_t = n_el = 0
    diffs = []
    for i, slot in sorted(slots.items()):
        j = hybrid.layer_map[i]
        mine, ref = slot.inner.state_dict(), ref_blocks[j].state_dict()
        if set(mine) != set(ref):
            diffs.append(f"A{i}<-B{j}: different tensor names")
            continue
        for name, t in mine.items():
            n_t += 1
            n_el += t.numel()
            if not torch.equal(t.detach().cpu(), ref[name].detach().cpu()):
                diffs.append(f"A{i}<-B{j} {name}: max |diff| {float((t.cpu().float() - ref[name].float()).abs().max()):.3g}")
    report("every MoE block in the hybrid == a fresh load of the donor (bit for bit)", not diffs and n_t > 0,
           f"{len(slots)} blocks, {n_t} tensors, {n_el:,} parameters compared"
           + (f"; differences: {diffs[:5]}" if diffs else ""))
    if load_info is not None:
        prefixes = [p for j in needed for p in block_prefixes(b_cfg, None, j).values()]
        missing = [k for k in load_info.get("missing_keys", []) if any(k.startswith(p) for p in prefixes)]
        mism = [k for k in load_info.get("mismatched_keys", []) if any(str(k).startswith(p) for p in prefixes)]
        report("no MoE / norm weight was missing from the checkpoint (nothing initialised at random)",
               not missing and not mism, f"missing in the mapped blocks: {missing[:5]}; shape mismatches: {mism[:5]}"
               f"; missing anywhere in the donor: {len(load_info.get('missing_keys', []))}")
    if not is_tiny(b_cfg):
        from huggingface_hub import hf_hub_download

        model_id, rev_cfg = b_cfg["model_id"], b_cfg.get("revision")
        try:
            index = Path(hf_hub_download(model_id, "model.safetensors.index.json", revision=rev_cfg, local_files_only=True))
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        except Exception:
            files = ["model.safetensors"]
        local = {f: Path(hf_hub_download(model_id, f, revision=rev_cfg, local_files_only=True)) for f in files}
        note(f"donor weight files in the local cache: {[str(p) for p in local.values()]}")
        # raw tensors of the checkpoint vs the block
        from safetensors import safe_open

        handles = {f: safe_open(str(p), framework="pt") for f, p in local.items()}
        where = {k: f for f, h in handles.items() for k in h.keys()}
        matched = unmatched = 0
        raw_diffs = []
        for i, slot in sorted(slots.items()):
            j = hybrid.layer_map[i]
            pref = block_prefixes(b_cfg, slot.inner, j)
            for name, t in slot.inner.state_dict().items():
                head = next(h for h in pref if name.startswith(h))
                key = pref[head] + name[len(head):]
                if key not in where:
                    unmatched += 1
                    continue
                raw = handles[where[key]].get_tensor(key)
                matched += 1
                if raw.shape != t.shape or not torch.equal(raw.to(t.dtype), t.detach().cpu()):
                    raw_diffs.append(key)
        if matched:
            report("MoE block tensors == the raw tensors in the donor's checkpoint files", not raw_diffs,
                   f"{matched} tensors found in the files and compared, {unmatched} stored under other names; "
                   f"different: {raw_diffs[:5]}")
        else:
            note(f"the checkpoint stores the block tensors under other names ({unmatched} not found); "
                 "the fresh-load and missing-weight checks above cover them")
        if not args.no_hub:
            try:
                from huggingface_hub import HfApi

                rev = (run_info.get("model_b") or {}).get("revision") or rev_cfg
                info = HfApi().model_info(model_id, revision=rev, files_metadata=True)
                def lfs_sha(s):
                    lfs = getattr(s, "lfs", None)
                    if lfs is None:
                        return None
                    return lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)

                hub = {s.rfilename: lfs_sha(s) for s in info.siblings}
                rows, ok = [], True
                for f, p in local.items():
                    mine = sha256(p)
                    same = hub.get(f) == mine
                    ok &= same
                    rows.append(f"{f}: local {mine[:16]}..., Hub@{str(rev)[:12]} {str(hub.get(f))[:16]}... "
                                f"{'same' if same else 'DIFFERENT'}")
                report("cached donor weight files == the Hub's files at the run's recorded revision (SHA-256)", ok,
                       "; ".join(rows))
            except Exception as e:
                note(f"Hub check skipped ({type(e).__name__}: {e}); run again online or pass --no-hub")

    # 4. What runs on a prompt
    print("\n-- 4. what runs at inference --", flush=True)
    counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    captured: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles_ = []

    def hook(i: int, part: str):
        def fn(mod, inp, out):
            counts[i][part] += 1
            if part == "slot":
                captured[i] = (inp[0].detach(), out.detach())
        return fn

    for i, layer in enumerate(layers):
        m = layer.mlp
        if i in slots:
            handles_.append(m.register_forward_hook(hook(i, "slot")))
            handles_.append(m.w_in.register_forward_hook(hook(i, "w_in")))
            handles_.append(m.inner.register_forward_hook(hook(i, "moe")))
            handles_.append(m.w_out.register_forward_hook(hook(i, "w_out")))
            handles_.append(m.ffn_a.register_forward_hook(hook(i, "gemma_ffn")))
            shared_name = getattr(m.inner, "shared_name", None)
            shared = resolve_attr(m.inner.moe, shared_name) if shared_name else None
            if isinstance(shared, nn.Module):
                handles_.append(shared.register_forward_hook(hook(i, "shared_expert")))
        else:
            handles_.append(m.register_forward_hook(hook(i, "gemma_ffn_kept")))
    tok = hybrid.tokenizer
    ids = torch.tensor([tok(args.prompt, add_special_tokens=True)["input_ids"]], device=device)
    with torch.no_grad(), surgery.mode(model, "transplant"), surgery.record_routing(model):
        logits = model(input_ids=ids).logits
        picks = {}
        for i, s in sorted(slots.items()):
            calls = getattr(s.inner, "router_calls", [])
            if calls:
                sel = s.inner.selected_experts(calls[-1])
                picks[i] = (tuple(sel.shape), len(set(sel.flatten().tolist())))
    for h in handles_:
        h.remove()
    bad = []
    for i in sorted(slots):
        c = counts[i]
        if not (c["w_in"] == c["moe"] == c["w_out"] == 1 and c["gemma_ffn"] == 0):
            bad.append(f"A{i}: {dict(c)}")
        if "shared_expert" in c and c["shared_expert"] != 1:
            bad.append(f"A{i}: shared expert ran {c['shared_expert']} times")
    for i in kept:
        if counts[i]["gemma_ffn_kept"] != 1:
            bad.append(f"kept A{i}: Gemma FFN ran {counts[i]['gemma_ffn_kept']} times")
    example = sorted(slots)[0]
    report("transplanted layers ran W_in -> MoE -> W_out, never Gemma's FFN; kept layers ran Gemma's FFN",
           not bad, f"{ids.shape[1]} tokens; layer {example}: {dict(counts[example])}"
           + (f"; kept layer {kept[0]}: {dict(counts[kept[0]])}" if kept else "") + (f"; problems: {bad}" if bad else ""))
    if picks:
        report("routers ran and picked experts in every transplanted layer", len(picks) == len(slots),
               "; ".join(f"A{i}: {shape[0]} tokens x top-{shape[1]}, {n} distinct experts" for i, (shape, n)
                         in list(picks.items())[:4]) + (" ..." if len(picks) > 4 else ""))
    # Recompute the first and last slot by hand from the fresh donor block and the file's projections.
    worst = 0.0
    lines = []
    for i in (min(slots), max(slots)):
        x, y = captured[i]
        j = hybrid.layer_map[i]
        s = slots[i]
        ref = ref_blocks[j].to(device)
        d_a, d_b = s.w_in.in_features, s.w_in.out_features
        w_in = nn.Linear(d_a, d_b, bias=False).to(device, s.proj_dtype)
        w_out = nn.Linear(d_b, d_a, bias=False).to(device, s.proj_dtype)
        with torch.no_grad():
            w_in.weight.copy_(from_file[f"{i}.w_in.weight"])
            w_out.weight.copy_(from_file[f"{i}.w_out.weight"])
            y_ref = w_out(ref(w_in(x.to(s.proj_dtype)).to(s.b_dtype)).to(s.proj_dtype)).to(x.dtype)
            r = rel(y, y_ref)
            line = f"A{i}<-B{j}: ||y - y_ref|| / ||y_ref|| = {r:.2e}"
            if init is not None and i in init:
                w_in.weight.copy_(init[i]["w_in"])
                w_out.weight.copy_(init[i]["w_out"])
                y_init = w_out(ref(w_in(x.to(s.proj_dtype)).to(s.b_dtype)).to(s.proj_dtype)).to(x.dtype)
                line += f" (with the ridge-init projections instead: {rel(y, y_init):.2e})"
        ref_blocks[j].to("cpu")
        worst = max(worst, r)
        lines.append(line)
    tol = 1e-2 if x.dtype in (torch.bfloat16, torch.float16) else 1e-4
    report("slot output == W_out(MoE_ref(N_ref(W_in x))) from the fresh donor block and projections.pt",
           worst < tol, "; ".join(lines))
    top = logits[0, -1].float().argmax().item()
    note(f"next token after {args.prompt!r}: {tok.decode([top])!r}")

    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n== {len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed ==")
    for n in failed:
        print(f"   FAILED: {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
