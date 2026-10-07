"""One training run on a fixed token budget, every token read once:

    ridge init -> local matching (first 25% of the tokens) -> end to end (the rest)
    -> full evaluation (loss, perplexity, gap recovery, accuracy, routing) -> sample outputs
    -> one row per evaluation set in results/runs.csv

An extension run (``run.extend_from``) starts from a finished run's projections and its
data-stream position and spends its whole budget end to end, so more tokens can be added
after looking at the results without reading any token twice.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import torch

from ..config import config_hash, save_config
from ..data.heldout import load_manifest, mixture_hash
from ..eval.evaluate import evaluate
from ..eval.samples import write_samples
from ..init.ridge import load_init, ridge_init
from ..models.load import build_hybrid
from ..utils.device import pick_device, resolve_precision
from ..utils.logging import RunLogger, append_runs_csv, git_commit
from ..utils.memory import count_params, empty_cache, peak_gb
from ..utils.seed import seed_everything
from .checkpoint import load_projections
from .end_to_end import train_e2e
from .local_match import train_local


def run_dir_of(cfg: dict) -> Path:
    return Path(cfg["run"].get("out_dir", "runs")) / cfg["run"]["name"]


def setup(cfg: dict):
    device = pick_device(cfg.get("device", "auto"))
    policy = resolve_precision(cfg.get("precision", {}), device)
    seed_everything(int(cfg["run"].get("seed", 0)))
    return device, policy


def token_budget(cfg: dict) -> tuple[int, int, Optional[Path]]:
    """(local-matching tokens, end-to-end tokens, run extended from)."""
    total = int(cfg["run"]["tokens"])
    extend = cfg["run"].get("extend_from")
    if extend:
        return 0, total, Path(extend)
    share = float(cfg["run"].get("local_share", 0.25)) if cfg["local"].get("enabled", True) else 0.0
    local = int(round(total * share))
    return local, total - local, None


def _device_name(device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        import platform

        return f"Apple GPU (MPS), {platform.machine()}"
    return device.type


def _init_matrices(cfg: dict, device, policy, run_dir: Path, log=print) -> Optional[dict]:
    tcfg = cfg["transplant"]
    if tcfg.get("init", "ridge") != "ridge":
        return None
    if tcfg.get("projection", "linear") == "mlp":
        log("[init] MLP projections keep their random init (ridge needs a linear map)")
        return None
    shared = tcfg.get("init_path")
    path = Path(shared) if shared else run_dir / "init.pt"
    if path.exists():
        log(f"[init] reusing {path}")
        return load_init(path)
    log(f"[init] ridge init -> {path}")
    return ridge_init(cfg, device, policy, out_path=path, log=log)


def run_training(cfg: dict, resume: bool = False, stop_after: Optional[str] = None, log=print) -> Path:
    device, policy = setup(cfg)
    run_dir = run_dir_of(cfg)
    if resume and (run_dir / "eval.json").exists() and (run_dir / "projections.pt").exists():
        log(f"[run] {run_dir} already finished; skipping (delete eval.json to re-evaluate)")
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    manifest = load_manifest(cfg["data"]["heldout_dir"])
    logger = RunLogger(run_dir, use_tensorboard=bool(cfg["log"].get("tensorboard", False)))
    local_tokens, e2e_tokens, extend_from = token_budget(cfg)
    prev_tokens = 0
    if extend_from is not None:
        prev_summary = json.loads((extend_from / "summary.json").read_text())
        prev_tokens = int(prev_summary.get("tokens_trained_total", 0))
    info = {
        "run_name": cfg["run"]["name"],
        "experiment": cfg["run"].get("experiment", ""),
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
        "seed": cfg["run"].get("seed", 0),
        "device": str(device),
        "gpu": _device_name(device),
        "precision": policy.describe(),
        "token_budget": {"local": local_tokens, "e2e": e2e_tokens, "extends": str(extend_from) if extend_from else None},
        "data_mixture_hash": mixture_hash(cfg["data"]),
        "data_sources": [s["name"] for s in cfg["data"]["sources"] if s.get("weight", 0) > 0],
        "heldout_head_docs": manifest["head_docs"],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    logger.write_json("run_info.json", info)
    log(f"[run] {info['run_name']} on {info['gpu']} with {info['precision']}")
    log(f"[run] token budget: local matching {local_tokens:,} + end to end {e2e_tokens:,}"
        + (f" (extends {extend_from}, {prev_tokens:,} tokens already trained)" if extend_from else ""))

    init = None if extend_from is not None else _init_matrices(cfg, device, policy, run_dir, log)
    empty_cache(device)
    if stop_after == "init":
        return run_dir
    hybrid = build_hybrid(cfg, device, policy, init_matrices=init)
    if extend_from is not None:
        load_projections(extend_from / "projections.pt", hybrid.model)
    info["trainable_params"] = count_params(hybrid.model, trainable_only=True)
    info["layer_map"] = hybrid.layer_map
    log(f"[run] transplanted layers {hybrid.a_layers}; trainable params {info['trainable_params']:,}")

    summary: dict = {}
    e2e_done = (run_dir / "projections.pt").exists() and resume
    local_done = (run_dir / "local_projections.pt").exists() and resume
    if e2e_done:
        load_projections(run_dir / "projections.pt", hybrid.model)
        prev = run_dir / "summary.json"
        if prev.exists():
            summary.update({k: v for k, v in json.loads(prev.read_text()).items() if k != "final"})
    else:
        if local_tokens > 0:
            if local_done:
                load_projections(run_dir / "local_projections.pt", hybrid.model)
            else:
                summary.update(train_local(hybrid, cfg, device, run_dir, logger, local_tokens, resume=resume, log=log))
            if cfg["local"].get("eval_after", True):
                res = evaluate(hybrid, cfg, device, sets=[cfg["eval"]["sets"][0]],
                               max_tokens=cfg["eval"].get("subset_tokens"), log=log)
                summary["after_local"] = res["sets"]
            if stop_after == "local":
                logger.write_json("summary.json", summary)
                return run_dir
        start_state = None
        if extend_from is not None and (extend_from / "e2e_data_state.pt").exists():
            start_state = torch.load(extend_from / "e2e_data_state.pt", weights_only=False)
        summary.update(train_e2e(hybrid, cfg, device, run_dir, logger, e2e_tokens, resume=resume,
                                 start_data_state=start_state, log=log))
        summary["tokens_trained_this_run"] = summary.get("local_tokens", 0) + summary.get("e2e_tokens", 0)
        summary["tokens_trained_total"] = prev_tokens + summary["tokens_trained_this_run"]
        logger.write_json("summary.json", summary)

    empty_cache(device)
    final = evaluate(hybrid, cfg, device, max_tokens=cfg["eval"].get("final_tokens"), log=log)
    summary["final"] = final
    summary["peak_gb"] = max(summary.get("peak_gb", 0) or 0, peak_gb(device) or 0)
    logger.write_json("eval.json", final)
    write_samples(hybrid, cfg, device, run_dir / "samples.md", label=f"({info['run_name']})")
    logger.write_json("summary.json", summary)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.write_json("run_info.json", info)
    for set_name, res in final["sets"].items():
        append_runs_csv(cfg["log"].get("runs_csv", "results/runs.csv"), {
            "time": info["finished"], "run_name": info["run_name"], "experiment": info["experiment"],
            "config_hash": info["config_hash"], "git_commit": info["git_commit"], "seed": info["seed"],
            "layers": json.dumps(hybrid.a_layers), "stage": "final",
            "tokens": summary.get("tokens_trained_total"), "metric_set": set_name,
            **{k: res.get(k) for k in ("ppl_a", "ppl_ablated", "ppl_hybrid", "gap_ablated_minus_a", "gr", "gr_ce",
                                       "ppl_ratio", "acc_a", "acc_hybrid")},
            "min_routing_entropy": final.get("min_routing_entropy"), "peak_gb": summary["peak_gb"],
            "tokens_per_s": summary.get("e2e_tokens_per_s"), "notes": cfg["run"].get("notes", ""),
        })
    logger.close()
    log(f"[run] done: {run_dir} (samples in {run_dir / 'samples.md'})")
    return run_dir


def run_eval(cfg: dict, projections: Optional[str] = None, routing_agreement: bool = False, log=print) -> dict:
    device, policy = setup(cfg)
    run_dir = run_dir_of(cfg)
    hybrid = build_hybrid(cfg, device, policy)
    load_projections(projections or run_dir / "projections.pt", hybrid.model)
    res = evaluate(hybrid, cfg, device, max_tokens=cfg["eval"].get("final_tokens"), log=log)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_samples(hybrid, cfg, device, run_dir / "samples.md", label=f"({cfg['run']['name']})")
    if routing_agreement:
        from ..data.heldout import read_jsonl_texts
        from ..eval.routing import routing_agreement as agree
        from ..models.load import load_model_b_full, load_tokenizer
        from ..models.moe_source import extract_blocks

        model_b = load_model_b_full(cfg["pair"]["b"], policy.b_dtype, device)
        blocks_b = extract_blocks(model_b, cfg["pair"]["b"], sorted(set(hybrid.layer_map.values())))
        texts = read_jsonl_texts(Path(cfg["data"]["heldout_dir"]) / "val_mix.jsonl", cfg["eval"].get("agreement_docs", 200))
        res["agreement"] = agree(hybrid, blocks_b, model_b, load_tokenizer(cfg["pair"]["b"], "b"), texts,
                                 int(cfg["data"]["seq_len"]), device)
        log(f"[eval] routing agreement: {json.dumps(res['agreement'], default=float)[:400]}")
    with open(run_dir / "eval_full.json", "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    return res
