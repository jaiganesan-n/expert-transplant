"""End-to-end training of every projection through Gemma's output head.

L2 = CE(y, p_hybrid) + beta * KL(p_A || p_hybrid), beta = 1.0 by default so the hybrid
recovers Gemma instead of drifting. p_A comes from a no-grad pass of the same model
with the transplants switched off (Gemma's own FFNs back in their slots).

The data stream continues from where local matching stopped (or, for an extension
run, from where the previous run stopped), so every token is read once.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from ..data import training_stream
from ..eval.evaluate import evaluate
from ..models import surgery
from ..utils.memory import empty_cache, peak_gb, peak_text, reset_peak, track_peak
from ..utils.progress import progress_bar, safe_log
from .checkpoint import ckpt_due, load_latest_checkpoint, save_checkpoint, save_projections
from .interrupt import PAUSE_FILE, StopGuard, StopRequested, TrainingPaused
from .losses import chunked_ce_kl, shift_for_lm
from .optim import make_optimizer, make_scheduler, steps_for_tokens

PHASE = "e2e"


def _direct_log(msg: str) -> None:
    """For signal handlers: straight to stderr, no locks."""
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _hidden(model, ids):
    return model.model(input_ids=ids, use_cache=False).last_hidden_state


def _past_checks(run_dir: Path, upto: int) -> list[dict]:
    """Subset evaluations already logged in metrics.jsonl at or before step ``upto``."""
    path = run_dir / "metrics.jsonl"
    if upto <= 0 or not path.exists():
        return []
    import json

    seen: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("stage") == "eval_subset" and int(row.get("step", 0)) <= upto:
            seen[int(row["step"])] = {k: v for k, v in row.items() if k not in ("time", "stage")}
    return [seen[k] for k in sorted(seen)]


def train_e2e(hybrid, cfg: dict, device: torch.device, run_dir: Path, logger, tokens: int, resume: bool = False,
              start_data_state: Optional[dict] = None, log=print) -> dict:
    """Train every projection end to end for ``tokens`` tokens.

    Stops on request (Ctrl-C, SIGTERM, SIGHUP, a PAUSE file in the run folder) at once, saves a
    checkpoint at the last finished step and raises ``TrainingPaused``; an error saves the same way
    and is raised again. Resuming continues from that step with the same data, and a held-out
    check that a stop cut short runs again first.
    """
    ec, data_cfg = cfg["e2e"], cfg["data"]
    model, policy = hybrid.model, hybrid.policy
    log = safe_log(log)
    seq_len = int(data_cfg["seq_len"])
    mb, accum = int(ec["micro_batch"]), int(ec["grad_accum"])
    steps = steps_for_tokens(tokens, mb, accum, seq_len)
    warmup = max(1, int(round(float(ec.get("warmup_frac", 0.02)) * steps)))
    params = surgery.trainable_parameters(model)
    opt = make_optimizer(params, float(ec["lr"]), float(ec.get("weight_decay", 0.01)), cfg["optim"])
    sched = make_scheduler(opt, ec.get("schedule", "cosine"), steps, warmup, float(ec.get("min_lr_frac", 0.1)))
    scaler = torch.amp.GradScaler("cuda", enabled=policy.grad_scaler)
    clip = float(cfg["optim"].get("grad_clip", 1.0))
    beta = float(ec.get("beta_kl", 1.0))
    chunk = int(ec.get("loss_chunk", 256))
    weight = model.lm_head.weight
    softcap = getattr(model.config, "final_logit_softcapping", None)
    if ec.get("grad_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ckpt_path = run_dir / "e2e_ckpt.pt"
    start, data_state = 0, start_data_state
    payload = None
    if resume:
        payload = load_latest_checkpoint(ckpt_path, model=model, optimizer=opt, scheduler=sched, scaler=scaler, log=log)
    if payload is not None:
        start, data_state = payload["step"], payload["data_state"]
        log(f"[e2e] resumed at step {start}")
    elif (run_dir / "local_data_state.pt").exists():
        data_state = torch.load(run_dir / "local_data_state.pt", weights_only=False)
    stream = training_stream(data_cfg, hybrid.tokenizer, data_state, seed=int(cfg["run"].get("seed", 0)))
    eval_every = int(cfg["eval"].get("every_steps", 100) or 0)
    eval_tokens = cfg["eval"].get("subset_tokens", 250_000)
    surgery.train_mode(model)
    reset_peak(device)
    t0, tokens_done = time.time(), 0
    last_save = t0
    history = _past_checks(run_dir, start)  # after a resume, the checks made before it
    per_step = mb * accum * seq_len
    log(f"[e2e] {steps} steps of {per_step:,} tokens ({mb} x {accum} sequences) from step {start}; a check on "
        f"{f'{int(eval_tokens):,}' if eval_tokens else 'all'} held-out tokens every {eval_every} steps. "
        f"Ctrl-C (or a file named PAUSE in {run_dir}) pauses with a checkpoint. "
        "The first step is slow while the GPU compiles its kernels.")
    guard = StopGuard(run_dir / PAUSE_FILE, log=_direct_log)
    if guard.pause_requested():  # a pause file left from before would stop this run at once
        guard.clear_pause_file()

    def check(at: int) -> None:
        res = evaluate(hybrid, cfg, device, sets=[cfg["eval"]["sets"][0]], max_tokens=eval_tokens, log=log)
        mix = next(iter(res["sets"].values()), {})
        with guard.critical():
            logger.log(at, "eval_subset", **{k: v for k, v in mix.items() if isinstance(v, (int, float))},
                       min_routing_entropy=res.get("min_routing_entropy"))
            history.append({"step": at, **mix})
        empty_cache(device)
        surgery.train_mode(model)

    def save(at: int, state: dict) -> None:
        save_checkpoint(ckpt_path, model=model, optimizer=opt, scheduler=sched, scaler=scaler, step=at,
                        data_state=state)

    bar = progress_bar(steps * per_step, "end to end", initial=start * per_step)
    safe_step, safe_state, updating = start, None, False  # the last finished step; None: the stream is there now
    with guard:
        try:
            if eval_every and 0 < start < steps and start % eval_every == 0 and all(h["step"] != start for h in history):
                log(f"[e2e] the check at step {start} was cut short before the stop; running it again")
                check(start)
            for step in range(start, steps):
                safe_step, safe_state = step, copy.deepcopy(stream.state_dict())
                ce_acc = kl_acc = acc = 0.0
                for _ in range(accum):
                    guard.check()
                    ids = stream.next_batch(mb).to(device)
                    h_t = None
                    if beta > 0:
                        with torch.no_grad(), surgery.mode(model, "original"):
                            h_t = _hidden(model, ids)[:, :-1]
                    with surgery.mode(model, "transplant"):
                        h_s = _hidden(model, ids)
                    hs, labels = shift_for_lm(h_s, ids)
                    ce_sum, kl_sum, n, correct = chunked_ce_kl(hs, labels, weight, h_teacher=h_t, beta=beta,
                                                               chunk=chunk, softcap=softcap, return_correct=True)
                    loss = (ce_sum + beta * kl_sum) / n
                    scaler.scale(loss / accum).backward()
                    ce_acc += float(ce_sum.detach()) / n / accum
                    kl_acc += float(kl_sum.detach()) / n / accum
                    acc += float(correct) / n / accum
                    tokens_done += ids.numel()
                    track_peak(device)
                    bar.update(ids.numel())
                with guard.critical():  # never half an optimizer step
                    updating = True
                    scaler.unscale_(opt)
                    gnorm = float(torch.nn.utils.clip_grad_norm_(params, clip))
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    sched.step()
                    updating = False
                    safe_step, safe_state = step + 1, None
                bar.set_postfix_str(f"step {step + 1}/{steps} CE {ce_acc:.3f} KL {kl_acc:.3f} acc {acc:.3f}"
                                    f"{peak_text(device)}")
                if step % int(cfg["log"].get("every_steps", 10)) == 0 or step == steps - 1:
                    rate = tokens_done / max(time.time() - t0, 1e-6)
                    with guard.critical():
                        logger.log(step, PHASE, ce=ce_acc, kl=kl_acc, loss=ce_acc + beta * kl_acc, accuracy=acc,
                                   grad_norm=gnorm, lr=sched.get_last_lr()[0], tokens_per_s=rate,
                                   peak_gb=peak_gb(device))
                    log(f"[e2e] step {step + 1}/{steps} CE {ce_acc:.4f} KL {kl_acc:.4f} acc {acc:.3f} "
                        f"|g| {gnorm:.3f} ({rate:.0f} tok/s{peak_text(device)})")
                if eval_every and (step + 1) % eval_every == 0 and step + 1 < steps:
                    check(step + 1)
                if ckpt_due(cfg, step, last_save) and step + 1 < steps:
                    with guard.critical():
                        save(step + 1, stream.state_dict())
                        last_save = time.time()
            guard.check()
        except KeyboardInterrupt:
            raise  # quit at once (third Ctrl-C): the last checkpoint stays
        except BaseException as e:
            with guard.shielded():  # a further Ctrl-C does not cut this short (a third one quits)
                bar.close()
                opt.zero_grad(set_to_none=True)
                if updating:
                    log("[e2e] stopped during an optimizer step; the last checkpoint stays")
                else:
                    try:
                        save(safe_step, safe_state if safe_state is not None else stream.state_dict())
                        log(f"[e2e] checkpoint saved at step {safe_step} of {steps}")
                    except Exception as save_error:
                        log(f"[e2e] could not save a checkpoint ({save_error}); the last saved one stays")
                if isinstance(e, StopRequested):
                    guard.clear_pause_file()
                    log(f"[e2e] paused at step {safe_step} of {steps} ({e.reason}). Run the same command to resume.")
                    raise TrainingPaused(safe_step, steps, e.reason) from None
                log(f"[e2e] stopped by an error at step {safe_step} of {steps}: {type(e).__name__}: {e}")
                raise
        with guard.shielded():
            bar.close()
            torch.save(stream.state_dict(), run_dir / "e2e_data_state.pt")  # an extension run continues from here
            save_projections(run_dir / "projections.pt", model)  # written last: it marks training as done
        if guard.reason is not None:  # asked to stop while the end was saved: stop before the final evaluation
            guard.clear_pause_file()
            raise TrainingPaused(steps, steps, guard.reason)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.eval()
    return {"e2e_steps": steps, "e2e_tokens": steps * mb * accum * seq_len,
            "e2e_tokens_per_s": tokens_done / max(time.time() - t0, 1e-6),
            "peak_gb": peak_gb(device), "eval_history": history}
