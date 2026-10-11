"""Local matching: every transplanted layer learns to imitate Gemma's FFN, in one pass.

Gemma runs once per batch, unmodified and without gradients (its FFNs in place). At
each transplanted layer i, hooks read x_i = N_A(h_i) and the target FFN_A(x_i). Each
layer's projections then learn, on their own, to make

    W_out MoE_B(N_B(W_in x_i))  ~  FFN_A(x_i)

Layers do not interact here, so one Gemma forward pass serves all of them and the
forward stops after the last transplanted layer. Gradient clipping is per layer.
This phase reads the first ``run.local_share`` of the run's tokens; end-to-end
training continues the same stream, so no token is read twice.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from ..data import training_stream
from ..models import surgery
from ..models.capture import capture_ffn_io, run_until
from ..utils.memory import peak_gb, peak_text, reset_peak, track_peak
from ..utils.progress import progress_bar, safe_log
from .checkpoint import ckpt_due, load_checkpoint, save_checkpoint, save_projections
from .losses import nmse
from .optim import make_optimizer, make_scheduler, steps_for_tokens

PHASE = "local"


def train_local(hybrid, cfg: dict, device: torch.device, run_dir: Path, logger, tokens: int,
                resume: bool = False, log=print) -> dict:
    lc, data_cfg = cfg["local"], cfg["data"]
    log = safe_log(log)
    if tokens <= 0:
        log("[local] skipped (no tokens assigned)")
        return {}
    model, policy = hybrid.model, hybrid.policy
    slots = surgery.slots(model)
    layers = surgery.decoder_layers(model)
    seq_len = int(data_cfg["seq_len"])
    mb, accum = int(lc["micro_batch"]), int(lc["grad_accum"])
    steps = steps_for_tokens(tokens, mb, accum, seq_len)
    params = surgery.trainable_parameters(model)
    opt = make_optimizer(params, float(lc["lr"]), float(lc.get("weight_decay", 0.0)), cfg["optim"])
    warmup = min(int(lc.get("warmup_steps", 100)), max(1, steps // 10))
    sched = make_scheduler(opt, lc.get("schedule", "constant"), steps, warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=policy.grad_scaler)
    clip = float(cfg["optim"].get("grad_clip", 1.0))
    skip_first = bool(lc.get("skip_first_position", True))
    post_norm = lc.get("loss", "nmse") == "nmse_post_norm"
    ckpt_path = run_dir / "local_ckpt.pt"
    start, data_state = 0, None
    if resume and ckpt_path.exists():
        payload = load_checkpoint(ckpt_path, model=model, optimizer=opt, scheduler=sched, scaler=scaler)
        start, data_state = payload["step"], payload["data_state"]
        log(f"[local] resumed at step {start}")
    stream = training_stream(data_cfg, hybrid.tokenizer, data_state, seed=int(cfg["run"].get("seed", 0)))
    last = max(slots)
    model.eval()
    reset_peak(device)
    t0, tokens_done = time.time(), 0
    last_save = t0
    mean_loss = None
    per_step = mb * accum * seq_len
    bar = progress_bar(steps * per_step, "local matching", initial=start * per_step)
    for step in range(start, steps):
        per_layer = {i: 0.0 for i in slots}
        for _ in range(accum):
            ids = stream.next_batch(mb).to(device)
            with torch.no_grad(), surgery.mode(model, "original"):
                with capture_ffn_io(model, slots, want_output=True, stop_after=last) as st:
                    run_until(model.model, input_ids=ids, use_cache=False)
            lo = 1 if skip_first else 0
            for i, slot in slots.items():
                x, y = st[i]["x"], st[i]["y"]
                pred = slot.transplant(x)
                if post_norm:
                    pn = layers[i].post_feedforward_layernorm
                    pred, y = pn(pred), pn(y)
                loss = nmse(pred[:, lo:], y[:, lo:])
                scaler.scale(loss / accum).backward()
                per_layer[i] += float(loss.detach()) / accum
            tokens_done += ids.numel()
            track_peak(device)
            bar.update(ids.numel())
        scaler.unscale_(opt)
        for slot in slots.values():  # layers train independently: clip each on its own
            torch.nn.utils.clip_grad_norm_(list(slot.trainable_parameters()), clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        mean_loss = sum(per_layer.values()) / len(per_layer)
        bar.set_postfix_str(f"step {step + 1}/{steps} L1 {mean_loss:.4f}{peak_text(device)}")
        if step % int(cfg["log"].get("every_steps", 10)) == 0 or step == steps - 1:
            rate = tokens_done / max(time.time() - t0, 1e-6)
            logger.log(step, PHASE, loss=mean_loss, lr=sched.get_last_lr()[0], tokens_per_s=rate,
                       peak_gb=peak_gb(device), **{f"loss_l{i}": v for i, v in per_layer.items()})
            log(f"[local] step {step + 1}/{steps} L1 {mean_loss:.4f} ({rate:.0f} tok/s{peak_text(device)})")
        if ckpt_due(cfg, step, last_save) and step + 1 < steps:
            save_checkpoint(ckpt_path, model=model, optimizer=opt, scheduler=sched, scaler=scaler,
                            step=step + 1, data_state=stream.state_dict())
            last_save = time.time()
    bar.close()
    save_projections(run_dir / "local_projections.pt", model)
    torch.save(stream.state_dict(), run_dir / "local_data_state.pt")  # end-to-end continues the stream
    return {"local_steps": steps, "local_tokens": steps * mb * accum * seq_len, "local_final_loss": mean_loss,
            "local_tokens_per_s": tokens_done / max(time.time() - t0, 1e-6)}
