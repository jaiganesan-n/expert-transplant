# Usage: data, hardware, setup, workflow, outputs and tests

The operational guide to the expert-transplant repo. The README describes the experiment, the architecture and the results of every stage; this page holds everything needed to run it again.

## Data: the mix MobileMoE was pre-trained on

Gemma 3's 2T-token training set was never released, so the training data is MobileMoE-M-Base's open pre-training mix, Table B.1 of [the MobileMoE paper](https://arxiv.org/abs/2605.27358) (Appendix B). Domain shares are exact; inside a domain the share is split evenly over the enabled sources, because the paper gives no per-dataset weights.

| Domain | Share | Enabled sources (Hugging Face, streamed) | Off until checked at M0 |
| --- | --- | --- | --- |
| Web | 62.0% | FineWeb-Edu `sample-10BT`; DCLM (`allenai/olmo-mix-1124`, `dclm`) | Common Crawl from Dolma 3 |
| Math | 11.6% | FineMath `finemath-3plus`; OpenWebMath and Algebraic Stack (`olmo-mix-1124`) | Nemotron Math |
| Code | 10.0% | StarCoder (`olmo-mix-1124`) | Stack-Edu from Dolma 3; Nemotron Code |
| Knowledge | 10.0% | FLAN (`allenai/dolmino-mix-1124`); Cosmopedia v2 (`smollm-corpus`); Wikipedia (`olmo-mix-1124`) | none |
| Science | 6.4% | arXiv and peS2o (`olmo-mix-1124`); Natural Reasoning (`facebook/natural_reasoning`) | OLMoCR science PDFs; Nemotron Science |

**Availability.** Enabled sources need no login and stream as plain text. The "off" sources are in the config at weight 0: Dolma 3 holds them only inside a 6T-token mix whose file pattern must be checked, Nemotron's are single large chat-format files, and `HuggingFaceTB/stack-edu` stores blob ids rather than text. Turn one on after `uv run transplant data-check --include-optional` shows it streams.

**Size.** The full mix is about 6T tokens; this project reads a small, fixed slice of it. Nothing is downloaded in full: a run streams exactly its budget (1M tokens for Stage 1, 20M for Stage 2, more only if you extend), and every token is read once. Disk use is the held-out files (about 60 MB), the Hugging Face model cache (about 8 GB for both models) and about 2 GB of checkpoints per run (projections, ridge init, optimizer state).

**Held out, never trained on.** `build-heldout` takes the first documents of every source, in proportion to its weight: 5M Gemma tokens of validation text and 2M tokens of calibration text (for ridge init). The training stream skips exactly those documents (recorded in `manifest.json`). Two shift sets are added for evaluation only: WikiText-103 test and 2M tokens of C4 validation.

## Hardware: MacBook Air M5, 16 GB

All runs use the M5's GPU through PyTorch MPS (`configs/hardware/mac_m5.yaml`): both models in bf16 (macOS 14 or later), the projections and their optimizer state in fp32, no loss scaling, never below 16-bit. Ops MPS lacks fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`, set by the scripts), and the scripts keep the Mac awake with `caffeinate`.

Unified memory during end-to-end training, all 26 layers, micro-batch 1 × 1,024 tokens:

| Component | Memory |
| --- | --- |
| Gemma 3 1B, bf16 (its FFNs stay loaded as local-matching targets and KL teacher) | 2.0 GB |
| MobileMoE MoE blocks only, bf16 (26 × 100.7M parameters) | 5.2 GB |
| Projections, fp32, with gradients and AdamW moments | 1.0 GB |
| One end-to-end micro-step: checkpointed activations, one layer recomputed at a time, one 256-position logit chunk (measured on a full-width replica on the CPU) | ~0.9 GB |
| **Total, live tensors** | **~9.2 GB of 16 GB** |

The loss is the part that grows: Gemma's 262,144-token vocabulary makes 256 positions of fp32 logits 0.27 GB. `chunked_ce_kl` keeps the logits of one chunk at a time and its backward pass is written by hand, so at most two such buffers are alive (autograd kept five or six: 2.6 GB at the earlier 512-position chunks).

MPS also keeps freed blocks cached, so what macOS sees is a little above the live figure. `uv run transplant memcheck` measures it in a fresh process: it loads the real models, runs a local-matching and an end-to-end micro-step with an AdamW step, prints both peaks and the speed, and fails above the GPU working set macOS recommends for the Mac (about two thirds of its memory; `TRANSPLANT_MEM_LIMIT_GB` overrides it). Gating test 5 runs it. Close browsers and other large apps during a run. Every log line and `summary.json` report the peak.

**Fixed shapes.** On the Apple GPU every new tensor shape compiles new kernels, and they stay in memory. MobileMoE's own MoE code gives each of its 60 experts a different number of tokens on every call and waits on the GPU once per expert; the first Stage 1 attempt grew to a 29 GB footprint in ridge init and swapped. The repo therefore runs MobileMoE's MoE blocks with a forward pass that computes the same thing but pads each expert's group of tokens to one of a few sizes and waits on the GPU once per layer (`bucketed_moe_forward` in `src/transplant/models/moe_source.py`, checked against the original in the tests and in gating test 2). Calibration batches and sample prompts are padded to fixed lengths for the same reason. `pair.b.moe_forward: native` switches back to MobileMoE's own code; `grouped` runs all 60 experts as one batched matrix product, padded to the largest group (fewer kernels, more padding). On the M5 it lost clearly (`transplant memcheck`): 21 against 98 tokens/s end to end, and 17.6 GB, over the memory limit, so `bucketed` stays the default. `memcheck` prints the speed of both training phases and how an end-to-end step splits into teacher pass, hybrid forward, backward and optimizer. Ridge init prints a progress line every 10 batches with speed, time left and memory.

**Time.** Measured on the M5 by `transplant memcheck` (all 26 layers): local matching about 475 tokens/s, end to end about 118 tokens/s; ridge init about 350 tokens/s for pass 1 and 550 for pass 2. So Stage 1 takes about 3 hours with ridge init and evaluation, and Stage 2 about 40 hours (5M tokens of local matching in 3 hours, 15M end to end in 35, plus about 1.5 hours of ridge init on the full calibration set). Stage 1's script prints the figure again from its own run. Training shows a live progress bar (tokens done, speed, time left, latest losses) with a log line every 10 steps; `TRANSPLANT_NO_PROGRESS=1` turns the bar off. Checkpoints every 200 steps or 30 minutes; if the Mac sleeps or the run stops, run the same script again and it resumes exactly (projections, optimizer, schedule, RNG and data-stream position). A finished run is skipped. Gemma's and the ablated model's held-out losses are computed once and cached in the held-out folder.

## Setup

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv`, or `curl -LsSf https://astral.sh/uv/install.sh | sh`). uv installs Python 3.12 and every package from `uv.lock` into `.venv`; there is no pip step.

```bash
cd expert-transplant
bash scripts/setup.sh        # uv sync, MPS check, unit tests, the whole pipeline on tiny models (~1 minute)
```

For the real models:

1. Accept Google's Gemma terms on [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) and request access to [facebook/MobileMoE-M-Base](https://huggingface.co/facebook/MobileMoE-M-Base).
2. Log in with your own token: `uv run hf auth login`.
3. `bash scripts/m0_checks.sh`: downloads both models (about 8 GB), records module names, checks every data source streams, builds the held-out sets, measures tokenizer alignment, and runs the seven gating tests on the real models (test 5 is `transplant memcheck`, which also prints a first speed estimate). Nothing trains until they pass.

## Workflow: stages, all 26 layers

There are no single-layer runs: every run replaces all 26 FFNs. Each run splits its tokens 25% local matching, 75% end to end, and reads every token once (one epoch). Stage 2 starts again from ridge init; it does not reuse Stage 1's projections.

| Step | Command | Tokens | What to check before going on |
| --- | --- | --- | --- |
| Stage 1: test run | `bash scripts/stage1_test_run.sh` | 1M (0.25M local matching, 0.75M end to end) | It finishes on the M5 with peak memory under the limit memcheck printed; local-matching loss falls below 1 (1 = as bad as deleting the FFN); end-to-end loss falls; nothing turns NaN; routing entropy stays at or above 0.7; resume works; `samples.md` is written. Its quality numbers are a sanity check, not the result. |
| Stage 2: full run | `bash scripts/stage2_full_run.sh` | 20M (5M local matching, 15M end to end) | `GR_CE`, perplexity ratio and accuracy on the validation mix, WikiText-103 and C4; the subset-evaluation curve in `metrics.jsonl`; `samples.md` against Gemma's. |
| Stage 2, short form | `bash scripts/stage2_full_run.sh --config configs/stages/stage2_5m.yaml` | 5M, all end to end; reuses Stage 1's ridge init; evaluation as in Stage 1 | Same as Stage 2; about 12 hours at 119 tokens/s |
| Stage 2: more tokens | `bash scripts/stage2_extend.sh 20000000` | +20M per call (any number); keeps the settings of the run it extends | Same as Stage 2. |
| Stage 3: 20M in total | `bash scripts/stage3_run.sh` | +15M on top of Stage 2's 5M, all end to end; continues Stage 2's projections and data stream; Stage 2's settings (`configs/stages/stage3_20m.yaml` lists the few that change) | Speed check, training, final evaluation, then the Q/A test of every stage (`results/qa_compare.md`). |

**When to add tokens.** Extend when the last subset evaluations of Stage 2 are still improving: end-to-end loss still falling, `GR_CE` and accuracy still rising. Stop when they flatten. An extension starts from the finished run's projections and continues its data stream, so no token is read twice; it is end to end only, with a fresh AdamW state and a new cosine schedule (2% warmup) over the added tokens. Each extension is its own run folder, named by the total tokens trained (`runs/stage2_full_40M`, `runs/stage2_full_60M`, …), so earlier results stay intact.

**Stage 3 and the micro-batch.** `scripts/stage3_run.sh` first measures end-to-end steps with micro-batches of 1, 2, 4 and 8 sequences on this Mac (`transplant memcheck --e2e-only` in a fresh process for each, about 10 to 15 minutes, once) and keeps the fastest that stays 0.75 GB below the GPU memory macOS recommends (`runs/stage3_20m/speed.json`; `scripts/stage3_speed.py`). Gradient accumulation goes down as the micro-batch goes up, so every optimizer step still averages over 16 sequences of 1,024 tokens: the training is the same (a test checks that 2 × 1 and 1 × 2 give the same projections) and only the speed changes. The speed comes from the experts: with one sequence per micro-batch each of the 60 experts gets about 70 tokens per layer, too few to keep the Apple GPU busy between the per-layer routing syncs.

**Pausing and interruptions (Stage 3).** Run `bash scripts/stage3_run.sh` (it writes its own log, `notes/stage3_console.log`; no `| tee` needed).

| What happens | What the run does | What you do |
| --- | --- | --- |
| Ctrl-C, or `touch runs/stage3_20m/PAUSE` from another terminal | Stops at once, saves a checkpoint at the last finished optimizer step (a few seconds) and exits; a step is never cut in half | Run the same command: it continues from that step with the same data |
| Ctrl-C during a held-out check or the final evaluation | The cut check runs again on resume; the final evaluation keeps every finished set (`eval_partial.json`) | Run the same command |
| Lid closed (sleep) | Frozen; carries on when the Mac wakes | Nothing |
| Terminal window closed, `kill`, logout or shutdown | SIGHUP or SIGTERM: same as Ctrl-C | Run the same command |
| Power cut or crash | At most the last 10 minutes are lost (checkpoint every 10 minutes, flushed to disk; the previous one is kept in case the newest is damaged) | Run the same command |
| An error, for example the network drops while data streams | Checkpoint at the last finished step; the script resumes by itself up to 5 times, a minute apart | Nothing, or run the same command later |
| Running the command while the old process is still saving | Refused: one process per run folder (`runs/stage3_20m/.lock`) | Wait a few seconds |

`python scripts/progress.py runs/stage3_20m` shows whether the run is going or paused, and the step of the last checkpoint. A test stops a run with Ctrl-C, by a PAUSE file, during a check, during the final evaluation and by an error, and checks that the resumed run ends with the same projections as one never stopped.

Single commands, for anything the scripts do not cover (`uv run` uses the project environment):

```bash
uv run python -m transplant train --config configs/base.yaml --config configs/stages/stage2_full.yaml \
    --set run.name=stage2_lr1e-4 --set e2e.lr=1.0e-4   # a variant of Stage 2
uv run python -m transplant train ... --resume         # continue a stopped run
uv run python -m transplant eval  ... --routing-agreement  # re-evaluate; agreement with MobileMoE's own routing and CKA
uv run python -m transplant summarize                  # one line per run and evaluation set from results/runs.csv
python scripts/progress.py runs/stage3_20m --watch     # a run's progress, speed and time left (safe while it trains)
bash scripts/ask_stages.sh                             # Q/A test, every stage: Gemma, hybrid, No FFN, then side by side
uv run transplant ask --run runs/stage2_5m --ablated  # Q/A test for one run: Gemma, hybrid and No FFN
uv run transplant ask-compare runs/stage1_test runs/stage2_5m  # stages side by side, from their qa.json
uv run transplant ask --run runs/stage1_test -i        # type your own questions (--raw: continue text as is)
```

`ask` is a question-and-answer test: 30 basic questions (geography, science, math, everyday, history), each answered in one full sentence. Both models are base models, not chat models, so every question follows three solved examples with one-sentence answers ("Q: What is the capital of Italy? A: The capital of Italy is Rome."), rotating from a pool of six so that copying the last example cannot score. It reports, for Gemma, the hybrid and No FFN (Gemma with all 26 FFNs deleted, `--ablated`):

- **correct:** the reply contains the key answer as a whole word ("The capital of France is Paris." contains "Paris");
- **fact log-prob:** the mean log-probability per token of the fact after the reference sentence's opening ("The capital of France is" → "Paris"). It isolates factual recall from phrasing and moves long before replies turn right, so it is the number to follow from Stage 1 to Stage 2 to Stage 3;
- **fact recovery:** where the hybrid's fact log-prob sits between No FFN (0%) and Gemma (100%), the question-answering counterpart of GR_CE;
- **sentence log-prob** of the whole reference sentence, and how many replies only **copied** an example or the question.

Results go to `qa.md` and `qa.json` in the run folder (`ask.md` / `ask.json` are from the earlier one-word test). `ask-compare` reads the `qa.json` of several runs (no model is loaded) and writes `results/qa_compare.md`: a summary row for No FFN, each stage's hybrid and Gemma, the fact log-prob of every question per stage, and every reply. `scripts/ask_stages.sh` does both for `runs/stage1_test`, `runs/stage2_5m` and any `runs/stage3*`, and skips runs whose `qa.json` is already up to date. Your own questions: `--questions file.txt`, one per line as `question | reference with [fact] | alias`.

Configs compose left to right (`--config` repeatable) and `--set key.sub=value` overrides anything. The knobs you are most likely to touch:

| Key | Default | Meaning |
| --- | --- | --- |
| `run.tokens` | 1M (Stage 1), 20M (Stage 2) | training tokens in this run |
| `run.local_share` | 0.25 | share of them used for local matching |
| `run.extend_from` | null | a finished run to continue (what `stage2_extend.sh` sets) |
| `e2e.lr`, `local.lr` | 3e-4, 1e-3 | peak learning rates |
| `e2e.beta_kl` | 1.0 | weight of the KL term to Gemma |
| `e2e.grad_accum`, `local.grad_accum` | 16, 4 | tokens per step: 16,384 end to end, 8,192 local matching |
| `eval.final_tokens` | 1M | tokens per evaluation set at the end of a run (null = whole set) |
| `run.seed` | 0 | seed for the projections and the data subsample |

## What a run writes

`runs/<name>/`:

| File | Contents |
| --- | --- |
| `samples.md` | Greedy continuations of five prompts (fact, science, code, math, history), Gemma beside the hybrid |
| `qa.md`, `qa.json` | Written by `transplant ask`: one-sentence replies to 30 basic questions from Gemma, the hybrid and No FFN; hits, fact and sentence log-probabilities, fact recovery |
| `eval.json` | Final evaluation on every held-out set |
| `summary.json` | Tokens trained in this run and in total, steps, tokens/s, peak memory, the evaluation after local matching, every subset evaluation |
| `metrics.jsonl` | Every 10 steps: loss (per layer in local matching), CE, KL, accuracy, gradient norm, learning rate, tokens/s, peak memory; plus the subset evaluations |
| `projections.pt` | The trained W_in and W_out for all 26 layers (about 245 MB) |
| `run_info.json`, `config.yaml` | Config and its hash, git commit, data sources, held-out head counts, seed, precision, device |
| `init.pt`, `init.json` | Ridge solution, chosen λ per layer, validation error, aligned share |
| `local_projections.pt`, `*_ckpt.pt`, `*_data_state.pt` | Checkpoints and data-stream positions for resuming and extending |

One row per evaluation set goes to `results/runs.csv`.

Every evaluation reports, on the same sequences, for Gemma, the ablated model (all 26 FFNs deleted) and the hybrid:

* perplexity, and `ppl_hybrid / ppl_A`;
* gap recovery in loss space, `GR_CE`, the headline number (1 = as good as Gemma, 0 = as bad as deleting the FFNs). The perplexity version `GR` is logged too, but it saturates near 1 when all 26 FFNs are deleted;
* next-token accuracy: the share of held-out positions where the most likely next token is the right one;
* routing health per layer: normalized load entropy (alarm below 0.7), largest expert share, dead experts.

## Tests

```bash
uv run pytest -q                                              # tiny stand-in models, CPU, ~20 s
TRANSPLANT_REAL=1 uv run pytest tests/test_real_models.py -v  # real models on the M5 (part of m0_checks.sh)
```

The seven gating tests run on tiny stand-ins under `pytest` and on the real models in `m0_checks.sh`: (1) no-op surgery reproduces Gemma, (2) the extracted MoE block reproduces MobileMoE's own output, (3) with transplants switched off the hybrid is exactly Gemma, (4) only projection parameters receive gradients, (5) real all-26-layer training steps, measured in a fresh process, peak below the GPU working set macOS recommends, (6) chunked cross-entropy and KL, with the hand-written backward, match plain autograd in value and gradient, (7) aligned positions cover the same character spans. Further tests cover ridge, the data mixture (token shares, determinism, exact resume, held-out disjointness), resuming a run from its checkpoint, and extending a finished run without reading a token twice.
