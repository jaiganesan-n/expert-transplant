# Expert Transplant (Project J'AI)

Training, testing and evaluation code for one question: **can a frozen Mixture-of-Experts layer from one pretrained language model replace the feed-forward network (FFN) of another, independently trained model, if only two small projection matrices per layer are trained?**

The host is Gemma 3 1B (dense, Model A). The donor is MobileMoE-M-Base (MoE, Model B). In all 26 layers, Gemma's FFN is removed and MobileMoE's MoE block takes its place between two trainable projections. Both models stay frozen. That replacement is the only change to Gemma. Everything trains and runs on a MacBook Air M5 (PyTorch MPS, bf16), with packages installed by [uv](https://docs.astral.sh/uv/).

![The hybrid end to end](docs/figures/fig1_end_to_end.png)

## Objective

Show whether pretrained MoE layers are functionally portable between transformers that never trained together. The project answers this before it looks at any benchmark:

| ID | Hypothesis | Measured by | Null result |
| --- | --- | --- | --- |
| H1 | With all 26 of Gemma's FFNs replaced by MobileMoE's MoE blocks, the hybrid recovers most of what Gemma loses when its FFNs are deleted | `GR_CE` on the held-out mix (proposed bar: ≥ 0.5), with perplexity ratio and next-token accuracy beside it | `GR_CE` near 0: the hybrid is no better than Gemma with its FFNs deleted |
| H3 | Inside the hybrid, routing stays diverse and agrees with MobileMoE's own routing above chance, on positions where both tokenizers align | Routing health on every evaluation; agreement with `eval --routing-agreement` | Router collapse (entropy below 0.7), or chance-level agreement (0.039) |

Knowledge benchmarks (MMLU, TriviaQA, NaturalQuestions and the rest) come only after the architecture is shown to work. See [Later, once the architecture works](#later-once-the-architecture-works).

## Method

Gemma 3 wraps its FFN in two norms: N_A before it, P_A after it. The transplant swaps only the module between them:

$$
h_{out} = h + P_A\Big(W_{out}\, \mathrm{MoE}_B\big(N_B(W_{in}\, N_A(h))\big)\Big)
$$

* **W_in** (1152 → 1024) maps Gemma's state into MobileMoE's space; **W_out** (1024 → 1152) maps it back. Each layer has its own pair: 2.36M parameters per layer, 61.3M for all 26. Nothing else trains.
* **N_B and MoE_B** are MobileMoE's frozen pre-MoE RMSNorm, router (sigmoid top-4 of 60), 60 routed experts and the always-on shared expert of the matching layer. Both models have 26 layers, so Gemma layer i borrows MobileMoE layer i.
* **Initialization: ridge regression.** W_in maps Gemma's N_A(h) to MobileMoE's own MoE-block input on the same calibration text (positions where both tokenizers end a token at the same character). W_out maps the MoE output to Gemma's own FFN output.
* **Local matching** (first 25% of a run's tokens): every layer learns to imitate Gemma's FFN with a normalized MSE. One Gemma pass feeds all 26 layers.
* **End to end** (the other 75%): next-token cross-entropy through Gemma's head plus β·KL to the original Gemma (β = 1), computed 256 positions at a time with a hand-written backward pass, because of Gemma's 262,144-token vocabulary.
* **Optimizer: AdamW**, betas (0.9, 0.95), eps 1e-8, gradient clipping 1.0. Gemma 3's technical report and model card name no optimizer, so the repo uses the one the plan's other dense models were pre-trained with: AdamW with betas (0.9, 0.95) in the Granite 3.0 report, AdamW in the Llama 3 paper.

![One transplanted layer](docs/figures/fig2_replace_layer.png)

More detail, with every equation: [docs/architecture.md](docs/architecture.md). Figures 3 and 4 show the routing of one token and the training pipeline.

## Models

| Role | Hugging Face id | Layers × width | FFN or experts | Tokenizer | License |
| --- | --- | --- | --- | --- | --- |
| A (host, dense) | `google/gemma-3-1b-pt` | 26 × 1152 | GeGLU, hidden 6912 | Gemma 3, 262,144 | Gemma Terms of Use (gated) |
| B (donor, MoE) | `facebook/MobileMoE-M-Base` | 26 × 1024 | 60 routed (hidden 512), top-4 sigmoid, + 1 shared (hidden 2048) | Llama-3, 128,256 | FAIR Noncommercial Research (gated) |

MobileMoE needs `trust_remote_code=True`; its card verifies transformers 4.57.6 and torch 2.8.0, which `uv.lock` pins. Its module names were not public, so `uv run transplant inspect` found them at M0 (6 Oct 2026). They are pinned in `configs/pairs/primary.yaml`, together with the inspected commit of its custom code.

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

**Time.** Speed on the M5 is unknown until Stage 1 measures it, and Stage 1's script then prints how long Stage 2 will take at that speed. Planning guess: 300 to 800 tokens/s end to end and faster for local matching, so Stage 1 takes about 1 to 2 hours with ridge init and evaluation, and Stage 2 about 7 to 16 hours (run it overnight). Checkpoints every 200 steps or 30 minutes; if the Mac sleeps or the run stops, run the same script again and it resumes exactly (projections, optimizer, schedule, RNG and data-stream position). A finished run is skipped. Gemma's and the ablated model's held-out losses are computed once and cached in the held-out folder.

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

## Workflow: two stages, all 26 layers

There are no single-layer runs: every run replaces all 26 FFNs. Each run splits its tokens 25% local matching, 75% end to end, and reads every token once (one epoch). Stage 2 starts again from ridge init; it does not reuse Stage 1's projections.

| Step | Command | Tokens | What to check before going on |
| --- | --- | --- | --- |
| Stage 1: test run | `bash scripts/stage1_test_run.sh` | 1M (0.25M local matching, 0.75M end to end) | It finishes on the M5 with peak memory under the limit memcheck printed; local-matching loss falls below 1 (1 = as bad as deleting the FFN); end-to-end loss falls; nothing turns NaN; routing entropy stays at or above 0.7; resume works; `samples.md` is written. Its quality numbers are a sanity check, not the result. |
| Stage 2: full run | `bash scripts/stage2_full_run.sh` | 20M (5M local matching, 15M end to end) | `GR_CE`, perplexity ratio and accuracy on the validation mix, WikiText-103 and C4; the subset-evaluation curve in `metrics.jsonl`; `samples.md` against Gemma's. |
| Stage 2: more tokens | `bash scripts/stage2_extend.sh 20000000` | +20M per call (any number) | Same as Stage 2. |

**When to add tokens.** Extend when the last subset evaluations of Stage 2 are still improving: end-to-end loss still falling, `GR_CE` and accuracy still rising. Stop when they flatten. An extension starts from the finished run's projections and continues its data stream, so no token is read twice; it is end to end only, with a fresh AdamW state and a new cosine schedule (2% warmup) over the added tokens. Each extension is its own run folder, named by the total tokens trained (`runs/stage2_full_40M`, `runs/stage2_full_60M`, …), so earlier results stay intact.

Single commands, for anything the scripts do not cover (`uv run` uses the project environment):

```bash
uv run python -m transplant train --config configs/base.yaml --config configs/stages/stage2_full.yaml \
    --set run.name=stage2_lr1e-4 --set e2e.lr=1.0e-4   # a variant of Stage 2
uv run python -m transplant train ... --resume         # continue a stopped run
uv run python -m transplant eval  ... --routing-agreement  # re-evaluate; agreement with MobileMoE's own routing and CKA
uv run python -m transplant summarize                  # one line per run and evaluation set from results/runs.csv
```

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

## Repository layout

```
expert-transplant/
├── README.md
├── LICENSE, THIRD_PARTY_LICENSES.md
├── pyproject.toml, uv.lock, .python-version   # uv; Python 3.12, torch 2.8.0, transformers 4.57.6
├── configs/
│   ├── base.yaml                               # every default, commented
│   ├── stages/       stage1_test.yaml (1M tokens), stage2_full.yaml (20M tokens)
│   ├── pairs/        primary.yaml (Gemma + MobileMoE, all 26 layers), tiny.yaml (tests)
│   ├── data/         mobilemoe_pretrain_mix.yaml (Table B.1)
│   └── hardware/     mac_m5.yaml (MPS, bf16)
├── src/transplant/
│   ├── models/   load.py, moe_source.py (MobileMoE block discovery), surgery.py (the FFN replacement),
│   │             projection.py, layer_map.py, align.py, capture.py, tiny.py
│   ├── data/     sources.py, mixture.py, pack.py, heldout.py
│   ├── init/     ridge.py
│   ├── train/    local_match.py, end_to_end.py, losses.py, optim.py, checkpoint.py, run.py
│   ├── eval/     perplexity.py, metrics.py, routing.py, cka.py, samples.py, baselines.py, evaluate.py
│   ├── utils/    device.py, memory.py, logging.py, seed.py
│   ├── tools.py  # M0 checks
│   └── cli.py    # python -m transplant <command>
├── scripts/      setup.sh, m0_checks.sh, stage1_test_run.sh, stage2_full_run.sh, stage2_extend.sh, env.sh
├── tests/
├── docs/         architecture.md, figures/ (4 figures, 13 equation cards, their sources)
└── results/      runs.csv
```

## Later, once the architecture works

* Controls: a random-weight MoE in the same slot (`transplant.inner=random_moe_b`) and Gemma's own FFN behind the same 1024-wide bottleneck (`transplant.inner=a_ffn_bottleneck`), to show the result comes from MobileMoE's trained experts. Both are implemented; run them as Stage 2 variants.
* More seeds of Stage 2 (`run.seed=1`, `run.seed=2`; each seed trains on a different 7-in-8 subsample of every source).
* Knowledge and reasoning benchmarks through lm-evaluation-harness (MMLU, TriviaQA, NaturalQuestions, ARC-Challenge, HellaSwag, GSM8K), with the benchmark-decontamination filter and the LoRA control that goes with them.
* Other model pairs: the code is pair-agnostic, but only the primary pair has a config.

## Licenses

Code: Apache 2.0 (`LICENSE`). Models and datasets keep their own terms, summarized in `THIRD_PARTY_LICENSES.md`: MobileMoE is non-commercial research only, Gemma has its own terms, and Natural Reasoning is CC-BY-NC-4.0. Trained projections are derived from both models; share them only as both licenses allow.
