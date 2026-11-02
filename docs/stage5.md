# Stage 5: the random-expert control (did MobileMoE-S's experts contribute?)

Stage 4 recovers most of what Gemma 3 1B loses when 20 of its FFNs are deleted, with MobileMoE-S's frozen
MoE blocks in those slots and only W_in / W_out trained. It does as well as Stages 1-3 with a donor 11 MMLU
points weaker, which raises the question this stage answers: **is the recovered quality coming from
MobileMoE-S's pretrained experts, or would any frozen block of the same shape, with the same trained
projections around it, do as well?**

Stage 5 repeats Stage 4 with one change: the 20 MoE blocks keep MobileMoE-S's architecture, but every
weight in them is random, and frozen. The difference between the two runs is what the pretrained experts
add:

$$
\text{contribution} = \Delta_{\mathrm{random}} - \Delta_{\mathrm{pretrained}}, \qquad \Delta = \mathrm{CE}_{\mathrm{hybrid}} - \mathrm{CE}_{\mathrm{Gemma}}
$$

## What stays, what changes

| | Stage 4 | Stage 5 |
| --- | --- | --- |
| Host, layout | Gemma 3 1B, `spread`: 20 FFNs replaced, 6 kept (2, 6, 10, 15, 19, 23) | same |
| Trained | W_in (1152 → 768), W_out (768 → 1152) in 20 layers: 35,389,440 | same |
| MoE block | MobileMoE-S-Base's own, frozen | MobileMoE-S's architecture, **random weights**, frozen |
| Init | ridge: W_in to MobileMoE-S's MoE input, W_out through the block to Gemma's FFN output | **Stage 4's W_in bit for bit**; W_out by the same ridge fit, through the random block |
| Data | the MobileMoE pre-training mix, 15M tokens from the start of the stream, each once | the same tokens in the same order (`run.seed` 0) |
| Recipe | CE + 1.0 · KL to Gemma, AdamW, 3e-4 cosine, 2% warmup, 915 steps of 16,384 tokens, micro-batch 8 × 2 | same |
| Evaluation | 9 checks on 100k held-out tokens; final 250k per set; Q/A test | same |

`tests/test_stage5.py` checks that the two stage configs differ in nothing but the run name, the inner
block, the random draw and the init.

## The random blocks

Each block is a copy of MobileMoE-S's block at that layer, the same classes and the same bucketed forward as
Stage 4, with every tensor redrawn (`randomize_like(..., full=True)` in `src/transplant/models/moe_source.py`):

| Tensor | Shape | Random version |
| --- | --- | --- |
| Routed experts, gate half of `gate_up_proj` | 60 × 768 × 384 | i.i.d. N(mean, std) of the pretrained gate half |
| Routed experts, up half of `gate_up_proj` | 60 × 768 × 384 | i.i.d. N(mean, std) of the pretrained up half |
| Routed experts, `down_proj` | 60 × 384 × 768 | i.i.d. N(mean, std) of the pretrained tensor |
| Shared expert, `gate_proj` / `up_proj` / `down_proj` | 1536 × 768, 1536 × 768, 768 × 1536 | each i.i.d. N(mean, std) of its pretrained tensor |
| Router | 60 × 768 | i.i.d. N(mean, std) of the pretrained router |
| Selection bias | 60 | 0 (a fresh MobileMoE starts at 0; a pretrained bias would tilt random routing) |
| Norm gain N_B | 768 | one constant: the RMS of the pretrained gain (input scale kept, per-channel pattern dropped) |

Only the scale of each tensor is taken from MobileMoE-S (two numbers per matrix); no learned value remains.
The scale matters here: MobileMoE-S's router has a standard deviation of 0.07-0.11, five times a standard
0.02 init, so a default init would change how sharply routing picks experts, not only what the experts know.
Routing stays MobileMoE-S's: sigmoid scores, top-4, the chosen four normalized and scaled by 2.5, the shared
expert always on.

The draw is seeded per layer (`transplant.random_moe.seed`, 0) and computed on the CPU in fp64, so ridge init,
training, `transplant eval` and `transplant ask` all rebuild the same blocks, bit for bit, on any device. A
fingerprint (SHA-256 over every block tensor) is written to the ridge `init.json` and the run's
`run_info.json`; `scripts/verify_stage5_random.py` checks it.

## Runs

| Step | Command | Time on the M5 |
| --- | --- | --- |
| Tests (before training) | `bash scripts/stage5_test.sh` | about 10 min |
| The run: ridge init, 15M tokens, final evaluation, validation | `bash scripts/stage5_run.sh` | about 14 h (twice that with macOS Low Power Mode on) |
| Validation of a finished run | `bash scripts/stage5_validate.sh [runs/stage5_random_15m]` | about 10 min |
| Report only (reads run files) | `.venv/bin/python scripts/stage5_compare.py` | seconds |
| Block checks on the real models | `.venv/bin/python scripts/verify_stage5_random.py runs/stage5_random_15m` | about 3 min |

Stage 4 must have finished first (`runs/stage4_spread_15m`, `runs/stage4_spread_init/init.pt`). The run
reuses Stage 4's micro-batch (`runs/stage4_spread_15m/speed.json`, 8 × 2), so both runs split every
optimizer step the same way. It stops cleanly on Ctrl-C or `touch runs/stage5_random_15m/PAUSE` and resumes
with the same command; the console goes to `notes/stage5_console.log`. Progress, in another terminal:
`.venv/bin/python scripts/progress.py runs/stage5_random_15m --watch`.

A second random draw, without changing the data: train with a new run name and
`--set transplant.random_moe.seed=1 --set transplant.init_path=runs/stage5_random_init_s1/init.pt`.

## What a run writes

* `runs/stage5_random_init/init.pt`, `init.json`: Stage 4's W_in, W_out fitted through the random blocks,
  the fingerprint of the draw.
* `runs/stage5_random_15m/`: `eval.json`, `summary.json`, `samples.md`, `metrics.jsonl`, `run_info.json`
  (with `inner` and `random_moe`), `qa.md` / `qa.json`, `verify_random.txt`.
* `results/stage5_compare.md`: the verdict, then text (CE gap per set, accuracy), the learning curves of
  both runs at the same steps, ridge W_out through each kind of block, the Q/A test (with a per-question
  sign test), routing and run cost.

## Reading the verdict

The decision margin is Stage 4's: 0.10 nats of CE gap on the validation mix, final evaluation.

| Random minus pretrained (mix) | Reading |
| --- | --- |
| ≥ +0.10 | The pretrained experts contributed: part of Stage 4's recovery comes from what MobileMoE-S learned. |
| between −0.10 and +0.10 | No clear contribution: the recovery comes mainly from Gemma (attention, embeddings, the 6 kept FFNs) and the trained projections around a frozen nonlinear block. |
| ≤ −0.10 | Random experts did better: MobileMoE-S's weights did not help this hybrid. |

WikiText-103 and C4 (text unlike the training mix) and the Q/A test are reported beside it; with 30
questions, read the fact log-probability and the sign test rather than the count of correct answers.

**Result (1 Nov 2026): +0.294 nats on the mix** (+0.326 WikiText-103, +0.301 C4; the right fact more likely
with the pretrained experts for 29 of 30 questions): the pretrained experts contributed. Numbers in
`results/stage5_compare.md`; what they show and do not show in section 7 of the README.

## Before training

Ridge init gives a first comparison, per layer and linear only: how well can a linear readout of each block's
output reproduce Gemma's own FFN output (validation NMSE, 0 = exact, 1 = no better than zero)? Mean over the
20 layers: **0.312 through MobileMoE-S's pretrained blocks, 0.312 through the random blocks**, with per-layer
differences of both signs (`results/stage5_compare.md`, section 3). Whether that holds end to end, after
training through all 20 layers at once, is what the run measures.

## Code

* `transplant.inner: random_moe_b` with `transplant.random_moe: {seed, full}`; `full: false` keeps control C5
  as it was first written (matrices redrawn, selection bias and norm gain kept).
* `ridge.w_in_from`: take W_in from another ridge init, fit only W_out (no MobileMoE forward pass needed).
* A ridge init is reused only for the blocks it was fitted through (`inner` and the random draw, from its
  `init.json`), so Stage 4's init is never picked up by Stage 5, nor overwritten.
* `tests/test_stage5.py` (tiny models; one test on the Apple GPU), `scripts/verify_stage5_random.py` (real models).
