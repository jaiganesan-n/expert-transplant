# Stage 4: partial transplant with MobileMoE-S (layout spread)

Gemma 3 1B keeps the FFNs of 6 layers (2, 6, 10, 15, 19, 23). Each of the other 20 has its FFN replaced by a
MobileMoE-S-Base MoE block, every MobileMoE-S layer used once and placed at the same relative depth
(MobileMoE-S layer j → Gemma layer round(25j/19)). Only the projections train: W_in (1152 → 768) and
W_out (768 → 1152) in the 20 transplanted layers, 35,389,440 parameters (Stages 1–3: 61,341,696).

![Stage 4 architecture](figures/stage4/stage4_architecture.png)

![Stage 4 equations](figures/stage4/stage4_equations.png)

Figures: `python docs/figures/src/make_stage4_figures.py` redraws both.

## Runs

| Step | Command | Tokens | Matched to |
| --- | --- | --- | --- |
| Tests (before training) | `bash scripts/stage4_test.sh` | – | – |
| 15M, one run | `bash scripts/stage4_run.sh 15m` | 15M from ridge init (915 steps) | the all-26 curve of Stages 2 then 3: the same tokens in the same order |
| Validation of a run | `bash scripts/stage4_validate.sh runs/<run> [--agreement]` | – | – |

Data and recipe are those of Stages 2–3: the MobileMoE pre-training mix, read from the start of the stream
in the same order, each token once; ridge init (recomputed for MobileMoE-S on the 20 layers, 400
calibration documents, in `runs/stage4_spread_init/`), no local matching, then end to
end with CE + 1.0 · KL to unmodified Gemma (all 26 FFNs), AdamW, peak LR 3e-4, cosine with 2% warmup,
16,384 tokens per step. A speed check picks the micro-batch on the Mac before the run.

Every run stops cleanly on Ctrl-C or `touch runs/<run>/PAUSE` and resumes with the same command; the
console goes to `notes/stage4_console.log`.

## What a run writes

* `runs/<run>/eval.json`, `summary.json`, `samples.md`, `metrics.jsonl`, `run_info.json` (with
  `kept_ffn_layers` and the MobileMoE-S commit used), `qa.md` / `qa.json`.
* `results/stage4_compare.md`: Stage 4 against Stages 1–3. Text (CE gap to Gemma, perplexity, accuracy),
  matched-token checks, GR_CE against two floors (all 26 FFNs deleted, and the layout floor: only the 20
  transplanted FFNs deleted), the Q/A test with fact recovery on both scales, routing health and, after
  `--agreement`, routing agreement with MobileMoE-S's own routing.

## Code

* `pair.layers: spread` (`models/layer_map.py: spread_layers`) chooses the layers; `transplant layout`
  prints the map and the parameter count from a config without loading a model.
* `configs/pairs/stage4_mobilemoe_s.yaml`: MobileMoE-S (20 × 768, 60 routed experts with FFN 384, top-4,
  1 shared expert with FFN 1536). `configs/stages/stage4_spread_15m.yaml`: the run.
* A ridge init made for other layers or another Model B is never reused (shapes are checked).
* `tests/test_stage4.py` (tiny models) and `tests/test_real_models.py` with
  `TRANSPLANT_REAL_CONFIG=configs/base.yaml,configs/stages/stage4a_spread_1m.yaml` (real models).
