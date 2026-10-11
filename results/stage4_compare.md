# Stage 4 report: Gemma 3 1B + MobileMoE-S (layout spread) against Stages 1-3

## 1 · Text (final evaluation, 250k held-out tokens per set)

| Run | FFNs replaced | Trainable | Tokens | Mix ppl (× Gemma) | CE gap mix | CE gap WikiText-103 | CE gap C4 | GR_CE mix, 26-deleted floor | GR_CE mix, layout floor | Accuracy mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 1B (target) | 0 of 26 | – | – | 10.25 (1.00) | 0 | 0 | 0 | 1 | 1 | 52.6% |
| Stage 1 (`stage1_test`) | 26 of 26 | 61.3M | 0.98M | 71.28 (6.95) | 1.939 | 2.502 | 2.076 | 0.852 | – | 31.0% |
| Stage 2 (`stage2_5m`) | 26 of 26 | 61.3M | 5.00M | 33.96 (3.31) | 1.198 | 1.794 | 1.486 | 0.909 | – | 38.5% |
| Stage 3 (`stage3_20m`) | 26 of 26 | 61.3M | 19.99M | 23.50 (2.29) | 0.829 | 1.406 | 1.154 | 0.937 | – | 42.5% |
| Stage 4 (15M) (`stage4_spread_15m`) | 20 of 26 | 35.4M | 14.99M | 23.17 (2.26) | 0.815 | 1.338 | 1.103 | 0.938 | 0.955 | 42.5% |
| All 26 FFNs deleted (floor of Stages 1-3) | – | – | – | 4,957,444 | 13.09 | 13.08 | 12.31 | – | – | – |
| Layout floor: the 20 transplanted FFNs deleted, 6 kept | – | – | – | 913,472,787 | 18.31 | 19.29 | 17.88 | – | – | – |

CE gap: hybrid minus Gemma, in nats (0 = Gemma; the headline). GR_CE against the 26-deleted floor compares with Stages 1-3; against the layout floor it is Stage 4's own.

Note: the layout floor is worse than deleting all 26 FFNs (the 6 kept FFNs, fed a residual stream the 20 deleted ones no longer shape, make it worse), so GR_CE and fact recovery against it overstate the recovery; read them against the 26-deleted floor.

## 2 · At matched tokens (same text, same recipe)

**Stage 4 (15M) vs all 26 layers (Stages 2 then 3, same tokens in the same order)**

| Tokens | All 26, Stages 2-3 (interpolated) CE gap | Stage 4 (15M) CE gap | Difference |
| --- | --- | --- | --- |
| 1.64M (check, 100k tokens) | 1.603 | 1.386 | -0.218 |
| 3.28M (check, 100k tokens) | 1.327 | 1.194 | -0.133 |
| 4.92M (check, 100k tokens) | 1.232 | 1.067 | -0.166 |
| 6.55M (check, 100k tokens) | 1.262 | 0.997 | -0.265 |
| 8.19M (check, 100k tokens) | 1.218 | 0.951 | -0.267 |
| 9.83M (check, 100k tokens) | 1.073 | 0.896 | -0.177 |
| 11.47M (check, 100k tokens) | 1.023 | 0.874 | -0.149 |
| 13.11M (check, 100k tokens) | 0.958 | 0.859 | -0.099 |
| 14.75M (check, 100k tokens) | 0.923 | 0.851 | -0.073 |

Final evaluation, 250k tokens per set (CE gap):

| Set | Stage 2 (5.0M, all 26) | Stage 3 (20.0M, all 26) | Stage 4 (15M) (15.0M) |
| --- | --- | --- | --- |
| mix | 1.198 | 0.829 | 0.815 |
| wikitext103 | 1.794 | 1.406 | 1.338 |
| c4 | 1.486 | 1.154 | 1.103 |

At 14.75M tokens the partial transplant is within 0.1 nats of all 26 layers (no clear difference) on the same text (CE gap 0.851 vs 0.923), with 35.4M trainable parameters against 61.3M. Its single learning-rate schedule differs from the two of Stages 2 and 3.

Against its layout floor: 17.49 nats better on the mix (clears the 0.1-nat margin).

## 3 · Q/A test (30 questions, one-sentence answers)

| Model | Correct | Fact log-prob | Fact recovery (26 deleted = 0%) | Fact recovery (layout floor = 0%) | Sentence log-prob | Copied |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 1B | 28/30 | -0.23 | 100% | 100% | -0.28 | – |
| Stage 1 hybrid | 1/30 | -5.33 | 65% | – | -2.29 | 11/30 |
| Stage 2 hybrid | 1/30 | -3.18 | 80% | – | -1.22 | 0/30 |
| Stage 3 hybrid | 6/30 | -1.90 | 89% | – | -0.92 | 0/30 |
| Stage 4 (15M) hybrid | 10/30 | -1.82 | 89% | 92% | -0.83 | 0/30 |
| All 26 FFNs deleted | 0/30 | -14.81 | 0% | – | -12.91 | – |
| Layout floor (20 deleted, 6 kept) | 0/30 | -20.16 | -37% | 0% | -16.46 | – |

Fact log-prob: mean log-probability per token of the fact after the reference sentence's opening. Run `bash scripts/stage4_validate.sh <run>` for a Stage 4 run without qa.json.

## 4 · Routing (final evaluation)

| Run | MoE layers | Lowest entropy (layer) | Mean entropy | Dead experts | Busiest expert (layer) | Agreement with Model B's own routing |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 1 | 26 | 0.856 (2) | 0.940 | 91 of 1560 | 8.3% (4) | – |
| Stage 2 | 26 | 0.829 (1) | 0.959 | 45 of 1560 | 14.2% (1) | – |
| Stage 3 | 26 | 0.806 (1) | 0.961 | 31 of 1560 | 15.6% (1) | – |
| Stage 4 (15M) | 20 | 0.724 (0) | 0.954 | 29 of 1200 | 18.3% (0) | – |

Entropy alarm: 0.7. Layer numbers are Gemma's. Agreement (mean Jaccard of the top-4 sets, chance 0.039) appears after `bash scripts/stage4_validate.sh <run> --agreement`.

