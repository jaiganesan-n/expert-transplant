# Stage 5 report: random experts against MobileMoE-S's pretrained experts (Stage 4)

## Verdict

**The pretrained experts contributed.** With random experts of the same architecture the hybrid is 0.29 nats worse on the mix (1.34 × the perplexity), with the same trained parameters, data, recipe and starting W_in. Part of Stage 4's recovery comes from what MobileMoE-S's experts learned in pre-training.

Contribution on text unlike the training mix (random minus pretrained, CE gap in nats): wikitext103 +0.326, c4 +0.301.

Facts: the right fact is more likely with the pretrained experts for 29 of 30 questions (sign test p = 5.77e-08); mean fact log-prob -1.82 (pretrained) vs -2.93 (random); correct 10/30 vs 3/30.

## 1 · Text (final evaluation, 250k held-out tokens per set)

| Model | Mix ppl (× Gemma) | CE gap mix | CE gap WikiText-103 | CE gap C4 | Accuracy mix | GR_CE mix (26 deleted = 0) |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 3 1B | 10.25 (1.00) | 0 | 0 | 0 | 52.6% | 1 |
| Stage 4, pretrained experts (`stage4_spread_15m`) | 23.17 (2.26) | 0.815 | 1.338 | 1.103 | 42.5% | 0.938 |
| Stage 5, random experts (`stage5_random_15m`) | 31.10 (3.03) | 1.110 | 1.665 | 1.404 | 39.6% | 0.915 |
| **Random minus pretrained** | × 1.34 | +0.294 | +0.326 | +0.301 | -2.9% | – |
| Layout floor (the 20 FFNs deleted, no block at all) | 913,472,787 | 18.31 | 19.29 | 17.88 | – | – |

CE gap: hybrid minus Gemma, in nats (0 = Gemma). Both runs: 20 of 26 FFNs replaced, 35.4M trained parameters, 15M tokens.

## 2 · Learning curve: the same tokens, in the same order, with the same schedule

| Tokens | Pretrained experts, CE gap | Random experts, CE gap | Random minus pretrained |
| --- | --- | --- | --- |
| 1.64M (check, 100k tokens) | 1.386 | 1.830 | +0.444 |
| 3.28M (check, 100k tokens) | 1.194 | 1.512 | +0.318 |
| 4.92M (check, 100k tokens) | 1.067 | 1.365 | +0.298 |
| 6.55M (check, 100k tokens) | 0.997 | 1.284 | +0.287 |
| 8.19M (check, 100k tokens) | 0.951 | 1.232 | +0.281 |
| 9.83M (check, 100k tokens) | 0.896 | 1.185 | +0.289 |
| 11.47M (check, 100k tokens) | 0.874 | 1.161 | +0.287 |
| 13.11M (check, 100k tokens) | 0.859 | 1.144 | +0.285 |
| 14.75M (check, 100k tokens) | 0.851 | 1.136 | +0.285 |

Both runs check at the same steps of the same cosine schedule, so every row compares the two at exactly the same point of training.

## 3 · Before training: ridge W_out through each kind of block

W_in is the same in both runs (Stage 4's ridge fit). W_out is fitted by ridge to Gemma's own FFN output through the block in the slot; validation NMSE (0 = exact, 1 = no better than zero):

| Gemma layer | W_in val NMSE (shared) | W_out val NMSE, pretrained block | W_out val NMSE, random block |
| --- | --- | --- | --- |
| 0 | 0.224 | 0.216 | 0.262 |
| 1 | 0.245 | 0.315 | 0.333 |
| 3 | 0.295 | 0.323 | 0.342 |
| 4 | 0.277 | 0.287 | 0.260 |
| 5 | 0.304 | 0.420 | 0.418 |
| 7 | 0.311 | 0.321 | 0.309 |
| 8 | 0.301 | 0.312 | 0.316 |
| 9 | 0.303 | 0.376 | 0.363 |
| 11 | 0.303 | 0.434 | 0.464 |
| 12 | 0.304 | 0.399 | 0.398 |
| 13 | 0.316 | 0.378 | 0.368 |
| 14 | 0.321 | 0.350 | 0.317 |
| 16 | 0.331 | 0.380 | 0.372 |
| 17 | 0.338 | 0.335 | 0.331 |
| 18 | 0.362 | 0.317 | 0.305 |
| 20 | 0.380 | 0.254 | 0.244 |
| 21 | 0.384 | 0.231 | 0.212 |
| 22 | 0.359 | 0.223 | 0.219 |
| 24 | 0.378 | 0.168 | 0.176 |
| 25 | 0.379 | 0.200 | 0.238 |
| **mean** | – | **0.312** | **0.312** |

## 4 · Q/A test (30 questions, one-sentence answers)

| Model | Correct | Fact log-prob | Fact recovery (26 deleted = 0%) | Sentence log-prob | Copied |
| --- | --- | --- | --- | --- | --- |
| Gemma 3 1B | 28/30 | -0.23 | 100% | -0.28 | – |
| Stage 4, pretrained experts | 10/30 | -1.82 | 89% | -0.83 | 0/30 |
| Stage 5, random experts | 3/30 | -2.93 | 81% | -1.18 | 0/30 |
| All 26 FFNs deleted | 0/30 | -14.81 | 0% | – | – |

Right with both: 3. Only with the pretrained experts (7): What is the capital of France?; What language is spoken in Brazil?; What organ pumps blood through the human body?; What force pulls objects toward the Earth?; How many months are in a year?; Who wrote Romeo and Juliet?; In which year did World War II end?. Only with random experts (0): –.

Every reply: `runs/stage4_spread_15m/qa.md` and `runs/stage5_random_15m/qa.md`.

## 5 · Routing (final evaluation, 20 MoE layers)

| Experts | Lowest entropy (Gemma layer) | Mean entropy | Dead experts | Busiest expert (layer) |
| --- | --- | --- | --- | --- |
| Stage 4, pretrained experts | 0.724 (0) | 0.954 | 29 of 1200 | 18.3% (0) |
| Stage 5, random experts | 0.596 (11) | 0.698 | 344 of 1200 | 24.8% (11) |

Entropy alarm: 0.7. W_in alone decides which experts a token reaches, in both runs.

## 6 · Runs

| Run | Trainable | Tokens | Training speed | Peak memory |
| --- | --- | --- | --- | --- |
| Stage 4, pretrained experts | 35.4M | 14.99M | 326 tok/s | 11.3 GB |
| Stage 5, random experts | 35.4M | 14.99M | 188 tok/s | 11.3 GB |

Random draw of Stage 5: seed 0, full True, fingerprint `0712576589269831` (`scripts/verify_stage5_random.py runs/stage5_random_15m` checks it).
