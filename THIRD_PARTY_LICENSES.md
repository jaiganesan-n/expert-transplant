# Third-party models and datasets

This repository's code is Apache 2.0. It downloads, but does not redistribute, the models and datasets below; each keeps its own terms. Read the full license of each before using or sharing anything derived from it (plan M0 checklist).

## Models

| Model | Use here | License | Notes |
| --- | --- | --- | --- |
| [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) | Model A, the host | Gemma Terms of Use | Gated: accept on Hugging Face. Check the terms on derived weights before sharing trained projections. |
| [facebook/MobileMoE-M-Base](https://huggingface.co/facebook/MobileMoE-M-Base) | Model B, the donor of MoE blocks | FAIR Noncommercial Research License | Gated (contact-info form). Non-commercial research only; anything derived (trained projections, hybrid outputs) inherits this. |

## Training data: MobileMoE's pre-training mix (Table B.1, arXiv:2605.27358)

| Source | Hugging Face id | License |
| --- | --- | --- |
| FineWeb-Edu | HuggingFaceFW/fineweb-edu | ODC-By 1.0, plus Common Crawl terms of use |
| DCLM, OpenWebMath, Algebraic Stack, StarCoder, Wikipedia, arXiv, peS2o | allenai/olmo-mix-1124 | ODC-By 1.0 collection; per-source terms apply (Common Crawl terms; The Stack per-file licenses; Wikipedia CC BY-SA; arXiv per-paper licenses) |
| FineMath | HuggingFaceTB/finemath | ODC-By 1.0, plus Common Crawl terms of use |
| FLAN | allenai/dolmino-mix-1124 | ODC-By 1.0 collection |
| Cosmopedia v2 | HuggingFaceTB/smollm-corpus | ODC-By 1.0 |
| Natural Reasoning | facebook/natural_reasoning | CC-BY-NC-4.0 (non-commercial) |
| Optional, off by default: Dolma 3 (Common Crawl, Stack-Edu, OLMoCR science PDFs) | allenai/dolma3_mix-6T-1025-7B | ODC-By 1.0 |
| Optional, off by default: Nemotron math, code, science | nvidia/Llama-Nemotron-Post-Training-Dataset | CC-BY-4.0 |

## Evaluation data

| Source | Hugging Face id | License |
| --- | --- | --- |
| WikiText-103 (test) | Salesforce/wikitext | CC BY-SA 3.0 |
| C4 (en, validation) | allenai/c4 | ODC-By 1.0, plus Common Crawl terms of use |

## Libraries

PyTorch (BSD-3-Clause), Hugging Face transformers, datasets, tokenizers, safetensors, accelerate, huggingface_hub (Apache 2.0), NumPy (BSD-3-Clause), PyYAML (MIT), zstandard (BSD-3-Clause), matplotlib (PSF-based) for figures.

Cite every model and dataset used in any write-up.
