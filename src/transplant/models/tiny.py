"""Tiny stand-ins for Gemma 3 1B (A) and MobileMoE-M (B), for tests and the CPU smoke run.

Tiny A is a real ``Gemma3ForCausalLM`` (same code path, same sandwich norms) with a
small config. Tiny B mimics a Llama-style MoE decoder: pre-norm attention stand-in,
then ``post_attention_layernorm`` and an ``mlp`` holding a sigmoid top-k router, routed
experts and one always-on shared expert. The two use different tokenizers trained on
the same synthetic text, so alignment is exercised the way it is for the real pair.
"""
from __future__ import annotations

import functools
import os
import random
from pathlib import Path

import torch
from torch import nn

WORDS = (
    "the of and to in is was for on that with as by at from his her an were are which this "
    "be or has had not but its their they one all new more first two after who been also "
    "time other some than into most only over such when would there can about up out him "
    "years made three them may later then city school world river energy number system "
    "theory model water light history music science expert layer token router gate value"
).split()


def synthetic_doc(seed: int, index: int, min_words: int = 30, max_words: int = 160) -> str:
    rng = random.Random(seed * 1_000_003 + index)
    n = rng.randint(min_words, max_words)
    words = []
    for k in range(n):
        w = rng.choice(WORDS)
        if rng.random() < 0.05:
            w = w + str(rng.randint(0, 99))
        words.append(w)
        if rng.random() < 0.08:
            words[-1] += "."
    text = " ".join(words)
    return text[0].upper() + text[1:]


def _corpus(n: int = 400) -> list[str]:
    return [synthetic_doc(7, i) for i in range(n)]


@functools.lru_cache(maxsize=None)
def tiny_tokenizer(kind: str):
    """``a``: byte-level BPE (subword pieces). ``b``: whitespace word-level. Deterministic."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    if kind == "a":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=specials, show_progress=False)
    elif kind == "b":
        tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(vocab_size=500, special_tokens=specials, show_progress=False)
    else:
        raise ValueError(kind)
    tok.train_from_iterator(_corpus(), trainer=trainer)
    from tokenizers.processors import TemplateProcessing

    bos = tok.token_to_id("<bos>")
    tok.post_processor = TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", bos)])
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"
    )
    return fast


def _cache_dir() -> Path:
    return Path(os.environ.get("TRANSPLANT_CACHE", Path.home() / ".cache" / "expert_transplant"))


def _pretrain(model, tok, steps: int, seed: int, seq_len: int = 64, batch: int = 8, lr: float = 3e-3):
    """A few hundred CPU steps on synthetic text, so the tiny models are not pure noise and
    removing an FFN visibly hurts (makes gap recovery meaningful in the smoke run)."""
    from ..data.pack import Packer

    torch.manual_seed(seed)
    packer = Packer(tok.bos_token_id, tok.eos_token_id, seq_len)
    doc = 10_000 * (seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        rows = []
        while len(rows) < batch:
            while not packer.ready():
                packer.add(tok(synthetic_doc(seed + 3, doc), add_special_tokens=False)["input_ids"])
                doc += 1
            rows.append(packer.pop())
        ids = torch.tensor(rows)
        logits = model(input_ids=ids).logits[:, :-1]
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def _cached(name: str, build, steps: int, tok, seed: int):
    model = build()
    if steps <= 0:
        return model
    path = _cache_dir() / f"{name}_seed{seed}_steps{steps}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model
    _pretrain(model, tok, steps, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def tiny_model_a(seed: int = 0, vocab_size: int | None = None, pretrain_steps: int = 0):
    return _cached("tiny_a", lambda: _tiny_model_a(seed, vocab_size), pretrain_steps, tiny_tokenizer("a"), seed)


def _tiny_model_a(seed: int = 0, vocab_size: int | None = None):
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    vocab = vocab_size or len(tiny_tokenizer("a"))
    cfg = Gemma3TextConfig(
        vocab_size=vocab,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        query_pre_attn_scalar=32,
        sliding_window=16,
        max_position_embeddings=512,
        rope_scaling=None,
        attn_implementation="eager",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    model = Gemma3ForCausalLM(cfg)
    # Non-zero norm gains so the sandwich norms are not identities in tests.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.normal_(0.0, 0.1)
    model.eval()
    return model


class TinyRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


class TinyGatedMLP(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, hidden, bias=False)
        self.up_proj = nn.Linear(d, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinySparseMoE(nn.Module):
    """Sigmoid top-k routing, normalized over the chosen k, plus an always-on shared expert.

    Shaped like MobileMoE's MobileMoETextMoe (read at M0): the router logits come from
    F.linear on the router's weight in fp32 (router.forward never runs), and the block
    returns (flattened (N, d) output, router logits); the decoder layer reshapes it back.
    """

    def __init__(self, d: int, num_experts: int, top_k: int, expert_hidden: int, shared_hidden: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d, num_experts, bias=False)
        self.experts = nn.ModuleList([TinyGatedMLP(d, expert_hidden) for _ in range(num_experts)])
        self.shared_expert = TinyGatedMLP(d, shared_hidden)

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        logits = nn.functional.linear(flat.float(), self.gate.weight.float())
        scores = torch.sigmoid(logits)
        top, idx = scores.topk(self.top_k, dim=-1)
        weights = (top / top.sum(-1, keepdim=True)).to(flat.dtype)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):  # dropless dispatch
            rows, slot = (idx == e).nonzero(as_tuple=True)
            if rows.numel():
                out.index_add_(0, rows, expert(flat[rows]) * weights[rows, slot, None])
        out = out + self.shared_expert(flat)
        return out, logits


class TinyMoELayer(nn.Module):
    def __init__(self, d: int, **moe_kw):
        super().__init__()
        self.input_layernorm = TinyRMSNorm(d)
        self.self_attn = nn.Linear(d, d, bias=False)  # stand-in for attention
        self.post_attention_layernorm = TinyRMSNorm(d)
        self.mlp = TinySparseMoE(d, **moe_kw)

    def forward(self, h):
        h = h + self.self_attn(self.input_layernorm(h))
        out, _ = self.mlp(self.post_attention_layernorm(h))
        return h + out.reshape(h.shape)


class TinyMoEInner(nn.Module):
    def __init__(self, vocab: int, d: int, n_layers: int, **moe_kw):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([TinyMoELayer(d, **moe_kw) for _ in range(n_layers)])
        self.norm = TinyRMSNorm(d)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        # causal mixing so that position t only sees <= t (like a decoder)
        h = torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1, device=h.device)[None, :, None]
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class TinyMoEForCausalLM(nn.Module):
    def __init__(self, vocab: int, d: int = 48, n_layers: int = 4, num_experts: int = 6, top_k: int = 2):
        super().__init__()
        self.model = TinyMoEInner(
            vocab, d, n_layers, num_experts=num_experts, top_k=top_k, expert_hidden=32, shared_hidden=64
        )
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = type("Cfg", (), {"hidden_size": d, "num_hidden_layers": n_layers, "vocab_size": vocab})()

    def forward(self, input_ids, attention_mask=None, **_):
        h = self.model(input_ids)
        logits = self.lm_head(h)
        return type("Out", (), {"logits": logits, "last_hidden_state": h})()


TINY_B_CFG = {"num_experts": 6, "top_k": 2, "layers_path": "auto", "moe_attr": "auto", "norm_attr": "auto",
              "router_attr": "auto", "shared_attr": "auto"}


def _tiny_model_b(seed: int = 1):
    torch.manual_seed(seed)
    model = TinyMoEForCausalLM(len(tiny_tokenizer("b")))
    model.eval()
    return model


def tiny_model_b(seed: int = 1, pretrain_steps: int = 0):
    return _cached("tiny_b", lambda: _tiny_model_b(seed), pretrain_steps, tiny_tokenizer("b"), seed)
