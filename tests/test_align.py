"""Gating test 7: aligned positions cover the same character spans in both tokenizers."""
from transplant.models.align import aligned_pairs, encode_aligned
from transplant.models.tiny import synthetic_doc, tiny_tokenizer


def test_handmade_offsets():
    a = [(0, 0), (0, 3), (3, 7), (7, 8), (8, 12)]  # <bos> "The" " cat" " " "sat."
    b = [(0, 0), (0, 7), (7, 12)]
    assert aligned_pairs(a, b) == [(2, 1), (4, 2)]


def test_aligned_positions_share_the_same_prefix():
    tok_a, tok_b = tiny_tokenizer("a"), tiny_tokenizer("b")
    texts = [synthetic_doc(3, i) for i in range(6)]
    ab = encode_aligned(tok_a, tok_b, texts, max_len=256, skip_first=False)
    assert ab.n_pairs > 0
    for r, text in enumerate(texts):
        oa = tok_a(text, return_offsets_mapping=True, truncation=True, max_length=256)["offset_mapping"]
        ob = tok_b(text, return_offsets_mapping=True, truncation=True, max_length=256)["offset_mapping"]
        for pa, pb in ab.pairs[r]:
            assert oa[pa][1] == ob[pb][1]  # same end character: both models read text[:end]
            assert oa[pa][1] > oa[pa][0] and ob[pb][1] > ob[pb][0]  # never a special token
    share = ab.n_pairs / ab.n_tokens_a
    assert 0.3 < share <= 1.0


def test_skip_first_drops_first_real_tokens():
    tok_a, tok_b = tiny_tokenizer("a"), tiny_tokenizer("b")
    texts = ["Expert layer token router gate."]
    with_first = encode_aligned(tok_a, tok_b, texts, 64, skip_first=False).pairs[0]
    without = encode_aligned(tok_a, tok_b, texts, 64, skip_first=True).pairs[0]
    assert len(without) <= len(with_first)
    assert all(a != 1 and b != 1 for a, b in without)
