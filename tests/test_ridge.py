import pytest
import torch

from transplant.init.ridge import RidgeStats


def test_ridge_recovers_a_linear_map():
    torch.manual_seed(0)
    d_in, d_out, n = 12, 7, 4000
    w_true = torch.randn(d_in, d_out)
    stats = RidgeStats(d_in, d_out, torch.device("cpu"))
    for k in range(10):
        x = torch.randn(n // 10, d_in)
        y = x @ w_true + 0.01 * torch.randn(n // 10, d_out)
        stats.add(x, y, "val" if k == 9 else "train")
    w, info = stats.solve([1e-6, 1e-3, 1e-1, 1.0])
    assert torch.allclose(w.float(), w_true, atol=5e-3)
    assert info["lambda"] <= 1e-3
    assert info["val_nmse"] < 1e-3


def test_val_nmse_matches_direct_computation():
    torch.manual_seed(1)
    stats = RidgeStats(5, 3, torch.device("cpu"))
    x, y = torch.randn(200, 5), torch.randn(200, 3)
    stats.add(x, y, "val")
    w = torch.randn(5, 3, dtype=torch.float64)
    direct = ((y.double() - x.double() @ w) ** 2).sum() / (y.double() ** 2).sum()
    assert abs(stats.val_nmse(w) - float(direct)) < 1e-9


def test_right_padding_without_a_mask_leaves_real_positions_unchanged():
    """Ridge runs right-padded batches with no attention mask (a mask makes padding rows beyond
    Gemma's sliding window NaN on the Apple GPU). Causal attention keeps real positions exact."""
    from transplant.models.tiny import tiny_model_a

    model = tiny_model_a(seed=0)  # sliding window 16, so padding runs far past it here
    ids = torch.randint(4, 300, (1, 64))
    ids[0, 0] = 1
    ids[0, 10:] = 0
    with torch.no_grad():
        alone = model.model(input_ids=ids[:, :10]).last_hidden_state
        padded = model.model(input_ids=ids).last_hidden_state
    assert torch.isfinite(padded).all()
    torch.testing.assert_close(padded[:, :10], alone, rtol=1e-5, atol=1e-5)


def test_ridge_sums_refuse_nan():
    from transplant.init.ridge import RidgeStats

    stats = RidgeStats(3, 2, torch.device("cpu"), name="W_in, layer 4")
    x = torch.randn(5, 3)
    x[2, 1] = float("nan")
    with pytest.raises(FloatingPointError, match="layer 4"):
        stats.add(x, torch.randn(5, 2))


def test_a_broken_init_file_is_computed_again(tiny_cfg):
    from transplant.data.heldout import build_heldout
    from transplant.models.tiny import tiny_tokenizer
    from transplant.train.run import run_dir_of, run_training

    build_heldout(tiny_cfg.data, tiny_tokenizer("a"), tiny_cfg.data.heldout_dir, log=lambda *a: None)
    path = run_dir_of(tiny_cfg) / "init.pt"
    path.parent.mkdir(parents=True)
    torch.save({0: {"w_in": torch.full((48, 64), float("nan")), "w_out": torch.zeros(64, 48)}}, path)
    lines = []
    run_training(tiny_cfg, stop_after="init", log=lines.append)
    assert any("NaN or inf" in line for line in lines)
    fresh = torch.load(path)
    assert all(torch.isfinite(m).all() for layer in fresh.values() for m in layer.values())
