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
