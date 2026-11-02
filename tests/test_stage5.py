"""Stage 5: the random-expert control of Stage 4 (MobileMoE-S's blocks, every weight random, frozen)."""
import importlib.util
import json

import pytest
import torch

from conftest import REPO
from test_stage4 import _fake_run
from transplant.config import load_config
from transplant.data.heldout import build_heldout
from transplant.init.ridge import load_init
from transplant.models.load import build_hybrid, random_moe_options
from transplant.models.moe_source import _stats, blocks_fingerprint, extract_blocks, randomize_like
from transplant.models.tiny import TINY_B_CFG, tiny_model_b, tiny_tokenizer
from transplant.train.optim import steps_for_tokens
from transplant.train.run import _blocks_of, _init_made_for, _init_matrices, run_training, setup, token_budget

STAGES = REPO / "configs" / "stages"


def _cfg(name: str):
    return load_config([REPO / "configs" / "base.yaml", STAGES / name])


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage5_config_is_stage4_with_random_experts():
    s4, s5 = _cfg("stage4_spread_15m.yaml").to_dict(), _cfg("stage5_random_15m.yaml").to_dict()
    assert s5["transplant"]["inner"] == "random_moe_b" and random_moe_options(s5) == {"seed": 0, "full": True}
    assert s5["ridge"]["w_in_from"] == s4["transplant"]["init_path"] == "runs/stage4_spread_init/init.pt"
    assert s5["transplant"]["init_path"] == "runs/stage5_random_init/init.pt"
    # Everything else is Stage 4's: the pair, the layout, the data and its order (run.seed), the tokens,
    # the recipe, the evaluation.
    for cfg in (s4, s5):
        for key in ("name", "experiment"):
            cfg["run"].pop(key)
        for key in ("inner", "random_moe", "init_path"):
            cfg["transplant"].pop(key)
        cfg["ridge"].pop("w_in_from")
    assert s4 == s5
    local, e2e, frm = token_budget(_cfg("stage5_random_15m.yaml"))
    assert (local, e2e, frm) == (0, 15_000_000, None) and steps_for_tokens(e2e, 8, 2, 1024) == 915
    run_sh = (REPO / "scripts" / "stage5_run.sh").read_text()
    assert "configs/stages/stage5_random_15m.yaml" in run_sh and "runs/stage4_spread_15m/speed.json" in run_sh
    # the old control C5 keeps its meaning: matrices only, seed = run.seed
    assert random_moe_options(load_config([REPO / "configs" / "base.yaml"])) == {"seed": 0, "full": False}


def _pretrained_like_block(seed: int = 1, layer: int = 1):
    """A tiny block with what pre-training leaves behind: a selection bias, an uneven norm gain, and gate
    and up halves at different scales (as in MobileMoE-S)."""
    block = extract_blocks(tiny_model_b(seed=seed), TINY_B_CFG, [layer])[layer]
    with torch.no_grad():
        block.moe.expert_bias.copy_(torch.linspace(-0.05, 0.02, block.moe.expert_bias.numel()))
        block.norm.weight.copy_(torch.linspace(0.2, 0.9, block.norm.weight.numel()))
        gu = block.moe.experts.gate_up_proj
        gu[..., : gu.shape[-1] // 2] *= 3.0
    return block


def test_full_random_block_keeps_only_the_scale():
    block = _pretrained_like_block()
    before = {n: t.clone() for n, t in block.state_dict().items()}
    rnd = randomize_like(block, seed=0, full=True)
    sd = rnd.state_dict()
    assert {n: t.shape for n, t in sd.items()} == {n: t.shape for n, t in before.items()}
    assert type(rnd.moe) is type(block.moe) and not any(p.requires_grad for p in rnd.parameters())
    for n, t in block.state_dict().items():  # the source is untouched
        assert torch.equal(t, before[n])
    assert not sd["moe.expert_bias"].any()
    gain = before["norm.weight"].double()
    torch.testing.assert_close(sd["norm.weight"].double(), torch.full_like(gain, gain.pow(2).mean().sqrt().item()))
    for n, t in sd.items():
        if t.ndim < 2:
            continue
        assert not torch.equal(t, before[n])
        r = torch.corrcoef(torch.stack([t.flatten().double(), before[n].flatten().double()]))[0, 1]
        assert abs(r) < 5 / t.numel() ** 0.5, n
        parts = [slice(None)]
        if n.endswith("gate_up_proj"):
            f = t.shape[-1] // 2
            parts = [slice(0, f), slice(f, None)]
        for p in parts:
            (m_new, s_new), (m_old, s_old) = _stats(t[..., p]), _stats(before[n][..., p])
            assert abs(s_new / s_old - 1) < 5 / (2 * t[..., p].numel()) ** 0.5, n
            assert abs(m_new - m_old) < 5 * s_old / t[..., p].numel() ** 0.5, n
    gu = sd["moe.experts.gate_up_proj"]
    f = gu.shape[-1] // 2
    assert 2.7 < gu[..., :f].std() / gu[..., f:].std() < 3.3  # gate and up matched apart
    rnd.record_routing = True
    with torch.no_grad():
        out = rnd(torch.randn(2, 5, gain.numel()))
    assert torch.isfinite(out).all() and len(rnd.router_calls) == 1


def test_default_random_block_is_c5_as_first_written():
    block = _pretrained_like_block()
    rnd = randomize_like(block, seed=0)
    assert torch.equal(rnd.moe.expert_bias, block.moe.expert_bias) and torch.equal(rnd.norm.weight, block.norm.weight)
    gu = rnd.moe.experts.gate_up_proj
    f = gu.shape[-1] // 2
    assert 0.9 < gu[..., :f].std() / gu[..., f:].std() < 1.1  # one scale for the whole fused tensor


def test_random_draw_is_reproducible():
    block = _pretrained_like_block()
    a, b = randomize_like(block, seed=0, full=True), randomize_like(block, seed=0, full=True)
    assert blocks_fingerprint({1: a}) == blocks_fingerprint({1: b})
    assert all(torch.equal(t, b.state_dict()[n]) for n, t in a.state_dict().items())
    assert blocks_fingerprint({1: randomize_like(block, seed=1, full=True)}) != blocks_fingerprint({1: a})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the Apple GPU")
def test_random_draw_is_the_same_on_the_gpu():
    """Ridge init draws the blocks on the GPU, training on the CPU: they must be the same blocks."""
    on_cpu = _pretrained_like_block().to(torch.bfloat16)
    on_gpu = _pretrained_like_block().to("mps", torch.bfloat16)
    a, b = randomize_like(on_cpu, seed=0, full=True), randomize_like(on_gpu, seed=0, full=True)
    assert blocks_fingerprint({1: a}) == blocks_fingerprint({1: b})


def test_random_control_starts_from_stage4_w_in(tiny_cfg):
    """Tiny Stage 4 (pretrained blocks, own ridge init), then tiny Stage 5 from its W_in."""
    build_heldout(tiny_cfg.data, tiny_tokenizer("a"), tiny_cfg.data.heldout_dir, log=lambda *a: None)
    tiny_cfg.pair.b.num_layers = 3
    tiny_cfg.pair.layers = "spread"
    tiny_cfg.local.enabled = False
    tiny_cfg.run.tokens = 64 * 4 * 6
    tiny_cfg.run.name = "stage4_tiny"
    s4 = run_training(tiny_cfg, stop_after="init", log=lambda *a: None)
    s4_init = s4 / "init.pt"
    s4_bytes = s4_init.read_bytes()

    tiny_cfg.run.name = "stage5_tiny"
    tiny_cfg.transplant.inner = "random_moe_b"
    tiny_cfg.transplant.random_moe = {"seed": 0, "full": True}
    tiny_cfg.ridge.w_in_from = str(s4_init)
    run_dir = run_training(tiny_cfg, log=lambda *a: None)

    init4, init5 = load_init(s4_init), load_init(run_dir / "init.pt")
    assert sorted(init5) == sorted(init4) == [0, 2, 3]
    for i in init4:  # the same W_in; W_out fitted through the random blocks
        assert torch.equal(init5[i]["w_in"], init4[i]["w_in"]) and not torch.equal(init5[i]["w_out"], init4[i]["w_out"])
    rep = json.loads((run_dir / "init.json").read_text())
    info = json.loads((run_dir / "run_info.json").read_text())
    assert rep["inner"] == info["inner"] == "random_moe_b" and rep["w_in_from"] == str(s4_init)
    assert info["random_moe"] == rep["random_moe"] and info["random_moe"]["full"] is True
    assert json.loads((run_dir / "eval.json").read_text())["sets"]["mix"]["ppl_hybrid"] > 0
    proj = torch.load(run_dir / "projections.pt")
    assert len(proj) == 6 and all(".w_in." in k or ".w_out." in k for k in proj)

    # a later process (eval, ask) rebuilds the blocks the run trained with
    device, policy = setup(tiny_cfg)
    hybrid = build_hybrid(tiny_cfg, device, policy)
    assert blocks_fingerprint(hybrid.blocks) == info["random_moe"]["fingerprint"]

    # Stage 4's init fits W_out through other blocks: never reused for random ones, and never overwritten
    assert _init_made_for(s4_init) == {"inner": "moe_b"} != _blocks_of(tiny_cfg)
    assert _init_made_for(run_dir / "init.pt") == _blocks_of(tiny_cfg)
    tiny_cfg.transplant.init_path = str(s4_init)
    other = run_dir.parent / "stage5_tiny_b"
    other.mkdir()
    logs = []
    init = _init_matrices(tiny_cfg, device, policy, other, log=logs.append)
    assert any("fits W_out to other MoE blocks" in line for line in logs)
    assert (other / "init.pt").exists() and s4_init.read_bytes() == s4_bytes
    assert all(torch.equal(init[i]["w_out"], init5[i]["w_out"]) for i in init5)  # the same random blocks again


def test_verify_script_on_tiny_models():
    verify = _script("verify_stage5_random")
    code = verify.main(["--config", str(REPO / "configs" / "base.yaml"), "--config", str(REPO / "configs" / "pairs" / "tiny.yaml"),
                        "--set", "transplant.inner=random_moe_b", "--set", "transplant.random_moe={seed: 0, full: true}",
                        "--set", "pair.b.num_layers=3", "--set", "pair.layers=spread"])
    assert code == 0


def _qa_facts(run_dir, facts):
    qa = json.loads((run_dir / "qa.json").read_text())
    qa["modes"]["hybrid"]["fact_logprob"] = facts
    qa["modes"]["hybrid"]["mean_fact_logprob"] = sum(facts) / len(facts)
    (run_dir / "qa.json").write_text(json.dumps(qa))


@pytest.mark.parametrize("gap5,expected", [
    (1.20, "**The pretrained experts contributed.** With random experts of the same architecture the hybrid is 0.38 nats worse"),
    (0.86, "**No clear contribution from the pretrained experts**"),
    (0.60, "**Random experts did better**, by 0.22 nats"),
])
def test_stage5_report(tmp_path, gap5, expected):
    mod = _script("stage5_compare")
    _fake_run(tmp_path, "stage3_20m", 0.83, [100, 200], 4_997_120)
    s4 = _fake_run(tmp_path, "stage4_spread_15m", 0.82, [100, 200, 300], 0, layers=20, trainable=35_389_440, floor=20.6)
    s5 = _fake_run(tmp_path, "stage5_random_15m", gap5, [100, 200, 300], 0, layers=20, trainable=35_389_440, floor=20.6)
    _qa_facts(s4, [-1.0] * 24 + [-3.0] * 6)
    _qa_facts(s5, [-2.0] * 24 + [-1.0] * 6)
    md = mod.report(tmp_path)
    assert expected in md
    assert "| Stage 4, pretrained experts (`stage4_spread_15m`) |" in md and "| Stage 5, random experts (`stage5_random_15m`) |" in md
    assert f"| **Random minus pretrained** | × {2.718281828 ** (gap5 - 0.82):.2f} | {gap5 - 0.82:+.3f} |" in md
    assert "| 1.64M (check, 100k tokens) |" in md  # step 100 of 8 x 2 x 1,024 tokens, both runs
    assert "more likely with the pretrained experts for 24 of 30 questions" in md
    assert mod.sign_test(24, 30) == pytest.approx(0.001430, abs=1e-6) and mod.sign_test(15, 30) == 1.0


def test_stage5_report_before_stage5(tmp_path):
    mod = _script("stage5_compare")
    _fake_run(tmp_path, "stage4_spread_15m", 0.82, [100], 0, layers=20, trainable=35_389_440, floor=20.6)
    md = mod.report(tmp_path)
    assert "Stage 5 has not finished yet" in md and "## Verdict" not in md and "No check of Stage 5 yet." in md
