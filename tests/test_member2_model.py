"""Member 2 verification: HGT, pair-conditioned gate, uniform / parameter-matched controls,
per-model frozen configs and the evaluation-time gate controls.

Run:  python -m pytest -q tests
"""
import numpy as np
import pytest
import torch

from pcgate.gates import _norm_entropy, gate_control, within_drug_variation
from pcgate.models import ENDPOINTS, GateDecoder, LinkModel, n_params
from pcgate.utils import run_cfg

N = {"drug": 6, "disease": 5, "gene_protein": 8}
EDGE_TYPES = [("drug", "drug_protein", "gene_protein"), ("gene_protein", "rev_drug_protein", "drug"),
              ("disease", "disease_protein", "gene_protein"), ("gene_protein", "rev_disease_protein", "disease"),
              ("drug", "drug_drug", "drug"), ("disease", "disease_disease", "disease")]
CFG = {"hidden": 16, "decoder_hidden": 16, "layers": 1, "heads": 2, "dropout": 0.0}


def tiny_graph(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = {t: torch.randn(n, 7, generator=g) for t, n in N.items()}
    ei = {}
    for st, r, dt in EDGE_TYPES:
        src = torch.randint(N[st], (12,), generator=g)
        dst = torch.randint(N[dt], (12,), generator=g)
        ei[(st, r, dt)] = torch.stack([src, dst])
    # drug 5 has no protein targets and no drug-drug edges -> no available drug-side relation
    for et in list(ei):
        if et[2] == "drug":
            keep = ei[et][1] != 5
            ei[et] = ei[et][:, keep]
    meta = (list(N), EDGE_TYPES)
    return x, ei, meta


def build(name, seed=0):
    torch.manual_seed(seed)
    x, ei, meta = tiny_graph()
    return LinkModel(name, meta, 7, CFG), x, ei


@pytest.mark.parametrize("name", ["hgt", "gated", "uniform", "param_matched"])
def test_every_variant_trains(name):
    m, x, ei = build(name)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    d, s = torch.tensor([0, 1, 2, 5]), torch.tensor([0, 1, 2, 3])
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    losses = []
    for _ in range(25):
        h, ctx = m.encode(x, ei)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m.score(h, ctx, d, s), y)
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"{name} did not reduce training loss"


def _alpha(name, d, s):
    m, x, ei = build(name)
    m.eval()
    with torch.no_grad():
        h, ctx = m.encode(x, ei)
        _, alpha = m.score(h, ctx, torch.tensor(d), torch.tensor(s), return_alpha=True)
    return m, ctx, alpha


@pytest.mark.parametrize("name", ["gated", "uniform"])
def test_weights_are_a_distribution_over_available_relations(name):
    d, s = [0, 1, 5], [0, 1, 2]
    m, ctx, alpha = _alpha(name, d, s)
    mask_d = ctx["drug"][1][torch.tensor(d)]
    a = alpha["drug"]
    assert torch.all(a[mask_d == 0] == 0), "unavailable relations must get zero weight"
    sums = a.sum(1)
    for i in range(len(d)):
        expect = 1.0 if mask_d[i].sum() > 0 else 0.0     # drug 5: no relation -> zero context
        assert sums[i].item() == pytest.approx(expect, abs=1e-5)
    if name == "uniform":
        k = mask_d.sum(1, keepdim=True).clamp(min=1)
        assert torch.allclose(a, mask_d / k)


def test_gate_is_pair_conditioned():
    # same drug, different diseases -> different drug-side weights (only the pair query differs)
    _, _, alpha = _alpha("gated", [0, 0, 0], [0, 1, 2])
    a = alpha["drug"]
    assert (a[0] - a[1]).abs().max() > 1e-6 or (a[0] - a[2]).abs().max() > 1e-6
    # the uniform control must NOT depend on the partner
    _, _, alpha_u = _alpha("uniform", [0, 0, 0], [0, 1, 2])
    assert torch.allclose(alpha_u["drug"][0], alpha_u["drug"][1])


def test_param_matched_decoder_size():
    g, _, _ = build("gated")
    p, _, _ = build("param_matched")
    target, got = n_params(g.decoder), n_params(p.decoder)
    assert abs(got - target) / target < 0.05, (got, target)
    # encoder and relation summaries are identical across the three summary-based models
    u, _, _ = build("uniform")
    assert n_params(g.encoder) == n_params(p.encoder) == n_params(u.encoder)
    assert n_params(g.summ) == n_params(p.summ) == n_params(u.summ)


def test_controls_share_the_encoder_with_hgt():
    h, _, _ = build("hgt")
    g, _, _ = build("gated")
    assert n_params(h.encoder) == n_params(g.encoder)
    assert h.summ is None and g.summ is not None


def test_run_cfg_overrides_and_control_fallback():
    cfg = {"model": {"hidden": 64, "dropout": 0.2}, "train": {"lr": 1e-3},
           "model_overrides": {"gated": {"model": {"hidden": 128}, "train": {"lr": 5e-4}},
                               "hgt": {"model": {"dropout": 0.4}}}}
    assert run_cfg(cfg, "gated")["model"] == {"hidden": 128, "dropout": 0.2}
    assert run_cfg(cfg, "gated")["train"]["lr"] == 5e-4
    # controls inherit the GATED model's frozen values
    assert run_cfg(cfg, "uniform")["model"]["hidden"] == 128
    assert run_cfg(cfg, "param_matched")["train"]["lr"] == 5e-4
    assert run_cfg(cfg, "hgt")["model"] == {"hidden": 64, "dropout": 0.4}
    assert run_cfg(cfg, "sage")["model"] == cfg["model"]              # untouched
    assert run_cfg({"model": {"hidden": 8}}, "gated")["model"] == {"hidden": 8}


def test_eval_time_gate_controls_restore_the_model():
    m, x, ei = build("gated")
    m.eval()
    d, s = torch.tensor([0, 0, 1, 2]), torch.tensor([0, 1, 2, 3])
    with torch.no_grad():
        h, ctx = m.encode(x, ei)
        base = m.score(h, ctx, d, s)
        bank = torch.randn(32, CFG["hidden"])
        outs = {}
        for mode in ("shuffled_query", "mean_query", "uniform_weights"):
            with gate_control(m, mode, bank=bank):
                outs[mode] = m.score(h, ctx, d, s)
            assert outs[mode].shape == base.shape
        after = m.score(h, ctx, d, s)
    assert torch.allclose(base, after), "control leaked into the trained model"
    assert isinstance(m.decoder, GateDecoder) and m.decoder.mode == "gated"
    # mean_query: every pair gets the same query -> drug 0's weights no longer depend on the disease
    with torch.no_grad(), gate_control(m, "mean_query", bank=bank):
        _, a = m.score(h, ctx, torch.tensor([0, 0]), torch.tensor([0, 1]), return_alpha=True)
    assert torch.allclose(a["drug"][0], a["drug"][1], atol=1e-6)
    with pytest.raises(ValueError):
        with gate_control(build("uniform")[0], "mean_query", bank=bank):
            pass


def test_gate_statistics():
    a = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    m = np.array([[1, 1, 0], [1, 1, 1]])
    ent = _norm_entropy(a, m)
    assert ent[0] == pytest.approx(1.0) and ent[1] == pytest.approx(0.0)
    # identical weights for every disease a drug is paired with -> zero within-drug variation
    assert within_drug_variation(np.array([[0.2, 0.8], [0.2, 0.8], [0.5, 0.5]]), np.array([0, 0, 1])) == 0.0
    assert within_drug_variation(np.array([[0.2, 0.8], [0.6, 0.4]]), np.array([0, 0])) > 0