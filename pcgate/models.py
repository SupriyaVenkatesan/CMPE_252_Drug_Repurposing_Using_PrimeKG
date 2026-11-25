"""Encoders, relation summaries and the decoder family (proposal section 7).

All learned models share one framework so the only difference between variants is
the decoder (proposal 8.3):

  model id         encoder     decoder input                                   role
  ---------------  ---------   ---------------------------------------------   -----------------------
  sage             GraphSAGE   [h_d, h_s, h_d*h_s]                             relation-agnostic baseline
  hgt              HGT         [h_d, h_s, h_d*h_s]                             standard HGT baseline
  gated            HGT         [.., c_d(d,s), c_s(d,s), miss_d, miss_s]        PROPOSED pair-conditioned gate
  uniform          HGT         same, alpha = 1/|R_v| over available rels       uniform-summary control
  param_matched    HGT         same summaries, non-gated flatten projection,   parameter-matched control
                               decoder widened to the gated parameter count
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, SAGEConv
from torch_geometric.utils import scatter

ENDPOINTS = ("drug", "disease")
MODEL_NAMES = ("sage", "hgt", "gated", "uniform", "param_matched")


# ----------------------------------------------------------------------------- encoders
class ContextEmbedding(nn.Module):
    """Optional learnable identity embeddings for CONTEXT node types only (genes,
    pathways, phenotypes, ...).  Drugs and diseases never get identity embeddings,
    so held-out drugs/diseases are represented by shared transformations only."""

    def __init__(self, num_nodes, hidden, exclude=ENDPOINTS):
        super().__init__()
        self.emb = nn.ModuleDict({t: nn.Embedding(n, hidden) for t, n in (num_nodes or {}).items()
                                  if t not in exclude})
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.1)

    def forward(self, t, z):
        return z + self.emb[t].weight if t in self.emb else z


class HGTEncoder(nn.Module):
    def __init__(self, metadata, in_dim, hidden=64, layers=2, heads=2, dropout=0.2, num_nodes=None):
        super().__init__()
        node_types, _ = metadata
        self.inp = nn.ModuleDict({t: nn.Linear(in_dim, hidden) for t in node_types})
        self.ctx_emb = ContextEmbedding(num_nodes, hidden)
        self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads=heads) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x_dict, ei_dict):
        h = {t: F.gelu(self.ctx_emb(t, self.inp[t](x))) for t, x in x_dict.items()}
        for conv in self.convs:
            out = conv(h, ei_dict)
            h = {t: (F.dropout(F.gelu(out[t]), self.dropout, self.training) + h[t])
                 if out.get(t) is not None else h[t] for t in h}
        return h


class SAGEEncoder(nn.Module):
    """GraphSAGE on the homogeneous view: node types are known (features + per-type
    input projection) but relation types are ignored during aggregation."""

    def __init__(self, metadata, in_dim, hidden=64, layers=2, dropout=0.2, num_nodes=None, **_):
        super().__init__()
        node_types, _ = metadata
        self.types = list(node_types)
        self.inp = nn.ModuleDict({t: nn.Linear(in_dim, hidden) for t in node_types})
        self.ctx_emb = ContextEmbedding(num_nodes, hidden)
        self.convs = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x_dict, ei_dict):
        offs, n = {}, 0
        for t in self.types:
            offs[t] = n
            n += x_dict[t].size(0)
        h = torch.cat([F.gelu(self.ctx_emb(t, self.inp[t](x_dict[t]))) for t in self.types], 0)
        ei = torch.cat([torch.stack([e[0] + offs[s], e[1] + offs[d]]) for (s, _, d), e in ei_dict.items()], 1)
        for conv in self.convs:
            h = F.dropout(F.gelu(conv(h, ei)), self.dropout, self.training) + h
        return {t: h[offs[t]:offs[t] + x_dict[t].size(0)] for t in self.types}


# ----------------------------------------------------------------------------- summaries
class RelationSummaries(nn.Module):
    """m_{v,r} = 1/|N_r(v)| sum_{u in N_r(v)} W_r h_u  for v in {drug, disease}.

    W_r is linear, so W_r(mean h_u) == mean(W_r h_u); we aggregate first (cheap).
    Unavailable relations get a zero summary and mask = 0.
    """

    def __init__(self, metadata, hidden):
        super().__init__()
        _, edge_types = metadata
        self.rels = {v: [et for et in edge_types if et[2] == v] for v in ENDPOINTS}
        self.W = nn.ModuleDict({"__".join(et): nn.Linear(hidden, hidden, bias=False)
                                for v in ENDPOINTS for et in self.rels[v]})

    def num_relations(self, v):
        return len(self.rels[v])

    def forward(self, h, ei_dict):
        out = {}
        for v in ENDPOINTS:
            n = h[v].size(0)
            Ms, masks = [], []
            for et in self.rels[v]:
                ei = ei_dict[et]
                if ei.numel() == 0:
                    Ms.append(h[v].new_zeros(n, h[v].size(1)))
                    masks.append(h[v].new_zeros(n))
                    continue
                mean = scatter(h[et[0]][ei[0]], ei[1], dim=0, dim_size=n, reduce="mean")
                deg = torch.bincount(ei[1], minlength=n)
                Ms.append(self.W["__".join(et)](mean))
                masks.append((deg > 0).float())
            out[v] = (torch.stack(Ms, 1), torch.stack(masks, 1))  # [N, R, H], [N, R]
        return out


# ----------------------------------------------------------------------------- decoders
def mlp(i, h, o, dropout):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


class PlainDecoder(nn.Module):
    uses_summaries = False

    def __init__(self, H, hid, dropout, **_):
        super().__init__()
        self.out = mlp(3 * H, hid, 1, dropout)

    def forward(self, hd, hs, **_):
        return self.out(torch.cat([hd, hs, hd * hs], -1)).squeeze(-1), None


class GateDecoder(nn.Module):
    """Pair-conditioned gate (proposal 7.3-7.5).

    q_ds    = MLP_q([h_d, h_s, h_d*h_s])
    a^(v)_r = w^T tanh(A q_ds + B m_{v,r} + C e_r)
    alpha   = softmax over available relations of v;  c_v = sum_r alpha_r m_{v,r}
    p_ds    = sigmoid(MLP_o([h_d, h_s, h_d*h_s, c_d, c_s, miss_d, miss_s]))
    mode="uniform" replaces alpha by 1/|available| (uniform-summary control).
    """
    uses_summaries = True

    def __init__(self, H, hid, dropout, n_rel, mode="gated", **_):
        super().__init__()
        self.mode = mode
        if mode == "gated":
            self.q = mlp(3 * H, H, H, dropout)
            self.A = nn.Linear(H, H)
            self.B = nn.Linear(H, H, bias=False)
            self.C = nn.Linear(H, H, bias=False)
            self.w = nn.Linear(H, 1, bias=False)
            self.e = nn.ParameterDict({v: nn.Parameter(torch.randn(n_rel[v], H) / math.sqrt(H)) for v in ENDPOINTS})
        self.out = mlp(5 * H + 2, hid, 1, dropout)

    def weights(self, q, M, mask, v):
        if self.mode == "uniform":
            return mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        a = self.w(torch.tanh(self.A(q).unsqueeze(1) + self.B(M) + self.C(self.e[v]).unsqueeze(0))).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e9)
        alpha = torch.softmax(a, 1)
        # endpoints with no available relation -> zero context (and finite gradients)
        return alpha * (mask.sum(1, keepdim=True) > 0).float()

    def forward(self, hd, hs, Md, maskd, Ms, masks):
        pair = torch.cat([hd, hs, hd * hs], -1)
        q = self.q(pair) if self.mode == "gated" else None
        ad = self.weights(q, Md, maskd, "drug")
        as_ = self.weights(q, Ms, masks, "disease")
        cd = (ad.unsqueeze(-1) * Md).sum(1)
        cs = (as_.unsqueeze(-1) * Ms).sum(1)
        miss = torch.stack([(maskd.sum(1) == 0).float(), (masks.sum(1) == 0).float()], -1)
        logit = self.out(torch.cat([pair, cd, cs, miss], -1)).squeeze(-1)
        return logit, {"drug": ad, "disease": as_}


class FlatDecoder(nn.Module):
    """Parameter-matched, non-gated control: c_v = Linear(concat_r mask*m_{v,r}).
    Same summary access as the gate, no pair-conditioned weighting."""
    uses_summaries = True

    def __init__(self, H, hid, dropout, n_rel, proj_dim=None, **_):
        super().__init__()
        k = proj_dim or H
        self.proj = nn.ModuleDict({v: nn.Linear(n_rel[v] * H, k) for v in ENDPOINTS})
        self.out = mlp(3 * H + 2 * k + 2, hid, 1, dropout)

    def forward(self, hd, hs, Md, maskd, Ms, masks):
        cd = self.proj["drug"]((Md * maskd.unsqueeze(-1)).flatten(1))
        cs = self.proj["disease"]((Ms * masks.unsqueeze(-1)).flatten(1))
        miss = torch.stack([(maskd.sum(1) == 0).float(), (masks.sum(1) == 0).float()], -1)
        logit = self.out(torch.cat([hd, hs, hd * hs, cd, cs, miss], -1)).squeeze(-1)
        return logit, None


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def matched_flat_decoder(target: int, H, dropout, n_rel):
    """Pick (proj_dim, hidden) so FlatDecoder's parameter count is closest to `target`."""
    best = None
    for k in sorted({H, H // 2, H // 4, 8}, reverse=True):
        proj = sum(n_rel[v] * H * k + k for v in ENDPOINTS)
        per_hid = (3 * H + 2 * k + 2) + 2          # first-layer weights+bias, output weight
        hid = max(4, round((target - proj - 1) / per_hid))
        diff = proj + per_hid * hid + 1 - target
        if best is None or abs(diff) < abs(best[0]):
            best = (diff, k, hid)
    _, k, hid = best
    dec = FlatDecoder(H, hid, dropout, n_rel, proj_dim=k)
    return dec


# ----------------------------------------------------------------------------- full model
class LinkModel(nn.Module):
    def __init__(self, name, metadata, in_dim, cfg, num_nodes=None):
        super().__init__()
        assert name in MODEL_NAMES, name
        H = cfg.get("hidden", 64)
        hid = cfg.get("decoder_hidden", H)
        dp = cfg.get("dropout", 0.2)
        self.name = name
        enc_cls = SAGEEncoder if name == "sage" else HGTEncoder
        self.encoder = enc_cls(metadata, in_dim, hidden=H, layers=cfg.get("layers", 2),
                               heads=cfg.get("heads", 2), dropout=dp,
                               num_nodes=num_nodes if cfg.get("context_embeddings", False) else None)
        self.summ = None
        if name in ("sage", "hgt"):
            self.decoder = PlainDecoder(H, hid, dp)
        else:
            self.summ = RelationSummaries(metadata, H)
            n_rel = {v: self.summ.num_relations(v) for v in ENDPOINTS}
            if name == "gated":
                self.decoder = GateDecoder(H, hid, dp, n_rel, mode="gated")
            elif name == "uniform":
                self.decoder = GateDecoder(H, hid, dp, n_rel, mode="uniform")
            else:
                target = n_params(GateDecoder(H, hid, dp, n_rel, mode="gated"))
                self.decoder = matched_flat_decoder(target, H, dp, n_rel)

    def encode(self, x, ei):
        h = self.encoder(x, ei)
        ctx = self.summ(h, ei) if self.summ is not None else None
        return h, ctx

    def score(self, h, ctx, d, s, return_alpha=False):
        hd, hs = h["drug"][d], h["disease"][s]
        if ctx is None:
            logit, alpha = self.decoder(hd, hs)
        else:
            (Md, md), (Ms, ms) = ctx["drug"], ctx["disease"]
            logit, alpha = self.decoder(hd, hs, Md=Md[d], maskd=md[d], Ms=Ms[s], masks=ms[s])
        return (logit, alpha) if return_alpha else logit

    def param_report(self):
        return {"total": n_params(self), "encoder": n_params(self.encoder),
                "summaries": n_params(self.summ) if self.summ is not None else 0,
                "decoder": n_params(self.decoder)}


class Popularity:
    """score(d, s) = #training indications of drug d (+ tiny graph-degree tie-break)."""
    name = "popularity"

    def __init__(self, n_drugs, train_pairs, drug_degree=None):
        cnt = torch.bincount(torch.as_tensor(train_pairs[:, 0]), minlength=n_drugs).float()
        tb = torch.zeros(n_drugs) if drug_degree is None else torch.log1p(drug_degree.float()) * 1e-3
        self.s = cnt + tb

    def score_pairs(self, d, s):
        return self.s[torch.as_tensor(d)]
