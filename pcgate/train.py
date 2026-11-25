"""Positive-unlabeled training loop with per-batch edge masking (proposal section 8).

Loss:  L = -1/|P_b| sum log p  - 1/|U_b| sum log(1-p)  + lambda ||theta||^2
       (positive and unlabeled terms averaged separately; lambda via weight decay)
Unlabeled pairs: corrupt the drug or the disease endpoint of a positive; known
indications from ANY partition are filtered out; held-out drugs/diseases are never
used as corruption targets, so they never appear in a training example.
Edge masking: while a training indication is scored, it and its reverse copy are
removed from message passing for that batch.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data.graph import GraphBundle
from .data.splits import Split
from .evaluate import evaluate_pool, make_nn_scorer
from .models import LinkModel
from .utils import run_cfg, set_seed


def _peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


class NegativeSampler:
    def __init__(self, split: Split, n_drugs: int, n_dis: int, device):
        self.n_dis = n_dis
        self.device = device
        drugs = np.setdiff1d(np.arange(n_drugs), split.heldout_drugs)
        dis = np.setdiff1d(np.arange(n_dis), split.heldout_diseases)
        self.drugs = torch.as_tensor(drugs, device=device)
        self.dis = torch.as_tensor(dis, device=device)
        k = split.all_known
        self.known = torch.as_tensor(k[:, 0] * n_dis + k[:, 1], device=device)

    def __call__(self, pos: torch.Tensor, k: int) -> torch.Tensor:
        # Over-sample by 20% and drop any pair that is a known indication.
        n = int(len(pos) * k * 1.2) + 1
        base = pos[torch.randint(len(pos), (n,), device=self.device)]
        flip = torch.rand(n, device=self.device) < 0.5
        nd = self.drugs[torch.randint(len(self.drugs), (n,), device=self.device)]
        ns = self.dis[torch.randint(len(self.dis), (n,), device=self.device)]
        out = torch.stack([torch.where(flip, nd, base[:, 0]), torch.where(flip, base[:, 1], ns)], 1)
        good = ~torch.isin(out[:, 0] * self.n_dis + out[:, 1], self.known)
        return out[good][: len(pos) * k]


def train_model(name: str, graph: GraphBundle, split: Split, cfg: dict, seed: int, device,
                log=print) -> tuple[LinkModel, dict]:
    set_seed(seed)
    cfg = run_cfg(cfg, name)          # per-model frozen hyperparameters, if any
    mcfg, tcfg = cfg.get("model", {}), cfg.get("train", {})
    in_dim = next(iter(graph.x.values())).size(1)
    model = LinkModel(name, graph.metadata(), in_dim, mcfg, graph.num_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.get("lr", 1e-3), weight_decay=tcfg.get("weight_decay", 1e-5))
    n_drugs, n_dis = graph.num_nodes["drug"], graph.num_nodes["disease"]
    sampler = NegativeSampler(split, n_drugs, n_dis, device)
    pos_all = torch.as_tensor(split.train, device=device)
    bs = tcfg.get("batch_size", 512)
    k_neg = tcfg.get("negatives", 5)
    epochs = tcfg.get("epochs", 100)
    patience = tcfg.get("patience", 10)
    select = tcfg.get("select_metric", "ap")
    mask_edges = tcfg.get("mask_batch_edges", True)

    best, best_state, bad_epochs, hist = -1.0, None, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(pos_all), device=device)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            pos = pos_all[idx]
            neg = sampler(pos, k_neg)
            ei = graph.masked_edges(idx) if mask_edges else graph.edge_index
            h, ctx = model.encode(graph.x, ei)
            lp = model.score(h, ctx, pos[:, 0], pos[:, 1])
            ln = model.score(h, ctx, neg[:, 0], neg[:, 1])
            loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
                F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 5.0))
            opt.step()
            tot += loss.item() * len(idx)
        rec = {"epoch": ep, "loss": tot / len(perm)}
        if ep % tcfg.get("eval_every", 1) == 0 or ep == epochs:
            vm = evaluate_pool(make_nn_scorer(model, graph, device), split.val_pool)
            rec.update({"val_ap": vm["ap"], "val_recall@10": vm["recall@10"]})
            cur = vm["ap"] if select == "ap" else vm["recall@10"]
            if cur > best:
                best, best_state, bad_epochs = cur, copy.deepcopy(model.state_dict()), 0
                rec["best"] = True
            else:
                bad_epochs += 1
        hist.append(rec)
        if tcfg.get("verbose", True):
            log("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
        if bad_epochs >= patience:
            break
    train_time = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"history": hist, "best_val": best, "epochs_run": len(hist), "train_time_s": train_time,
            "peak_memory_mb": _peak_memory_mb(device), "params": model.param_report(),
            "hparams": {"model": mcfg, "train": {k: v for k, v in tcfg.items() if k != "verbose"}}}
    return model, info