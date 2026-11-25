"""Bounded hyperparameter tuning on VALIDATION data only (Member 2: HGT and gated HGT).

    python -m pcgate tune --config configs/primekg_eligible_tune.yaml --only hgt gated \
        --freeze-to configs/primekg_eligible_frozen.yaml

Protocol
  * Every model gets the same grid and the same trial budget (`tune.max_trials`), so no model
    is favoured by more search.
  * Selection uses the best validation AP reached during training (early stopping is on
    validation AP too).  The test pool is never scored here.
  * Trials run on `tune.settings` (default: disease-held-out, the setting the gate targets)
    with `tune.seeds` (default: [0]).  When several settings/seeds are given, the mean
    validation AP decides.
  * uniform / param_matched are NOT tuned: they reuse the gated model's frozen values so the
    controls differ from the gate only in the decoder (see utils.OVERRIDE_FALLBACK).

Outputs (under <results>/tuning/)
  trials.csv              one row per (model, trial, setting, seed): hparams, val AP, val R@10, time
  best.json               selected hparams per model + the validation score that chose them
  frozen_overrides.yaml   `model_overrides:` block, ready to paste into a config
  --freeze-to PATH        also writes a complete config  {base: <tuned config>, model_overrides: ...}
"""
from __future__ import annotations

import itertools
import os
import json
import random
import time
from pathlib import Path

import pandas as pd
import yaml

from .utils import deep_update, write_json

# Default grid: small on purpose (bounded tuning, proposal 10.6).  Keys are "block.param".
DEFAULT_GRID = {
    "model.hidden": [64, 128],
    "train.lr": [1.0e-3, 5.0e-4],
}


def _trials(grid: dict, max_trials: int, seed: int) -> list[dict]:
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    if len(combos) > max_trials:
        # fixed-seed subsample, identical for every model -> identical budget AND identical points
        combos = random.Random(seed).sample(combos, max_trials)
    return combos


def _as_override(trial: dict) -> dict:
    out: dict = {}
    for k, v in trial.items():
        block, param = k.split(".", 1)
        out.setdefault(block, {})[param] = v
    # the decoder MLP width follows the encoder width unless the grid sets it explicitly
    if "hidden" in out.get("model", {}) and "decoder_hidden" not in out["model"]:
        out["model"]["decoder_hidden"] = out["model"]["hidden"]
    return out


def tune(cfg: dict, load_fn, models, device, log=print, freeze_to: str | None = None,
         config_path: str | None = None) -> Path:
    from .train import train_model

    t = cfg.get("tune", {})
    grid = t.get("grid", DEFAULT_GRID)
    settings = t.get("settings", ["disease"])
    seeds = t.get("seeds", [0])
    max_trials = int(t.get("max_trials", 4))   # proposal 9.2: <= 4 configs per model
    trials = _trials(grid, max_trials, t.get("sample_seed", 0))
    out_dir = Path(cfg["paths"]["results"]) / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = out_dir / "trials.csv"
    done = pd.read_csv(trials_csv) if trials_csv.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    log(f"tuning {list(models)} on {settings} seeds={seeds}: {len(trials)} trials each "
        f"(grid size {len(list(itertools.product(*grid.values())))})  [validation only]")

    for setting in settings:
        split, graph = load_fn(cfg, setting, device)
        for name in models:
            for ti, trial in enumerate(trials):
                for seed in seeds:
                    key = json.dumps(trial, sort_keys=True)
                    if len(done) and ((done["model"] == name) & (done["setting"] == setting) &
                                      (done["seed"] == seed) & (done["trial"] == key)).any():
                        continue                                       # resumable
                    tcfg = deep_update(cfg, _as_override(trial))
                    tcfg["model_overrides"] = {}                        # the trial IS the config
                    tcfg["train"] = dict(tcfg["train"], verbose=False,
                                         epochs=t.get("epochs", tcfg["train"].get("epochs", 100)),
                                         patience=t.get("patience", tcfg["train"].get("patience", 10)))
                    t0 = time.time()
                    _, info = train_model(name, graph, split, tcfg, seed, device, log=log)
                    best_row = max((h for h in info["history"] if "val_ap" in h), key=lambda h: h["val_ap"])
                    rows.append({"model": name, "setting": setting, "seed": seed, "trial_id": ti, "trial": key,
                                 **trial, "val_ap": info["best_val"], "val_recall@10": best_row["val_recall@10"],
                                 "best_epoch": best_row["epoch"], "epochs_run": info["epochs_run"],
                                 "params": info["params"]["total"], "time_s": time.time() - t0})
                    pd.DataFrame(rows).to_csv(trials_csv, index=False)
                    log(f"  [{setting}] {name} trial {ti + 1}/{len(trials)} seed {seed}: "
                        f"val AP={info['best_val']:.4f}  R@10={best_row['val_recall@10']:.4f}  {trial}")
        del split, graph

    df = pd.DataFrame(rows)
    df = df[df["model"].isin(models)]
    best, overrides = {}, {}
    for name, g in df.groupby("model"):
        agg = g.groupby("trial").agg(val_ap=("val_ap", "mean"), val_r10=("val_recall@10", "mean"),
                                     n=("val_ap", "size")).reset_index()
        # tie-break on validation Recall@10, then on the smaller model (fewer params is cheaper)
        agg = agg.sort_values(["val_ap", "val_r10"], ascending=False)
        win = json.loads(agg.iloc[0]["trial"])
        best[name] = {"hparams": win, "mean_val_ap": float(agg.iloc[0]["val_ap"]),
                      "mean_val_recall@10": float(agg.iloc[0]["val_r10"]), "n_runs": int(agg.iloc[0]["n"]),
                      "n_trials": int(len(agg)), "settings": settings, "seeds": seeds}
        overrides[name] = _as_override(win)
    write_json(best, out_dir / "best.json")
    with open(out_dir / "frozen_overrides.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"model_overrides": overrides}, f, sort_keys=True)
    if freeze_to:
        fp = Path(freeze_to)
        base = Path(config_path).resolve()
        try:
            base_ref = str(base.relative_to(fp.resolve().parent))
        except ValueError:
            base_ref = str(base)
        with open(fp, "w", encoding="utf-8") as f:
            f.write("# FROZEN configuration - written by `pcgate tune` from validation data only.\n"
                    "# Do not edit by hand; do not re-tune after test results are seen.\n"
                    f"# Source: {os.path.relpath(out_dir / 'best.json')}\n")
            yaml.safe_dump({"base": base_ref, "model_overrides": overrides}, f, sort_keys=False)
        log(f"wrote frozen config {fp}")
    log(json.dumps(best, indent=2))
    return out_dir