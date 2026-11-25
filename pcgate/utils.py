"""Small shared helpers: seeding, hashing, JSON IO, config loading."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_array(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_json(obj: Any, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def load_config(path: str | os.PathLike) -> dict:
    """Load a YAML config; supports one level of `base:` inheritance."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        parent = load_config(path.parent / base)
        cfg = deep_update(parent, cfg)
    if "paths" in cfg:
        cfg["paths"] = {k: long_path(v) for k, v in cfg["paths"].items()}
    return cfg


def long_path(p: str | os.PathLike) -> str:
    """Absolute path; on Windows prefixed with \\\\?\\ so deep result files are not
    limited to 260 characters when long-path support is disabled."""
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + p
    return p


def deep_update(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# Controls must use exactly the gated model's frozen hyperparameters, otherwise a difference
# between "gated" and "uniform"/"param_matched" could come from tuning rather than the gate.
OVERRIDE_FALLBACK = {"uniform": "gated", "param_matched": "gated"}


def run_cfg(cfg: dict, name: str) -> dict:
    """Config for one model: `model_overrides.<name>` (written by `pcgate tune`) merged over the
    shared `model:` / `train:` blocks.  An override may hold `model:` and/or `train:` keys.
    Models without an override (and every config without `model_overrides`) are unchanged."""
    ov = cfg.get("model_overrides") or {}
    key = name if name in ov else OVERRIDE_FALLBACK.get(name)
    o = ov.get(key) or {}
    out = dict(cfg)
    out["model"] = deep_update(cfg.get("model", {}), o.get("model", {}))
    out["train"] = deep_update(cfg.get("train", {}), o.get("train", {}))
    return out


def get_logger(log_file: str | os.PathLike | None = None, name: str = "pcgate"):
    """Console + (optional) append-mode file logger with timestamps."""
    import logging

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = os.path.abspath(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def sanitize_type(t: str) -> str:
    """PrimeKG types such as 'gene/protein' -> 'gene_protein' (safe as module keys)."""
    return t.strip().replace("/", "_").replace(" ", "_").replace(".", "_")


def sanitize_relation(r: str) -> str:
    return r.strip().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")