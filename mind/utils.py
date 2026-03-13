# -*- coding: utf-8 -*-
"""Shared utilities: seed, I/O, layer selection, fMRI file resolver."""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Episode list I/O
# --------------------------------------------------------------------------- #
def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]


# --------------------------------------------------------------------------- #
#  Layer selection helpers
# --------------------------------------------------------------------------- #
def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    """Group layers by fraction boundaries and average within each group."""
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions)) or [L - 1]
    if idxs[-1] != L - 1:
        idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]
    starts = [0] + bounds[:-1]
    groups = []
    for s, e in zip(starts, bounds):
        s = max(0, min(s, L))
        e = max(0, min(e, L))
        if e <= s:
            s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(groups, axis=0)


def parse_layers_arg(layers_arg: str, probe_L: int) -> Tuple[str, list]:
    """
    Parse a ``--layers`` string into ``(mode, payload)``.

    Examples::

        ""           -> ("indices", [probe_L - 1])       # last layer only
        "all"        -> ("indices", [0, 1, ..., L-1])
        "last4"      -> ("indices", [L-4, ..., L-1])
        "idx:0,3,7"  -> ("indices", [0, 3, 7])
        "0.25,0.75"  -> ("fractions", [0.25, 0.75])      # group_mean boundaries
    """
    s = (layers_arg or "").strip().lower()
    if not s:
        return "indices", [probe_L - 1]
    if s == "all":
        return "indices", list(range(probe_L))
    if s.startswith("last"):
        try:
            k = int(s.replace("last", ""))
        except ValueError:
            k = 1
        k = max(1, min(k, probe_L))
        return "indices", list(range(max(0, probe_L - k), probe_L))
    if s.startswith("idx:"):
        idxs = []
        for p in s[4:].split(","):
            p = p.strip()
            if not p:
                continue
            try:
                i = int(p)
                if 0 <= i < probe_L:
                    idxs.append(i)
            except ValueError:
                pass
        return "indices", sorted(set(idxs or [probe_L - 1]))
    try:
        fracs = [min(1.0, max(0.0, float(x))) for x in s.split(",") if x.strip()]
        return "fractions", (fracs or [1.0])
    except ValueError:
        return "indices", [probe_L - 1]


# --------------------------------------------------------------------------- #
#  fMRI file resolver
# --------------------------------------------------------------------------- #
_task_rx = re.compile(r"(task-[A-Za-z0-9]+(?:_[^.]*)?)", re.IGNORECASE)


def fmri_canonical(root: Path, ds: str) -> Path:
    """Resolve an fMRI .npy path that may use different naming conventions."""
    p = Path(root) / f"{ds}.npy"
    if p.exists():
        return p
    m = _task_rx.search(ds)
    if m:
        key = m.group(1)
        cands = sorted(Path(root).glob(f"*_{key}.npy")) + sorted(Path(root).glob(f"*{key}.npy"))
        if cands:
            return cands[0]
    parts = ds.split("_", 1)
    if len(parts) == 2:
        suf = parts[1]
        cands = sorted(Path(root).glob(f"*_{suf}.npy")) + sorted(Path(root).glob(f"*{suf}.npy"))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"fMRI GT not found for '{ds}' under '{root}'")


def load_fmri_flexible(root: Path, ds: str) -> np.ndarray:
    """Load fMRI ground truth, resolving filename variations."""
    return np.load(fmri_canonical(root, ds))


def orient_fmri(arr: np.ndarray) -> np.ndarray:
    """Ensure fMRI array has shape [O=1000, T]."""
    if 1000 in arr.shape:
        return arr if arr.shape[0] == 1000 else arr.T
    return arr.T if arr.shape[0] > arr.shape[1] else arr
