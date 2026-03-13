# -*- coding: utf-8 -*-
"""Parcel-wise evaluation metrics (Pearson r, Spearman rho, R^2)."""
from __future__ import annotations

import numpy as np
import torch


def _rank1d(x: np.ndarray) -> np.ndarray:
    """Rank with averaged ties."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sx = x[order]
    n = x.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Pearson r per output parcel.  Input shape: [T, O]."""
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred ** 2).sum(axis=0) * (true ** 2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)


@torch.no_grad()
def voxelwise_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Spearman rho per output parcel.  Input shape: [T, O]."""
    _N, O = pred.shape
    rp = np.empty_like(pred, dtype=np.float64)
    rt = np.empty_like(true, dtype=np.float64)
    for o in range(O):
        rp[:, o] = _rank1d(pred[:, o])
        rt[:, o] = _rank1d(true[:, o])
    return voxelwise_pearson(rp.astype(np.float32), rt.astype(np.float32))


@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """R^2 per output parcel.  Input shape: [T, O]."""
    yt_mean = true.mean(axis=0, keepdims=True)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - yt_mean) ** 2).sum(axis=0) + 1e-8
    return (1.0 - ss_res / ss_tot).astype(np.float32)
