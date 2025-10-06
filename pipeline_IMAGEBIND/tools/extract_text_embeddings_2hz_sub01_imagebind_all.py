# -*- coding: utf-8 -*-
"""
ImageBind TEXT → 2Hz timed text embeddings for sub-01 (run ALL datasets by default).

- 对每个词 w，用上文 k=CONTEXT_WORDS 个词（含 w）组成一句，喂入 ImageBind TEXT（CLIP式整句向量）
- 将落入同一 2Hz bin 的词向量在 GPU 上求和
- 输出每个 dataset： (T_2Hz, N_LAYERS, D)

ENV:
  RUN_DATASETS                   e.g. "ses-001_task-s01e02a,ses-035_task-s04e23a"
  ALGONAUTS_TEXT_CTX_WORDS       default 1024
  ALGONAUTS_TEXT_BATCH           default 64
  ALGONAUTS_IMAGEBIND_LAYERS     default 1
"""

from __future__ import annotations

# ---- pytorchvideo 兼容 shim（必须最顶部）----
import sys, types
try:
    import torchvision.transforms.functional_tensor as _ft
except Exception:
    from torchvision.transforms import functional as _f
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.__dict__.update(_f.__dict__)
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
# ---- 兼容 shim 结束 ----

from pathlib import Path
import os
import argparse
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ImageBind
from imagebind.models import imagebind_model
try:
    from imagebind.models.imagebind_model import ModalityType as _MBModalityType
    TEXT_KEY = _MBModalityType.TEXT
except Exception:
    TEXT_KEY = "text"
from imagebind import data as ib_data

# ───────── paths ─────────
PIPE_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
GRID_2HZ  = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"
TEXT_EVT  = PIPE_ROOT / "timelines" / "text_events_sub-01.parquet"

# 输出目录（按你的要求）
OUT_DIR   = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_IMAGEBIND/features/text_2hz/sub-01")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FULL     = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8_full_episodes.txt"
SPLIT_FALLBACK = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8.txt"

SUBJECT     = "sub-01"
SAVE_DTYPE  = np.float32

# ───────── cfg ─────────
RUN_DATASETS    = [s.strip() for s in os.environ.get("RUN_DATASETS", os.environ.get("ALGONAUTS_RUN_DATASETS","")).split(",") if s.strip()]
BATCH_WORDS     = int(os.environ.get("ALGONAUTS_TEXT_BATCH", 64))
CONTEXT_WORDS   = int(os.environ.get("ALGONAUTS_TEXT_CTX_WORDS", 1024))
N_LAYERS        = max(1, int(os.environ.get("ALGONAUTS_IMAGEBIND_LAYERS", "1")))

# ───────── helpers ─────────
def build_contexts(words: List[str], k: int) -> List[str]:
    ctx = []
    for i in range(len(words)):
        st = max(0, i - k + 1)
        ctx.append(" ".join(words[st:i+1]))
    return ctx

def _load_text_batch(texts: List[str], device: str) -> Union[torch.Tensor, Dict]:
    if hasattr(ib_data, "load_and_transform_text"):
        try:
            return ib_data.load_and_transform_text(texts, device)
        except TypeError:
            return ib_data.load_and_transform_text(texts, device=device)
    if hasattr(ib_data, "load_and_transform_text_data"):
        try:
            return ib_data.load_and_transform_text_data(texts, device)
        except TypeError:
            return ib_data.load_and_transform_text_data(texts, device=device)
    raise RuntimeError("imagebind.data: no text loader found.")

def as_ib_dict(x: Union[torch.Tensor, Dict]) -> Dict:
    if isinstance(x, dict):
        if TEXT_KEY in x: return x
        if "text" in x:   return {TEXT_KEY: x["text"]}
        k, v = next(iter(x.items()))
        return {TEXT_KEY: v}
    elif isinstance(x, torch.Tensor):
        return {TEXT_KEY: x}
    else:
        raise TypeError(type(x))

def find_overlapping_bins(bin_starts: np.ndarray, bin_ends: np.ndarray, w0: float, w1: float) -> np.ndarray:
    lo = np.maximum(bin_starts, w0)
    hi = np.minimum(bin_ends, w1)
    return np.nonzero(lo < hi)[0]

# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--respect-split", action="store_true",
                    help="Use split files to filter datasets. By default we ignore split and run ALL datasets in grid.")
    args = ap.parse_args()

    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass

    # 读取 parquet
    try:
        grid = pd.read_parquet(GRID_2HZ, engine="pyarrow")
    except Exception:
        grid = pd.read_parquet(GRID_2HZ, engine="fastparquet")
    try:
        txt  = pd.read_parquet(TEXT_EVT, engine="pyarrow")
    except Exception:
        txt  = pd.read_parquet(TEXT_EVT, engine="fastparquet")

    # 统一时标列：支持 (w_start,w_end) / (onset,duration) / (onset,offset)
    cols = set(txt.columns)
    if {"w_start", "w_end"}.issubset(cols):
        txt = txt.rename(columns={"w_start": "onset", "w_end": "offset"})
    elif {"onset", "duration"}.issubset(cols):
        txt["offset"] = txt["onset"].astype(float) + np.maximum(0.0, txt["duration"].astype(float).to_numpy())
    elif {"onset", "offset"}.issubset(cols):
        # 已经是最终格式
        txt["onset"]  = txt["onset"].astype(float)
        txt["offset"] = txt["offset"].astype(float)
    else:
        raise SystemExit(f"Text events parquet needs (w_start,w_end) or (onset,duration) or (onset,offset). Got: {list(txt.columns)}")

    # subject 过滤
    if "subject" not in grid.columns:
        grid["subject"] = SUBJECT
    if "subject" not in txt.columns:
        txt["subject"] = SUBJECT
    grid = grid[grid["subject"] == SUBJECT].copy()
    txt  = txt[txt["subject"] == SUBJECT].copy()

    # 是否使用 split
    if args.respect_split:
        split_file = SPLIT_FULL if SPLIT_FULL.exists() else (SPLIT_FALLBACK if SPLIT_FALLBACK.exists() else None)
        if split_file is not None:
            keep = {l.strip() for l in open(split_file, "r", encoding="utf-8") if l.strip()}
            grid = grid[grid["dataset"].isin(keep)].copy()
            print(f"[INFO] Respect split: {split_file.name} ({len(keep)} datasets)")
    else:
        print("[INFO] Ignoring split; running ALL datasets present in grid parquet.")

    # 可选只跑子集
    if RUN_DATASETS:
        keep2 = set(RUN_DATASETS)
        grid = grid[grid["dataset"].isin(keep2)].copy()
        print(f"[INFO] Further restricting to {len(keep2)} via RUN_DATASETS")

    if grid.empty:
        raise SystemExit("Empty 2Hz grid after filtering.")
    if txt.empty:
        print("[WARN] No text rows for subject after filtering.")

    # 设备与模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print("[INFO] Loading ImageBind model (FP32): imagebind_huge")
    model = imagebind_model.imagebind_huge(pretrained=True).to(device).eval()

    # 探测 D
    with torch.inference_mode():
        probe = as_ib_dict(_load_text_batch(["hello world"], device))
        pout = model(probe)
        D = int(pout.get(TEXT_KEY, next(iter(pout.values()))).shape[-1])
    print(f"[INFO] feature dim (ImageBind TEXT) = {D}; N_LAYERS (repeat) = {N_LAYERS}; batch={BATCH_WORDS}")

    index_rows = []

    # 逐 dataset
    for ds, g in grid.groupby("dataset", sort=True):
        g = g.sort_values("bin_idx").reset_index(drop=True)
        bins_start = g["win_start"].to_numpy(dtype=float)
        bins_end   = g["win_end"].to_numpy(dtype=float)
        T_bins = len(g)

        words_df = txt[txt["dataset"] == ds].copy().sort_values(["onset","offset"]).reset_index(drop=True)

        # GPU 累加缓存 (T, N, D)
        out_gpu = torch.zeros((T_bins, N_LAYERS, D), device=device, dtype=torch.float32)

        if words_df.empty:
            out_path = OUT_DIR / f"{ds}.npy"
            np.save(out_path, out_gpu.cpu().numpy().astype(SAVE_DTYPE))
            index_rows.append([ds, str(out_path), T_bins, N_LAYERS, D])
            print(f"[INFO] {ds}: no text; zeros -> {(T_bins, N_LAYERS, D)}")
            continue

        words = words_df["word"].astype(str).tolist()
        w_on  = words_df["onset"].to_numpy(dtype=float)
        w_off = words_df["offset"].to_numpy(dtype=float)
        contexts = build_contexts(words, CONTEXT_WORDS)

        for bstart in tqdm(range(0, len(words), BATCH_WORDS), desc=f"{ds} text->2Hz", leave=False):
            bpos = list(range(bstart, min(len(words), bstart + BATCH_WORDS)))
            batch_ctx = [contexts[i] for i in bpos]

            with torch.inference_mode():
                t_in  = as_ib_dict(_load_text_batch(batch_ctx, device))
                tout  = model(t_in)
                vec   = tout.get(TEXT_KEY, next(iter(tout.values())))   # [B, D] on device
                vec   = vec.float()

                if N_LAYERS == 1:
                    vec_rep = vec.unsqueeze(1)
                else:
                    vec_rep = vec.unsqueeze(1).expand(-1, N_LAYERS, -1)

            for j, pos in enumerate(bpos):
                hit = find_overlapping_bins(bins_start, bins_end, float(w_on[pos]), float(w_off[pos]))
                if hit.size == 0:
                    continue
                hit_t = torch.from_numpy(hit.astype(np.int64)).to(device, non_blocking=True)  # [H]
                add_blk = vec_rep[j].unsqueeze(0).expand(hit_t.numel(), N_LAYERS, D)          # [H,N,D]
                out_gpu.index_add_(0, hit_t, add_blk)

        out_path = OUT_DIR / f"{ds}.npy"
        np.save(out_path, out_gpu.detach().cpu().numpy().astype(SAVE_DTYPE))
        index_rows.append([ds, str(out_path), T_bins, N_LAYERS, D])
        print(f"[OK] {ds}: saved {out_path.name} shape={(T_bins, N_LAYERS, D)}")

    idx = pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"])
    idx.to_csv(OUT_DIR / "index_imagebind_text_all.csv", index=False)
    print("\n[DONE] text_2hz (ImageBind, ALL datasets) index ->", OUT_DIR)

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()