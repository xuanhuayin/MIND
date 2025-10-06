# -*- coding: utf-8 -*-
"""
Build a 2 Hz alignment grid for sub-01 that all modalities will share.
For each dataset, we create bins at 2 Hz (every 0.5 s). Each bin 't' uses a
causal context window [t-4.0, t) as in TRIBE.
Outputs:
  - grid_2hz_sub-01.parquet
      columns: [dataset, bin_idx, t_center, win_start, win_end, n_words(optional)]
  - map_2hz_to_tr_sub-01.parquet
      columns: [dataset, bin_idx, tr_index, overlap_sec, weight]
    where 'weight' is normalized per (dataset, bin_idx) so sum(weight)=1
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

PIPE_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
TIMELINE_PARQUET = PIPE_ROOT / "timelines" / "timeline_fmri_sub-01.parquet"
TEXT_EVENTS_PARQUET = PIPE_ROOT / "timelines" / "text_events_sub-01.parquet"  # 可不存在
OUT_GRID = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"
OUT_MAP  = PIPE_ROOT / "timelines" / "map_2hz_to_tr_sub-01.parquet"

SUBJECT = "sub-01"
BIN_HZ = 2.0          # 2 Hz
BIN_STEP = 1.0 / BIN_HZ   # 0.5s
WIN_SEC = 4.0         # window length [t-4, t)

def _build_bins_for_dataset(ds_grid: pd.DataFrame) -> pd.DataFrame:
    """Return 2Hz bins for a dataset with columns:
       [dataset, bin_idx, t_center, win_start, win_end]
    """
    ds = ds_grid["dataset"].iloc[0]
    t0 = float(ds_grid["t_start"].min())
    t1 = float(ds_grid["t_end"].max())
    # 以 BIN_STEP 采样中心点；窗口是 [t-4, t)
    # 为了保证第一个窗口不空，中心点最早可到 t0+WIN_SEC
    t_start_center = t0 + WIN_SEC + 0.5*1e-6
    centers = np.arange(t_start_center, t1, BIN_STEP, dtype=np.float64)
    if centers.size == 0:
        return pd.DataFrame(columns=["dataset","bin_idx","t_center","win_start","win_end"])
    df = pd.DataFrame({
        "dataset": ds,
        "bin_idx": np.arange(len(centers), dtype=int),
        "t_center": centers,
    })
    df["win_start"] = df["t_center"] - WIN_SEC
    df["win_end"]   = df["t_center"]
    return df

def _interval_overlap(a0, a1, b0, b1):
    """
    Vectorized intersection length between intervals [a0, a1) and [b0, b1)
    a0, a1, b0, b1 可以是标量或 np.ndarray
    返回与 b0-b1 区间的交集长度
    """
    lo = np.maximum(a0, b0)
    hi = np.minimum(a1, b1)
    overlap = np.maximum(0, hi - lo)
    return overlap

def _map_bins_to_tr(ds_grid: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Compute overlap of each 2Hz bin window with TR intervals.
       Returns columns: [dataset, bin_idx, tr_index, overlap_sec, weight]
    """
    out_rows = []
    ds = ds_grid["dataset"].iloc[0]
    tr_starts = ds_grid["t_start"].to_numpy(dtype=np.float64)
    tr_ends   = ds_grid["t_end"].to_numpy(dtype=np.float64)
    tr_idx    = ds_grid["tr_index"].to_numpy(dtype=int)

    for _, r in bins.iterrows():
        w0 = float(r["win_start"]); w1 = float(r["win_end"])
        overlaps = _interval_overlap(tr_starts, tr_ends, w0, w1)  # vectorized
        # 计算每个 TR 的 overlap（逐元素）
        if np.isscalar(overlaps):
            # 标量是广播不成功的信号；回退手算
            overlaps = np.array([_interval_overlap(a0, a1, w0, w1) for a0, a1 in zip(tr_starts, tr_ends)], dtype=np.float64)
        total = float(overlaps.sum())
        if total <= 0:
            # 不与任何 TR 相交，仍写一条 weight=0 的记录，避免后续崩
            out_rows.append([ds, int(r["bin_idx"]), int(tr_idx[0]), 0.0, 0.0])
        else:
            for tr_i, ov in zip(tr_idx, overlaps):
                if ov > 0:
                    out_rows.append([ds, int(r["bin_idx"]), int(tr_i), float(ov), float(ov/total)])

    return pd.DataFrame(out_rows, columns=["dataset","bin_idx","tr_index","overlap_sec","weight"])

def main():
    assert TIMELINE_PARQUET.exists(), f"missing {TIMELINE_PARQUET}"
    tl = pd.read_parquet(TIMELINE_PARQUET)
    tl = tl[tl["subject"] == SUBJECT].copy()
    if tl.empty:
        raise SystemExit("timeline empty for sub-01")

    # 为每个 dataset 生成 2Hz bins
    grids = []
    maps  = []
    for ds, ds_grid in tl.groupby("dataset", sort=True):
        ds_grid = ds_grid.sort_values("tr_index").reset_index(drop=True)
        bins = _build_bins_for_dataset(ds_grid)
        if bins.empty:
            continue
        grids.append(bins)
        m = _map_bins_to_tr(ds_grid, bins)
        maps.append(m)

    if not grids:
        raise SystemExit("no bins generated")

    grid_all = pd.concat(grids, ignore_index=True)
    map_all  = pd.concat(maps,  ignore_index=True)

    # 如果有文本事件，统计每个 bin 内的词数（可选）
    if TEXT_EVENTS_PARQUET.exists():
        txt = pd.read_parquet(TEXT_EVENTS_PARQUET)
        txt = txt[txt["subject"] == SUBJECT]
        if not txt.empty:
            # 假设文本事件里有列：dataset, word_t（词的时间戳，或 onsets_per_tr 展开后的绝对时间）
            # 如果你的 text_events 没有绝对时间列，请先在构建它时加入一列 't_abs'
            time_col_candidates = [c for c in ["t_abs","time","onset","t"] if c in txt.columns]
            if time_col_candidates:
                tcol = time_col_candidates[0]
                # 关联计数：bin 的 [win_start, win_end) 内的词数
                # 为了效率，逐 dataset 做区间计数
                counts = []
                for ds, g in grid_all.groupby("dataset"):
                    g = g.copy()
                    # 该 dataset 的全部词
                    tt = txt[txt["dataset"] == ds].get(tcol)
                    if tt is None or tt.empty:
                        g["n_words"] = 0
                    else:
                        arr = tt.to_numpy(dtype=np.float64)
                        # 对每个 bin 计数
                        n = []
                        for _, r in g.iterrows():
                            w0 = float(r["win_start"]); w1 = float(r["win_end"])
                            n.append(int(((arr >= w0) & (arr < w1)).sum()))
                        g["n_words"] = n
                    counts.append(g)
                grid_all = pd.concat(counts, ignore_index=True)
            else:
                # 没有时间列就先不计数
                grid_all["n_words"] = np.nan
        else:
            grid_all["n_words"] = np.nan
    else:
        grid_all["n_words"] = np.nan

    # 落盘
    OUT_GRID.parent.mkdir(parents=True, exist_ok=True)
    grid_all.to_parquet(OUT_GRID, index=False)
    map_all.to_parquet(OUT_MAP, index=False)

    print("[OK] saved:", OUT_GRID, "rows=", len(grid_all))
    print("[OK] saved:", OUT_MAP,  "rows=", len(map_all))
    # 小预览
    print("\nGrid preview:")
    print(grid_all.head(5).to_string(index=False))
    print("\nMap preview:")
    print(map_all.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
