# -*- coding: utf-8 -*-
from pathlib import Path
import sys

import pandas as pd

PARQUET = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/timelines/grid_2hz_sub-01.parquet")
TXT_OUT = PARQUET.with_name(PARQUET.stem + "_videos.txt")
CSV_OUT = PARQUET.with_name(PARQUET.stem + "_videos.csv")

# 读 parquet（优先 pyarrow，失败再尝试 fastparquet）
def read_parquet_any(p):
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return pd.read_parquet(p, engine="fastparquet")

df = read_parquet_any(PARQUET)

print("[INFO] columns:", list(df.columns))

# 自动选择“视频名字”列
candidates = ["stimulus_name", "stimulus", "video_name", "video", "clip", "movie", "dataset"]
name_col = next((c for c in candidates if c in df.columns), None)
if name_col is None:
    print("[ERROR] 没找到可用的名字列（尝试了:", candidates, ")")
    sys.exit(1)

print(f"[INFO] using column: {name_col!r} 作为视频片段名字")

# 计算基本列表
names = (
    df[name_col]
    .astype(str)
    .dropna()
    .drop_duplicates()
    .sort_values()
    .tolist()
)

# 可选统计：每个名字出现的 bin 数 & 总时长（若列存在）
has_time = ("win_start" in df.columns) and ("win_end" in df.columns)
if has_time:
    dur = (df["win_end"] - df["win_start"]).clip(lower=0)
    g = df.assign(_dur=dur).groupby(name_col, as_index=False).agg(
        n_bins=("bin_idx", "nunique") if "bin_idx" in df.columns else (name_col, "size"),
        total_seconds=("_dur", "sum"),
    )
    g = g.sort_values(name_col)
else:
    g = pd.DataFrame({name_col: names, "n_bins": pd.NA, "total_seconds": pd.NA})

# 保存 TXT（只名字）
with open(TXT_OUT, "w", encoding="utf-8") as f:
    for n in names:
        f.write(n + "\n")

# 保存 CSV（名字 + 统计）
g.rename(columns={name_col: "video_name"}, inplace=True)
g.to_csv(CSV_OUT, index=False)

print(f"[DONE] 共 {len(names)} 个视频片段名字")
print(f"       TXT: {TXT_OUT}")
print(f"       CSV: {CSV_OUT}")
print()
print(g.head(10).to_string(index=False))