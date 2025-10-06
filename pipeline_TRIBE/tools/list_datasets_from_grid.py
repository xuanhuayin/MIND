# /home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/tools/list_datasets_from_grid.py
from pathlib import Path
import pandas as pd, re

GRID = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/timelines/grid_2hz_sub-01.parquet")
OUT_DIR = GRID.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] loading {GRID} …")
df = pd.read_parquet(GRID)

# 1) 唯一的 dataset 列表
ds = (df["dataset"].dropna().astype(str).unique().tolist())
ds.sort()
print(f"[INFO] unique datasets: {len(ds)}")

# 2) 可选：转成 Friends 文件名（friends_sXXeYY[a-d]）
def ds2friends(x: str):
    m = re.search(r"task-(s\d{2}e\d{2}[a-d])", x, flags=re.I)
    return f"friends_{m.group(1).lower()}" if m else None

friends = [ds2friends(x) for x in ds]

# 3) 每个 dataset 的网格行数（可理解为 2Hz bin 数）
counts = (df.groupby("dataset").size()
            .sort_values(ascending=False)
            .rename("bins_2hz"))

# 保存
(ds, friends, counts.to_frame()).__class__  # no-op
Path(OUT_DIR / "datasets_unique.txt").write_text("\n".join(ds), encoding="utf-8")
Path(OUT_DIR / "datasets_unique_friends.txt").write_text(
    "\n".join([x for x in friends if x]), encoding="utf-8"
)
counts.to_csv(OUT_DIR / "dataset_lengths_2hz.csv")

print("[DONE]")
print("  - 剧集ID列表:", OUT_DIR / "datasets_unique.txt")
print("  - Friends文件名列表:", OUT_DIR / "datasets_unique_friends.txt")
print("  - 每集2Hz bin数统计:", OUT_DIR / "dataset_lengths_2hz.csv")