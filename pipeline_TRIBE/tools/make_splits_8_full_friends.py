from __future__ import annotations
from pathlib import Path
import re, json
import pandas as pd

PIPE_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
GRID_PATH  = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"
SPLITS_DIR = PIPE_ROOT / "TRIBE_8features" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)

def main():
    grid = pd.read_parquet(GRID_PATH)
    ds_all = grid["dataset"].drop_duplicates().tolist()

    ep_part_to_ds = {}
    ep_to_parts   = {}
    for ds in ds_all:
        m = RE_FRIENDS.search(ds)
        if not m:
            continue
        ep   = m.group(1).lower()
        part = m.group(2).lower()
        ep_part_to_ds[(ep, part)] = ds
        ep_to_parts.setdefault(ep, set()).add(part)

    # 不检查文件存在性：按完整度优先（4>3>2>1），再按 ep 排序
    candidates = sorted(
        ((ep, sorted(parts)) for ep, parts in ep_to_parts.items()),
        key=lambda x: (-len(x[1]), x[0])
    )

    picked = candidates[:8]
    if len(picked) < 8:
        print(f"[WARN] 只有 {len(picked)} 集可用。")

    ds_list = []
    episode_splits = []
    for ep, parts in picked:
        for p in parts:
            ds_list.append(ep_part_to_ds[(ep, p)])
        episode_splits.append({"episode": ep, "parts": parts})

    train_eps = set(ep for ep, _ in picked[:7])
    val_eps   = set(ep for ep, _ in picked[7:8])

    train_ds = [ds for ds in ds_list if RE_FRIENDS.search(ds).group(1).lower() in train_eps]
    val_ds   = [ds for ds in ds_list if RE_FRIENDS.search(ds).group(1).lower() in val_eps]

    (SPLITS_DIR / "datasets_8_full_episodes.txt").write_text("\n".join(ds_list), encoding="utf-8")
    (SPLITS_DIR / "train_full_episodes.txt").write_text("\n".join(train_ds), encoding="utf-8")
    (SPLITS_DIR / "val_full_episodes.txt").write_text("\n".join(val_ds), encoding="utf-8")
    (SPLITS_DIR / "episodes_8.json").write_text(json.dumps(episode_splits, indent=2), encoding="utf-8")

    print(f"[OK] 选中 {len(picked)} 集：")
    for ep, parts in picked:
        print(f" - {ep}: parts={''.join(parts)}")
    print("[OK] splits 写入：", SPLITS_DIR)

if __name__ == "__main__":
    main()
