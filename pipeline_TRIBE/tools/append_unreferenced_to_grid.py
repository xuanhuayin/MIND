# /home/lawrence/Desktop/algonauts-2025/algonauts2025/tools/append_unreferenced_to_grid.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import re, json, subprocess, sys
import cv2

GRID_PATH = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/timelines/grid_2hz_sub-01.parquet")
STIM_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors/stimuli")
UNREF_TXT = Path("/home/lawrence/Desktop/algonauts-2025/unreferenced_mkvs.txt")

# 网格参数：2Hz 步长 0.5s，窗口长度 4.0s（win=[t-4.0, t]）
STEP = 0.5
WIN  = 4.0
EPS_START = 5e-7  # 和你现有 parquet 一致的小偏移
EPS_END   = 1e-6

re_friend = re.compile(r"friends_(s\d{2}e\d{2})([a-d])\.mkv$", re.IGNORECASE)

def parse_episode_from_mkv_name(name: str):
    """
    friends_s07e01a.mkv -> ("s07e01", "a")
    """
    m = re_friend.search(name)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower()

def get_duration_sec_ffprobe(path: Path) -> float | None:
    try:
        # ffprobe -v error -show_entries format=duration -of json INPUT
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path)
        ], stderr=subprocess.STDOUT)
        info = json.loads(out.decode("utf-8"))
        dur = float(info["format"]["duration"])
        if np.isfinite(dur) and dur > 0:
            return float(dur)
    except Exception:
        pass
    return None

def get_duration_sec_cv2(path: Path) -> float | None:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        if fps > 0 and nfr > 0:
            return float(nfr / fps)
    except Exception:
        pass
    return None

def get_duration_sec(path: Path) -> float:
    d = get_duration_sec_ffprobe(path)
    if d is None:
        d = get_duration_sec_cv2(path)
    if d is None:
        raise RuntimeError(f"Cannot read duration for {path}")
    return d

def next_free_dataset_id(existing: set[str], season_episode: str, part: str, start_idx: int = 900) -> str:
    """
    生成不与现有 dataset 冲突的新 ID：ses-9xx_task-sXXeYYp
    """
    # 如果恰好已有同名的 ses-xxx，我们用新 ses 号
    i = start_idx
    while True:
        ds = f"ses-{i:03d}_task-{season_episode}{part}"
        if ds not in existing:
            return ds
        i += 1

def build_grid_rows_for_video(dataset: str, duration: float) -> pd.DataFrame:
    """
    生成该视频所有 bin 行：bin_idx 从 0 开始，t_center 从 4.0 到 duration 之间，步长 0.5s。
    """
    if duration < WIN + 1e-3:
        # 太短的视频，直接空表
        return pd.DataFrame(columns=["dataset","bin_idx","t_center","win_start","win_end","n_words"])
    # k_max = floor((D - WIN)/STEP)
    k_max = int(np.floor((duration - WIN) / STEP))
    centers = 4.0 + np.arange(k_max + 1, dtype=np.float64) * STEP
    win_start = np.maximum(centers - WIN, 0.0) + EPS_START
    win_end   = centers + EPS_END
    df = pd.DataFrame({
        "dataset":  [dataset] * len(centers),
        "bin_idx":  np.arange(len(centers), dtype=np.int64),
        "t_center": centers,
        "win_start": win_start,
        "win_end":   win_end,
        "n_words":   np.nan,  # 保持与现有表一致
    })
    return df

def main():
    assert GRID_PATH.exists(), f"Grid parquet not found: {GRID_PATH}"
    assert UNREF_TXT.exists(), f"Missing list: {UNREF_TXT}"
    print("[INFO] Loading original grid…")
    grid = pd.read_parquet(GRID_PATH)
    have = set(grid["dataset"].unique())

    # 读取未引用的 MKV（相对 STIM_ROOT 的路径）
    lines = [ln.strip() for ln in UNREF_TXT.read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    mkv_rel_paths = [Path(ln) for ln in lines]
    # 拼出绝对路径
    mkvs = [STIM_ROOT / p for p in mkv_rel_paths]
    mkvs = [p for p in mkvs if p.suffix.lower()==".mkv" and p.exists()]

    print(f"[INFO] Grid datasets (before): {len(have)}")
    print(f"[INFO] Unreferenced MKVs to add: {len(mkvs)}")
    if not mkvs:
        print("[INFO] Nothing to add. Exit.")
        return

    new_rows = []
    mapping = []  # 记录新增 dataset ↔ 相对 mkv 路径
    for p_abs, p_rel in zip(mkvs, mkv_rel_paths):
        parsed = parse_episode_from_mkv_name(p_abs.name)
        if parsed is None:
            print(f"[WARN] Skip (name not matched): {p_rel}")
            continue
        season_episode, part = parsed  # sXXeYY, a/b/c/d
        # 分配一个不冲突的 dataset id
        ds = next_free_dataset_id(have, season_episode, part, start_idx=900)
        have.add(ds)

        try:
            dur = get_duration_sec(p_abs)
        except Exception as e:
            print(f"[WARN] duration fail for {p_rel}: {e}; skip.")
            continue

        df_rows = build_grid_rows_for_video(ds, dur)
        if df_rows.empty:
            print(f"[WARN] too short or no bins: {p_rel}")
            continue

        new_rows.append(df_rows)
        mapping.append({"dataset": ds, "mkv_rel": p_rel.as_posix(), "duration_sec": float(dur), "n_bins": int(df_rows.shape[0])})
        print(f"[OK] {p_rel} -> {ds}  bins={df_rows.shape[0]}  dur={dur:.1f}s")

    if not new_rows:
        print("[INFO] No rows generated. Exit.")
        return

    add_df = pd.concat(new_rows, axis=0, ignore_index=True)
    print(f"[INFO] Total new rows: {add_df.shape[0]}")

    # 备份再写回
    backup = GRID_PATH.with_suffix(".backup.parquet")
    GRID_PATH.replace(backup)
    print(f"[INFO] Backup saved: {backup.name}")

    merged = pd.concat([grid, add_df], axis=0, ignore_index=True)
    # 保持列顺序一致
    merged = merged[["dataset","bin_idx","t_center","win_start","win_end","n_words"]]
    merged.to_parquet(GRID_PATH, index=False)
    print(f"[INFO] Updated parquet written: {GRID_PATH}")

    # 写入映射表（便于后续对账）
    map_csv = GRID_PATH.with_name("added_datasets_map.csv")
    pd.DataFrame(mapping).to_csv(map_csv, index=False)
    print(f"[INFO] Mapping saved: {map_csv}")

    # 小结
    print("\n=== DONE ===")
    print(f"New datasets added: {len(mapping)}")
    print(f"Grid rows before: {len(grid)}  after: {len(merged)}")

if __name__ == "__main__":
    main()