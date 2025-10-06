# -*- coding: utf-8 -*-
# pipeline_TRIBE/tools/generate_timeline_sub01.py

from pathlib import Path
import re
import numpy as np
import pandas as pd
import h5py

# 数据根（competitors 数据集）
DATA_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")
# 输出到 pipeline_TRIBE/timelines
OUT_DIR   = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/timelines")
SUBJECT   = "sub-01"

DEFAULT_TR_SEC = 1.49  # 后续如拿到精确 TR，可替换

RE_DS   = re.compile(r"^ses-(?P<session>\d+)_task-(?P<episode>s\d+e\d+)(?P<part>[a-d])$", re.IGNORECASE)
RE_TASK = re.compile(r"task-(?P<task>friends|movie10)", re.IGNORECASE)

def parse_ds_name(ds_name: str):
    m = RE_DS.match(ds_name)
    if not m:
        return None, None, None
    return m.group("session"), m.group("episode").lower(), m.group("part").lower()

def parse_task_from_filename(fname: str):
    m = RE_TASK.search(fname)
    return (m.group("task").lower() if m else None)

def expand_one_file(h5_path: Path, subject: str) -> pd.DataFrame:
    task = parse_task_from_filename(h5_path.name)
    rows = []
    with h5py.File(h5_path, "r") as f:
        for ds_name, ds in f.items():
            if not isinstance(ds, h5py.Dataset):  # 只要 dataset
                continue
            if ds.ndim != 2 or ds.shape[1] != 1000:
                continue
            n_tr = int(ds.shape[0])
            TR   = float(ds.attrs.get("RepetitionTime", DEFAULT_TR_SEC))
            session, episode, part = parse_ds_name(ds_name)

            t = np.arange(n_tr, dtype=float) * TR
            df = pd.DataFrame({
                "subject":  subject,
                "task":     task,          # friends / movie10
                "file":     h5_path.name,
                "dataset":  ds_name,       # ses-xxx_task-sxxexx[a-d]
                "session":  session,       # "003"
                "episode":  episode,       # "s01e01"
                "part":     part,          # "a"/"b"/"c"/"d"
                "tr_index": np.arange(n_tr, dtype=int),
                "t_start":  t,
                "t_end":    t + TR,
                "TR":       TR,
                "n_parcels": 1000,
                "type":     "Fmri",
            })
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def main():
    fmri_dir = DATA_ROOT / "fmri" / SUBJECT / "func"
    files = sorted(fmri_dir.glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 under {fmri_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    parts = []
    for fp in files:
        df = expand_one_file(fp, SUBJECT)
        if df.empty:
            print(f"[SKIP] {fp.name}: no usable datasets."); continue
        # 排序：task -> session(int) -> episode -> part -> tr
        df["_session_int"] = pd.to_numeric(df["session"], errors="coerce").fillna(-1).astype(int)
        df.sort_values(["task", "_session_int", "episode", "part", "tr_index"], inplace=True)
        df.drop(columns=["_session_int"], inplace=True)
        parts.append(df)
        print(f"[OK] {fp.name}: datasets={df['dataset'].nunique()} rows={len(df)} TR≈{df['TR'].iloc[0]}")

    if not parts:
        raise SystemExit("No datasets expanded.")

    timeline = pd.concat(parts, ignore_index=True)
    timeline["global_index"] = np.arange(len(timeline), dtype=int)

    p_parquet = OUT_DIR / "timeline_fmri_sub-01.parquet"
    p_csv     = OUT_DIR / "timeline_fmri_sub-01.csv"
    timeline.to_parquet(p_parquet, index=False)
    timeline.to_csv(p_csv, index=False)
    print("Saved:", p_parquet)
    print("Saved:", p_csv)

    print("\nSummary head:")
    print(timeline.head())

if __name__ == "__main__":
    main()
