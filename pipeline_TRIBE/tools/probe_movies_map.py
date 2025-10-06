# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json, subprocess, shlex
import pandas as pd
import numpy as np

PIPE_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
DATA_ROOT = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")

TL = PIPE_ROOT / "timelines" / "timeline_fmri_sub-01.parquet"
TRAIN_TXT = PIPE_ROOT / "TRIBE_8features" / "text" / "sub-01" / "train.txt"
VAL_TXT   = PIPE_ROOT / "TRIBE_8features" / "text" / "sub-01" / "val.txt"
MOVIES_DIR = DATA_ROOT / "stimuli" / "movies"

OUT_CSV = PIPE_ROOT / "timelines" / "movies_map.csv"

def read_lines(p: Path) -> list[str]:
    return [x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def ffprobe_duration_seconds(path: Path) -> float | None:
    # 需要系统安装 ffprobe（来自 ffmpeg）
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries format=duration -of json {shlex.quote(str(path))}'
    try:
        out = subprocess.check_output(cmd, shell=True, text=True)
        data = json.loads(out)
        dur = float(data.get("format", {}).get("duration", "nan"))
        return None if np.isnan(dur) else dur
    except Exception:
        return None

def main():
    assert TL.exists(), f"missing {TL}"
    assert MOVIES_DIR.exists(), f"missing {MOVIES_DIR}"
    # 选我们的 8 个 dataset
    ds_list = []
    for p in (TRAIN_TXT, VAL_TXT):
        if p.exists():
            ds_list += read_lines(p)
    ds_list = list(dict.fromkeys(ds_list))  # 去重保序

    tl = pd.read_parquet(TL)
    tl = tl[tl["subject"] == "sub-01"]
    # 每个 dataset 的期望时长（用 t_end 最后一行 - t_start 第一行）
    exp = (tl.groupby("dataset")
             .agg(t0=("t_start","min"), t1=("t_end","max"),
                  episode=("episode","first"), part=("part","first"))
             .reset_index())
    exp["expected_sec"] = exp["t1"] - exp["t0"]

    # 只保留我们要跑的 8 个
    exp = exp[exp["dataset"].isin(ds_list)].copy().reset_index(drop=True)

    # 列出所有 mkv
    mkvs = [p for p in MOVIES_DIR.rglob("*.mkv")]
    print(f"Scanning {len(mkvs)} mkv files for duration (this can take a while the first time)...")
    video_meta = []
    for p in mkvs:
        dur = ffprobe_duration_seconds(p)
        if dur is not None:
            video_meta.append((p, dur))
    vm = pd.DataFrame(video_meta, columns=["path","duration_sec"])
    if vm.empty:
        raise SystemExit("No mkv durations obtainable; ensure ffprobe/ffmpeg is installed and files are readable.")

    # 逐 dataset 找最近时长的 mkv
    rows = []
    for _, r in exp.iterrows():
        ds = r["dataset"]; need = r["expected_sec"]
        # 绝对差最小
        vm["diff"] = (vm["duration_sec"] - need).abs()
        best = vm.sort_values("diff").iloc[0]
        rows.append({
            "dataset": ds,
            "episode": r["episode"],
            "part": r["part"],
            "expected_sec": float(need),
            "matched_path": str(best["path"]),
            "matched_duration_sec": float(best["duration_sec"]),
            "abs_diff_sec": float(best["diff"]),
        })

    out = pd.DataFrame(rows).sort_values("dataset")
    out.to_csv(OUT_CSV, index=False)
    print("Saved movies map ->", OUT_CSV)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
