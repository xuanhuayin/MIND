# -*- coding: utf-8 -*-
"""
把 sub-01 的 friends fMRI HDF5 批量导出为每集/半集的 .npy：
- 从 .h5 中找出所有 2D 且包含 1000 parcel 的数据集（键名一般就是 ses-XXX_task-sYYeZZ[a-d]）
- 统一导出为 [1000, T] float32
- 文件名：friends_sYYeZZ[a-d]_fmri.npy
- 可选：用 --id_list 仅导出指定列表里的 ds（如 train/val 的 txt）
"""

from __future__ import annotations
import re, argparse
from pathlib import Path
import numpy as np
import h5py

RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2}[a-d])$", re.IGNORECASE)

def ds2friends(ds: str) -> str | None:
    m = RE_FRIENDS.search(ds)
    if not m:
        return None
    return f"friends_{m.group(1).lower()}"

def iter_candidate_dsets(h: h5py.File):
    """遍历所有 2D 且一维为 1000 的 dataset，返回 (key_path, dset)"""
    def _walk(g, path=""):
        for k, v in g.items():
            p = f"{path}/{k}" if path else k
            if isinstance(v, h5py.Dataset):
                if len(v.shape) == 2 and (1000 in v.shape):
                    yield p, v
            elif isinstance(v, h5py.Group):
                yield from _walk(v, p)
    yield from _walk(h, "")

def read_id_list(txt: str) -> set[str]:
    return {ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True,
                    help="e.g. .../sub-01/...desc-s123456_bold.h5 (friends)")
    ap.add_argument("--out_dir", required=True,
                    help="输出目录，比如 .../algonauts2025/fmri_data/sub1")
    ap.add_argument("--id_list", default="",
                    help="可选：只导出这个 txt 列表中的 ds（如 train_full_episodes.txt）")
    ap.add_argument("--channels_first", action="store_true",
                    help="导出 [1000, T]（默认 True）；若关闭则导出 [T,1000]")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    id_allow = read_id_list(args.id_list) if args.id_list else None
    to_channels_first = True if args.channels_first or args.channels_first is None else False
    # 默认导出 [1000, T]
    to_channels_first = True

    h5_path = Path(args.h5).resolve()
    print(f"[INFO] open: {h5_path}")
    n_ok = n_skip = 0

    with h5py.File(h5_path, "r") as h:
        for key, dset in iter_candidate_dsets(h):
            ds = key.split("/")[-1]  # 例如 ses-001_task-s01e02a
            if id_allow is not None and ds not in id_allow:
                n_skip += 1
                continue
            fr = ds2friends(ds)
            if fr is None:
                # 非 friends（如果 h5 里有别的任务会被跳过）
                n_skip += 1
                continue

            # 读取并标准化形状
            arr = np.asarray(dset)           # 可能是 (T,1000) 或 (1000,T)
            if arr.ndim != 2 or 1000 not in arr.shape:
                n_skip += 1
                continue
            if to_channels_first:
                if arr.shape[0] != 1000:
                    arr = arr.T              # -> (1000, T)
            else:
                if arr.shape[1] != 1000:
                    arr = arr.T              # -> (T,1000)
            arr = arr.astype(np.float32, copy=False)

            # 保存
            out = out_dir / f"{fr}_fmri.npy"
            np.save(out, arr)
            n_ok += 1
            print(f"[OK] {ds} -> {out.name}  shape={arr.shape}")

    print(f"\n[DONE] exported={n_ok}  skipped={n_skip}  -> {out_dir}")

if __name__ == "__main__":
    main()