#!/usr/bin/env python3
from pathlib import Path
import numpy as np

VIDEO = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/TRIBE_8features/video_2hz/sub-01")
ALL_LIST = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/TRIBE_8features/all_list.txt")
OUT_BAD = VIDEO.parent / "bad_video_ids.txt"

def is_npy_magic(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(6) == b"\x93NUMPY"
    except:
        return False

def main():
    ids = [ln.strip() for ln in open(ALL_LIST, "r", encoding="utf-8") if ln.strip()]
    bad = []
    for stem in ids:
        f = VIDEO / f"{stem}.npy"
        if (not f.exists()) or (not is_npy_magic(f)):
            bad.append(stem); continue
        try:
            arr = np.load(f)  # 不允许 allow_pickle
            if arr.ndim != 3:
                bad.append(stem)
        except Exception:
            bad.append(stem)
    OUT_BAD.write_text("\n".join(bad), encoding="utf-8")
    print(f"[DONE] {len(bad)} bad/missing video files -> {OUT_BAD}")

if __name__ == "__main__":
    main()