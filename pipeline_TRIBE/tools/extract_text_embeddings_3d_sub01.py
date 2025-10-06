# -*- coding: utf-8 -*-
"""
Fast single-GPU extract: 2Hz video embeddings for sub-01 using V-JEPA-2 Gigantic @ 256.
- Keeps the same outputs as before: per dataset (T_2Hz, N_LAYERS, 1408) float16
- Faster by:
  * larger default batch size
  * cudnn.benchmark=True
  * autocast(fp16) + inference_mode
  * avoid frequent empty_cache

ENV overrides:
  ALGONAUTS_VJEPA_ID           default "facebook/vjepa2-vitg-fpc64-256"
  ALGONAUTS_VIDEO_BATCH        default 12       # ↑ bigger default
  ALGONAUTS_VIDEO_KEEP_LAYERS  "last41" | "all" | "0,1,2,..."
  ALGONAUTS_VIDEO_FRAMES       default 64
"""

from __future__ import annotations
from pathlib import Path
import os, re, time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import cv2
import torch
from transformers import AutoVideoProcessor, AutoModel

# ───────── paths (portable) ─────────
from pathlib import Path
import os

SUBJECT = "sub-01"  # 放在前面，下面拼 grid 用得到

# 当前文件：.../algonauts2025/pipeline_TRIBE/tools/xxx.py
THIS_FILE  = Path(__file__).resolve()
PIPE_ROOT  = THIS_FILE.parents[1]                      # .../algonauts2025/pipeline_TRIBE
PROJ_ROOT  = PIPE_ROOT.parent                          # .../algonauts2025
DATA_ROOT  = PROJ_ROOT / "download" / "algonauts_2025.competitors"

GRID_2HZ   = PIPE_ROOT / "timelines" / f"grid_2hz_{SUBJECT}.parquet"
MOVIES_ROOT = DATA_ROOT / "stimuli" / "movies"

OUT_DIR    = PIPE_ROOT / "TRIBE_8features" / "video_2hz" / SUBJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FULL     = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8_full_episodes.txt"
SPLIT_FALLBACK = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8.txt"
SAVE_DTYPE = np.float16

# ───────── model / sampling cfg ─────────
MODEL_ID       = os.environ.get("ALGONAUTS_VJEPA_ID", "facebook/vjepa2-vitg-fpc64-256")
BATCH_SIZE     = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", 12))  # default bigger
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", 64))
KEEP_SPEC      = os.environ.get("ALGONAUTS_VIDEO_KEEP_LAYERS", "last41")

# ───────── tqdm ─────────
try:
    from tqdm.auto import tqdm
    def _tqdm(it, **kw): return tqdm(it, **kw)
except Exception:
    def _tqdm(it, **kw): return it

# ───────── utils ─────────
RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)

def try_parse_friends(ds: str) -> Optional[Tuple[str, str]]:
    m = RE_FRIENDS.search(ds)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower()  # ("s01e02","a")

# —— movies 解析：从 dataset 提取 task 名（去掉 _run-x），生成 basename 和可选系列名 ——
def parse_movie_task(ds: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    ds: like 'ses-001_task-bourne02' / 'ses-006_task-life01_run-1' / 'ses-003_task-wolf07' / 'ses-007_task-figures03_run-1'
    returns: (basename, series) e.g. ('bourne02','bourne'), ('life01','life'), ('wolf07','wolf'), ('figures03','figures')
    """
    if "_task-" not in ds:
        return None
    task = ds.split("_task-", 1)[1]
    # 去掉 _run-x
    task = task.split("_run-", 1)[0]
    basename = task.lower()
    # series: 去掉结尾数字与可选字母，比如 'bourne02' -> 'bourne', 'wolf07' -> 'wolf'
    m = re.match(r"([a-zA-Z]+)", basename)
    series = m.group(1).lower() if m else None
    return basename, series

# —— 为了高效查找，建立一次 movies 根目录下所有 .mkv 的索引（小写文件名 -> 绝对路径）——
_VIDEO_INDEX: Dict[str, Path] = {}

def build_video_index(root: Path):
    global _VIDEO_INDEX
    if _VIDEO_INDEX:
        return
    for dirpath, _, filenames in os.walk(str(root)):
        for fn in filenames:
            if fn.lower().endswith(".mkv"):
                key = fn.lower()
                p = Path(dirpath) / fn
                # 若重复文件名，保留先发现的；需要可改成列表
                if key not in _VIDEO_INDEX:
                    _VIDEO_INDEX[key] = p

def resolve_video_path(ds: str) -> Optional[Path]:
    """
    兼容 Friends + 其它 movies：
    - Friends：保留原有规则，在 movies/**/friends_* 结构下找。
    - Movies：从 task 中提取 basename，如 'bourne02'，优先找 */<series>/<basename>.mkv，找不到再全局索引。
    """
    # 先尝试 Friends
    parsed = try_parse_friends(ds)
    if parsed is not None:
        ep, part = parsed            # ("s01e02","a")
        season_num = int(ep[1:3])   # s01 -> 1
        friends_candidates = [
            MOVIES_ROOT / "friends" / f"s{season_num}"          / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"s{season_num:02d}"      / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"season{season_num}"     / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"season{season_num:02d}" / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"friends_{ep}{part}.mkv",
        ]
        for p in friends_candidates:
            if p.exists():
                return p
        # 如果 friends_* 规则失败，再落回通用搜索
        basename = f"{ep}{part}".lower()
        build_video_index(MOVIES_ROOT)
        return _VIDEO_INDEX.get(f"friends_{basename}.mkv")

    # Movies（非 Friends）
    pm = parse_movie_task(ds)
    if pm is None:
        return None
    basename, series = pm  # e.g. ('bourne02','bourne')

    # 先试 “*/<series>/<basename>.mkv”
    if series:
        series_dir_hits = list(MOVIES_ROOT.glob(f"**/{series}/{basename}.mkv"))
        if series_dir_hits:
            return series_dir_hits[0]

    # 再用全局索引兜底
    build_video_index(MOVIES_ROOT)
    hit = _VIDEO_INDEX.get(f"{basename}.mkv")
    if hit:
        return hit

    # 最后再做一轮全局 glob（避免极端大小写/软链接情况）
    hits = list(MOVIES_ROOT.glob(f"**/{basename}.mkv"))
    return hits[0] if hits else None

def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or nframes <= 0:
        cap.release()
        raise RuntimeError(f"Bad video meta (fps={fps}, nframes={nframes}) for {path}")
    return cap, fps, nframes

def sample_frame_indices_uniform(fps: float, a: float, b: float, k: int) -> List[int]:
    if k <= 0 or b <= a or fps <= 0: return []
    step = (b - a) / k
    times = [a + (i + 0.5) * step for i in range(k)]
    idxs = [int(round(t * fps)) for t in times]
    return [max(0, min(i, int(1e12))) for i in idxs]

def grab_frame_rgb(cap: cv2.VideoCapture, frame_idx: int, nframes: int) -> np.ndarray | None:
    frame_idx = min(max(0, frame_idx), max(0, nframes - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None: return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def choose_layer_indices(hidden_states_len: int) -> List[int]:
    spec = KEEP_SPEC.strip().lower()
    if spec == "with_embed":
        return list(range(0, hidden_states_len))
    all_blocks = list(range(1, hidden_states_len))  # drop embeddings @0
    if spec == "all":
        return all_blocks
    if spec == "last41":
        return all_blocks[-41:] if len(all_blocks) >= 41 else all_blocks
    try:
        ids = [int(x) for x in spec.split(",") if x.strip() != ""]
        out = [all_blocks[i] for i in ids if 0 <= i < len(all_blocks)]
        if out: return out
    except Exception:
        pass
    return all_blocks

def safe_mean_over_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise RuntimeError(f"unexpected hidden shape {tuple(x.shape)}")
    dims = list(range(1, x.ndim - 1))
    return x.mean(dim=dims)

# ───────── main ─────────
def main():
    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    # ❗去掉 split 二次过滤：不过滤，按 grid 全量跑
    # split_file = SPLIT_FULL if SPLIT_FULL.exists() else (SPLIT_FALLBACK if SPLIT_FALLBACK.exists() else None)
    # if split_file is not None:
    #     keep = {l.strip() for l in open(split_file, "r", encoding="utf-8") if l.strip()}
    #     grid = grid[grid["dataset"].isin(keep)].copy()
    #     print(f"[INFO] Filtering to {len(keep)} datasets from {split_file.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True  # speed up for fixed shapes
    print(f"[INFO] Loading V-JEPA-2 model: {MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to(device).eval()

    # probe hidden_states & dim
    dummy = np.zeros((FRAMES_PER_BIN, 256, 256, 3), dtype=np.uint8)
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        d_inputs = processor(videos=[dummy], return_tensors="pt").to(device, non_blocking=True)
        d_out = model(**d_inputs, output_hidden_states=True)
    hs_len = len(d_out.hidden_states)
    layer_ids = choose_layer_indices(hs_len)
    D = getattr(getattr(model, "config", model), "hidden_size", 1408)
    N_LAYERS = len(layer_ids)
    print(f"[INFO] hidden_states len={hs_len}, keep {N_LAYERS} layers, feature dim={D}")
    del d_out, d_inputs, dummy

    index_rows = []

    # prebuild index once to speed movie lookup
    build_video_index(MOVIES_ROOT)

    # per dataset
    for ds, g in grid.groupby("dataset", sort=True):
        g = g.sort_values("bin_idx").reset_index(drop=True)
        vpath = resolve_video_path(ds)
        if vpath is None:
            print(f"[WARN] Skip dataset without resolvable video file: {ds}")
            continue

        try:
            cap, fps, nframes = open_video(vpath)
        except Exception as e:
            print(f"[WARN] Cannot open video for {ds}: {e}")
            continue

        T_bins = len(g)
        out_arr = np.zeros((T_bins, N_LAYERS, int(D)), dtype=SAVE_DTYPE)

        batch_clips: List[np.ndarray] = []
        batch_slots: List[int] = []

        def flush():
            if not batch_clips:
                return
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                inputs = processor(videos=batch_clips, return_tensors="pt")
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                vecs_per_layer = []
                for lid in layer_ids:
                    h = outputs.hidden_states[lid]           # [B, ..., D]
                    pooled = safe_mean_over_tokens(h)        # [B, D]
                    vecs_per_layer.append(pooled)
                all_layers = torch.stack(vecs_per_layer, dim=1)   # [B, N_LAYERS, D]
                arr = all_layers.float().cpu().numpy().astype(SAVE_DTYPE)

            for i, bidx in enumerate(batch_slots):
                out_arr[bidx, :, :] = arr[i]
            batch_clips.clear()
            batch_slots.clear()

        print(f"[DATASET] {ds}  bins={T_bins}  video={vpath.name}  fps={fps:.2f} frames={nframes}")
        t0 = time.time()
        for _, r in _tqdm(g.iterrows(), total=T_bins, desc=f"{ds}"):
            bidx = int(r["bin_idx"])
            a = float(r["win_start"]); b = float(r["win_end"])
            fids = sample_frame_indices_uniform(fps, a, b, FRAMES_PER_BIN)
            if not fids:
                continue

            frames = []
            ok = True
            for fi in fids:
                rgb = grab_frame_rgb(cap, fi, nframes)
                if rgb is None:
                    ok = False; break
                frames.append(rgb)
            if not ok or len(frames) != FRAMES_PER_BIN:
                continue  # keep zeros

            clip = np.stack(frames, axis=0)  # [T, H, W, 3], uint8
            batch_clips.append(clip)
            batch_slots.append(bidx)

            if len(batch_clips) >= BATCH_SIZE:
                flush()

        flush()
        cap.release()
        dt = time.time() - t0

        out_path = OUT_DIR / f"{ds}.npy"
        np.save(out_path, out_arr)
        index_rows.append([ds, str(out_path), T_bins, N_LAYERS, int(D)])
        print(f"[OK] {ds}: saved {out_path.name} shape={out_arr.shape}  elapsed={dt:.1f}s")

    if index_rows:
        idx = pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"])
        idx.to_csv(OUT_DIR / "index.csv", index=False)
        print("\n[DONE] video_2hz index.csv ->", OUT_DIR)
    else:
        print("[WARN] No dataset processed. Check grid and video files.")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()