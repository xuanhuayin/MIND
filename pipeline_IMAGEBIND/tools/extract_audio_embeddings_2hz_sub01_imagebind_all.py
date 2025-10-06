# -*- coding: utf-8 -*-
"""
Fast single-GPU extract: 2Hz video embeddings for sub-01 using V-JEPA-2 Gigantic @ 256.
- portable paths
- threaded prefetch & pipeline feeding
- memory-safe flush with gc/empty_cache
- skip datasets already extracted (shape-validated)
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext
import os, re, time

import numpy as np
import pandas as pd
import cv2
import torch
from transformers import AutoModel

# ---- transformers processor: 兼容老版本 ----
try:
    from transformers import AutoVideoProcessor as _AutoProc  # 需要 >=4.45
except Exception:
    from transformers import AutoImageProcessor as _AutoProc  # 4.4x 也可用

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ======== 并行/CPU 线程设置 ========
PREFETCH_WORKERS = int(os.environ.get("ALGONAUTS_PREFETCH", "4"))
MAX_INFLIGHT     = int(os.environ.get("ALGONAUTS_MAX_INFLIGHT", "16"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
cv2.setNumThreads(2)

# ======== 路径（相对 + 可移植） ========
SUBJECT   = "sub-01"
THIS_FILE = Path(__file__).resolve()
PIPE_ROOT = THIS_FILE.parents[1]                                  # .../algonauts2025/pipeline_TRIBE
PROJ_ROOT = PIPE_ROOT.parent                                      # .../algonauts2025
DATA_ROOT = PROJ_ROOT / "download" / "algonauts_2025.competitors"

GRID_2HZ     = PIPE_ROOT / "timelines" / f"grid_2hz_{SUBJECT}.parquet"
MOVIES_ROOT  = DATA_ROOT / "stimuli" / "movies"
OUT_DIR      = PIPE_ROOT / "TRIBE_8features" / "video_2hz" / SUBJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FULL     = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8_full_episodes.txt"
SPLIT_FALLBACK = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8.txt"
SAVE_DTYPE     = np.float16

# ======== 模型/采样配置（支持 ENV 覆盖） ========
MODEL_ID       = os.environ.get("ALGONAUTS_VJEPA_ID", "facebook/vjepa2-vitg-fpc64-256")
BATCH_SIZE     = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", 12))
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", 64))
KEEP_SPEC      = os.environ.get("ALGONAUTS_VIDEO_KEEP_LAYERS", "last41")

# ======== 正则工具 ========
RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)

def try_parse_friends(ds: str) -> Optional[Tuple[str, str]]:
    m = RE_FRIENDS.search(ds)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower()  # ("s01e02","a")

def parse_movie_task(ds: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    ds: 'ses-001_task-bourne02' / 'ses-006_task-life01_run-1' ...
    returns: (basename, series) e.g. ('bourne02','bourne')
    """
    if "_task-" not in ds:
        return None
    task = ds.split("_task-", 1)[1]
    task = task.split("_run-", 1)[0]
    basename = task.lower()
    m = re.match(r"([a-zA-Z]+)", basename)
    series = m.group(1).lower() if m else None
    return basename, series

# ======== 视频索引 ========
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
                if key not in _VIDEO_INDEX:
                    _VIDEO_INDEX[key] = p

def resolve_video_path(ds: str) -> Optional[Path]:
    # Friends
    parsed = try_parse_friends(ds)
    if parsed is not None:
        ep, part = parsed            # ("s01e02","a")
        season_num = int(ep[1:3])    # s01 -> 1
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
        basename = f"{ep}{part}".lower()
        build_video_index(MOVIES_ROOT)
        return _VIDEO_INDEX.get(f"friends_{basename}.mkv")

    # Movies
    pm = parse_movie_task(ds)
    if pm is None:
        return None
    basename, series = pm

    if series:
        series_dir_hits = list(MOVIES_ROOT.glob(f"**/{series}/{basename}.mkv"))
        if series_dir_hits:
            return series_dir_hits[0]

    build_video_index(MOVIES_ROOT)
    hit = _VIDEO_INDEX.get(f"{basename}.mkv")
    if hit:
        return hit

    hits = list(MOVIES_ROOT.glob(f"**/{basename}.mkv"))
    return hits[0] if hits else None

# ======== I/O & 采样 ========
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

def read_clip_cv(path: Path, a: float, b: float, k: int) -> Optional[np.ndarray]:
    """独立打开视频，取 [a,b) 时间窗内均匀采样 k 帧，返回 [T,H,W,3] uint8。失败返回 None。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or nframes <= 0:
        cap.release(); return None
    fids = sample_frame_indices_uniform(fps, a, b, k)
    frames = []
    for fi in fids:
        fi = min(max(0, fi), max(0, nframes - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release(); return None
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) != k:
        return None
    return np.stack(frames, axis=0)

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

# ======== 主流程 ========
def main():
    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Loading V-JEPA-2 model: {MODEL_ID}")
    processor = _AutoProc.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to(device).eval()

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()

    # probe hidden_states & dim
    dummy = np.zeros((FRAMES_PER_BIN, 256, 256, 3), dtype=np.uint8)
    with torch.inference_mode(), amp_ctx:
        d_inputs = processor(videos=[dummy], return_tensors="pt")
        d_inputs = {k: (v.pin_memory() if hasattr(v, "pin_memory") else v) for k, v in d_inputs.items()}
        d_inputs = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in d_inputs.items()}
        d_out = model(**d_inputs, output_hidden_states=True)
    hs_len = len(d_out.hidden_states)
    layer_ids = choose_layer_indices(hs_len)
    D = getattr(getattr(model, "config", model), "hidden_size", 1408)
    N_LAYERS = len(layer_ids)
    print(f"[INFO] hidden_states len={hs_len}, keep {N_LAYERS} layers, feature dim={D}")
    del d_out, d_inputs, dummy
    torch.cuda.empty_cache()
    import gc; gc.collect()

    index_rows = []
    build_video_index(MOVIES_ROOT)

    # ====== 每个数据集（视频）循环 ======
    for ds, g in grid.groupby("dataset", sort=True):
        out_path = OUT_DIR / f"{ds}.npy"

        # ---- 跳过已存在的结果（shape 校验通过才跳）----
        if out_path.exists():
            try:
                arr_exist = np.load(out_path, mmap_mode="r")
                if arr_exist.ndim == 3 and arr_exist.shape[0] == len(g):
                    print(f"[SKIP] {ds}: {out_path.name} 已存在 shape={arr_exist.shape}")
                    index_rows.append([ds, str(out_path), arr_exist.shape[0], arr_exist.shape[1], arr_exist.shape[2]])
                    continue
                else:
                    print(f"[WARN] {ds}: {out_path.name} 形状 {arr_exist.shape} 与 bins {len(g)} 不匹配，重新提取")
            except Exception as e:
                print(f"[WARN] {ds}: 读取已有文件失败 ({e})，重新提取")

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

        # ---------- flush：送入模型并写回、释放内存 ----------
        def flush():
            """把已累积的 batch_clips 送入模型，写回 out_arr，并及时释放显存/内存。"""
            if not batch_clips:
                return
            with torch.inference_mode(), amp_ctx:
                inputs = processor(videos=batch_clips, return_tensors="pt")
                inputs = {k: (v.pin_memory() if hasattr(v, "pin_memory") else v) for k, v in inputs.items()}
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                outputs = model(**inputs, output_hidden_states=True)
                vecs_per_layer = []
                for lid in layer_ids:
                    h = outputs.hidden_states[lid]      # [B, ..., D]
                    pooled = safe_mean_over_tokens(h)   # [B, D]
                    vecs_per_layer.append(pooled)
                all_layers = torch.stack(vecs_per_layer, dim=1)   # [B, N_LAYERS, D]
                arr = all_layers.to(dtype=torch.float16).cpu().numpy().astype(SAVE_DTYPE, copy=False)

            for i, bidx in enumerate(batch_slots):
                out_arr[bidx, :, :] = arr[i]

            batch_clips.clear()
            batch_slots.clear()

            # 显式释放大对象 + 回收显存/内存
            del outputs, inputs, vecs_per_layer, all_layers, arr
            torch.cuda.empty_cache()
            import gc; gc.collect()

        print(f"[DATASET] {ds}  bins={T_bins}  video={vpath.name}  fps={fps:.2f} frames={nframes}")
        t0 = time.time()

        # ---------- 并行预取与流水化 ----------
        tasks = [(int(r["bin_idx"]), float(r["win_start"]), float(r["win_end"])) for _, r in g.iterrows()]

        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
            inflight: Dict = {}
            it = iter(tasks)

            # 先填满 in-flight 队列
            while len(inflight) < min(MAX_INFLIGHT, len(tasks)):
                try:
                    bidx, a, b = next(it)
                except StopIteration:
                    break
                fut = ex.submit(read_clip_cv, vpath, a, b, FRAMES_PER_BIN)
                inflight[fut] = bidx

            pbar = tqdm(total=T_bins, desc=f"{ds}")
            while inflight:
                for fut in as_completed(list(inflight.keys()), timeout=None):
                    bidx = inflight.pop(fut)
                    clip = fut.result()
                    if clip is not None:
                        batch_clips.append(clip)
                        batch_slots.append(bidx)
                        if len(batch_clips) >= BATCH_SIZE:
                            flush()
                    pbar.update(1)

                    # 补位，保持 in-flight 满载
                    try:
                        nbidx, na, nb = next(it)
                        nfut = ex.submit(read_clip_cv, vpath, na, nb, FRAMES_PER_BIN)
                        inflight[nfut] = nbidx
                    except StopIteration:
                        pass
                    break
            pbar.close()

        # 处理尾批
        flush()
        cap.release()

        dt = time.time() - t0
        np.save(out_path, out_arr)
        index_rows.append([ds, str(out_path), T_bins, N_LAYERS, int(D)])
        print(f"[OK] {ds}: saved {out_path.name} shape={out_arr.shape}  elapsed={dt:.1f}s")

        # —— 每个数据集结束后，再做一次全局清理，防止长跑内存上涨 ——
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # 索引文件
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