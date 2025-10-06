# -*- coding: utf-8 -*-
"""
Fast single-GPU extract: 2Hz video embeddings for sub-01 using V-JEPA-2 Gigantic @ 256.

新增功能：
- --batch_size 控制批大小（优先于环境变量）
- --cuda 选择GPU编号（如 0/1；负数或无GPU则用CPU）
- --order forward/reverse 控制根据 --only_list 的顺序正反处理（若未给 only_list，则对 grid 的 dataset 顺序正反）
- 其余保持：GPU 上预处理、float32 推理与保存、每个数据集打印 mean/var/std
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

# 兼容不同 transformers 版本拿 processor
try:
    from transformers import AutoVideoProcessor as _AutoProc  # >=4.45
except Exception:
    from transformers import AutoImageProcessor as _AutoProc  # 4.4x 可用

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ======== 并行/CPU 线程设置 ========
PREFETCH_WORKERS = int(os.environ.get("ALGONAUTS_PREFETCH", "4"))
MAX_INFLIGHT     = int(os.environ.get("ALGONAUTS_MAX_INFLIGHT", "16"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
cv2.setNumThreads(2)

# ======== 路径 ========
SUBJECT   = "sub-01"
THIS_FILE = Path(__file__).resolve()
PIPE_ROOT = THIS_FILE.parents[1]                                  # .../algonauts2025/pipeline_TRIBE
PROJ_ROOT = PIPE_ROOT.parent                                      # .../algonauts2025
DATA_ROOT = PROJ_ROOT / "download" / "algonauts_2025.competitors"

GRID_2HZ     = PIPE_ROOT / "timelines" / f"grid_2hz_{SUBJECT}.parquet"
MOVIES_ROOT  = DATA_ROOT / "stimuli" / "movies"
OUT_DIR      = PIPE_ROOT / "TRIBE_8features" / "video_2hz" / SUBJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_DTYPE = np.float32  # 输出 float32

# ======== 模型/采样配置（仍支持环境变量，参数优先覆盖） ========
MODEL_ID       = os.environ.get("ALGONAUTS_VJEPA_ID", "facebook/vjepa2-vitg-fpc64-256")
ENV_BATCH      = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", 12))
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", 64))
KEEP_SPEC      = os.environ.get("ALGONAUTS_VIDEO_KEEP_LAYERS", "last41")

# ======== 仅重算坏样本：参数与默认路径 ========
DEFAULT_BAD_LIST = PIPE_ROOT / "TRIBE_8features" / "video_2hz" / "bad_video_ids.txt"

# ======== 正则工具 ========
RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)

def try_parse_friends(ds: str) -> Optional[Tuple[str, str]]:
    m = RE_FRIENDS.search(ds)
    if not m: return None
    return m.group(1).lower(), m.group(2).lower()  # ("s01e02","a")

def parse_movie_task(ds: str) -> Optional[Tuple[str, Optional[str]]]:
    if "_task-" not in ds: return None
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
    if _VIDEO_INDEX: return
    for dirpath, _, filenames in os.walk(str(root)):
        for fn in filenames:
            if fn.lower().endswith(".mkv"):
                _VIDEO_INDEX[fn.lower()] = Path(dirpath) / fn

def resolve_video_path(ds: str) -> Optional[Path]:
    parsed = try_parse_friends(ds)
    if parsed:
        ep, part = parsed
        season_num = int(ep[1:3])
        candidates = [
            MOVIES_ROOT / "friends" / f"s{season_num}"          / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"s{season_num:02d}"      / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"season{season_num}"     / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"season{season_num:02d}" / f"friends_{ep}{part}.mkv",
            MOVIES_ROOT / "friends" / f"friends_{ep}{part}.mkv",
        ]
        for p in candidates:
            if p.exists(): return p
        build_video_index(MOVIES_ROOT)
        return _VIDEO_INDEX.get(f"friends_{ep}{part}.mkv")

    pm = parse_movie_task(ds)
    if not pm: return None
    basename, series = pm
    if series:
        hits = list(MOVIES_ROOT.glob(f"**/{series}/{basename}.mkv"))
        if hits: return hits[0]
    build_video_index(MOVIES_ROOT)
    return _VIDEO_INDEX.get(f"{basename}.mkv") or next(iter(MOVIES_ROOT.glob(f"**/{basename}.mkv")), None)

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
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): return None
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
    if spec == "with_embed": return list(range(0, hidden_states_len))
    all_blocks = list(range(1, hidden_states_len))  # drop embeddings @0
    if spec == "all": return all_blocks
    if spec == "last41": return all_blocks[-41:] if len(all_blocks) >= 41 else all_blocks
    try:
        ids = [int(x) for x in spec.split(",") if x.strip() != ""]
        out = [all_blocks[i] for i in ids if 0 <= i < len(all_blocks)]
        return out or all_blocks
    except Exception:
        return all_blocks

def safe_mean_over_tokens(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise RuntimeError(f"unexpected hidden shape {tuple(x.shape)}")
    dims = list(range(1, x.ndim - 1))
    return x.mean(dim=dims)

# ======== 主流程 ========
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_list", type=str, default=str(DEFAULT_BAD_LIST),
                    help="仅处理该列表中的 ID（每行一个 stem）；若文件不存在则处理全部。")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="每次 flush 的视频 clip 数；默认为环境变量 ALGONAUTS_VIDEO_BATCH 或 12。")
    ap.add_argument("--cuda", type=int, default=0,
                    help="选择 GPU 编号；<0 或不可用时使用 CPU。")
    ap.add_argument("--order", type=str, default="forward", choices=["forward","reverse"],
                    help="当提供 --only_list 时按该文件顺序正向/反向处理；未提供时对 grid 的 dataset 顺序正向/反向处理。")
    args = ap.parse_args()

    # 批大小：参数优先，未给则用环境变量
    BATCH_SIZE = int(args.batch_size if args.batch_size and args.batch_size > 0 else ENV_BATCH)
    print(f"[CFG] batch_size={BATCH_SIZE}")

    # 设备选择
    if torch.cuda.is_available() and args.cuda is not None and args.cuda >= 0:
        device = torch.device(f"cuda:{args.cuda}")
        print(f"[DEV] Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print("[DEV] Using CPU")

    # 读取 only_list 顺序（如存在）
    only_list_path = Path(args.only_list) if args.only_list else DEFAULT_BAD_LIST
    only_ids_ordered: Optional[List[str]] = None
    if only_list_path.exists():
        with open(only_list_path, "r", encoding="utf-8") as f:
            only_ids_ordered = [ln.strip() for ln in f if ln.strip()]
        if args.order == "reverse":
            only_ids_ordered = list(reversed(only_ids_ordered))
        print(f"[FILTER] Only processing {len(only_ids_ordered)} IDs from {only_list_path} (order={args.order})")

    # 载入 grid
    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    # 模型与 processor（float32）
    torch.backends.cudnn.benchmark = True
    processor = _AutoProc.from_pretrained(MODEL_ID)  # 仅取参数
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device).eval()

    # 从 processor 取归一化与尺寸信息
    ip = getattr(processor, "image_processor", processor)
    mean = torch.tensor(getattr(ip, "image_mean", [0.485, 0.456, 0.406]),
                        device=device, dtype=torch.float32).view(1,1,3,1,1)
    std  = torch.tensor(getattr(ip, "image_std",  [0.229, 0.224, 0.225]),
                        device=device, dtype=torch.float32).view(1,1,3,1,1)
    do_rescale     = bool(getattr(ip, "do_rescale", True))
    rescale_factor = float(getattr(ip, "rescale_factor", 1.0/255.0))
    tgt_size = int(getattr(ip, "size", {}).get("shortest_edge", 256)) or 256

    amp_ctx = nullcontext()  # 全程 float32，不用混合精度

    # 干跑拿 hidden_states 维度
    dummy = np.zeros((FRAMES_PER_BIN, tgt_size, tgt_size, 3), dtype=np.uint8)
    with torch.inference_mode(), amp_ctx:
        x = torch.from_numpy(dummy[None]).to(device).float()        # [1,T,H,W,3] -> fp32
        x = x.permute(0,1,4,2,3).contiguous()                       # [1,T,3,H,W]
        if do_rescale: x = x * rescale_factor
        x = (x - mean) / std
        d_out = model(pixel_values_videos=x, output_hidden_states=True)
    hs_len = len(d_out.hidden_states)
    layer_ids = choose_layer_indices(hs_len)
    D = getattr(getattr(model, "config", model), "hidden_size", 1408)
    N_LAYERS = len(layer_ids)
    print(f"[INFO] hidden_states={hs_len}, keep={N_LAYERS}, dim={D}")
    del d_out, x, dummy

    # 决定处理的 dataset 顺序
    if only_ids_ordered is not None:
        datasets = only_ids_ordered
    else:
        # 按 grid 出现顺序（groupby 保持原顺序），并可根据 --order 反转
        datasets = list(dict.fromkeys(grid["dataset"].tolist()))
        if args.order == "reverse":
            datasets = list(reversed(datasets))
        print(f"[ORDER] Processing {len(datasets)} datasets from grid (order={args.order})")

    # 主循环
    build_video_index(MOVIES_ROOT)

    for ds in tqdm(datasets, desc="Datasets"):
        # 若提供 only_list，但 grid 不含该 ds（理论少见），跳过
        if ds not in set(grid["dataset"]):
            print(f"[WARN] {ds} not found in grid; skip")
            continue

        g = grid[grid["dataset"] == ds].sort_values("bin_idx").reset_index(drop=True)
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
            np_batch = np.stack(batch_clips, axis=0)               # [B,T,H,W,3] uint8
            B, T, H, W, C = np_batch.shape
            with torch.inference_mode(), amp_ctx:
                x = torch.from_numpy(np_batch).to(device).float()  # fp32
                x = x.permute(0,1,4,2,3).contiguous()              # [B,T,3,H,W]
                x = torch.nn.functional.interpolate(
                        x.view(B*T, 3, H, W),
                        size=(tgt_size, tgt_size),
                        mode="bilinear", align_corners=False
                    ).view(B, T, 3, tgt_size, tgt_size)
                if do_rescale:
                    x = x * rescale_factor
                x = (x - mean) / std
                outputs = model(pixel_values_videos=x, output_hidden_states=True)
                vecs = [safe_mean_over_tokens(outputs.hidden_states[lid]) for lid in layer_ids]
                all_layers = torch.stack(vecs, dim=1)              # [B,N_LAYERS,D]
                arr = all_layers.detach().cpu().numpy().astype(SAVE_DTYPE, copy=False)
            for i, bidx in enumerate(batch_slots):
                out_arr[bidx, :, :] = arr[i]
            batch_clips.clear()
            batch_slots.clear()

        print(f"[DATASET] {ds}  bins={T_bins}  video={vpath.name}  fps={fps:.2f} frames={nframes}")
        t0 = time.time()

        # 预取流水
        tasks = [(int(r["bin_idx"]), float(r["win_start"]), float(r["win_end"])) for _, r in g.iterrows()]
        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
            inflight: Dict = {}
            it = iter(tasks)
            # 先填满 inflight
            while len(inflight) < min(MAX_INFLIGHT, len(tasks)):
                try:
                    bidx, a, b = next(it)
                except StopIteration:
                    break
                inflight[ex.submit(read_clip_cv, vpath, a, b, FRAMES_PER_BIN)] = bidx

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

                    # 补位
                    try:
                        nbidx, na, nb = next(it)
                        inflight[ex.submit(read_clip_cv, vpath, na, nb, FRAMES_PER_BIN)] = nbidx
                    except StopIteration:
                        pass
                    break
            pbar.close()

        # 收尾
        flush()
        cap.release()
        dt = time.time() - t0

        # 保存 + 统计
        out_path = OUT_DIR / f"{ds}.npy"
        np.save(out_path, out_arr)
        finite = out_arr[np.isfinite(out_arr)]
        mean_v = float(finite.mean()) if finite.size else np.nan
        var_v  = float(finite.var())  if finite.size else np.nan
        std_v  = float(finite.std())  if finite.size else np.nan
        print(f"[OK] {ds}: saved {out_path.name} shape={out_arr.shape}  elapsed={dt:.1f}s")
        print(f"  ↳ mean={mean_v:.4f}, var={var_v:.4f}, std={std_v:.4f}")

    print("\n[DONE] Extraction finished.")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()