# -*- coding: utf-8 -*-
"""
FP32 + Safe defaults (pure torch resize + CLIP norm) + dataset filter:
2Hz video embeddings for sub-01 using ImageBind (VISION only).

- Data pipeline: decord (CPU) + PyTorch DataLoader (num_workers, prefetch) + GPU torch preprocess
- Model inference: FP32 (no AMP/TF32)
- Outputs per dataset: [T_2Hz, N_LAYERS, D] float32
  * D is usually 1024 for imagebind_huge
  * N_LAYERS is configurable via env ALGONAUTS_IMAGEBIND_LAYERS (default 1)
- Optional filters:
    --run-datasets ses-035_task-s04e23a,ses-001_task-s01e02a
    --max-datasets 5
"""

from __future__ import annotations

# ---- pytorchvideo 旧接口兼容 shim（必须放在最顶部，早于任何 imagebind/pytorchvideo 导入）----
import sys, types
try:
    import torchvision.transforms.functional_tensor as _ft
except Exception:
    from torchvision.transforms import functional as _f
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.__dict__.update(_f.__dict__)
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
# ---- 兼容 shim 结束 ----

# ---- ImageBind Modality key 兼容（有枚举则用枚举；否则退回字符串）--------------------------
try:
    from imagebind.models.imagebind_model import ModalityType as _MBModalityType  # noqa: E402
    VISION_KEY = _MBModalityType.VISION
except Exception:
    VISION_KEY = "vision"
# -------------------------------------------------------------------------------------

from pathlib import Path
import os, re, time, argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import cv2
except Exception:
    cv2 = None

import decord
decord.bridge.set_bridge('torch')  # get torch tensors from decord

# ───────── ImageBind ─────────
from imagebind.models import imagebind_model

# ───────── paths ─────────
PIPE_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
DATA_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")
GRID_2HZ   = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"
MOVIES_DIR = DATA_ROOT / "stimuli" / "movies" / "friends"

OUT_DIR = Path("/algonauts2025/pipeline_IMAGEBIND/TRIBE_8features/video_2hz/sub-01")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT    = "sub-01"
SAVE_DTYPE = np.float32   # 如需更小体积可改 np.float16（会有量化误差）

# —— 吞吐/占用相关（不影响结果） ——
BATCH_SIZE     = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", 8))
NUM_WORKERS    = int(os.environ.get("ALGONAUTS_NUM_WORKERS", 1))
PREFETCH       = int(os.environ.get("ALGONAUTS_PREFETCH", 1))
MAX_DATASETS   = int(os.environ.get("MAX_DATASETS", 0))  # 0 表示不限制
DEC_T          = int(os.environ.get("ALGONAUTS_DECORD_THREADS", "2"))  # decord 解码线程

# —— 结果相关：保持采样策略一致 ——
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", 64))

# CLIP 归一化参数（可覆盖）
IMG_SIZE       = int(os.environ.get("ALGONAUTS_IMG_SIZE", 224))
IMG_MEAN       = [float(x) for x in os.environ.get("ALGONAUTS_IMG_MEAN", "0.48145466,0.4578275,0.40821073").split(",")]
IMG_STD        = [float(x) for x in os.environ.get("ALGONAUTS_IMG_STD",  "0.26862954,0.26130258,0.27577711").split(",")]

# “层”维度（为了与旧版三维输出对齐），默认 1，可按需提高（会简单重复该向量）
N_LAYERS       = max(1, int(os.environ.get("ALGONAUTS_IMAGEBIND_LAYERS", "1")))

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

def resolve_video_path(ds: str) -> Optional[Path]:
    parsed = try_parse_friends(ds)
    if parsed is None:
        return None
    ep, part = parsed
    season_num = int(ep[1:3])   # s01 -> 1
    candidates = [
        MOVIES_DIR / f"s{season_num}"          / f"friends_{ep}{part}.mkv",
        MOVIES_DIR / f"s{season_num:02d}"      / f"friends_{ep}{part}.mkv",
        MOVIES_DIR / f"season{season_num}"     / f"friends_{ep}{part}.mkv",
        MOVIES_DIR / f"season{season_num:02d}" / f"friends_{ep}{part}.mkv",
        MOVIES_DIR / f"friends_{ep}{part}.mkv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def sample_frame_indices_uniform(fps: float, a: float, b: float, k: int, nframes: Optional[int] = None) -> List[int]:
    """均匀取 k 个时间点（窗口中心），四舍五入到帧，并夹到 [0, nframes-1]。"""
    if k <= 0 or b <= a or fps <= 0:
        return []
    step = (b - a) / k
    times = [a + (i + 0.5) * step for i in range(k)]
    idxs = [int(round(t * fps)) for t in times]
    if nframes is not None and nframes > 0:
        last = nframes - 1
        idxs = [0 if i < 0 else (last if i > last else i) for i in idxs]
    else:
        idxs = [max(0, i) for i in idxs]
    return idxs

# ───────── Dataset / DataLoader ─────────
class EpisodeBinsDataset(Dataset):
    def __init__(self, ds_name: str, ds_grid: pd.DataFrame, frames_per_bin: int):
        self.ds = ds_name
        vpath = resolve_video_path(ds_name)
        if vpath is None:
            raise FileNotFoundError(f"Cannot resolve video for dataset: {ds_name}")
        self.vpath = str(vpath)

        vr = decord.VideoReader(self.vpath, num_threads=max(1, DEC_T))  # 可调线程
        fps = float(vr.get_avg_fps())
        nframes = int(len(vr))
        self.items = []
        for _, r in ds_grid.sort_values("bin_idx").iterrows():
            a, b = float(r["win_start"]), float(r["win_end"])
            idxs = sample_frame_indices_uniform(fps, a, b, frames_per_bin, nframes=nframes)
            if idxs:
                self.items.append((int(r["bin_idx"]), idxs))
        del vr

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        bidx, idxs = self.items[i]
        vr = decord.VideoReader(self.vpath, num_threads=max(1, DEC_T))
        n = int(len(vr))
        safe_idxs = [0 if j < 0 else (n - 1 if j >= n else j) for j in idxs]
        frames_t_hw3 = vr.get_batch(safe_idxs)  # [T,H,W,3] uint8 torch CPU
        return bidx, frames_t_hw3

def collate_frames(batch):
    bidxs, frames_list = zip(*batch)
    frames_bt_hw3 = torch.stack(frames_list, dim=0)  # [B,T,H,W,3] uint8 CPU
    return torch.tensor(bidxs, dtype=torch.long), frames_bt_hw3

# ───────── 预处理到 ImageBind 输入（GPU，FP32，纯 torch 实现） ─────────
def preprocess_frames_for_imagebind(
    frames_bt_hw3_uint8_cpu: torch.Tensor,  # [B,T,H,W,3] uint8 (CPU)
    device: str,
    img_size: int,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    assert frames_bt_hw3_uint8_cpu.ndim == 5 and frames_bt_hw3_uint8_cpu.dtype == torch.uint8
    B, T, H, W, _ = frames_bt_hw3_uint8_cpu.shape

    x = frames_bt_hw3_uint8_cpu.to(device, non_blocking=True).float() / 255.0  # [B,T,H,W,3]
    x = x.permute(0,1,4,2,3).reshape(B*T, 3, H, W).contiguous()               # [B*T,3,H,W]
    # 直接双线性 resize 到正方形（更鲁棒，避免依赖 torchvision transforms）
    x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t

    return x.view(B, T, 3, img_size, img_size).contiguous()  # [B,T,3,H,W]

# ───────── main ─────────
def main():
    # CLI args
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-datasets", type=str, default=os.environ.get("RUN_DATASETS", "").strip(),
                    help="Comma-separated dataset names to run, e.g., 'ses-035_task-s04e23a'")
    ap.add_argument("--max-datasets", type=int, default=MAX_DATASETS,
                    help="Process at most N datasets in this run (0 = no limit)")
    args = ap.parse_args()

    run_filter = set([s.strip() for s in args.run_datasets.split(",") if s.strip()]) if args.run_datasets else None
    max_datasets = int(args.max_datasets) if args.max_datasets is not None else 0

    # 限制 BLAS/OMP/OpenCV 线程，避免CPU打满
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass
    if cv2 is not None:
        try:
            cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "1")))
        except Exception:
            pass

    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    # 默认：处理 grid 里所有出现的 dataset；如果提供了 --run-datasets，则只跑指定子集
    if run_filter:
        grid = grid[grid["dataset"].isin(run_filter)].copy()
        print(f"[INFO] Running only {sorted(run_filter)}; filtered rows = {len(grid)}")

    # 限制处理数量
    if max_datasets > 0:
        kept_ds = []
        for ds in grid["dataset"].drop_duplicates().tolist():
            kept_ds.append(ds)
            if len(kept_ds) >= max_datasets:
                break
        grid = grid[grid["dataset"].isin(set(kept_ds))].copy()
        print(f"[INFO] Limiting to first {len(kept_ds)} datasets (max-datasets={max_datasets})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    print(f"[INFO] Loading ImageBind model (FP32): imagebind_huge")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(device)

    # 探测输出维度 D（标准前向）
    with torch.inference_mode():
        dummy_img = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), device=device, dtype=torch.float32)
        out_dict = model({VISION_KEY: dummy_img})
        if VISION_KEY in out_dict:
            vision_embed = out_dict[VISION_KEY]
        elif "vision" in out_dict:
            vision_embed = out_dict["vision"]
        else:
            vision_embed = next(iter(out_dict.values()))
        D = int(vision_embed.shape[-1])
    print(f"[INFO] feature dim (ImageBind VISION) = {D}; n_layers = {N_LAYERS}")

    def imagebind_encode_vision(x_bt_chw: torch.Tensor) -> torch.Tensor:
        """
        x_bt_chw: [B,T,3,H,W] float32 normalized
        returns:  [B,D]  (对 T 做平均池化)
        """
        B, T, C, H, W = x_bt_chw.shape
        x_flat = x_bt_chw.view(B * T, C, H, W)
        with torch.inference_mode():
            out = model({VISION_KEY: x_flat})
            if VISION_KEY in out:
                z = out[VISION_KEY]     # [B*T, D]
            elif "vision" in out:
                z = out["vision"]
            else:
                z = next(iter(out.values()))
        z = z.view(B, T, -1).mean(dim=1)          # [B,D]
        return z

    index_rows = []
    manifest_rows = []

    # 逐 dataset 处理
    for ds, g in grid.groupby("dataset", sort=True):
        vpath = resolve_video_path(ds)
        if vpath is None:
            print(f"[WARN] Skip dataset without resolvable video file: {ds}")
            continue
        manifest_rows.append([ds, str(vpath)])

        vr_meta = decord.VideoReader(str(vpath), num_threads=max(1, DEC_T))
        fps = float(vr_meta.get_avg_fps()); nframes = int(len(vr_meta)); del vr_meta
        T_bins = len(g)
        out_arr = np.zeros((T_bins, int(N_LAYERS), int(D)), dtype=SAVE_DTYPE)

        dataset = EpisodeBinsDataset(ds, g, FRAMES_PER_BIN)
        if len(dataset) == 0:
            print(f"[WARN] No valid bins for {ds}.")
            continue

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH,
            persistent_workers=True,   # 固定 worker，减少重复开销
            collate_fn=collate_frames,
        )

        print(f"[DATASET] {ds}  bins={T_bins}  video={Path(vpath).name}  fps={fps:.2f} frames={nframes}")
        t0 = time.time()
        with torch.inference_mode():
            for bidxs, frames_bt_hw3_cpu in _tqdm(
                loader, total=((len(dataset)+BATCH_SIZE-1)//BATCH_SIZE), desc=ds
            ):
                # 预处理到 ImageBind 期望的张量（已归一化）
                x_bt_chw = preprocess_frames_for_imagebind(
                    frames_bt_hw3_cpu, device=device,
                    img_size=IMG_SIZE, mean=IMG_MEAN, std=IMG_STD
                )  # [B,T,3,H,W]

                # 编码并对 T 做平均（得到 [B,D]）
                feats_bd = imagebind_encode_vision(x_bt_chw)  # [B,D]

                # 扩展/重复到 [B,N_LAYERS,D] 以匹配旧版三维输出
                if N_LAYERS == 1:
                    arr = feats_bd.unsqueeze(1).cpu().numpy().astype(SAVE_DTYPE)  # [B,1,D]
                else:
                    arr = feats_bd.unsqueeze(1).repeat(1, N_LAYERS, 1).cpu().numpy().astype(SAVE_DTYPE)  # [B,N,D]

                for i, bidx in enumerate(bidxs.tolist()):
                    if 0 <= bidx < T_bins:
                        out_arr[bidx, :, :] = arr[i]

        dt = time.time() - t0
        out_path = OUT_DIR / f"{ds}.npy"
        np.save(out_path, out_arr)
        index_rows.append([ds, str(out_path), T_bins, int(N_LAYERS), int(D)])
        print(f"[OK] {ds}: saved {out_path.name} shape={out_arr.shape}  elapsed={dt:.1f}s")

    # 写索引与清单
    if index_rows:
        idx = pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"])
        idx.to_csv(OUT_DIR / "index.csv", index=False)
        print("\n[DONE] video_2hz index.csv ->", OUT_DIR)
    else:
        print("[WARN] No dataset processed. Check grid and video files.")

    if manifest_rows:
        man = pd.DataFrame(manifest_rows, columns=["dataset","video_path"])
        man.to_csv(OUT_DIR / "manifest.csv", index=False)
        print("      manifest.csv (dataset → video_path) has been written.")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()