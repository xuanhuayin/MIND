# -*- coding: utf-8 -*-
"""
Movies-only (non-Friends) 2Hz video embeddings for sub-01 using ImageBind (VISION only).
FP32, outputs: [T_2Hz, N_LAYERS, D] float32

- Uses a manifest CSV to map dataset -> absolute video_path
- Excludes Friends datasets by default (task-sXXeYY[a-d]); pass --include-friends to override
- Skips datasets whose output .npy already exists (can disable with --no-skip-existing)
- Backend switchable: ALGONAUTS_VID_BACKEND=decord|opencv (default: decord)
- T-chunk microbatch to reduce peak memory: ALGONAUTS_T_CHUNK (default: 64, e.g., set 16)

CLI example (single dataset debug):
  RUN_DATASETS=ses-001_task-bourne02 \
  ALGONAUTS_VIDEO_BATCH=2 ALGONAUTS_NUM_WORKERS=1 ALGONAUTS_PREFETCH=1 \
  ALGONAUTS_VID_BACKEND=decord ALGONAUTS_T_CHUNK=16 \
  python extract_video_embeddings_2hz_sub01_movies_only.py \
      --video-manifest /home/lawrence/Desktop/algonauts-2025/manifest.csv \
      --debug
"""

from __future__ import annotations

# ---- torchvision legacy shim (must be first) ----
import sys, types
try:
    import torchvision.transforms.functional_tensor as _ft
except Exception:
    from torchvision.transforms import functional as _f
    _ft = types.ModuleType("torchvision.transforms.functional_tensor")
    _ft.__dict__.update(_f.__dict__)
    sys.modules["torchvision.transforms.functional_tensor"] = _ft
# -----------------------------------------------

# ---- ImageBind Modality key compat ----
try:
    from imagebind.models.imagebind_model import ModalityType as _MBModalityType  # noqa: E402
    VISION_KEY = _MBModalityType.VISION
except Exception:
    VISION_KEY = "vision"

from pathlib import Path
import os, re, time, argparse, csv
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info

try:
    import cv2
except Exception:
    cv2 = None

# decord (optional)
_DECORD_OK = True
try:
    import decord
    decord.bridge.set_bridge('torch')
except Exception:
    _DECORD_OK = False

# ───────── ImageBind ─────────
from imagebind.models import imagebind_model

# ───────── paths ─────────
PIPE_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
DATA_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")
GRID_2HZ   = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"

OUT_DIR = Path("/algonauts2025/pipeline_IMAGEBIND/TRIBE_8features/video_2hz/sub-01")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT    = "sub-01"
SAVE_DTYPE = np.float32

# —— performance knobs ——
BATCH_SIZE     = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", 8))
NUM_WORKERS    = int(os.environ.get("ALGONAUTS_NUM_WORKERS", 4))
PREFETCH       = int(os.environ.get("ALGONAUTS_PREFETCH", 4))
MAX_DATASETS   = int(os.environ.get("MAX_DATASETS", 0))  # 0 = unlimited
DEC_T          = int(os.environ.get("ALGONAUTS_DECORD_THREADS", "2"))
VID_BACKEND    = os.environ.get("ALGONAUTS_VID_BACKEND", "decord").lower()  # decord | opencv
T_CHUNK        = max(1, int(os.environ.get("ALGONAUTS_T_CHUNK", "64")))     # microbatch over T

# —— sampling ——
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", 64))

# —— CLIP norm ——
IMG_SIZE       = int(os.environ.get("ALGONAUTS_IMG_SIZE", 224))
IMG_MEAN       = [float(x) for x in os.environ.get("ALGONAUTS_IMG_MEAN", "0.48145466,0.4578275,0.40821073").split(",")]
IMG_STD        = [float(x) for x in os.environ.get("ALGONAUTS_IMG_STD",  "0.26862954,0.26130258,0.27577711").split(",")]

# —— "layers" dim to match 3D output ——
N_LAYERS       = max(1, int(os.environ.get("ALGONAUTS_IMAGEBIND_LAYERS", "1")))

# ───────── tqdm ─────────
try:
    from tqdm.auto import tqdm
    def _tqdm(it, **kw): return tqdm(it, **kw)
except Exception:
    def _tqdm(it, **kw): return it

# ───────── utils ─────────
RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)
def is_friends_dataset(ds: str) -> bool:
    return RE_FRIENDS.search(ds) is not None

def sample_frame_indices_uniform(fps: float, a: float, b: float, k: int, nframes: Optional[int] = None) -> List[int]:
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

# ───────── manifest ─────────
VIDEO_MANIFEST: Dict[str, str] = {}
DEBUG = False

def load_video_manifest(path: Optional[str]) -> int:
    VIDEO_MANIFEST.clear()
    if not path:
        return 0
    csv_path = Path(path)
    if not csv_path.exists():
        print(f"[WARN] video manifest not found: {csv_path}")
        return 0
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = (row.get("dataset") or "").strip()
            vp = (row.get("video_path") or "").strip()
            if ds and vp and Path(vp).exists():
                VIDEO_MANIFEST[ds] = vp
            elif ds and vp:
                print(f"[WARN] manifest entry path not exists: {ds} -> {vp}")
    print(f"[INFO] loaded {len(VIDEO_MANIFEST)} entries from manifest: {csv_path}")
    return len(VIDEO_MANIFEST)

def resolve_video_path_from_manifest(ds: str) -> Optional[Path]:
    vp = VIDEO_MANIFEST.get(ds)
    if vp:
        p = Path(vp)
        if p.exists():
            return p
    return None

# ───────── Dataset / DataLoader ─────────
class EpisodeBinsDataset(Dataset):
    """
    - Keeps a per-worker cached decoder (decord or OpenCV).
    - Avoids reopening decoder per __getitem__.
    """
    def __init__(self, ds_name: str, ds_grid: pd.DataFrame, frames_per_bin: int, video_path: str, backend: str):
        self.ds = ds_name
        self.vpath = str(video_path)
        self.backend = backend
        self.frames_per_bin = frames_per_bin

        # Probe meta once
        if backend == "opencv":
            if cv2 is None:
                raise RuntimeError("OpenCV not available but ALGONAUTS_VID_BACKEND=opencv")
            cap = cv2.VideoCapture(self.vpath)
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV cannot open {self.vpath}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        else:
            if not _DECORD_OK:
                raise RuntimeError("decord not available; set ALGONAUTS_VID_BACKEND=opencv")
            vr = decord.VideoReader(self.vpath, num_threads=max(1, DEC_T))
            fps = float(vr.get_avg_fps())
            nframes = int(len(vr))
            del vr

        if DEBUG:
            print(f"[DEBUG] META ds={ds_name} backend={backend} fps={fps:.3f} nframes={nframes}")

        # bin sampling plan
        self.items = []
        for _, r in ds_grid.sort_values("bin_idx").iterrows():
            a, b = float(r["win_start"]), float(r["win_end"])
            idxs = sample_frame_indices_uniform(fps, a, b, frames_per_bin, nframes=nframes)
            if idxs:
                self.items.append((int(r["bin_idx"]), idxs))

        # per-worker decoder cache
        self._vr = None  # for decord
        self._cap = None # for opencv

    def __len__(self): return len(self.items)

    def _get_decord(self):
        if self._vr is None:
            self._vr = decord.VideoReader(self.vpath, num_threads=max(1, DEC_T))
            if DEBUG:
                try:
                    _ = self._vr.get_batch([0])
                    print(f"[DEBUG] decord probe ok: {self.vpath}")
                except Exception as e:
                    print(f"[DEBUG] decord probe failed: {e}")
        return self._vr

    def _get_opencv(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.vpath)
            if not self._cap.isOpened():
                raise RuntimeError(f"OpenCV cannot open {self.vpath}")
        return self._cap

    def __getitem__(self, i):
        bidx, idxs = self.items[i]
        if self.backend == "opencv":
            cap = self._get_opencv()
            frames = []
            prev = None
            for j in idxs:
                j = max(0, j)
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                ok, frame = cap.read()
                if not ok:
                    frame_rgb = prev if prev is not None else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[1] != IMG_SIZE or frame.shape[0] != IMG_SIZE:
                        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                    frame_rgb = frame
                    prev = frame_rgb
                frames.append(torch.from_numpy(frame_rgb).to(torch.uint8))
            frames_t_hw3 = torch.stack(frames, 0)
        else:
            vr = self._get_decord()
            n = int(len(vr))
            safe_idxs = [0 if j < 0 else (n - 1 if j >= n else j) for j in idxs]
            try:
                frames_t_hw3 = vr.get_batch(safe_idxs)  # [T,H,W,3] uint8 torch CPU
            except Exception:
                # fallback per-frame
                frames = []
                prev = None
                for j in safe_idxs:
                    try:
                        f = vr[j]
                        prev = f
                    except Exception:
                        f = prev if prev is not None else torch.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=torch.uint8)
                    frames.append(f)
                frames_t_hw3 = torch.stack(frames, dim=0)
        return bidx, frames_t_hw3

def collate_frames(batch):
    bidxs, frames_list = zip(*batch)
    frames_bt_hw3 = torch.stack(frames_list, dim=0)  # [B,T,H,W,3] uint8 CPU
    return torch.tensor(bidxs, dtype=torch.long), frames_bt_hw3

# ───────── preprocess (GPU, FP32) ─────────
def preprocess_frames_for_imagebind(
    frames_bt_hw3_uint8_cpu: torch.Tensor,  # [B,T,H,W,3] uint8 CPU
    device: str,
    img_size: int,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    assert frames_bt_hw3_uint8_cpu.ndim == 5 and frames_bt_hw3_uint8_cpu.dtype == torch.uint8
    B, T, H, W, _ = frames_bt_hw3_uint8_cpu.shape
    x = frames_bt_hw3_uint8_cpu.to(device, non_blocking=True).float() / 255.0
    x = x.permute(0,1,4,2,3).reshape(B*T, 3, H, W).contiguous()
    if (H, W) != (img_size, img_size):
        x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=device).view(1, 3, 1, 1)
    x = (x - mean_t) / std_t
    return x.view(B, T, 3, img_size, img_size).contiguous()  # [B,T,3,H,W]

# ───────── main ─────────
def main():
    global DEBUG
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-datasets", type=str, default=os.environ.get("RUN_DATASETS", "").strip(),
                    help="Comma-separated dataset names to run, e.g., 'ses-001_task-bourne02'")
    ap.add_argument("--max-datasets", type=int, default=MAX_DATASETS,
                    help="Process at most N datasets in this run (0 = no limit)")
    ap.add_argument("--video-manifest", type=str, required=True,
                    help="CSV with columns: dataset,video_path (absolute paths).")
    ap.add_argument("--include-friends", action="store_true",
                    help="Include Friends episodes (task-sXXeYY[a-d]). Default: exclude.")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Do not skip datasets whose output npy already exists.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    DEBUG = bool(args.debug)
    include_friends = bool(args.include_friends)
    skip_existing = not bool(args.no_skip_existing)

    # Threads
    try:
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    except Exception:
        pass
    if cv2 is not None:
        try:
            cv2.setNumThreads(int(os.environ.get("OPENCV_NUM_THREADS", "1")))
        except Exception:
            pass

    # grid
    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    try:
        grid = pd.read_parquet(GRID_2HZ, engine="pyarrow")
    except Exception:
        grid = pd.read_parquet(GRID_2HZ, engine="fastparquet")
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    # manifest
    load_video_manifest(args.video_manifest)

    # filter datasets
    ds_all = grid["dataset"].drop_duplicates().tolist()
    ds_has_video = [ds for ds in ds_all if ds in VIDEO_MANIFEST]
    if not include_friends:
        ds_has_video = [ds for ds in ds_has_video if not is_friends_dataset(ds)]

    run_filter = set([s.strip() for s in args.run_datasets.split(",") if s.strip()]) if args.run_datasets else None
    if run_filter:
        ds_has_video = [ds for ds in ds_has_video if ds in run_filter]

    max_datasets = int(args.max_datasets) if args.max_datasets is not None else 0
    if max_datasets > 0:
        ds_has_video = ds_has_video[:max_datasets]

    if not ds_has_video:
        print("[WARN] No datasets to process after filtering.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    print(f"[INFO] Output dir: {OUT_DIR}")
    print(f"[INFO] Video backend: {VID_BACKEND}")
    print(f"[INFO] Loading ImageBind model (FP32): imagebind_huge")
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(device)

    # probe D
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
    print(f"[INFO] feature dim (ImageBind VISION) = {D}; n_layers = {N_LAYERS}; T_CHUNK={T_CHUNK}")

    def imagebind_encode_vision_chunked(x_bt_chw: torch.Tensor) -> torch.Tensor:
        """
        x_bt_chw: [B,T,3,H,W], returns [B,D]
        Process T frames in chunks to reduce peak memory. Average across T.
        """
        B, T, C, H, W = x_bt_chw.shape
        chunk = max(1, min(T, T_CHUNK))
        acc = None
        count = 0
        with torch.inference_mode():
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                x_flat = x_bt_chw[:, s:e].reshape(B*(e-s), C, H, W)
                out = model({VISION_KEY: x_flat})
                if VISION_KEY in out: z = out[VISION_KEY]
                elif "vision" in out: z = out["vision"]
                else: z = next(iter(out.values()))
                z = z.view(B, e-s, -1).sum(dim=1)  # sum over this slice of T
                acc = z if acc is None else (acc + z)
                count += (e - s)
        return acc / float(count)  # [B,D]

    index_rows = []
    manifest_rows = []
    skipped_rows = []

    for ds in ds_has_video:
        vpath = resolve_video_path_from_manifest(ds)
        if DEBUG:
            print(f"[DEBUG] dataset={ds} -> vpath={vpath}")
        if vpath is None:
            print(f"[WARN] Skip (not in manifest or path missing): {ds}")
            skipped_rows.append([ds, "manifest_missing_or_path_invalid"])
            continue

        g = grid[grid["dataset"] == ds].copy()
        T_bins = len(g)
        out_path = OUT_DIR / f"{ds}.npy"

        if skip_existing and out_path.exists():
            print(f"[SKIP] exists -> {out_path.name}")
            index_rows.append([ds, str(out_path), T_bins, int(N_LAYERS), -1])
            manifest_rows.append([ds, str(vpath)])
            continue

        # meta (best-effort)
        fps, nframes = 0.0, 0
        if VID_BACKEND == "decord" and _DECORD_OK:
            try:
                vr_meta = decord.VideoReader(str(vpath), num_threads=max(1, DEC_T))
                fps = float(vr_meta.get_avg_fps()); nframes = int(len(vr_meta)); del vr_meta
            except Exception:
                pass
        elif VID_BACKEND == "opencv" and cv2 is not None:
            cap = cv2.VideoCapture(str(vpath))
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

        # dataset & loader
        try:
            dataset = EpisodeBinsDataset(ds, g, FRAMES_PER_BIN, str(vpath), backend=VID_BACKEND)
        except Exception as e:
            print(f"[WARN] dataset init failed for {ds}: {e}")
            skipped_rows.append([ds, "init_failed"])
            continue

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            prefetch_factor=PREFETCH,
            persistent_workers=True,
            collate_fn=collate_frames,
        )

        print(f"[DATASET] {ds}  bins={T_bins}  video={Path(vpath).name}  fps={fps:.2f} frames={nframes}")
        out_arr = np.zeros((T_bins, int(N_LAYERS), int(D)), dtype=SAVE_DTYPE)
        t0 = time.time()

        with torch.inference_mode():
            for bidxs, frames_bt_hw3_cpu in _tqdm(
                loader, total=((len(dataset)+BATCH_SIZE-1)//BATCH_SIZE), desc=ds
            ):
                x_bt_chw = preprocess_frames_for_imagebind(
                    frames_bt_hw3_cpu, device=device,
                    img_size=IMG_SIZE, mean=IMG_MEAN, std=IMG_STD
                )  # [B,T,3,H,W]

                feats_bd = imagebind_encode_vision_chunked(x_bt_chw)  # [B,D], T-chunk averaging

                if N_LAYERS == 1:
                    arr = feats_bd.unsqueeze(1).cpu().numpy().astype(SAVE_DTYPE)
                else:
                    arr = feats_bd.unsqueeze(1).repeat(1, N_LAYERS, 1).cpu().numpy().astype(SAVE_DTYPE)

                for i, bidx in enumerate(bidxs.tolist()):
                    if 0 <= bidx < T_bins:
                        out_arr[bidx, :, :] = arr[i]

        dt = time.time() - t0
        np.save(out_path, out_arr)
        index_rows.append([ds, str(out_path), T_bins, int(N_LAYERS), int(D)])
        manifest_rows.append([ds, str(vpath)])
        print(f"[OK] {ds}: saved {out_path.name} shape={out_arr.shape}  elapsed={dt:.1f}s")

    if index_rows:
        pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"]).to_csv(
            OUT_DIR / "index_movies.csv", index=False
        )
        print("\n[DONE] index_movies.csv ->", OUT_DIR)
    else:
        print("[WARN] No dataset processed. Check manifest/filtering/backends.")

    if manifest_rows:
        pd.DataFrame(manifest_rows, columns=["dataset","video_path"]).to_csv(
            OUT_DIR / "manifest_used_movies.csv", index=False
        )
        print("      manifest_used_movies.csv has been written.")

    if skipped_rows:
        pd.DataFrame(skipped_rows, columns=["dataset","reason"]).to_csv(
            OUT_DIR / "skipped_movies.csv", index=False
        )
        print("      skipped_movies.csv has been written.")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()