# -*- coding: utf-8 -*-
"""
Extract 2Hz video embeddings for sub-01 using V-JEPA-2 @ 256.

本版改动（应你的要求）：
- **完全禁用 memmap**：无论任何环境变量，统一使用内存数组 out_arr，最后 np.save()。
- 逐条/小子批 flush，避免 RAM 峰值；修复负步长（negative strides）导致 from_numpy 报错。
- 写后三重校验（魔数 + 能 np.load + 形状一致），失败会重写或标记 .corrupt。
- 仅重算坏样本：--only_list / ALGONAUTS_ONLY_LIST（每行一个 dataset stem）
- 可选 NVDEC（ALGONAUTS_DECODER=gpu）或 CPU 解码（cpu）
- 可控 pinned memory（ALGONAUTS_PIN=0/1）

可用环境变量（可选）：
- ALGONAUTS_VJEPA_ID         默认 "facebook/vjepa2-vitg-fpc64-256"
- ALGONAUTS_VIDEO_BATCH      默认 12（修复期建议 1~2）
- ALGONAUTS_VIDEO_FRAMES     默认 64（修复期可降到 16）
- ALGONAUTS_QUEUE            默认 1（解码队列）
- ALGONAUTS_DECODER          "auto|gpu|cpu"（默认 auto）
- ALGONAUTS_PIN              默认 1；内存紧张先设 0
- ALGONAUTS_ONLY_LIST        仅处理该列表中的 ID
- ALGONAUTS_TARGET_SIZE      默认 256
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from contextlib import nullcontext
import os, re, time, gc

import numpy as np
import pandas as pd
import cv2
import torch
from transformers import AutoModel

try:
    from transformers import AutoVideoProcessor as _AutoProc  # 4.45+
except Exception:
    from transformers import AutoImageProcessor as _AutoProc  # 4.4x 兼容

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ======== 解码后端：优先 Decord，失败回退 OpenCV ========
USE_DECORD = True
try:
    if USE_DECORD:
        import decord
        from decord import VideoReader, cpu, gpu, bridge
        try:
            bridge.set_bridge('torch')
        except Exception:
            pass
except Exception:
    USE_DECORD = False

# ======== 并行/CPU 线程设置 ========
PREFETCH_WORKERS = int(os.environ.get("ALGONAUTS_PREFETCH", "4"))
MAX_INFLIGHT     = int(os.environ.get("ALGONAUTS_MAX_INFLIGHT", "8"))
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

SAVE_DTYPE     = np.float16

# ======== 模型/采样配置（支持 ENV 覆盖） ========
MODEL_ID       = os.environ.get("ALGONAUTS_VJEPA_ID", "facebook/vjepa2-vitg-fpc64-256")
BATCH_SIZE     = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", "12"))
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", "64"))
KEEP_SPEC      = os.environ.get("ALGONAUTS_VIDEO_KEEP_LAYERS", "last41")  # 实际 40 层
TARGET_SIZE    = int(os.environ.get("ALGONAUTS_TARGET_SIZE", "256"))

# ======== 仅重算坏样本：从参数或 ENV 读取 ========
ONLY_LIST_PATH = os.environ.get("ALGONAUTS_ONLY_LIST", "")
ONLY_SET: set[str] = set()

# ======== 正则工具 ========
RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)

def try_parse_friends(ds: str) -> Optional[Tuple[str, str]]:
    m = RE_FRIENDS.search(ds)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower()  # ("s01e02","a")

def parse_movie_task(ds: str) -> Optional[Tuple[str, Optional[str]]]:
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
        ep, part = parsed
        season_num = int(ep[1:3])
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

# ======== 数组步长 & 校验 ========
def has_negative_strides(a: np.ndarray) -> bool:
    try:
        return any(s < 0 for s in a.strides)
    except Exception:
        return False

def ensure_poscontig(a: np.ndarray) -> np.ndarray:
    """确保 C 连续且无负步长；否则复制为正步长副本"""
    if (a is None) or (not isinstance(a, np.ndarray)):
        return a
    if (not a.flags.c_contiguous) or has_negative_strides(a):
        return np.ascontiguousarray(a)
    return a

def is_npy_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(6) == b"\x93NUMPY"
    except Exception:
        return False

def verify_npy(path: Path, expected_shape: Tuple[int,int,int]) -> bool:
    if not path.exists() or not is_npy_magic(path):
        return False
    try:
        arr = np.load(path, mmap_mode=None)
    except Exception:
        return False
    return (arr.ndim == 3) and (tuple(arr.shape) == expected_shape)

# —— decord：使用已持有的 VR 读取（避免频繁打开/seek）——
def read_clip_decord_vr(vr, a: float, b: float, k: int) -> Optional[np.ndarray]:
    if not USE_DECORD or vr is None:
        return None
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        return None
    if fps <= 0 or len(vr) <= 0:
        return None
    step = (b - a) / k
    times = [a + (i + 0.5) * step for i in range(k)]
    idxs = [int(round(t * fps)) for t in times]
    idxs = [max(0, min(i, len(vr) - 1)) for i in idxs]
    try:
        batch = vr.get_batch(idxs)  # torch.Tensor 或 NDArray
    except Exception:
        return None

    # 统一转 RGB，确保正步长 & C 连续
    try:
        import torch as _torch
        if isinstance(batch, _torch.Tensor):
            batch = batch[..., [2, 1, 0]]             # BGR->RGB
            arr = batch.cpu().numpy()
            return ensure_poscontig(arr)
    except Exception:
        pass

    arr = batch.asnumpy()
    arr = arr[..., ::-1].copy(order="C")              # BGR->RGB + 正步长副本
    if arr.shape[0] != k:
        return None
    return arr

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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_list", type=str, default=ONLY_LIST_PATH,
                    help="仅处理该列表中的 ID（每行一个 stem）；为空则处理所有")
    args = ap.parse_args()

    # 仅处理坏样本
    global ONLY_SET
    if args.only_list and Path(args.only_list).exists():
        ONLY_SET = {ln.strip() for ln in open(args.only_list, "r", encoding="utf-8") if ln.strip()}
        print(f"[FILTER] Only processing {len(ONLY_SET)} IDs from {args.only_list}")

    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Loading V-JEPA-2 model: {MODEL_ID}")
    processor = _AutoProc.from_pretrained(MODEL_ID)  # 仅用来取配置（比如 mean/std/size）
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    model.to(device).eval()

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if device.startswith("cuda") else nullcontext()

    # 取 mean/std（如不可取，就用常见值）
    try:
        ip = getattr(processor, "image_processor", processor)
        image_mean = torch.tensor(getattr(ip, "image_mean", [123.675,116.28,103.53]),
                                  device=device, dtype=torch.float16).view(1,1,3,1,1)
        image_std  = torch.tensor(getattr(ip, "image_std",  [58.395,57.12,57.375]),
                                  device=device, dtype=torch.float16).view(1,1,3,1,1)
        tgt_size = int(getattr(ip, "size", {}).get("shortest_edge", TARGET_SIZE))
        if tgt_size <= 0: tgt_size = TARGET_SIZE
    except Exception:
        image_mean = torch.tensor([123.675,116.28,103.53], device=device, dtype=torch.float16).view(1,1,3,1,1)
        image_std  = torch.tensor([58.395,57.12,57.375],  device=device, dtype=torch.float16).view(1,1,3,1,1)
        tgt_size = TARGET_SIZE

    # probe hidden_states & dim
    dummy = np.zeros((FRAMES_PER_BIN, tgt_size, tgt_size, 3), dtype=np.uint8)
    with torch.inference_mode(), amp_ctx:
        cpu_tensor = torch.from_numpy(np.expand_dims(dummy, 0))
        x = cpu_tensor.to(device, non_blocking=True).to(torch.float16)
        x = x.permute(0,1,4,2,3).contiguous()
        x = torch.nn.functional.interpolate(
                x.view(1*FRAMES_PER_BIN,3,tgt_size,tgt_size),
                size=(tgt_size, tgt_size), mode="bilinear", align_corners=False
            ).view(1, FRAMES_PER_BIN, 3, tgt_size, tgt_size)
        x = (x - image_mean) / image_std
        outputs = model(pixel_values_videos=x, output_hidden_states=True)
    hs_len = len(outputs.hidden_states)
    layer_ids = choose_layer_indices(hs_len)
    D = getattr(getattr(model, "config", model), "hidden_size", 1408)
    N_LAYERS = len(layer_ids)
    print(f"[INFO] hidden_states len={hs_len}, keep {N_LAYERS} layers, feature dim={D}")
    del outputs, x, cpu_tensor, dummy
    torch.cuda.empty_cache(); gc.collect()

    build_video_index(MOVIES_ROOT)

    # ====== 每个数据集（视频）循环 ======
    groups = grid.groupby("dataset", sort=True)
    total_ids = sum(1 for _ in groups)
    groups = grid.groupby("dataset", sort=True)

    for ds, g in tqdm(groups, total=total_ids, desc="Datasets"):
        if ONLY_SET and (ds not in ONLY_SET):
            continue

        g = g.sort_values("bin_idx").reset_index(drop=True)
        vpath = resolve_video_path(ds)
        if vpath is None:
            print(f"[WARN] Skip dataset without resolvable video file: {ds}")
            continue

        # 读视频元信息（decord 路径下不依赖它）
        try:
            cap, fps, nframes = open_video(vpath)
        except Exception as e:
            print(f"[WARN] Cannot open video for {ds}: {e}")
            continue

        T_bins = len(g)
        expected = (T_bins, N_LAYERS, int(D))
        out_file = OUT_DIR / f"{ds}.npy"

        print(f"[DATASET] {ds}  bins={T_bins}  video={vpath.name}  fps={fps:.2f} frames={nframes}")
        t0 = time.time()

        # —— 输出容器：统一用内存数组（不用 memmap） —— #
        out_arr = np.zeros(expected, dtype=SAVE_DTYPE)

        # —— 累积容器（仅存“待 flush 的 clips 的索引”）—— #
        batch_clips: List[np.ndarray] = []
        batch_slots: List[int] = []

        # ---------- flush：逐条/小子批送入模型，避免 RAM 峰值 ----------
        def flush():
            if not batch_clips:
                return
            use_pin = os.environ.get("ALGONAUTS_PIN", "1") == "1" and torch.cuda.is_available()

            # 逐条最稳（也可改为 2-4 条小子批）
            for i in range(len(batch_clips)):
                clip = batch_clips[i]               # [T,H,W,3] uint8
                bslot = batch_slots[i]
                sub = clip[np.newaxis, ...]         # [1,T,H,W,3]
                sub = ensure_poscontig(sub)

                B, T, H, W, C = sub.shape
                try:
                    with torch.inference_mode(), amp_ctx:
                        cpu_tensor = torch.from_numpy(sub)
                        if use_pin:
                            cpu_tensor = cpu_tensor.pin_memory()

                        x = cpu_tensor.to(device, non_blocking=True).to(torch.float16)  # [1,T,H,W,3]
                        x = x.permute(0,1,4,2,3).contiguous()                           # [1,T,3,H,W]
                        x = torch.nn.functional.interpolate(
                                x.view(B*T, 3, H, W),
                                size=(tgt_size, tgt_size),
                                mode="bilinear", align_corners=False
                            ).view(B, T, 3, tgt_size, tgt_size)
                        x = (x - image_mean) / image_std

                        outputs = model(pixel_values_videos=x, output_hidden_states=True)

                        vecs = []
                        for lid in layer_ids:
                            h = outputs.hidden_states[lid]    # [1, tokens, D] or alike
                            pooled = safe_mean_over_tokens(h) # [1, D]
                            vecs.append(pooled)
                        all_layers = torch.stack(vecs, dim=1) # [1, N_LAYERS, D]
                        arr = (all_layers
                               .to(dtype=torch.float16)
                               .detach().cpu().numpy()
                               .astype(SAVE_DTYPE, copy=False))   # [1, L, D]

                    out_arr[bslot, :, :] = arr[0]

                    # 清理
                    del cpu_tensor, x, outputs, vecs, all_layers, arr
                    torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[OOM] single item OOM at slot {bslot}; skipping.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        print(f"[ERR] flush item {bslot} failed: {e}")
                        torch.cuda.empty_cache()
                        continue

            batch_clips.clear()
            batch_slots.clear()

        # ---------- 读取 + 预取 ----------
        tasks = [(int(r["bin_idx"]), float(r["win_start"]), float(r["win_end"])) for _, r in g.iterrows()]
        pbar = tqdm(total=T_bins, desc=f"{ds}", leave=False)

        if USE_DECORD:
            DECODER = os.environ.get("ALGONAUTS_DECODER", "auto")  # auto|gpu|cpu
            vr = None
            if DECODER in ("auto", "gpu"):
                try:
                    vr = VideoReader(str(vpath), ctx=gpu(0), num_threads=1)
                except Exception:
                    vr = None
            if vr is None:
                vr = VideoReader(str(vpath), ctx=cpu(0), num_threads=1)

            import queue, threading
            QSIZE = int(os.environ.get("ALGONAUTS_QUEUE", "1"))
            q: "queue.Queue[tuple[int, Optional[np.ndarray]]]" = queue.Queue(maxsize=max(1, QSIZE))
            SENTINEL = (-1, None)

            def producer():
                for bidx, a, b in tasks:
                    clip = read_clip_decord_vr(vr, a, b, FRAMES_PER_BIN)
                    if clip is not None:
                        clip = ensure_poscontig(clip)
                    q.put((bidx, clip), block=True)
                q.put(SENTINEL, block=True)

            th = threading.Thread(target=producer, daemon=True)
            th.start()

            while True:
                bidx, clip = q.get()
                if bidx == -1:
                    break
                if clip is not None:
                    batch_clips.append(clip)
                    batch_slots.append(bidx)
                    if len(batch_clips) >= max(1, min(BATCH_SIZE, 2)):
                        flush()
                pbar.update(1)

            flush()
            pbar.close()

        else:
            # OpenCV + 线程池（并发尽量小）
            reader = read_clip_cv
            with ThreadPoolExecutor(max_workers=max(1, min(2, PREFETCH_WORKERS))) as ex:
                inflight: Dict = {}
                it = iter(tasks)

                while len(inflight) < min(max(1, min(2, MAX_INFLIGHT)), len(tasks)):
                    try:
                        bidx, a, b = next(it)
                    except StopIteration:
                        break
                    fut = ex.submit(reader, vpath, a, b, FRAMES_PER_BIN)
                    inflight[fut] = bidx

                while inflight:
                    for fut in as_completed(list(inflight.keys()), timeout=None):
                        bidx = inflight.pop(fut)
                        clip = fut.result()
                        if clip is not None:
                            clip = ensure_poscontig(clip)
                            batch_clips.append(clip)
                            batch_slots.append(bidx)
                            if len(batch_clips) >= max(1, min(BATCH_SIZE, 2)):
                                flush()
                        pbar.update(1)

                        try:
                            nbidx, na, nb = next(it)
                            nfut = ex.submit(reader, vpath, na, nb, FRAMES_PER_BIN)
                            inflight[nfut] = nbidx
                        except StopIteration:
                            pass
                        break
                pbar.close()

        # 释放 cap（decord 路径下不依赖它）
        try:
            cap.release()
        except Exception:
            pass

        # ===== 落盘（并校验） =====
        np.save(out_file, out_arr)

        ok = verify_npy(out_file, expected)
        if not ok:
            print(f"[RETRY] {ds}: verifying failed, rewriting via np.save() ...")
            try:
                np.save(out_file, np.array(out_arr, copy=True))
            except Exception:
                pass
            ok = verify_npy(out_file, expected)
            if not ok:
                corrupt_path = out_file.with_suffix(".corrupt")
                try:
                    out_file.rename(corrupt_path)
                except Exception:
                    pass
                print(f"[ERR] {ds}: still invalid; marked {corrupt_path.name}")
            else:
                print(f"[OK] {ds}: saved {out_file.name} (rewritten) shape={expected}  elapsed={time.time()-t0:.1f}s")
        else:
            print(f"[OK] {ds}: saved {out_file.name} shape={expected}  elapsed={time.time()-t0:.1f}s")

        # 清理
        del out_arr
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[DONE] Extraction finished.")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()