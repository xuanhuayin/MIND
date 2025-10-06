# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/extract_imagebind_from_grid.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys, re, subprocess, tempfile, warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchaudio
import cv2

# ──────────────────────────────────────────────────────────────────────────────
# 先做兼容补丁：让 pytorchvideo 旧版能找到 functional_tensor
try:
    import torchvision.transforms.functional as _F
    sys.modules.setdefault("torchvision.transforms.functional_tensor", _F)
except Exception:
    pass
warnings.filterwarnings("ignore", message="The 'torchvision.transforms._functional_video'")
warnings.filterwarnings("ignore", message="The 'torchvision.transforms._transforms_video'")
# ──────────────────────────────────────────────────────────────────────────────

# 项目根路径
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# 现在再导入 ImageBind（避免其 __init__ 连带导入 data 触发旧接口）
try:
    from imagebind.models.imagebind_model import imagebind_huge, ModalityType
except Exception:
    # 有些安装把符号挂在 models.__init__ 下
    from imagebind.models import imagebind_model as _im
    imagebind_huge = _im.imagebind_huge
    ModalityType = _im.ModalityType

# 固定路径
PIPE_ROOT  = PROJ / "pipeline_TRIBE"
DATA_ROOT  = PROJ / "download" / "algonauts_2025.competitors"
GRID_2HZ   = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"
MOVIES_DIR = DATA_ROOT / "stimuli" / "movies" / "friends"

OUT_ROOT_DEFAULT = PROJ / "pipeline_IMAGEBIND" / "TRIBE_8features"
OUT_VISION = "video_2hz/sub-01"
OUT_AUDIO  = "audio_2hz/sub-01"

RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)

# 视觉预处理（CLIP 风格）
IMG_SIZE = 224
_VISION_TX = T.Compose([
    T.ToPILImage(),
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                std =(0.26862954, 0.26130258, 0.27577711)),
])

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def ds2friends(ds: str) -> str:
    m = RE_FRIENDS.search(ds)
    if not m: raise ValueError(f"Bad ds id: {ds}")
    return f"friends_{m.group(1).lower()}{m.group(2).lower()}"

def resolve_video_path(ds: str) -> str:
    ep = RE_FRIENDS.search(ds)
    assert ep, f"Cannot parse ds: {ds}"
    season = int(ep.group(1)[1:3])
    basename = f"friends_{ep.group(1).lower()}{ep.group(2).lower()}.mkv"
    cands = [
        MOVIES_DIR / f"s{season}" / basename,
        MOVIES_DIR / f"s{season:02d}" / basename,
        MOVIES_DIR / f"season{season}" / basename,
        MOVIES_DIR / f"season{season:02d}" / basename,
        MOVIES_DIR / basename,
    ]
    for p in cands:
        if p.exists(): return str(p.resolve())
    raise FileNotFoundError(f"Video not found for {ds}\n" + "\n".join(map(str, cands)))

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

def grab_center_frames(video_path: str, centers_s: np.ndarray) -> torch.Tensor:
    """读取中心时刻的一帧：返回 [T,3,224,224]"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    for c in centers_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(c)*1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            # 回退最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)-1)))
            ok2, frame = cap.read()
            if not ok2 or frame is None:
                raise RuntimeError(f"Read frame failed at {c}s in {video_path}")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = _VISION_TX(img)  # [3,224,224]
        frames.append(img)
    cap.release()
    return torch.stack(frames, dim=0)  # [T,3,224,224]

def extract_audio_wav(video_path: str, out_wav: Path, sr: int = 16000):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(sr), str(out_wav)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def slice_audio_by_grid(wav: torch.Tensor, sr: int, starts: np.ndarray, ends: np.ndarray) -> torch.Tensor:
    """
    每个 bin 用 [start,end) 秒，该函数把所有切片统一成固定长度（全局中位长度），
    不足右侧0填充，超出截断；返回 [T,1,fixed_len]。
    """
    win_lens = (ends - starts) * sr
    fixed_len = int(np.round(np.median(win_lens)))
    fixed_len = max(fixed_len, 1)
    out = []
    for s, e in zip(starts, ends):
        s_idx = int(np.round(s * sr))
        e_idx = s_idx + fixed_len
        seg = wav[:, s_idx:e_idx]  # [1,L?]
        if seg.shape[1] < fixed_len:
            pad = fixed_len - seg.shape[1]
            seg = torch.nn.functional.pad(seg, (0, pad))
        elif seg.shape[1] > fixed_len:
            seg = seg[:, :fixed_len]
        out.append(seg)
    return torch.stack(out, dim=0)  # [T,1,fixed_len]

def main():
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--id_list", type=str, required=True, help="ds id 列表（如 train_full_episodes.txt 或 val_full_episodes.txt）")
    ap.add_argument("--out_root", type=str, default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--do_vision", action="store_true")
    ap.add_argument("--do_audio",  action="store_true")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--audio_sr", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=64, help="前向 batch size")
    args = ap.parse_args()

    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    ids = set(read_ids(args.id_list))
    grid = grid[grid["dataset"].isin(ids)].copy()
    if grid.empty:
        raise SystemExit("No rows after filtering by id_list; check your list vs parquet.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = imagebind_huge(pretrained=True).to(device).eval()

    out_root = Path(args.out_root)
    out_vis = out_root / OUT_VISION
    out_aud = out_root / OUT_AUDIO
    for p in [out_vis, out_aud]:
        ensure_dir(p)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ib_audio_"))
    print(f"[INFO] temp dir: {tmp_dir}")

    ds_iter = list(grid["dataset"].unique())
    pbar_ds = tqdm(ds_iter, desc="Datasets", unit="ds")

    for ds in pbar_ds:
        pbar_ds.set_postfix_str(ds)
        g = grid[grid["dataset"] == ds].sort_values("bin_idx").reset_index(drop=True)
        starts = g["win_start"].to_numpy(dtype=np.float64)
        ends   = g["win_end"].to_numpy(dtype=np.float64)
        centers= 0.5 * (starts + ends)
        T_bins = len(g)

        vpath = resolve_video_path(ds)

        # ----- Vision -----
        if args.do_vision:
            save_p = out_vis / f"{ds}.npy"
            if not save_p.exists():
                try:
                    frames = grab_center_frames(vpath, centers)  # [T,3,224,224]
                    outs = []
                    for i in tqdm(range(0, T_bins, args.batch), leave=False, desc=f"vision {ds}"):
                        with torch.no_grad():
                            inp = {ModalityType.VISION: frames[i:i+args.batch].to(device)}
                            out = model(inp)
                            emb = out[ModalityType.VISION]  # [b,1024]
                        outs.append(emb.cpu())
                    arr = torch.cat(outs, dim=0).numpy().astype(np.float32)  # [T,1024]
                    arr = arr[:, None, :]  # [T,1,1024]
                    np.save(save_p, arr)
                except Exception as e:
                    print(f"[WARN] vision fail for {ds}: {e}; writing zeros")
                    np.save(save_p, np.zeros((T_bins, 1, 1024), dtype=np.float32))

        # ----- Audio -----
        if args.do_audio:
            save_p = out_aud / f"{ds}.npy"
            if not save_p.exists():
                try:
                    wav_path = tmp_dir / f"{ds}.wav"
                    extract_audio_wav(vpath, wav_path, sr=args.audio_sr)
                    wav, sr = torchaudio.load(str(wav_path))  # wav: [1, L]
                    # 统一到 float32 / [-1,1]
                    wav = wav.to(torch.float32)

                    # 按 grid 裁波形并统一长度 → [T,1,fixed_len]
                    win_segs = slice_audio_by_grid(wav, sr, starts, ends)

                    outs = []
                    for i in tqdm(range(0, T_bins, args.batch), leave=False, desc=f"audio {ds}"):
                        with torch.no_grad():
                            # ImageBind 期望 [B,1,T] float32
                            batch_wavs = win_segs[i:i+args.batch].to(device)  # [B,1,T]
                            out_dict = model({ModalityType.AUDIO: batch_wavs})
                            emb = out_dict[ModalityType.AUDIO]  # [B,1024]
                        outs.append(emb.cpu())
                    arr = torch.cat(outs, dim=0).numpy().astype(np.float32)  # [T,1024]
                    arr = arr[:, None, :]  # [T,1,1024]
                    np.save(save_p, arr)
                except Exception as e:
                    print(f"[WARN] audio fail for {ds}: {e}; writing zeros")
                    np.save(save_p, np.zeros((T_bins, 1, 1024), dtype=np.float32))
                finally:
                    try: os.remove(wav_path)
                    except: pass

    print("\n[DONE] 输出目录：", str(out_root))
    print("  视觉：", str(out_vis))
    print("  音频：", str(out_aud))

if __name__ == "__main__":
    # 为了可复现，避免 cuBLAS nondeterministic 报警（可选）
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()