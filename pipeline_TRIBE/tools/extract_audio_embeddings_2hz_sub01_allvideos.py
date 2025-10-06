# -*- coding: utf-8 -*-
"""
Extract 2Hz audio embeddings for sub-01 from ALL videos (Friends + other movies).
- Use ffmpeg to decode audio.
- Model: facebook/w2v-bert-2.0 (≈50 Hz), downsample to 2 Hz by mean-pooling (factor=25).
- Keep intermediate hidden layers (exclude embedding layer 0) -> (T_2Hz, N_LAYERS, 1024).
- Align to grid_2hz_sub-01.parquet via bin_idx (2 Hz).

Priority of video path resolution (per dataset):
1) --video-manifest CSV (dataset,video_path)
2) Friends pattern: friends_sXXeYY[a-d].mkv under .../stimuli/movies/friends/
3) Auto-scan index under .../stimuli/movies/** and match basename guessed from dataset (e.g., bourne02.mkv, wolf05.mkv, life03.mkv, figures10.mkv).

Outputs:
  PIPE_ROOT/TRIBE_8features/audio_2hz/sub-01/<dataset>.npy   # (T_2Hz, N_LAYERS, 1024) float32
  PIPE_ROOT/TRIBE_8features/audio_2hz/sub-01/index.csv
  .../manifest_used_audio.csv, .../skipped_audio.csv (日志)
"""

from __future__ import annotations
from pathlib import Path
import os, re, subprocess, argparse, csv
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ───────── paths ─────────
PIPE_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE")
DATA_ROOT  = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/download/algonauts_2025.competitors")
GRID_2HZ   = PIPE_ROOT / "timelines" / "grid_2hz_sub-01.parquet"

MOVIES_ROOT = DATA_ROOT / "stimuli" / "movies"           # 总根目录（friends + 其它 movies）

SPLIT_FULL     = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8_full_episodes.txt"
SPLIT_FALLBACK = PIPE_ROOT / "TRIBE_8features" / "splits" / "datasets_8.txt"

OUT_DIR   = PIPE_ROOT / "TRIBE_8features" / "audio_2hz" / "sub-01"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT    = "sub-01"
SAVE_DTYPE = np.float32  # 输出精度（与之前一致）

# ───────── cfg ─────────
MODEL_ID      = os.environ.get("ALGONAUTS_AUDIO_ID", "facebook/w2v-bert-2.0")
SR            = int(os.environ.get("ALGONAUTS_AUDIO_SR", 16000))
CHUNK_SECS    = int(os.environ.get("ALGONAUTS_AUDIO_BATCHSECS", 60))
KEEP_SPEC     = os.environ.get("ALGONAUTS_AUDIO_KEEP", "all")  # "all" | "last24" | "0,1,2,..."
SPLIT_OVERRIDE= os.environ.get("ALGONAUTS_AUDIO_SPLIT", "")
FFMPEG_BIN    = os.environ.get("ALGONAUTS_FFMPEG", "ffmpeg")
RUN_DATASETS  = [s.strip() for s in os.environ.get("ALGONAUTS_RUN_DATASETS", "").split(",") if s.strip()]

# transformers 兼容导入（新旧版本）
try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception:
    from transformers import AutoProcessor as AutoFeatureExtractor
    from transformers import AutoModel

# ───────── utils: dataset → video path ─────────
RE_FRIENDS = re.compile(r"task-(s\d{2}e\d{2})([a-d])$", re.IGNORECASE)
RE_MOVIE_TOKEN = re.compile(r"task-([a-z]+)(\d+)", re.IGNORECASE)   # e.g., bourne02 / wolf05 / life03 / figures10

def parse_friends(ds: str) -> Optional[Tuple[str, str, int]]:
    m = RE_FRIENDS.search(ds)
    if not m: return None
    ep, part = m.group(1).lower(), m.group(2).lower()
    season_num = int(ep[1:3])  # s01 -> 1
    return ep, part, season_num

def friends_candidates(ds: str) -> List[Path]:
    parsed = parse_friends(ds)
    if parsed is None:
        return []
    ep, part, season = parsed
    base = f"friends_{ep}{part}.mkv"
    friends_dir = MOVIES_ROOT / "friends"
    return [
        friends_dir / f"s{season}"          / base,
        friends_dir / f"s{season:02d}"      / base,
        friends_dir / f"season{season}"     / base,
        friends_dir / f"season{season:02d}" / base,
        friends_dir / base,
    ]

def guess_movie_basename(ds: str) -> Optional[str]:
    """
    从 dataset 名里提取诸如 bourne02 / wolf07 / life03 / figures10 之类的 token，
    生成 basename：<token>.mkv
    """
    m = RE_MOVIE_TOKEN.search(ds)
    if not m:
        return None
    token = (m.group(1) + m.group(2)).lower()  # e.g., bourne02
    return f"{token}.mkv"

# ───────── manifest support ─────────
VIDEO_MANIFEST: Dict[str, str] = {}

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

# ───────── build movies file index (one-time scan) ─────────
_MOVIE_INDEX: Dict[str, Path] = {}

def build_movies_index(root: Path):
    """
    遍历 stimuli/movies 下所有 mkv/mp4/mov/avi，建立 basename -> 绝对路径的索引
    （只记录第一次出现的路径，若重名则保留首个并告警）
    """
    global _MOVIE_INDEX
    _MOVIE_INDEX.clear()
    if not root.exists():
        return
    exts = (".mkv", ".mp4", ".mov", ".avi", ".webm")
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            name = p.name.lower()
            if name not in _MOVIE_INDEX:
                _MOVIE_INDEX[name] = p
            else:
                # 保留首个，同时记录冲突（可选打印）
                pass
    print(f"[INFO] movies index built: {len(_MOVIE_INDEX)} files indexed under {root}")

def resolve_video_path(ds: str) -> Optional[Path]:
    """
    按优先级解析：manifest -> friends -> index(猜文件名)。
    """
    # 1) manifest
    if ds in VIDEO_MANIFEST:
        p = Path(VIDEO_MANIFEST[ds])
        return p if p.exists() else None

    # 2) friends candidates
    for c in friends_candidates(ds):
        if c.exists():
            return c

    # 3) guessed basename via index
    if not _MOVIE_INDEX:
        build_movies_index(MOVIES_ROOT)
    base = guess_movie_basename(ds)
    if base and base.lower() in _MOVIE_INDEX:
        return _MOVIE_INDEX[base.lower()]

    return None

# ───────── decode audio with ffmpeg ─────────
def load_audio_ffmpeg(path: Path, sr: int = 16000) -> np.ndarray:
    """
    Use ffmpeg to decode audio to mono float32 PCM at given sample rate.
    Returns shape [T], float32 in [-1,1].
    """
    cmd = [
        FFMPEG_BIN, "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr),
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio  # [T]

# ───────── layer keeping ─────────
def choose_layers(hs_len: int, spec: str = "all") -> List[int]:
    # hidden_states: [emb, layer1, ..., layerN] -> valid layers are 1..hs_len-1
    all_layers = list(range(1, hs_len))
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return all_layers
    if spec.startswith("last"):
        try:
            k = int(spec.replace("last", ""))
            return all_layers[-k:] if k <= len(all_layers) else all_layers
        except Exception:
            return all_layers
    try:
        ids_rel = [int(x) for x in spec.split(",") if x.strip()!=""]
        out = []
        for i in ids_rel:
            if 0 <= i < len(all_layers):
                out.append(all_layers[i])
        return out if out else all_layers
    except Exception:
        return all_layers

# ───────── downsample 50 Hz → 2 Hz ─────────
def downsample_to_2hz(feats_50hz: torch.Tensor) -> torch.Tensor:
    """
    feats_50hz: [T50, D] or [T50, L, D]
    Returns 2Hz by mean pooling along time with factor 25.
    Output: [T2, D] or [T2, L, D]
    """
    if feats_50hz.ndim == 2:
        T, D = feats_50hz.shape
        K = 25
        T2 = T // K
        if T2 == 0:
            return torch.zeros((0, D), dtype=feats_50hz.dtype, device=feats_50hz.device)
        x = feats_50hz[:T2*K].reshape(T2, K, D).mean(dim=1)
        return x
    elif feats_50hz.ndim == 3:
        T, L, D = feats_50hz.shape
        K = 25
        T2 = T // K
        if T2 == 0:
            return torch.zeros((0, L, D), dtype=feats_50hz.dtype, device=feats_50hz.device)
        x = feats_50hz[:T2*K].reshape(T2, K, L, D).mean(dim=1)
        return x
    else:
        raise ValueError(f"unexpected ndim {feats_50hz.ndim}")

# ───────── main ─────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-manifest", type=str, default=os.environ.get("ALGONAUTS_VIDEO_MANIFEST", ""),
                    help="CSV with columns: dataset,video_path (absolute paths). Highest priority.")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Do not skip datasets whose output npy already exists.")
    ap.add_argument("--max-datasets", type=int, default=int(os.environ.get("MAX_DATASETS", "0")),
                    help="Process at most N datasets (0 = no limit).")
    ap.add_argument("--run-datasets", type=str, default=os.environ.get("ALGONAUTS_RUN_DATASETS", ""),
                    help="Comma-separated dataset names to run.")
    args = ap.parse_args()

    run_filter = set([s.strip() for s in args.run_datasets.split(",") if s.strip()]) if args.run_datasets else None
    skip_existing = not bool(args.no_skip_existing)
    max_datasets = int(args.max_datasets) if args.max_datasets else 0

    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid = pd.read_parquet(GRID_2HZ)
    if grid.empty:
        raise SystemExit("Empty 2Hz grid parquet.")

    # 可选 split 过滤（保持和你原逻辑一致；如不想要，可注释掉）
    split_path = Path(SPLIT_OVERRIDE) if SPLIT_OVERRIDE else (SPLIT_FULL if SPLIT_FULL.exists() else SPLIT_FALLBACK)
    if split_path and split_path.exists():
        keep = {l.strip() for l in open(split_path, "r", encoding="utf-8") if l.strip()}
        grid = grid[grid["dataset"].isin(keep)].copy()
        print(f"[INFO] Filtering to {len(keep)} datasets from {split_path.name}")

    if run_filter:
        grid = grid[grid["dataset"].isin(run_filter)].copy()
        print(f"[INFO] Further restricting to {len(run_filter)} dataset(s) via --run-datasets")

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # 加载模型
    print(f"[INFO] Loading audio model: {MODEL_ID}")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)  # FP32
    model.to(device).eval()

    # 探层数/维度
    with torch.no_grad():
        dummy = np.zeros((SR,), dtype=np.float32)
        fe = feature_extractor(dummy, sampling_rate=SR, return_tensors="pt")
        fe = {k: v.to(device) for k, v in fe.items()}
        out = model(**fe, output_hidden_states=True)
    hs_len = len(out.hidden_states)
    layer_ids = choose_layers(hs_len, KEEP_SPEC)
    D = int(getattr(getattr(model, "config", model), "hidden_size", 1024))
    L = len(layer_ids)
    print(f"[INFO] hidden_states len={hs_len}, keep {L} layers, dim={D}")

    # 载入 manifest（可选）
    if args.video_manifest:
        load_video_manifest(args.video_manifest)

    # 数据集列表（去重，排序）
    datasets_all = grid["dataset"].drop_duplicates().tolist()
    if max_datasets > 0:
        datasets_all = datasets_all[:max_datasets]

    index_rows, manifest_used, skipped = [], [], []

    for ds in tqdm(datasets_all, desc="Datasets"):
        g = grid[grid["dataset"] == ds].sort_values("bin_idx").reset_index(drop=True)
        out_path = OUT_DIR / f"{ds}.npy"
        if skip_existing and out_path.exists():
            # 保持 index 完整性（dim 填 -1 表示本次未新写）
            index_rows.append([ds, str(out_path), len(g), L, -1])
            continue

        vpath = resolve_video_path(ds)
        if vpath is None:
            print(f"[WARN] Skip {ds}: video path not resolved")
            skipped.append([ds, "video_not_found"])
            continue

        # 解码音频
        try:
            audio = load_audio_ffmpeg(vpath, sr=SR)  # float32 [T]
        except subprocess.CalledProcessError as e:
            print(f"[WARN] ffmpeg decode failed for {ds}: {e}")
            skipped.append([ds, "ffmpeg_failed"])
            continue

        T = len(audio)
        if T <= SR // 2:  # <= 0.5s
            print(f"[WARN] very short audio for {ds}; writing zeros.")
            out_arr = np.zeros((len(g), L, D), dtype=SAVE_DTYPE)
            np.save(out_path, out_arr)
            index_rows.append([ds, str(out_path), len(g), L, D])
            manifest_used.append([ds, str(vpath)])
            continue

        # 60s 切块过模型，拼 50Hz 时序
        chunk_samps = CHUNK_SECS * SR
        hs_accum = []   # list of [T50, L, D] (CPU)
        with torch.no_grad():
            for st in tqdm(range(0, T, chunk_samps), desc=f"{ds} audio->50Hz", leave=False):
                ed = min(T, st + chunk_samps)
                wav = audio[st:ed]
                fe = feature_extractor(wav, sampling_rate=SR, return_tensors="pt")
                fe = {k: v.to(device) for k, v in fe.items()}
                out = model(**fe, output_hidden_states=True)
                per_layers = []
                for lid in layer_ids:
                    h = out.hidden_states[lid]  # [B, T50, D]
                    per_layers.append(h[0])     # [T50, D]
                Hs = torch.stack(per_layers, dim=1)  # [T50, L, D]
                hs_accum.append(Hs.cpu())

        if not hs_accum:
            print(f"[WARN] model produced no frames for {ds}")
            out_arr = np.zeros((len(g), L, D), dtype=SAVE_DTYPE)
            np.save(out_path, out_arr)
            index_rows.append([ds, str(out_path), len(g), L, D])
            manifest_used.append([ds, str(vpath)])
            continue

        hs_50 = torch.cat(hs_accum, dim=0)  # [T50_total, L, D]
        hs_2  = downsample_to_2hz(hs_50)    # [T2, L, D]
        T2 = hs_2.shape[0]

        # 按 grid 对齐（bin_idx 是 2Hz 连续索引）
        out_arr = np.zeros((len(g), L, D), dtype=SAVE_DTYPE)
        for i, r in g.iterrows():
            bidx = int(r["bin_idx"])
            if 0 <= bidx < T2:
                out_arr[i, :, :] = hs_2[bidx].numpy().astype(SAVE_DTYPE)

        np.save(out_path, out_arr)
        index_rows.append([ds, str(out_path), len(g), L, D])
        manifest_used.append([ds, str(vpath)])
        print(f"[OK] {ds}: saved {out_path.name} shape={out_arr.shape} src={vpath.name} dur={T/SR:.1f}s")

    # 写索引与日志
    if index_rows:
        idx = pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"])
        idx.to_csv(OUT_DIR / "index.csv", index=False)
        print("\n[DONE] audio_2hz index.csv ->", OUT_DIR)
    if manifest_used:
        pd.DataFrame(manifest_used, columns=["dataset","video_path"]).to_csv(OUT_DIR / "manifest_used_audio.csv", index=False)
        print("      manifest_used_audio.csv has been written.")
    if skipped:
        pd.DataFrame(skipped, columns=["dataset","reason"]).to_csv(OUT_DIR / "skipped_audio.csv", index=False)
        print("      skipped_audio.csv has been written.")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    main()