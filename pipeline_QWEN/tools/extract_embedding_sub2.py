# -*- coding: utf-8 -*-
from __future__ import annotations
"""
2Hz fused features using Qwen2.5-Omni —— 无 memmap + fp32 存盘（显存友好、吞吐优化）
- Video + Audio 必须；Text 可选（若存在则强制带上）
- decord/OpenCV 解码 + 负 stride 修复（C 连续 RGB）
- 每 bin：Qwen generate() 输出 hidden_states，时序平均池化 -> [D]；汇总成 (T_bins, D)
- 控内存：限制每 bin 帧数（QWEN_MAX_FRAMES），帧下采样（TARGET_SIZE），帧微批（VIDEO_CHUNK）
- 吞吐：pin_memory + non_blocking 传输；decord GPU 解码；预取与队列可调
- 仅处理 ../timelines/all_list.txt 中列出的 dataset
- 若 OUT_DIR 下目标 .npy 已存在则跳过；若全部已存在则不加载模型，仅刷新 index.csv 并退出
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import os, re, time, gc, io, subprocess
import numpy as np
import pandas as pd
import cv2
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

# 兼容性探测（不参与预处理）
try:
    from transformers import AutoVideoProcessor as _AutoProc
except Exception:
    from transformers import AutoImageProcessor as _AutoProc

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# ========= 解码后端：优先 Decord，失败回退 OpenCV =========
USE_DECORD = True
try:
    if USE_DECORD:
        import decord
        from decord import VideoReader, cpu, gpu, bridge
        try:
            bridge.set_bridge('torch')  # 让 get_batch 返回 torch.Tensor（CPU）；不支持则忽略
        except Exception:
            pass
except Exception:
    USE_DECORD = False

# ========= 音频抽取 =========
USE_AUDIO = True
try:
    import torchaudio
    _TA_OK = True
except Exception:
    _TA_OK = False

# ========= 并行/CPU 线程 =========
PREFETCH_WORKERS = int(os.environ.get("ALGONAUTS_PREFETCH", "4"))   # OpenCV 路径线程池
MAX_INFLIGHT     = int(os.environ.get("ALGONAUTS_MAX_INFLIGHT", "16"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
cv2.setNumThreads(2)

# ========= 路径 =========
SUBJECT   = "sub-01"  # 改这里可切换 subject
THIS_FILE = Path(__file__).resolve()
TOOLS_DIR = THIS_FILE.parent
PIPE_ROOT = TOOLS_DIR.parent
PROJ_ROOT = PIPE_ROOT.parent

def _resolve_first_exist(paths: List[Path], must_exist: bool = True) -> Path:
    for p in paths:
        if p.exists():
            return p
    if must_exist:
        raise FileNotFoundError("None of candidate paths exist:\n" + "\n".join(str(p) for p in paths))
    return paths[0]

TIMELINES_PRIMARY   = PIPE_ROOT / "timelines"
TIMELINES_FALLBACK  = PROJ_ROOT / "pipeline_TRIBE" / "timelines"

GRID_2HZ = _resolve_first_exist([
    TIMELINES_PRIMARY / f"grid_2hz_{SUBJECT}.parquet",
    TIMELINES_FALLBACK / f"grid_2hz_{SUBJECT}.parquet",
])
TEXT_EVENTS = _resolve_first_exist([
    TIMELINES_PRIMARY / f"text_events_{SUBJECT}.parquet",
    TIMELINES_FALLBACK / f"text_events_{SUBJECT}.parquet",
])
DATASETS_TXT = _resolve_first_exist([
    TIMELINES_PRIMARY / "all_list.txt",
], must_exist=True)

DATA_ROOT   = PROJ_ROOT / "download" / "algonauts_2025.competitors"
MOVIES_ROOT = DATA_ROOT / "stimuli" / "movies"

OUT_DIR = PIPE_ROOT / "Qwen_8features" / "qwen2p5_fused_2hz" / SUBJECT
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= 存储精度 =========
SAVE_DTYPE = np.float32   # fp32 存盘

# ========= 模型/采样配置 =========
MODEL_ID       = os.environ.get("ALGONAUTS_QWEN_ID", "Qwen/Qwen2.5-Omni-3B")
BATCH_SIZE     = int(os.environ.get("ALGONAUTS_VIDEO_BATCH", "1"))
FRAMES_PER_BIN = int(os.environ.get("ALGONAUTS_VIDEO_FRAMES", "64"))
QWEN_MAX_FRM   = int(os.environ.get("ALGONAUTS_QWEN_MAX_FRAMES", "12"))
EFFECTIVE_FRAMES_PER_BIN = max(1, min(FRAMES_PER_BIN, QWEN_MAX_FRM))
TARGET_SIZE    = int(os.environ.get("ALGONAUTS_TARGET_SIZE", "224"))    # 224/256/288
TEXT_GRACE_SEC = float(os.environ.get("ALGONAUTS_TEXT_GRACE", "5.0"))
MAX_NEW_TOKENS = int(os.environ.get("ALGONAUTS_MAX_NEW_TOKENS", "1"))
VIDEO_CHUNK    = int(os.environ.get("ALGONAUTS_VIDEO_CHUNK", "6"))      # 帧微批
DECODER_MODE   = os.environ.get("ALGONAUTS_DECODER", "auto")            # auto|gpu|cpu
DECODER_THREADS= int(os.environ.get("ALGONAUTS_DEC_THREADS", "2"))
QUEUE_SIZE     = int(os.environ.get("ALGONAUTS_QUEUE", "2"))

# ========= 正则工具 =========
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

# ========= 视频索引 =========
_VIDEO_INDEX: Dict[str, Path] = {}
def build_video_index(root: Path):
    global _VIDEO_INDEX
    if _VIDEO_INDEX: return
    for dirpath, _, filenames in os.walk(str(root)):
        for fn in filenames:
            if fn.lower().endswith(".mkv"):
                key = fn.lower()
                p = Path(dirpath) / fn
                if key not in _VIDEO_INDEX:
                    _VIDEO_INDEX[key] = p

def resolve_video_path(ds: str) -> Optional[Path]:
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
            if p.exists(): return p
        basename = f"{ep}{part}".lower()
        build_video_index(MOVIES_ROOT)
        return _VIDEO_INDEX.get(f"friends_{basename}.mkv")

    pm = parse_movie_task(ds)
    if pm is None: return None
    basename, series = pm
    if series:
        series_dir_hits = list(MOVIES_ROOT.glob(f"**/{series}/{basename}.mkv"))
        if series_dir_hits: return series_dir_hits[0]
    build_video_index(MOVIES_ROOT)
    hit = _VIDEO_INDEX.get(f"{basename}.mkv")
    if hit: return hit
    hits = list(MOVIES_ROOT.glob(f"**/{basename}.mkv"))
    return hits[0] if hits else None

# ========= I/O & 采样 =========
def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise IOError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or nframes <= 0:
        cap.release(); raise RuntimeError(f"Bad video meta (fps={fps}, nframes={nframes}) for {path}")
    return cap, fps, nframes

def sample_frame_indices_uniform(fps: float, a: float, b: float, k: int) -> List[int]:
    if k <= 0 or b <= a or fps <= 0: return []
    step = (b - a) / k
    times = [a + (i + 0.5) * step for i in range(k)]
    idxs = [int(round(t * fps)) for t in times]
    return [max(0, min(i, int(1e12))) for i in idxs]

def read_clip_decord_vr(vr, a: float, b: float, k: int) -> Optional[np.ndarray]:
    if not USE_DECORD or vr is None: return None
    try:
        fps = float(vr.get_avg_fps())
    except Exception:
        return None
    if fps <= 0 or len(vr) <= 0: return None
    step = (b - a) / k
    times = [a + (i + 0.5) * step for i in range(k)]
    idxs = [int(round(t * fps)) for t in times]
    idxs = [max(0, min(i, len(vr) - 1)) for i in idxs]
    try:
        batch = vr.get_batch(idxs)
    except Exception:
        return None
    try:
        import torch as _torch
        if isinstance(batch, _torch.Tensor):
            batch = batch[..., [2, 1, 0]].contiguous()  # BGR->RGB
            arr = batch.cpu().numpy().copy(order="C")
            if arr.shape[0] != k: return None
            return arr
    except Exception:
        pass
    arr = batch.asnumpy()[..., ::-1].copy(order="C")
    if arr.shape[0] != k: return None
    return arr

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
    if len(frames) != k: return None
    return np.stack(frames, axis=0)

def ensure_contig_uint8_rgb(x: np.ndarray) -> np.ndarray:
    if not (isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.ndim == 4 and x.shape[-1] == 3):
        x = np.ascontiguousarray(x)
    elif not x.flags["C_CONTIGUOUS"]:
        x = np.ascontiguousarray(x)
    return x

# ========= 文本事件 =========
def _normalize_text_events(df: pd.DataFrame) -> pd.DataFrame:
    cols_lc = {c.lower(): c for c in df.columns}
    ds_col    = cols_lc.get("dataset") or cols_lc.get("ds")
    start_col = cols_lc.get("start")   or cols_lc.get("onset")  or cols_lc.get("win_start") or cols_lc.get("ts")
    end_col   = cols_lc.get("end")     or cols_lc.get("offset") or cols_lc.get("win_end")   or cols_lc.get("te")
    text_col  = (cols_lc.get("text") or cols_lc.get("content") or
                 cols_lc.get("caption") or cols_lc.get("utterance") or
                 cols_lc.get("word"))
    if not all([ds_col, start_col, end_col, text_col]):
        raise ValueError(f"Unrecognized text_events schema: {list(df.columns)}")
    out = df[[ds_col, start_col, end_col, text_col]].copy()
    out.columns = ["dataset", "start", "end", "text"]
    out = out.sort_values(["dataset", "start", "end"]).reset_index(drop=True)
    return out

def build_text_map(text_events_path: Path) -> Dict[str, pd.DataFrame]:
    if not text_events_path.exists():
        raise FileNotFoundError(f"Missing text_events parquet: {text_events_path}")
    raw = pd.read_parquet(text_events_path)
    raw.columns = [c.lower() for c in raw.columns]
    if "subject" in raw.columns:
        raw = raw[raw["subject"].astype(str).str.lower() == SUBJECT.lower()].copy()
    norm = _normalize_text_events(raw)
    out: Dict[str, pd.DataFrame] = {}
    for ds, g in norm.groupby("dataset", sort=False):
        out[ds] = g.reset_index(drop=True)
    return out

def pick_text_for_window(ds_events: pd.DataFrame, a: float, b: float, grace: float) -> str:
    hits = ds_events[~((ds_events["end"] <= a) | (ds_events["start"] >= b))]
    if not hits.empty:
        return " ".join(hits.sort_values("start")["text"].astype(str).tolist())
    cand = ds_events[(ds_events["start"] < a) & (ds_events["start"] >= a - max(0.0, grace))]
    if not cand.empty:
        return str(cand.sort_values("start").iloc[-1]["text"])
    return ""

# ========= 音频（修复：优先同名 .wav；否则 ffmpeg 抽段） =========
def read_audio_segment_from_video(path: Path, a: float, b: float, target_sr: int = 16000) -> np.ndarray:
    """
    从视频容器中读取 [a,b) 音频切片：
    1) 若存在同名 .wav，直接读并切片（更快、更稳）
    2) 否则用 ffmpeg 抽取该时段音频到内存（WAV），再用 torchaudio 读取
    返回 numpy.float32 [T]，采样率为 target_sr
    """
    if not (USE_AUDIO and _TA_OK):
        raise RuntimeError(f"[ERROR] torchaudio 不可用或 USE_AUDIO 被禁用，无法从 {path} 读取音频。")

    path = Path(path)
    t0 = max(0.0, float(a))
    duration = max(1e-2, float(b - a))
    wav_path = path.with_suffix(".wav")

    # 1) 优先同名 .wav
    if wav_path.exists():
        wav, sr = torchaudio.load(str(wav_path))  # [C, T]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        s = int(t0 * target_sr)
        e = s + int(duration * target_sr)
        audio = wav.squeeze(0)[s:e].contiguous().cpu().numpy().astype(np.float32, copy=False)
        if audio.size == 0:
            raise RuntimeError(f"[ERROR] 从 {wav_path} 切片得到空音频：{t0:.2f}-{t0+duration:.2f}s")
        return audio

    # 2) ffmpeg 现抽（避免 libsndfile 无法解析 MKV/MP4）
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{t0:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(target_sr),
        "-f", "wav", "-"
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"[ERROR] ffmpeg 抽取音频失败：{e.stderr.decode(errors='ignore')}")

    buf = io.BytesIO(proc.stdout)
    wav, sr = torchaudio.load(buf)  # [1, T]
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    audio = wav.squeeze(0).contiguous().cpu().numpy().astype(np.float32, copy=False)
    if audio.size == 0:
        raise RuntimeError(f"[ERROR] ffmpeg 抽取后读取到空音频：{path}  {t0:.2f}-{t0+duration:.2f}s")
    return audio

# ========= 构建输入 =========
def build_single_inputs(processor: Qwen2_5OmniProcessor,
                        text: Optional[str],
                        video_arr: np.ndarray,
                        audio_arr: np.ndarray,
                        expect_has_text: bool):
    video_arr = ensure_contig_uint8_rgb(video_arr)
    has_text = bool(text and str(text).strip())
    user_contents = []
    if has_text:
        user_contents.append({"type": "text", "text": str(text)})
    user_contents.append({"type": "video", "video": video_arr})
    user_contents.append({"type": "audio", "audio": audio_arr})
    if expect_has_text:
        assert has_text, "expect_has_text=True 但未将文本加入对话！"
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text",
                         "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]
        },
        {"role": "user", "content": user_contents}
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=prompt, videos=[video_arr], audio=[audio_arr], return_tensors="pt")
    return inputs, ("TVA" if has_text else "VA")

def _to_device_inplace(inputs: Dict[str, torch.Tensor], device: torch.device | str):
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            try: v = v.pin_memory()
            except Exception: pass
            inputs[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor):
                    try: sv = sv.pin_memory()
                    except Exception: pass
                    inputs[k][sk] = sv.to(device, non_blocking=True)

def generate_and_pool(model: Qwen2_5OmniForConditionalGeneration,
                      inputs: Dict[str, torch.Tensor],
                      max_new_tokens: int = 1):
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
        return_audio=False
    )
    hs = out.hidden_states
    last_layers = hs[-1] if isinstance(hs[-1], (list, tuple)) else hs
    last_layer = last_layers[-1]  # [B, S, D]
    pooled = last_layer.mean(dim=1)  # [B, D]
    return pooled

# ========= 主流程 =========
def main():
    # 目标 datasets
    with open(DATASETS_TXT, "r") as f:
        target_datasets = set(line.strip() for line in f if line.strip())
    if not target_datasets:
        raise SystemExit(f"No datasets found in list: {DATASETS_TXT}")

    # grid 过滤
    assert GRID_2HZ.exists(), f"Missing grid parquet: {GRID_2HZ}"
    grid_all = pd.read_parquet(GRID_2HZ)
    if grid_all.empty: raise SystemExit("Empty 2Hz grid parquet.")
    grid = grid_all[grid_all["dataset"].isin(target_datasets)].copy()
    if grid.empty: raise SystemExit("No matching datasets found in grid parquet for the given list.")

    # 文本事件
    assert TEXT_EVENTS.exists(), f"Missing text events parquet: {TEXT_EVENTS}"
    text_map = build_text_map(TEXT_EVENTS)

    # 预扫描
    all_ds = sorted(grid["dataset"].unique().tolist())
    todo_ds = [ds for ds in all_ds if not (OUT_DIR / f"{ds}.npy").exists()]
    if len(todo_ds) == 0:
        print("[INFO] 目标输出下所有 datasets 的文件均已存在，跳过提取，仅刷新 index.csv")
        index_rows = []
        for ds in all_ds:
            out_path = OUT_DIR / f"{ds}.npy"
            try:
                arr_exist = np.load(out_path, mmap_mode="r")
                T_2Hz = int(arr_exist.shape[0]) if arr_exist.ndim >= 1 else -1
                dim   = int(arr_exist.shape[1]) if arr_exist.ndim == 2 else -1
                index_rows.append([ds, str(out_path), T_2Hz, 1, dim])
            except Exception as e:
                print(f"[WARN] 读取 {out_path.name} 失败：{e}")
                index_rows.append([ds, str(out_path), -1, 1, -1])
        if index_rows:
            pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"])\
              .to_csv(OUT_DIR / "index.csv", index=False)
            print("\n[DONE] qwen2p5_fused_2hz index.csv ->", OUT_DIR)
        else:
            print("[WARN] 没有可写入 index.csv 的条目。")
        return

    # 设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Loading Qwen2.5-Omni model: {MODEL_ID} on {device}")

    try: _ = _AutoProc.from_pretrained(MODEL_ID)
    except Exception: pass

    # 推理用 fp16（省显存），最终保存 fp32（SAVE_DTYPE）
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
    try: model.disable_talker()
    except Exception: pass

    # probe 估计维度
    with torch.inference_mode():
        if not _TA_OK:
            raise RuntimeError("[ERROR] torchaudio 不可用，无法运行（音频为必需）。")
        dummy_vid = np.zeros((max(1, EFFECTIVE_FRAMES_PER_BIN), TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        dummy_audio = np.zeros((16000,), dtype=np.float32)
        inputs, _ = build_single_inputs(processor, None, dummy_vid, dummy_audio, expect_has_text=False)
        _to_device_inplace(inputs, device)
        D = int(generate_and_pool(model, inputs, max_new_tokens=MAX_NEW_TOKENS).shape[-1])
        print(f"[INFO] feature dim (pooled last hidden) = {D}")
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    build_video_index(MOVIES_ROOT)
    index_rows = []

    # each dataset
    for ds, g in grid.groupby("dataset", sort=True):
        out_path = OUT_DIR / f"{ds}.npy"
        if out_path.exists():
            print(f"[SKIP] {ds}: {out_path.name} 已存在，跳过提取")
            try:
                arr_exist = np.load(out_path, mmap_mode="r")
                T_2Hz = int(arr_exist.shape[0]) if arr_exist.ndim >= 1 else -1
                dim   = int(arr_exist.shape[1]) if arr_exist.ndim == 2 else -1
                index_rows.append([ds, str(out_path), T_2Hz, 1, dim])
            except Exception as e:
                print(f"[WARN] 读取已存在文件失败（{e}），仍然跳过该数据集")
                index_rows.append([ds, str(out_path), -1, 1, -1])
            continue

        g = g.sort_values("bin_idx").reset_index(drop=True)
        vpath = resolve_video_path(ds)
        if vpath is None: raise FileNotFoundError(f"[ERROR] 无法为 {ds} 解析视频路径。")
        try:
            cap, fps, nframes = open_video(vpath)
        except Exception as e:
            raise RuntimeError(f"[ERROR] 打开视频失败 {ds}: {e}")

        ds_events = text_map.get(ds, pd.DataFrame(columns=["dataset", "start", "end", "text"]))
        T_bins = len(g)
        out_path_tmp = OUT_DIR / f"{ds}_tmp.npy"
        out_arr = np.zeros((T_bins, D), dtype=SAVE_DTYPE)   # fp32 存盘

        # —— 新增：逐 bin 的均值/方差统计 ——
        stats_rows: List[Tuple[int, float, float]] = []
        stats_path = OUT_DIR / f"{ds}_embed_stats.tsv"

        n_tva, n_va = 0, 0
        batch_videos: List[np.ndarray] = []
        batch_texts:  List[Optional[str]] = []
        batch_audios: List[np.ndarray] = []
        batch_slots:  List[int] = []

        def flush():
            nonlocal n_tva, n_va
            if not batch_videos: return
            feats_list, slots_list = [], []
            with torch.inference_mode():
                for i in range(len(batch_videos)):
                    txt  = batch_texts[i]
                    vid0 = batch_videos[i]   # [T,H,W,3] uint8
                    aud  = batch_audios[i]
                    bidx = batch_slots[i]

                    # 双保险：下采样
                    if TARGET_SIZE > 0 and (vid0.shape[1] != TARGET_SIZE or vid0.shape[2] != TARGET_SIZE):
                        vid0 = np.stack(
                            [cv2.resize(fr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for fr in vid0],
                            axis=0
                        ).astype(np.uint8)

                    # 微批帧
                    if VIDEO_CHUNK and VIDEO_CHUNK > 0 and vid0.shape[0] > VIDEO_CHUNK:
                        chunks = [vid0[j:j+VIDEO_CHUNK] for j in range(0, vid0.shape[0], VIDEO_CHUNK)]
                    else:
                        chunks = [vid0]

                    pooled_chunks = []
                    for sub in chunks:
                        inputs, _mode = build_single_inputs(processor, txt, sub, aud, expect_has_text=(txt is not None))
                        _to_device_inplace(inputs, device)
                        pooled = generate_and_pool(model, inputs, max_new_tokens=MAX_NEW_TOKENS)  # [1,D]
                        pooled_chunks.append(pooled.to(dtype=torch.float32))
                    pooled_bin = torch.mean(torch.stack(pooled_chunks, dim=0), dim=0)  # [1,D]

                    # ====== 新增：每个 embedding 的均值/方差 ======
                    emb = pooled_bin.squeeze(0)                 # [D]
                    emb_mean = float(emb.mean().item())
                    emb_std  = float(emb.std().item())
                    print(f"[EMB] {ds} bin={bidx} mean={emb_mean:.6f} std={emb_std:.6f}")
                    try:
                        pbar.set_postfix({"μ": f"{emb_mean:.4f}", "σ": f"{emb_std:.4f}"})
                    except Exception:
                        pass
                    stats_rows.append((int(bidx), emb_mean, emb_std))
                    # ===========================================

                    feats_list.append(pooled_bin)
                    slots_list.append(bidx)
                    if txt: n_tva += 1
                    else:   n_va  += 1

                feats = torch.cat(feats_list, dim=0)  # [B,D]
                arr = feats.detach().cpu().numpy().astype(SAVE_DTYPE, copy=False)

            for i, bidx in enumerate(slots_list):
                out_arr[bidx, :] = arr[i]

            batch_videos.clear(); batch_texts.clear(); batch_audios.clear(); batch_slots.clear()
            del feats_list, slots_list
            gc.collect();
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        print(f"[DATASET] {ds}  bins={T_bins}  video={vpath.name}  fps={fps:.2f} frames={nframes}  frames/bin={EFFECTIVE_FRAMES_PER_BIN}")
        t0 = time.time()

        tasks = [(int(r["bin_idx"]), float(r["win_start"]), float(r["win_end"])) for _, r in g.iterrows()]
        pbar = tqdm(total=T_bins, desc=f"{ds}")

        if USE_DECORD:
            vr = None
            if DECODER_MODE in ("auto", "gpu"):
                try:
                    vr = VideoReader(str(vpath), ctx=gpu(0), num_threads=max(1, DECODER_THREADS))
                except Exception:
                    vr = None
            if vr is None:
                vr = VideoReader(str(vpath), ctx=cpu(0), num_threads=max(1, DECODER_THREADS))

            import queue, threading
            q: "queue.Queue[tuple[int, Optional[np.ndarray], float, float]]" = queue.Queue(maxsize=max(1, QUEUE_SIZE))
            SENTINEL = (-1, None, 0.0, 0.0)

            def producer():
                for bidx, a, b in tasks:
                    clip = read_clip_decord_vr(vr, a, b, EFFECTIVE_FRAMES_PER_BIN)
                    if clip is not None and TARGET_SIZE > 0:
                        clip = np.stack(
                            [cv2.resize(fr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for fr in clip],
                            axis=0
                        ).astype(np.uint8)
                    q.put((bidx, clip, a, b), block=True)
                q.put(SENTINEL, block=True)

            threading.Thread(target=producer, daemon=True).start()

            while True:
                bidx, clip, a, b = q.get()
                if bidx == -1: break
                if clip is None or not isinstance(clip, np.ndarray) or clip.size == 0:
                    raise RuntimeError(f"[ERROR] {ds} bin={bidx} 在 {a:.2f}-{b:.2f}s 抽取视频帧失败")
                txt = pick_text_for_window(ds_events, a, b, TEXT_GRACE_SEC) or ""
                aud = read_audio_segment_from_video(vpath, a, b)
                batch_videos.append(clip)
                batch_texts.append(txt if txt.strip() else None)
                batch_audios.append(aud)
                batch_slots.append(bidx)
                if len(batch_videos) >= BATCH_SIZE: flush()
                pbar.update(1)

            flush(); pbar.close()

        else:
            reader = read_clip_cv
            with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as ex:
                inflight: Dict = {}
                it = iter(tasks)
                while len(inflight) < min(MAX_INFLIGHT, len(tasks)):
                    try: bidx, a, b = next(it)
                    except StopIteration: break
                    fut = ex.submit(reader, vpath, a, b, EFFECTIVE_FRAMES_PER_BIN)
                    inflight[fut] = (bidx, a, b)

                while inflight:
                    for fut in as_completed(list(inflight.keys()), timeout=None):
                        bidx, a, b = inflight.pop(fut)
                        clip = fut.result()
                        if clip is None or not isinstance(clip, np.ndarray) or clip.size == 0:
                            raise RuntimeError(f"[ERROR] {ds} bin={bidx} 在 {a:.2f}-{b:.2f}s 抽取视频帧失败")
                        if TARGET_SIZE > 0:
                            clip = np.stack(
                                [cv2.resize(fr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA) for fr in clip],
                                axis=0
                            ).astype(np.uint8)
                        txt = pick_text_for_window(ds_events, a, b, TEXT_GRACE_SEC) or ""
                        aud = read_audio_segment_from_video(vpath, a, b)
                        batch_videos.append(clip)
                        batch_texts.append(txt if txt.strip() else None)
                        batch_audios.append(aud)
                        batch_slots.append(bidx)
                        if len(batch_videos) >= BATCH_SIZE: flush()
                        pbar.update(1)
                        try:
                            nbidx, na, nb = next(it)
                            nfut = ex.submit(reader, vpath, na, nb, EFFECTIVE_FRAMES_PER_BIN)
                            inflight[nfut] = (nbidx, na, nb)
                        except StopIteration:
                            pass
                        break
                pbar.close()

        try: cap.release()
        except Exception: pass

        dt = time.time() - t0
        np.save(out_path_tmp, out_arr); os.replace(out_path_tmp, out_path)
        print(f"[OK] {ds}: saved {out_path.name} shape=({T_bins}, {D})  elapsed={dt:.1f}s")
        print(f"[STATS] {ds}: TVA(含文本)={n_tva}, VA(无文本)={n_va}")

        # —— 写出每-bin统计到 TSV ——
        try:
            if stats_rows:
                pd.DataFrame(stats_rows, columns=["bin_idx", "mean", "std"])\
                  .to_csv(stats_path, sep="\t", index=False)
                print(f"[STATS] {ds}: per-embedding mean/std -> {stats_path}")
        except Exception as e:
            print(f"[WARN] 写入 {stats_path.name} 失败：{e}")

        # 自检
        try:
            arr_test = np.load(out_path, mmap_mode="r")
            all_zero_rows = int(np.sum(np.all(arr_test == 0, axis=1)))
            print(f"[TEST] {ds}: shape={arr_test.shape}, dtype={arr_test.dtype}, "
                  f"min={arr_test.min():.6f}, max={arr_test.max():.6f}, "
                  f"mean={arr_test.mean():.6f}, std={arr_test.std():.6f}, "
                  f"zero_rows={all_zero_rows}")
            if all_zero_rows > 0:
                print("⚠️  发现全 0 行，请检查该数据段的输入或显存设置。")
        except Exception as e:
            print(f"[TEST] {ds}: 读取检测失败：{e}")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if index_rows:
        pd.DataFrame(index_rows, columns=["dataset","npy_path","T_2Hz","n_layers","dim"])\
          .to_csv(OUT_DIR / "index.csv", index=False)
        print("\n[DONE] qwen2p5_fused_2hz index.csv ->", OUT_DIR)
    else:
        raise RuntimeError("[ERROR] 没有任何 dataset 被处理。")

if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    main()