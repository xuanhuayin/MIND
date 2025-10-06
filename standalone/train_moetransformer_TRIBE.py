# -*- coding: utf-8 -*-
# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/train_moetransformer_TRIBE.py
from __future__ import annotations
import argparse, os, sys, re, math, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

# ---- project root ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---- model ----
from algonauts2025.standalone.moe_transformer_encoder import (
    FmriEncoder_MoETransformer,
    FmriEncoderMoEConfig,
)

# 可选：提升 matmul 吞吐（不改变 FP32/AMP 策略）
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# ---------------- utils ----------------
def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]


# ---------- feature layer helpers ----------
def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    """
    输入 [L,D,T]，按 fractions 作为右边界分组做均值；输出 [G,D,T]。
    """
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions))
    if not idxs:
        idxs = [L - 1]
    if idxs[-1] != L - 1:
        idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]
    starts = [0] + bounds[:-1]
    ends = bounds
    groups = []
    for s, e in zip(starts, ends):
        s = max(0, min(s, L))
        e = max(0, min(e, L))
        if e <= s:
            s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))  # [D,T]
    return np.stack(groups, axis=0)


def parse_layers_arg(layers_arg: str, probe_L: int):
    s = (layers_arg or "").strip().lower()
    if not s:
        return ("indices", [probe_L - 1])
    if s == "all":
        return ("indices", list(range(probe_L)))
    if s.startswith("last"):
        try:
            k = int(s.replace("last", ""))
        except Exception:
            k = 1
        k = max(1, min(k, probe_L))
        return ("indices", list(range(max(0, probe_L - k), probe_L)))
    if s.startswith("idx:"):
        parts = [p for p in s[4:].split(",") if p.strip()]
        idxs = []
        for p in parts:
            try:
                i = int(p)
                if 0 <= i < probe_L:
                    idxs.append(i)
            except Exception:
                pass
        if not idxs:
            idxs = [probe_L - 1]
        return ("indices", sorted(set(idxs)))
    # fractions
    try:
        fracs = [float(x) for x in s.split(",") if x.strip() != ""]
        fracs = [min(1.0, max(0.0, f)) for f in fracs]
        if not fracs:
            fracs = [1.0]
        return ("fractions", fracs)
    except Exception:
        return ("indices", [probe_L - 1])


# ---------------- canonical file resolver ----------------
_task_rx = re.compile(r"(task-[A-Za-z0-9]+(?:_[^.]*)?)", re.IGNORECASE)
_ses_rx = re.compile(r"ses-(\d+)", re.IGNORECASE)

def task_key_from_name(name: str) -> Optional[str]:
    m = _task_rx.search(name)
    return m.group(1).lower() if m else None

def pick_maxses(paths: List[Path]) -> Optional[Path]:
    if not paths: return None
    best = None
    best_ses = -1
    for p in paths:
        m = _ses_rx.search(p.name)
        ses = int(m.group(1)) if m else -1
        if ses > best_ses:
            best_ses = ses
            best = p
    return best

def build_task_map(root: Path) -> Dict[str, Path]:
    """
    从 subject 的 fmri 目录收集所有 .npy，按 task-key 分组，取 max ses。
    返回: { 'task-xxx': /path/to/file.npy }
    """
    root = Path(root)
    files = sorted(root.glob("*.npy"))
    buckets: Dict[str, List[Path]] = {}
    for p in files:
        tk = task_key_from_name(p.name)
        if tk is None:
            continue
        buckets.setdefault(tk, []).append(p)
    out: Dict[str, Path] = {}
    for tk, lst in buckets.items():
        out[tk] = pick_maxses(lst)
    return out


# ---------------- dataset (multi-subject) ----------------
SUBS = ["sub1", "sub2", "sub3", "sub5"]

class WindowedDatasetMS(Dataset):
    """
    一个样本 = 一个 episode 的一个窗口（N 个 TR）
    - 输入 (video/text/audio): [G*, D*, N*frames_per_tr]
    - 输出 fmri: [4, 1000, N] 以及存在掩码 present[4] (子被试缺失则 0)
    """
    def __init__(
        self,
        ids: List[str],
        video_root: Path,
        text_root: Path,
        audio_root: Path,
        fmri_roots: Dict[str, Path],  # {'sub1': Path(...), ...}
        layers: str,
        layer_agg: str,
        window_tr: int,
        stride_tr: int,
        frames_per_tr: int,
    ):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root  = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_roots = {k: Path(v) for k, v in fmri_roots.items()}
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.f = int(frames_per_tr)

        # 预构建 subject 的 task map（task-xxx -> file）
        self.task_maps: Dict[str, Dict[str, Path]] = {}
        for s in SUBS:
            root = self.fmri_roots.get(s, None)
            if root is None or (not root.exists()):
                self.task_maps[s] = {}
            else:
                self.task_maps[s] = build_task_map(root)

        # 探测层数模式（从视频第一条）
        probe_key = ids[0]
        v0 = np.load(self.video_root / f"{probe_key}.npy")  # [T,L,D]
        probe_L = v0.shape[1]
        self.layer_mode, payload = parse_layers_arg(layers, probe_L)
        if self.layer_mode == "fractions":
            self.fracs = [float(x) for x in payload]
            self.sel_indices = None
        else:
            self.fracs = None
            self.sel_indices = [int(i) for i in payload]
        self.layer_agg = (layer_agg or "none").lower()

        # 构建样本索引（(ds, start_tr)）
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        self._present_subjects: Dict[str, List[int]] = {}

        for ds in ids:
            # feature 时间
            v_path = self.video_root / f"{ds}.npy"
            if not v_path.exists():
                raise FileNotFoundError(f"Missing video feature: {v_path}")
            v = np.load(v_path)  # [T,L,D]
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f

            # 找 task key
            tk = task_key_from_name(ds)
            present_subs = []
            sub_T = []
            for si, s in enumerate(SUBS):
                fmap = self.task_maps.get(s, {})
                p = fmap.get(tk, None)
                if p is None:
                    continue
                arr = np.load(p)
                if 1000 in arr.shape:
                    fmri = arr if arr.shape[0] == 1000 else arr.T
                else:
                    fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
                if fmri.shape[0] != 1000:
                    continue
                present_subs.append(si)
                sub_T.append(fmri.shape[1])

            # 至少一个 subject 有此 ds 才纳入
            if not present_subs:
                continue

            # 取所有存在 subject 与 feature 的最短 T
            if sub_T:
                T_tr = min(T_tr_feat, min(sub_T))
            else:
                T_tr = T_tr_feat
            self._episode_len_tr[ds] = T_tr
            self._present_subjects[ds] = present_subs

            for start_tr in range(0, T_tr - self.N + 1, self.S):
                self._index.append((ds, start_tr))

        # 记录每模态各自 (G,D)
        first_ds, _ = self._index[0]
        v_LDT = self._load_feature_LDT(self.video_root / f"{first_ds}.npy")
        t_LDT = self._load_feature_LDT(self.text_root  / f"{first_ds}.npy")
        a_LDT = self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")

        v_GDT = self._maybe_pick_layers(v_LDT)
        t_GDT = self._maybe_pick_layers(t_LDT)
        a_GDT = self._maybe_pick_layers(a_LDT)

        self.Gv, self.Dv = v_GDT.shape[0], v_GDT.shape[1]
        self.Gt, self.Dt = t_GDT.shape[0], t_GDT.shape[1]
        self.Ga, self.Da = a_GDT.shape[0], a_GDT.shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        arr = np.load(path_npy)
        if arr.ndim != 3:
            raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))  # [L,D,T]

    def _maybe_pick_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L]
            if not sel: sel = [L - 1]
            if self.layer_agg in ("group_mean","groupmean"):
                return lat_LDT[sel]
            else:
                return lat_LDT[sel]
        # fractions
        if self.layer_agg in ("group_mean","groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)
        else:
            sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs))
            sel = [min(L - 1, max(0, i)) for i in sel]
            if not sel: sel = [L - 1]
            return lat_LDT[sel]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        tk = task_key_from_name(ds)

        # features
        win_frames = self.N * self.f
        s_frame = start_tr * self.f
        e_frame = s_frame + win_frames

        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_LDT = self._load_feature_LDT(root / f"{ds}.npy")
            lat_GDT = self._maybe_pick_layers(lat_LDT)  # [G,D,Tf]
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]
                s_frame = e_frame - win_frames
            lat = lat_GDT[..., s_frame:e_frame]         # [G,D,win_frames]
            feats[name] = torch.from_numpy(lat.astype(np.float32))

        # fmri: [4,1000,N], mask: [4]
        fmri_4 = np.zeros((4, 1000, self.N), dtype=np.float32)
        mask_4 = np.zeros((4,), dtype=np.float32)
        for si, s in enumerate(SUBS):
            fmap = self.task_maps.get(s, {})
            p = fmap.get(tk, None)
            if p is None:
                continue
            arr = np.load(p)
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            if fmri.shape[0] != 1000:
                continue
            Y = fmri[:, start_tr:start_tr + self.N]  # [1000,N]
            if Y.shape[1] == self.N:
                fmri_4[si] = Y.astype(np.float32)
                mask_4[si] = 1.0

        sample = {
            "video": feats["video"],
            "text" : feats["text" ],
            "audio": feats["audio"],
            "fmri" : torch.from_numpy(fmri_4),     # [4,1000,N]
            "mask" : torch.from_numpy(mask_4),     # [4]
            "ds": ds,
            "start_tr": int(start_tr),
        }
        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    data = {}
    for k in ["video","text","audio"]:
        data[k] = torch.stack([b[k] for b in batch], dim=0)  # [B,G,D,T]
    data["fmri"]  = torch.stack([b["fmri"] for b in batch], dim=0)  # [B,4,1000,N]
    data["mask"]  = torch.stack([b["mask"] for b in batch], dim=0)  # [B,4]
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [b["start_tr"] for b in batch]
    return Batch(data)


class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]): self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device, non_blocking=True)
        return self


# ---------------- metrics ----------------
@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)

def _rank1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sx = x[order]
    n = x.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks

@torch.no_grad()
def voxelwise_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    N, O = pred.shape
    rp = np.empty_like(pred, dtype=np.float64)
    rt = np.empty_like(true, dtype=np.float64)
    for o in range(O):
        rp[:, o] = _rank1d(pred[:, o])
        rt[:, o] = _rank1d(true[:,  o])
    return voxelwise_pearson(rp.astype(np.float32), rt.astype(np.float32))

@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    yt_mean = true.mean(axis=0, keepdims=True)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - yt_mean) ** 2).sum(axis=0) + 1e-8
    r2 = 1.0 - (ss_res / ss_tot)
    return r2.astype(np.float32)

@torch.no_grad()
def compute_metrics(preds_np: np.ndarray, trues_np: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if preds_np.shape[0] == 0:
        for k in ["pearson","spearman","r2"]:
            out[k] = float("nan")
        return out
    pear = voxelwise_pearson(preds_np, trues_np)
    spear = voxelwise_spearman(preds_np, trues_np)
    r2v = voxelwise_r2(preds_np, trues_np)
    out["pearson"]  = float(np.nanmean(pear))
    out["spearman"] = float(np.nanmean(spear))
    out["r2"]       = float(np.nanmean(r2v))
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()

    # lists & roots
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="")
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # layers
    ap.add_argument("--layers", type=str, default="last41",
                    help="lastK | all | idx:0,1,5 | 0.6,0.8,1.0 (with group_mean)")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean","none","None"])

    # windows
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)

    # optimization
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--modality_dropout", type=float, default=0.2)

    # SWA / ckpt
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)
    ap.add_argument("--disable_swa", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")

    # MoE
    ap.add_argument("--moe_ffn_where", type=str, default="last2")
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_expert_layers", type=int, default=1)
    ap.add_argument("--moe_hidden_mult", type=float, default=4.0)
    ap.add_argument("--moe_dropout", type=float, default=0.1)
    ap.add_argument("--moe_token_chunk", type=int, default=8192)
    ap.add_argument("--moe_aux_weight", type=float, default=0.01)
    ap.add_argument("--moe_share_dense", action="store_true")
    ap.add_argument("--opt", type=str, default="adamw_no_foreach",
                    choices=["adamw_no_foreach", "adamw", "adafactor"])
    ap.add_argument("--moe_share_alpha", type=float, default=1.0)

    # subject embedding
    ap.add_argument("--subject_embedding", action="store_true")

    # AMP
    ap.add_argument("--no_amp", action="store_true", help="Disable autocast + GradScaler.")

    # logging / out
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "standalone_windows"))
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "standalone_windows_imagebind"))
    ap.add_argument("--seed", type=int, default=33)

    args = ap.parse_args()
    set_seed(args.seed)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        pin_mem = True
        print(f"[DEV] Using cuda:0")
    else:
        device = torch.device("cpu")
        pin_mem = False
        print("[DEV] Using CPU")

    # dirs
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_windows").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_episodes").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_episodes_gt").mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = Path(args.log_dir) / ts
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TB] Logging to: {tb_dir}")

    # split
    if args.all_list.strip():
        all_ids = read_ids(args.all_list.strip())
        train_ids, val_ids = (lambda l, r: (l[:r], l[r:]))(random.Random(args.seed).sample(all_ids, len(all_ids)), int(round(len(all_ids)*0.9)))
        # 为了与前面的打印一致
        print(f"[SPLIT] Using --all_list, split to train={len(train_ids)}  val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise ValueError("Provide --all_list or both --train_list/--val_list")
        train_ids = read_ids(args.train_list)
        val_ids   = read_ids(args.val_list)
        print(f"[SPLIT] Using provided lists: train={len(train_ids)} val={len(val_ids)}")

    # fmri roots
    fmri_roots = {
        "sub1": Path(args.fmri_root_sub1),
        "sub2": Path(args.fmri_root_sub2),
        "sub3": Path(args.fmri_root_sub3),
        "sub5": Path(args.fmri_root_sub5),
    }

    # dataset/loader
    def build_loader(ids: List[str], shuffle: bool):
        ds = WindowedDatasetMS(
            ids=ids,
            video_root=Path(args.video_root),
            text_root =Path(args.text_root),
            audio_root=Path(args.audio_root),
            fmri_roots=fmri_roots,
            layers=args.layers,
            layer_agg=args.layer_aggregation,
            window_tr=args.window_tr,
            stride_tr=args.stride_tr,
            frames_per_tr=args.frames_per_tr,
        )
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                        num_workers=args.num_workers, collate_fn=collate_fn,
                        pin_memory=pin_mem, drop_last=False)
        return ds, ld

    train_set, train_loader = build_loader(train_ids, shuffle=True)
    val_set,   val_loader   = build_loader(val_ids, shuffle=False)

    # model config
    feat_dims = {
        "video": (train_set.Gv, train_set.Dv),
        "text" : (train_set.Gt, train_set.Dt),
        "audio": (train_set.Ga, train_set.Da),
    }
    n_outputs = 1000

    cfg = FmriEncoderMoEConfig(
        n_subjects=4,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=bool(args.subject_embedding),
        modality_dropout=args.modality_dropout,
        hidden=3072,
        transformer_depth=8,
        n_heads=8,
        attn_dropout=0.0,
        resid_dropout=0.0,
        layer_dropout=0.0,
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
        moe_expert_layers=args.moe_expert_layers,
        moe_hidden_mult=args.moe_hidden_mult,
        moe_dropout=args.moe_dropout,
        moe_token_chunk=args.moe_token_chunk,
        moe_ffn_where=args.moe_ffn_where,
        moe_share_dense=bool(args.moe_share_dense),
        moe_share_alpha=float(args.moe_share_alpha),
    )

    model = FmriEncoder_MoETransformer(
        feature_dims=feat_dims,
        n_outputs=n_outputs,
        n_output_timesteps=args.window_tr,
        config=cfg,
    ).to(device)

    # 对齐时间位置编码长度
    with torch.no_grad():
        if hasattr(model, "time_pos_embed"):
            want = args.window_tr * args.frames_per_tr
            cur = model.time_pos_embed.shape[1]
            if cur != want:
                pos = model.time_pos_embed
                pos = torch.nn.functional.interpolate(
                    pos.transpose(1,2), size=want, mode="linear", align_corners=False
                ).transpose(1,2)
                model.time_pos_embed = nn.Parameter(pos)

    # gradient checkpoint（可选）
    if args.grad_ckpt:
        try:
            import torch.utils.checkpoint as ckpt
            if hasattr(model.encoder, "layers") and isinstance(model.encoder.layers, torch.nn.ModuleList):
                for blk in model.encoder.layers:
                    fwd = blk.forward
                    def wrapper(*x, _f=fwd, **kw):
                        return ckpt.checkpoint(_f, *x, use_reentrant=False, **kw)
                    blk.forward = wrapper
                print("[CKPT] Enabled on encoder layers.")
            else:
                print("[CKPT][WARN] encoder.layers not found; skip.")
        except Exception as e:
            print(f"[CKPT][WARN] enabling failed: {e}")

    # optimizer / scheduler
    criterion = nn.MSELoss(reduction="none")

    # ---------- drop-in: extremely memory-friendly Adafactor ----------
    class Adafactor(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, eps1=1e-30, eps2=1e-3, clip_threshold=1.0):
            defaults = dict(lr=lr, weight_decay=weight_decay, eps1=eps1, eps2=eps2, clip_threshold=clip_threshold)
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                lr = group['lr'];
                wd = group['weight_decay'];
                eps1 = group['eps1'];
                eps2 = group['eps2'];
                ct = group['clip_threshold']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if wd != 0.0:
                        g = g.add(p, alpha=wd)

                    state = self.state[p]
                    if 'step' not in state:
                        state['step'] = 0
                    state['step'] += 1

                    if p.ndim >= 2:
                        r = state.get('r')
                        c = state.get('c')
                        if r is None or c is None:
                            r = state['r'] = torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
                            c = state['c'] = torch.zeros(p.shape[-1], device=p.device, dtype=p.dtype)
                        grad_sq = g.pow(2) + eps1
                        r.mul_(0.95).add_(grad_sq.mean(dim=-1), alpha=0.05)
                        c.mul_(0.95).add_(grad_sq.mean(dim=tuple(range(0, g.ndim - 1))), alpha=0.05)
                        v = (r.unsqueeze(-1) * c.unsqueeze(0 if g.ndim == 2 else -2)).sqrt_()
                    else:
                        v = state.get('v')
                        if v is None:
                            v = state['v'] = torch.zeros((), device=p.device, dtype=p.dtype)
                        v.mul_(0.95).add_(g.pow(2).mean(), alpha=0.05)
                        v = v.sqrt()

                    u = g / (v + eps2)
                    # 简单的全张量归一 + 裁剪（避免数值爆炸）
                    denom = u.norm().clamp(min=1e-6)
                    if denom > ct:
                        u = u * (ct / denom)
                    p.add_(u, alpha=-lr)
            return loss

    # ---------- build optimizer with no-foreach AdamW by default ----------
    def build_optimizer(model, lr, weight_decay, opt="adamw_no_foreach"):
        # no-decay: LayerNorm & bias
        decay_params, nodecay_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.endswith("bias") or "norm" in n.lower() or "ln" in n.lower():
                nodecay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        if opt == "adafactor":
            return Adafactor(param_groups, lr=lr, weight_decay=weight_decay)
        elif opt == "adamw_no_foreach":
            # 关键：foreach=False，fused=False，避免 _foreach_* 路径的内存峰值
            return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95),
                                     eps=1e-8, foreach=False, fused=False)
        else:  # "adamw"
            # 即使是普通 adamw，也建议关掉 foreach
            return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95),
                                     eps=1e-8, foreach=False, fused=False)

    # —— 然后在 main() 里用：
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, opt=args.opt)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_pct, anneal_strategy="cos")

    # AMP
    from torch.cuda.amp import GradScaler, autocast
    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    scaler = GradScaler(enabled=amp_enabled)

    # SWA
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = (not args.disable_swa) and (swa_start_epoch < args.epochs)
    swa_model = AveragedModel(model) if use_swa else None

    # track best
    best_val_mean_pearson = float("-inf")
    best_per_subject: Dict[str, float] = {s: float("-inf") for s in SUBS}
    global_step = 0

    # 训练集 Friends 片段（仅显示）
    train_probe_ds = ""
    for ds in train_ids:
        if "friends" in ds.lower():
            train_probe_ds = ds
            break
    if train_probe_ds:
        print(f"[TRAIN-PROBE] Friends episode: {train_probe_ds}")

    # -------- training loop --------
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                # 前向：特征 -> 编码 -> TR 表征
                x_tr = model.forward_features(batch)              # [B,N,H]
                # 各 subject 头
                y_all = model.decode_all_subjects(x_tr)           # [B,4,N,O]
                y_all = y_all.permute(0,1,3,2)                    # [B,4,O,N] 与 fmri 对齐

                fmri = batch.data["fmri"]                         # [B,4,O,N]
                mask = batch.data["mask"]                         # [B,4]

                # 掩码加权 MSE（与原始一致）
                diff = y_all - fmri                               # [B,4,O,N]
                mse  = diff.pow(2)                                # [B,4,O,N]
                mask_ = mask[:, :, None, None]                    # [B,4,1,1]
                mse = mse * mask_
                denom = (mask_.sum() * float(n_outputs) * float(args.window_tr)).clamp(min=1.0)
                loss_main = mse.sum() / denom

                # MoE 辅助损失
                aux, zloss = model.gather_moe_losses()
                loss = loss_main + float(args.moe_aux_weight) * (aux + 0.1 * zloss)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()

            # logging（注意：这里只在日志里 .item()，不会影响梯度）
            train_loss_epoch += loss_main.detach().item() * y_all.size(0)
            pbar.set_postfix(loss=f"{loss_main.detach().item():.4f}")
            writer.add_scalar("loss/train_step", float(loss_main.detach().item()), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss_epoch /= max(1, len(train_set))
        writer.add_scalar("loss/train_epoch", float(train_loss_epoch), epoch)

        # ---- Val ----
        model.eval()
        preds_cat = {s: [] for s in SUBS}
        trues_cat = {s: [] for s in SUBS}
        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val", leave=False)
            for batch in pbar_v:
                batch = batch.to(device)
                x_tr = model.forward_features(batch)              # [B,N,H]
                y_all = model.decode_all_subjects(x_tr)           # [B,4,N,O]
                fmri = batch.data["fmri"]                         # [B,4,O,N]
                mask = batch.data["mask"]                         # [B,4]

                B, S, N, O = y_all.shape
                for si, s in enumerate(SUBS):
                    if mask[:,si].sum().item() == 0:
                        continue
                    valid_b = (mask[:,si] > 0.5).nonzero(as_tuple=False).flatten()
                    if valid_b.numel() == 0:
                        continue
                    yp = y_all[valid_b, si].reshape(-1, O).cpu().numpy()
                    yt = fmri [valid_b, si].permute(0,2,1).reshape(-1, O).cpu().numpy()
                    preds_cat[s].append(yp)
                    trues_cat[s].append(yt)

        # 计算 subject-wise metrics + 平均
        sub_metrics: Dict[str, Dict[str, float]] = {}
        sub_pearsons = []
        for s in SUBS:
            if preds_cat[s]:
                preds_np = np.concatenate(preds_cat[s], axis=0)
                trues_np = np.concatenate(trues_cat[s], axis=0)
                m = compute_metrics(preds_np, trues_np)
            else:
                m = {"pearson": float("nan"), "spearman": float("nan"), "r2": float("nan")}
            sub_metrics[s] = m
            if not math.isnan(m["pearson"]):
                sub_pearsons.append(m["pearson"])
            writer.add_scalar(f"val/{s}_pearson",  m["pearson"],  epoch)
            writer.add_scalar(f"val/{s}_spearman", m["spearman"], epoch)
            writer.add_scalar(f"val/{s}_r2",       m["r2"],       epoch)

        # Inter-Subject Generalization (ISG)
        isg_vals = []
        for i, si in enumerate(SUBS):
            if not preds_cat[si]:
                continue
            pred_i = np.concatenate(preds_cat[si], axis=0)
            for j, sj in enumerate(SUBS):
                if i == j: continue
                if not trues_cat[sj]:
                    continue
                true_j = np.concatenate(trues_cat[sj], axis=0)
                L = min(pred_i.shape[0], true_j.shape[0])
                if L <= 0: continue
                isg_vals.append(float(np.nanmean(voxelwise_pearson(pred_i[:L], true_j[:L]))))
        isg_mean = float(np.nanmean(isg_vals)) if isg_vals else float("nan")
        writer.add_scalar("val/ISG_pearson_mean", isg_mean, epoch)

        mean_val_pearson = float(np.nanmean(sub_pearsons)) if sub_pearsons else float("nan")
        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f} | VAL mean Pearson={mean_val_pearson:.6f} | ISG={isg_mean:.6f}")
        for s in SUBS:
            m = sub_metrics[s]
            print(f"    [{s}] r={m['pearson']:.6f}  ρ={m['spearman']:.6f}  R²={m['r2']:.6f}")

        # 保存 best（state_dict + 尝试整模型）
        improved = False
        if not math.isnan(mean_val_pearson) and mean_val_pearson > best_val_mean_pearson:
            best_val_mean_pearson = mean_val_pearson
            improved = True
        for s in SUBS:
            m = sub_metrics[s]["pearson"]
            if not math.isnan(m) and m > best_per_subject[s]:
                best_per_subject[s] = m
                improved = True

        if improved:
            ckpt_dir = out_dir / "checkpoints"
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            try:
                torch.save(model, ckpt_dir / "best_full.pt")
            except Exception as e:
                print(f"[SAVE][WARN] full model save failed: {e}")

        writer.add_scalar("val/mean_pearson", mean_val_pearson, epoch)
        writer.flush()

    # SWA finalize（如启用）
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best VAL mean Pearson: {best_val_mean_pearson:.6f}")
    print("Best per-subject Pearson:")
    for s in SUBS:
        print(f"  - {s}: {best_per_subject[s]:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    print(f"TensorBoard: {tb_dir}")


if __name__ == "__main__":
    main()