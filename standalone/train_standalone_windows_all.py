# -*- coding: utf-8 -*-
# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/train_standalone_windows_all.py
from __future__ import annotations
import argparse, os, sys, re, math, random, subprocess
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

# ---- model (你的新版 fmri_model_min) ----
from algonauts2025.standalone.fmri_model_min1 import FmriEncoder, FmriEncoderConfig

# 可选：提升 matmul 吞吐（不改变你使用的 FP32 策略）
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def safe_save_full_model(model: nn.Module, path: Path):
    """Best-effort 保存整模型：失败则仅提示，不抛异常。"""
    try:
        torch.save(model, path)
        return True
    except Exception as e:
        print(f"[SAVE][WARN] torch.save(model) failed: {e} -> skip saving full model.")
        return False

# ---------------- utils ----------------
def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 可选：更强确定性
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
    # 取 ses 最大的；没有 ses 则当 0
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
        self._present_subjects: Dict[str, List[int]] = {}  # 哪些 subject 有该 ds

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
                # indices + group_mean 等价于直接 gather
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


# ---------------- model adapters ----------------
def compute_hidden_tr(model: nn.Module, batch: Batch) -> torch.Tensor:
    """
    聚合三模态 -> transformer -> 池化到 TR
    返回 [B, N_TR, H]
    """
    x = model.aggregate_features(batch)                      # [B,T2,H]
    x = model.transformer_forward(x, subject_id=None)        # [B,T2,H]（你的模型里 subject_embed 可选）
    x = x.transpose(1, 2)                                    # [B,H,T2]
    x_tr = model.pooler(x).transpose(1, 2)                   # [B,N,H]
    return x_tr


def predict_all_heads_from_hidden(model: nn.Module, x_tr: torch.Tensor) -> torch.Tensor:
    """
    输入 x_tr: [B,N,H]
    输出 y_all: [B,4,N,1000] 对应 sub1,sub2,sub3,sub5
    兼容：
      - model.pred_head(x, subject_id)
      - model.pred_heads[...] -> [B,N,O]
      - 单头复用
    """
    B, N, _ = x_tr.shape
    device = x_tr.device

    # subject-conditional 单头
    if hasattr(model, "pred_head") and isinstance(model.pred_head, nn.Module):
        ys = []
        for sid in (0,1,2,3):
            sid_vec = torch.full((B,), sid, dtype=torch.long, device=device)
            y = model.pred_head(x_tr, sid_vec)    # [B,N,O]
            ys.append(y.unsqueeze(1))             # [B,1,N,O]
        return torch.cat(ys, dim=1)               # [B,4,N,O]

    # 多头
    if hasattr(model, "pred_heads"):
        heads = getattr(model, "pred_heads")
        items = None
        if isinstance(heads, (nn.ModuleList, list, tuple)):
            items = list(enumerate(heads))
        elif isinstance(heads, (nn.ModuleDict, dict)):
            items = list(heads.items())
        else:
            raise TypeError(f"Unsupported pred_heads type: {type(heads)}")
        if not items:
            raise RuntimeError("Empty pred_heads")
        # 取前 4 个（不够就复用最后一个）
        ordered = [v for _, v in items]
        if len(ordered) < 4:
            ordered = ordered + [ordered[-1]] * (4 - len(ordered))
        if len(ordered) > 4:
            ordered = ordered[:4]
        ys = []
        for head in ordered:
            y = head(x_tr)                         # 期望 [B,N,O]
            if y.dim()==3 and y.shape[1]==N: pass
            elif y.dim()==3 and y.shape[2]==N: y = y.transpose(1,2)
            else: raise RuntimeError(f"head out shape {tuple(y.shape)}")
            ys.append(y.unsqueeze(1))
        return torch.cat(ys, dim=1)

    # 单头复用
    if hasattr(model, "pred") and callable(model.pred):
        y = model.pred(x_tr)                       # [B,N,O]
        if y.dim()==3 and y.shape[1]==N: pass
        elif y.dim()==3 and y.shape[2]==N: y = y.transpose(1,2)
        else: raise RuntimeError(f"single head out shape {tuple(y.shape)}")
        return y.unsqueeze(1).repeat(1,4,1,1)

    raise AttributeError("Model lacks pred_head/pred_heads/pred interfaces.")


# ---------------- eval full episode per subject ----------------
@torch.no_grad()
def eval_full_episode_subject(
    model: nn.Module,
    ds: str,
    video_root: Path, text_root: Path, audio_root: Path,
    fmap_sub: Dict[str, Path],    # { 'sub1': Path, ... } 已经挑好该 ds 对应的文件
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    返回 per-subject { 'sub1': (pred[T,1000], gt[T,1000]), ... }（缺失的不返回）
    """
    # 临时 dataset 仅该 ds
    fmri_roots = {s: p.parent if p is not None else None for s, p in fmap_sub.items()}
    ds_tmp = WindowedDatasetMS(
        ids=[ds], video_root=video_root, text_root=text_root, audio_root=audio_root,
        fmri_roots=fmri_roots, layers=layers, layer_agg=layer_agg,
        window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr
    )
    loader = DataLoader(ds_tmp, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type=="cuda"))

    # 找 T
    T_ds = ds_tmp._episode_len_tr[ds]
    n_outputs = 1000
    # 为四个被试分配累加器
    acc = {s: np.zeros((T_ds, n_outputs), dtype=np.float32) for s in SUBS}
    cnt = {s: np.zeros((T_ds,), dtype=np.int32) for s in SUBS}
    gt_all = {s: None for s in SUBS}

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            x_tr = compute_hidden_tr(model, batch)           # [B,N,H]
            y_all = predict_all_heads_from_hidden(model, x_tr)  # [B,4,N,O]
        B, S, N, O = y_all.shape
        start_tr = int(batch.data["start_tr_list"][0])
        end_tr = min(start_tr + N, T_ds)

        # GT
        fmri = batch.data["fmri"][0].cpu().numpy()  # [4,1000,N]
        for si, s in enumerate(SUBS):
            if batch.data["mask"][0, si].item() < 0.5:
                continue
            yp = y_all[0, si, :end_tr-start_tr, :].detach().cpu().numpy()  # [n,1000]
            acc[s][start_tr:end_tr] += yp
            cnt[s][start_tr:end_tr] += 1
            # 收 GT
            gt_slice = fmri[si, :, :end_tr-start_tr].T  # [n,1000]
            if gt_all[s] is None:
                gt_all[s] = np.zeros((T_ds, n_outputs), dtype=np.float32)
            gt_all[s][start_tr:end_tr] = gt_slice

    # 组装输出（缺失的被试跳过）
    out = {}
    for si, s in enumerate(SUBS):
        if gt_all[s] is None:  # 没有该被试
            continue
        c = np.maximum(cnt[s][:,None], 1)
        pred = acc[s] / c
        out[s] = (pred.astype(np.float32), gt_all[s].astype(np.float32))
    return out


# ---------------- helpers ----------------
def resolve_friends_episode(ids: List[str]) -> str:
    # 优先包含 "friends" 的 ds，否则最长的（按字符长度近似）
    for ds in ids:
        if "friends" in ds.lower():
            return ds
    # 回退：返回第一个（稳定）
    return ids[0] if ids else ""


def split_ids(all_ids: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_ratio)))
    train_ids = ids[:n_train]
    val_ids   = ids[n_train:]
    return train_ids, val_ids


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
    ap.add_argument("--modality_dropout", type=float, default=0.2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # memory helpers（不改变数学结果）
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="对 Transformer 层启用梯度检查点，省显存，不改变结果")
    ap.add_argument("--disable_swa", action="store_true",
                    help="关闭 SWA 平均模型，节省一份参数显存")

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
        print(f"[DEV] Using CUDA device: {device}")
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
        train_ids, val_ids = split_ids(all_ids, 0.9, args.seed)
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

    # model
    Gv, Gt, Ga = train_set.Gv, train_set.Gt, train_set.Ga
    Dv, Dt, Da = train_set.Dv, train_set.Dt, train_set.Da
    feat_dims = {
        "video": (Gv, Dv),
        "text" : (Gt, Dt),
        "audio": (Ga, Da),
    }
    n_outputs = 1000

    cfg = FmriEncoderConfig(
        n_subjects=4,                    # 4 个受试者
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=False,
        modality_dropout=args.modality_dropout,
    )
    model = FmriEncoder(
        feature_dims=feat_dims,
        n_outputs=n_outputs,
        n_output_timesteps=args.window_tr,
        config=cfg,
    ).to(device)

    # 对齐时间位置编码长度（若存在）
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

    # optimizer / sched
    criterion = nn.MSELoss(reduction="none")   # 我们要按 subject 掩码加权
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_pct, anneal_strategy="cos"
    )

    # SWA
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = (not args.disable_swa) and (swa_start_epoch < args.epochs)
    swa_model = AveragedModel(model) if use_swa else None

    # track best
    best_val_mean_pearson = float("-inf")
    best_per_subject: Dict[str, float] = {s: float("-inf") for s in SUBS}
    global_step = 0

    # 训练集 Friends 片段（仅显示，不参与选择）
    train_probe_ds = resolve_friends_episode(train_ids)
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

            # 前向：统一产生四个被试的预测
            x_tr = compute_hidden_tr(model, batch)              # [B,N,H]
            y_all = predict_all_heads_from_hidden(model, x_tr)  # [B,4,N,O]
            y_all = y_all.permute(0,1,3,2)                      # [B,4,O,N]

            fmri = batch.data["fmri"]                           # [B,4,O,N]
            mask = batch.data["mask"]                           # [B,4]
            # loss: 对每个 subject 有效的样本计算 MSE
            diff = y_all - fmri                                 # [B,4,O,N]
            mse = (diff**2)                                     # [B,4,O,N]
            # 按 subject 掩码加权
            mask_ = mask[:, :, None, None]                      # [B,4,1,1]
            mse = mse * mask_
            # 平均：除以有效元素个数
            denom = (mask_.sum() * float(n_outputs) * float(args.window_tr)).clamp(min=1.0)
            loss = mse.sum() / denom

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_epoch += loss.item() * y_all.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss_epoch /= max(1, len(train_set))
        writer.add_scalar("loss/train_epoch", float(train_loss_epoch), epoch)

        # ---- Val ----
        model.eval()
        # 收集每个 subject 的窗口级拼接
        preds_cat = {s: [] for s in SUBS}
        trues_cat = {s: [] for s in SUBS}

        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val", leave=False)
            for batch in pbar_v:
                batch = batch.to(device)
                x_tr = compute_hidden_tr(model, batch)              # [B,N,H]
                y_all = predict_all_heads_from_hidden(model, x_tr)  # [B,4,N,O]
                fmri = batch.data["fmri"]                           # [B,4,O,N]
                mask = batch.data["mask"]                           # [B,4]

                # reshape to [B*N, O]
                B, S, N, O = y_all.shape
                for si, s in enumerate(SUBS):
                    if mask[:,si].sum().item() == 0:
                        continue
                    # 只取该 batch 中该 subject 有效的样本（mask[b,si]==1）
                    valid_b = (mask[:,si] > 0.5).nonzero(as_tuple=False).flatten()
                    if valid_b.numel() == 0:
                        continue
                    yp = y_all[valid_b, si].reshape(-1, O).detach().cpu().numpy()   # [*,O]
                    yt = fmri [valid_b, si].permute(0,2,1).reshape(-1, O).detach().cpu().numpy()
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

        # Inter-Subject Generalization (ISG): head_i 对应 GT_j (i!=j)
        isg_vals = []
        for i, si in enumerate(SUBS):
            if not preds_cat[si]:
                continue
            pred_i = np.concatenate(preds_cat[si], axis=0)  # 这是 head_i 的预测
            for j, sj in enumerate(SUBS):
                if i == j: continue
                if not trues_cat[sj]:
                    continue
                true_j = np.concatenate(trues_cat[sj], axis=0)
                # 取对齐长度
                L = min(pred_i.shape[0], true_j.shape[0])
                if L <= 0: continue
                isg_vals.append(float(np.nanmean(voxelwise_pearson(pred_i[:L], true_j[:L]))))
        isg_mean = float(np.nanmean(isg_vals)) if isg_vals else float("nan")
        writer.add_scalar("val/ISG_pearson_mean", isg_mean, epoch)

        # 打印
        mean_val_pearson = float(np.nanmean(sub_pearsons)) if sub_pearsons else float("nan")
        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f} | "
              f"VAL mean Pearson={mean_val_pearson:.6f} | ISG={isg_mean:.6f}")
        for s in SUBS:
            m = sub_metrics[s]
            print(f"    [{s}] r={m['pearson']:.6f}  ρ={m['spearman']:.6f}  R²={m['r2']:.6f}")

        # 最佳跟踪（逐 subject + 平均）
        improved_any = False
        if not math.isnan(mean_val_pearson) and mean_val_pearson > best_val_mean_pearson:
            best_val_mean_pearson = mean_val_pearson
            improved_any = True
        for s in SUBS:
            m = sub_metrics[s]["pearson"]
            if not math.isnan(m) and m > best_per_subject[s]:
                best_per_subject[s] = m
                improved_any = True

        # 保存 best
        if improved_any:
            # 始终保存 state_dict（可靠）
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")

            # 试着保存整模型，失败就算了，不中断训练
            safe_save_full_model(model, out_dir / "checkpoints" / "best_full.pt")

        writer.add_scalar("val/mean_pearson", mean_val_pearson, epoch)
        writer.flush()

        # ---- 可选：训练集 Friends 片段评估（仅打印） ----
        try:
            if train_probe_ds:
                # 构造该 ds 的各 subject 文件映射
                tk = task_key_from_name(train_probe_ds)
                fmap_sub = {}
                for s in SUBS:
                    mp = build_task_map(fmri_roots[s])
                    fmap_sub[s] = mp.get(tk, None)
                per_sub = eval_full_episode_subject(
                    model, train_probe_ds,
                    Path(args.video_root), Path(args.text_root), Path(args.audio_root),
                    fmap_sub,
                    args.layers, args.layer_aggregation,
                    args.window_tr, args.stride_tr, args.frames_per_tr,
                    device,
                )
                # 打分
                print("[Train-Probe/Friends] metrics:")
                for s, (pred, gt) in per_sub.items():
                    r = float(np.nanmean(voxelwise_pearson(pred, gt)))
                    print(f"    {s}: Pearson={r:.6f}")
        except Exception as e:
            print(f"[Train-Probe][WARN] failed: {e}")

    # SWA finalize
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
    # 用法示例（与你当前命令一致）
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # python -m algonauts2025.standalone.train_standalone_windows2 \
    #   --all_list .../all_list.txt \
    #   --video_root .../video_2hz/sub-01 \
    #   --text_root  .../text_2hz/sub-01 \
    #   --audio_root .../audio_2hz/sub-01 \
    #   --fmri_root_sub1 .../fmri_data/sub1 \
    #   --fmri_root_sub2 .../fmri_data/sub2 \
    #   --fmri_root_sub3 .../fmri_data/sub3 \
    #   --fmri_root_sub5 .../fmri_data/sub5 \
    #   --layers 0.6,0.8,1.0 --layer_aggregation group_mean \
    #   --window_tr 100 --stride_tr 50 --frames_per_tr 3 \
    #   --epochs 25 --batch_size 1 --num_workers 0 \
    #   --grad_ckpt --disable_swa
    main()