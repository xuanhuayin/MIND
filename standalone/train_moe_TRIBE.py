# -*- coding: utf-8 -*-
# algonauts2025/standalone/train_moe_TRIBE.py
from __future__ import annotations
import argparse, os, sys, re, math, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

# ---- project root ----
CUR_FILE = Path(__file__).resolve()
PROJ = CUR_FILE.parents[1]     # .../algonauts2025
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---- model ----
# 需要你的 moe_decoder.py 中的 FmriEncoder_MoE 支持 expert_layers / moe_dropout / top_k / subject_embedding
from algonauts2025.standalone.moe_decoder import FmriEncoder_MoE

# matmul 吞吐
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
SUB_TO_NUM = {"sub1":"1","sub2":"2","sub3":"3","sub5":"5"}  # 用于保存路径/打印

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
        fmri_roots: Dict[str, Path],
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

        self.task_maps: Dict[str, Dict[str, Path]] = {}
        for s in SUBS:
            root = self.fmri_roots.get(s, None)
            if root is None or (not root.exists()):
                self.task_maps[s] = {}
            else:
                self.task_maps[s] = build_task_map(root)

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

        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        self._present_subjects: Dict[str, List[int]] = {}

        for ds in ids:
            v_path = self.video_root / f"{ds}.npy"
            if not v_path.exists():
                raise FileNotFoundError(f"Missing video feature: {v_path}")
            v = np.load(v_path)  # [T,L,D]
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f

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

            if not present_subs:
                continue
            T_tr = min(T_tr_feat, min(sub_T)) if sub_T else T_tr_feat
            self._episode_len_tr[ds] = T_tr
            self._present_subjects[ds] = present_subs

            for start_tr in range(0, T_tr - self.N + 1, self.S):
                self._index.append((ds, start_tr))

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
            return lat_LDT[sel]
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


# ---------------- episode reconstruction (subject-conditioned MoE) ----------------
@torch.no_grad()
def reconstruct_episode_subject_conditioned(
    model: FmriEncoder_MoE,
    ds: str,
    video_root: Path, text_root: Path, audio_root: Path,
    fmri_roots: Dict[str, Path],
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    返回：
      preds_by_sub[s]: [T,1000]（subject_id=s 条件化得到，s in SUBS）
      gts_by_sub[s]:   [T,1000]
    """
    # 用 windowed dataset 确定切窗
    ds_tmp = WindowedDatasetMS(
        ids=[ds],
        video_root=video_root, text_root=text_root, audio_root=audio_root,
        fmri_roots=fmri_roots, layers=layers, layer_agg=layer_agg,
        window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr
    )
    loader = DataLoader(ds_tmp, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type=='cuda'))

    T_ds = ds_tmp._episode_len_tr[ds]
    O = 1000
    preds_sum = {s: np.zeros((T_ds, O), dtype=np.float32) for s in SUBS}
    preds_cnt = np.zeros((T_ds,), dtype=np.int32)
    gt_full   = {s: np.zeros((T_ds, O), dtype=np.float32) for s in SUBS}
    any_gt    = {s: False for s in SUBS}

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        st = int(batch.data["start_tr_list"][0])
        N = batch.data["fmri"].shape[-1]
        ed = min(st + N, T_ds)
        span = ed - st
        # 对每个 subject_id 分别 forward
        for si, s in enumerate(SUBS):
            if batch.data["mask"][0, si].item() < 0.5:
                continue
            batch.data["subject_id"] = torch.full((1,), si, dtype=torch.long, device=device)
            y = model(batch, pool_outputs=True)    # [B, O, N]
            yp = y[0, :, :span].transpose(0,1).detach().cpu().numpy()  # [span,1000]
            preds_sum[s][st:ed] += yp
            any_gt[s] = True
            gt_slice = batch.data["fmri"][0, si, :, :span].T.detach().cpu().numpy()
            gt_full[s][st:ed] = gt_slice
        preds_cnt[st:ed] += 1

    cnt = np.maximum(preds_cnt[:, None], 1)
    preds_by_sub = {s: (preds_sum[s] / cnt).astype(np.float32) for s in SUBS if any_gt[s]}
    gts_by_sub   = {s: gt_full[s].astype(np.float32)            for s in SUBS if any_gt[s]}
    return preds_by_sub, gts_by_sub


# ---------------- evaluate a list of episodes ----------------
@torch.no_grad()
def evaluate_episodes(
    model: FmriEncoder_MoE,
    episodes: List[str],
    roots_feat: Dict[str, Path],
    fmri_roots_by_subject: Dict[str, Path],
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
    save_root: Path | None = None, save_split_name: str = "val",
):
    """
    返回：
      per_sub_means: {si: {'r','rho','r2'}}
      isg_means:     {si: mean}
      used_counts:   {si: #episodes}
    若 save_root 不为 None，会把每个 episode 的 pred/gt 保存到：
      save_root/sub0X/preds_{split}_episodes/*.npy  & *_gt/*.npy
    """
    agg = {i: {"r": [], "rho": [], "r2": []} for i in range(4)}
    agg_isg = {i: [] for i in range(4)}
    used_counts = {i: 0 for i in range(4)}
    name_by_idx = {0:"sub01", 1:"sub02", 2:"sub03", 3:"sub05"}

    for ds in episodes:
        preds_by_sub, gts_by_sub = reconstruct_episode_subject_conditioned(
            model, ds,
            roots_feat["video"], roots_feat["text"], roots_feat["audio"],
            fmri_roots_by_subject,
            layers, layer_agg, window_tr, stride_tr, frames_per_tr, device
        )
        if not preds_by_sub:
            continue
        # per-subject metrics
        for i, s_name in enumerate(SUBS):
            if s_name not in preds_by_sub:
                continue
            pred = preds_by_sub[s_name]; gt = gts_by_sub[s_name]  # [T,1000]
            r   = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2  = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[i]["r"].append(r); agg[i]["rho"].append(rho); agg[i]["r2"].append(r2)
            used_counts[i] += 1

        # ISG: 对每个 s，用其他 t!=s 的预测与 s 的 GT 做相关并取均值
        have = [i for i, s_name in enumerate(SUBS) if s_name in preds_by_sub]
        for si in have:
            s_name = SUBS[si]
            gt = gts_by_sub[s_name]
            r_list = []
            for tj in have:
                if tj == si: continue
                t_name = SUBS[tj]
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t_name], gt))))
            if r_list:
                agg_isg[si].append(float(np.mean(r_list)))

        # 保存每集 pred/gt
        if save_root is not None:
            for i, s_name in enumerate(SUBS):
                if s_name not in preds_by_sub:
                    continue
                subdir_pred = save_root / name_by_idx[i] / f"preds_{save_split_name}_episodes"
                subdir_gt   = save_root / name_by_idx[i] / f"preds_{save_split_name}_episodes_gt"
                subdir_pred.mkdir(parents=True, exist_ok=True)
                subdir_gt.mkdir(parents=True, exist_ok=True)
                np.save(subdir_pred / f"{ds}_pred.npy", preds_by_sub[s_name])
                np.save(subdir_gt   / f"{ds}_gt.npy",   gts_by_sub[s_name])

    per_sub_means, isg_means = {}, {}
    for i in range(4):
        if used_counts[i] > 0:
            per_sub_means[i] = {
                "r":   float(np.mean(agg[i]["r"])),
                "rho": float(np.mean(agg[i]["rho"])),
                "r2":  float(np.mean(agg[i]["r2"])),
            }
            isg_means[i] = float(np.mean(agg_isg[i])) if agg_isg[i] else float("nan")
        else:
            per_sub_means[i] = {"r": float("nan"), "rho": float("nan"), "r2": float("nan")}
            isg_means[i] = float("nan")
    return per_sub_means, isg_means, used_counts


# ---------------- helpers ----------------
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
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # MoE hyper-params (decoder)
    ap.add_argument("--num_layers", type=int, default=1, help="每个专家 MLP 的层数")
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_dropout", type=float, default=0.1)
    ap.add_argument("--moe_aux_weight", type=float, default=0.01,
                    help="MoE 负载均衡辅助损失权重（0关闭）")
    ap.add_argument("--subject_embedding", action="store_true",
                    help="让 MoE 路由在隐藏表征上加 subject embedding 偏置")

    # memory helpers
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="对 Transformer 层启用梯度检查点")
    ap.add_argument("--disable_swa", action="store_true",
                    help="关闭 SWA")

    # logging / out
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "MoE_TRIBE"))
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "MoE_TRIBE"))
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
    for sname in ["sub01","sub02","sub03","sub05"]:
        (out_dir / sname / "preds_val_episodes").mkdir(parents=True, exist_ok=True)
        (out_dir / sname / "preds_val_episodes_gt").mkdir(parents=True, exist_ok=True)

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
    feat_dims = {
        "video": (train_set.Gv, train_set.Dv),
        "text":  (train_set.Gt, train_set.Dt),
        "audio": (train_set.Ga, train_set.Da),
    }
    n_outputs = 1000

    print(f"[MODEL] FmriEncoder_MoE | experts={args.moe_num_experts}, topk={args.moe_top_k}, expert_layers={args.num_layers}, subj_embed={args.subject_embedding}")
    model = FmriEncoder_MoE(
        feature_dims=feat_dims,
        n_outputs=n_outputs,
        n_output_timesteps=args.window_tr,
        n_subjects=4,
        num_experts=args.moe_num_experts,
        top_k=args.moe_top_k,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=args.subject_embedding,
        moe_dropout=args.moe_dropout,
        expert_layers=args.num_layers,   # ← 关键：专家 MLP 层数
    ).to(device)

    # 对齐时间位置编码长度（安全起见）
    with torch.no_grad():
        if hasattr(model, "time_pos_embed"):
            want = args.window_tr * args.frames_per_tr
            cur = model.time_pos_embed.shape[1]
            if cur < want:
                pos = torch.nn.functional.interpolate(
                    model.time_pos_embed.transpose(1,2), size=want, mode="linear", align_corners=False
                ).transpose(1,2)
                model.time_pos_embed = nn.Parameter(pos)

    # gradient checkpoint（可选）
    if args.grad_ckpt:
        try:
            import torch.utils.checkpoint as ckpt
            if hasattr(model, "encoder") and hasattr(model.encoder, "layers") and isinstance(model.encoder.layers, torch.nn.ModuleList):
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
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
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
    global_step = 0

    roots_feat = {"video": Path(args.video_root), "text": Path(args.text_root), "audio": Path(args.audio_root)}

    # -------- training loop --------
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # 针对 4 个 subject，分别 forward 并计算窗口级 MSE
            loss_terms = []
            aux_terms  = []
            B, _, O, N = batch.data["fmri"].shape
            for si in range(4):
                mask_si = batch.data["mask"][:, si] > 0.5
                if mask_si.sum().item() == 0:
                    continue
                # forward
                batch.data["subject_id"] = torch.full((B,), si, dtype=torch.long, device=device)
                y = model(batch, pool_outputs=True)                    # [B,O,N]
                # 仅对 valid 样本计算
                yv = y[mask_si]                                        # [Bv,O,N]
                gt = batch.data["fmri"][mask_si, si, :, :]             # [Bv,O,N]
                if yv.numel() == 0: continue
                diff = (yv - gt)
                mse = (diff**2).mean()                                 # 简单平均
                loss_terms.append(mse)
                # MoE 负载均衡辅助损失（若启用权重）
                if args.moe_aux_weight > 0 and getattr(model, "last_aux_loss", None) is not None:
                    aux_terms.append(model.last_aux_loss)

            if not loss_terms:
                continue

            loss = torch.stack(loss_terms).mean()
            if aux_terms:
                loss = loss + float(args.moe_aux_weight) * torch.stack(aux_terms).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_epoch += loss.item() * B
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss_epoch /= max(1, len(train_set))
        writer.add_scalar("loss/train_epoch", float(train_loss_epoch), epoch)

        # ---- Evaluate on validation episodes (no saving during training) ----
        model.eval()
        with torch.no_grad():
            per_sub_means, isg_means, used_counts = evaluate_episodes(
                model=model, episodes=val_ids, roots_feat=roots_feat,
                fmri_roots_by_subject=fmri_roots, layers=args.layers, layer_agg=args.layer_aggregation,
                window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
                device=device, save_root=None, save_split_name="val"
            )

        # ====== 打印/记录每个 subject 的 4 个指标（r / ρ / R² / ISG）======
        val_key_acc = []
        print(f"[VAL][Epoch {epoch}] metrics per subject:")
        for s in range(4):
            r   = per_sub_means[s]["r"]
            rho = per_sub_means[s]["rho"]
            r2  = per_sub_means[s]["r2"]
            isg = isg_means[s]
            n_used = used_counts[s]

            # 控制台打印
            print(f"  S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={n_used}")

            # TensorBoard
            writer.add_scalar(f"val/sub{s+1:02d}_pearson_mean",   0.0 if np.isnan(r)   else r,   epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_spearman_mean",  0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_r2_mean",        0.0 if np.isnan(r2)  else r2,  epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_ISG_pearson",    0.0 if np.isnan(isg) else isg, epoch)

            if not np.isnan(r):
                val_key_acc.append(r)

        # 作为选择 best 的 key（用 r 的平均）
        mean_val_pearson = float(np.mean(val_key_acc)) if val_key_acc else float("-inf")
        writer.add_scalar("val/mean_pearson", 0.0 if np.isnan(mean_val_pearson) else mean_val_pearson, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss_epoch:.6f} | VAL mean Pearson={mean_val_pearson:.6f}")

        # 保存 best
        improved = (mean_val_pearson > best_val_mean_pearson)
        if improved:
            best_val_mean_pearson = mean_val_pearson
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            try:
                torch.save(model, out_dir / "checkpoints" / "best_full.pt")
            except Exception as e:
                print(f"[SAVE][WARN] torch.save(model) failed: {e}")

        writer.flush()

    # SWA finalize（如启用）
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    # ---------------- After Training: 用最佳模型在所有验证集上保存 pred/gt ----------------
    print("\n[FINAL] Saving predictions & GT for all validation episodes with best model...")
    # 重新加载最佳参数（更稳妥）
    best_ckpt = out_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        model.eval()
    else:
        print(f"[WARN] best checkpoint not found at {best_ckpt}, use last-in-memory model.")

    with torch.no_grad():
        per_sub_means_final, isg_means_final, used_counts_final = evaluate_episodes(
            model=model, episodes=val_ids, roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers=args.layers, layer_agg=args.layer_aggregation,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="val"
        )

    print("\n[FINAL] VAL metrics per subject (best model):")
    vals = []
    for s in range(4):
        r   = per_sub_means_final[s]["r"]
        rho = per_sub_means_final[s]["rho"]
        r2  = per_sub_means_final[s]["r2"]
        isg = isg_means_final[s]
        print(f"  S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={used_counts_final[s]}")
        if not np.isnan(r):
            vals.append(r)
    if vals:
        print(f"[FINAL] VAL mean Pearson: {float(np.mean(vals)):.6f}")

    writer.close()
    print("\n[Done]")
    print(f"Best VAL mean Pearson (during training): {best_val_mean_pearson:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    for sname in ["sub01","sub02","sub03","sub05"]:
        print(f"VAL preds dir: {out_dir / sname / 'preds_val_episodes'}")
        print(f"VAL GT dir:   {out_dir / sname / 'preds_val_episodes_gt'}")
    print(f"TensorBoard: {Path(args.log_dir) / ts}")


if __name__ == "__main__":
    main()