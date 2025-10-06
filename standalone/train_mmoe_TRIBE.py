# -*- coding: utf-8 -*-
"""
Standalone windowed training using TRIBE-like 2Hz features with FmriEncoder_MMoE (your MMoE decoder).

- 输入: video/text/audio 三模态 2Hz 特征，按窗口训练；GT: 4 subjects 的 fMRI（1000 parcels）
- 模型: 共享专家 + 多门门控 (MMoE) + 每 subject 读出；subject_id 作为路由差异（建议开启 subject_embedding）
- 评测: 整段 episode Pearson / Spearman / R² / ISG；保存每 subject 的 pred/gt
- 可视化: 每 epoch 结束，对“本 epoch 上 VAL 均值 r 最好”的 episode + 训练集的 Friends 片段进行可视化（阈值 0.1）
"""

from __future__ import annotations
import argparse, os, sys, random, re, subprocess, shlex
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---- repo root ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
PKG_PARENT = PROJ.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# ---- import your MMoE decoder model ----
from algonauts2025.standalone.mmoe_decoder import FmriEncoder_MMoE

# ---------------- utils ----------------
def set_seed(seed: int = 33):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

# ---------------- layer helpers ----------------
def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions)) or [L - 1]
    if idxs[-1] != L - 1:
        idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]
    starts = [0] + bounds[:-1]
    groups = []
    for s, e in zip(starts, bounds):
        s = max(0, min(s, L)); e = max(0, min(e, L))
        if e <= s: s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))
    return np.stack(groups, axis=0)

def parse_layers_arg(layers_arg: str, probe_L: int):
    s = (layers_arg or "").strip().lower()
    if not s: return "indices", [probe_L - 1]
    if s == "all": return "indices", list(range(probe_L))
    if s.startswith("last"):
        try: k = int(s.replace("last", ""))
        except: k = 1
        k = max(1, min(k, probe_L))
        return "indices", list(range(max(0, probe_L - k), probe_L))
    if s.startswith("idx:"):
        idxs = []
        for p in [p for p in s[4:].split(",") if p.strip()]:
            try:
                i = int(p);
                if 0 <= i < probe_L: idxs.append(i)
            except: pass
        return "indices", sorted(set(idxs or [probe_L - 1]))
    try:
        fracs = [min(1.0, max(0.0, float(x))) for x in s.split(",") if x.strip() != ""]
        return "fractions", (fracs or [1.0])
    except:
        return "indices", [probe_L - 1]

# ---------------- metrics ----------------
@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)

def _rankdata_1d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(x, dtype=np.float64)
    sx = x[order]; n = x.size; i = 0
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]: j += 1
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
        rp[:, o] = _rankdata_1d(pred[:, o])
        rt[:, o] = _rankdata_1d(true[:,  o])
    return voxelwise_pearson(rp.astype(np.float32), rt.astype(np.float32))

@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    yt_mean = true.mean(axis=0, keepdims=True)
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - yt_mean) ** 2).sum(axis=0) + 1e-8
    return (1.0 - (ss_res / ss_tot)).astype(np.float32)

# ---------------- filename resolver ----------------
def resolve_fmri_file(root: Path, ds: str) -> Path:
    p = Path(root) / f"{ds}.npy"
    if p.exists():
        return p
    m = re.search(r"(task-[A-Za-z0-9_-]+)", ds)
    key = m.group(1) if m else None
    candidates = []
    if key:
        candidates += sorted(Path(root).glob(f"*_{key}.npy"))
        candidates += sorted(Path(root).glob(f"*{key}.npy"))
    if not candidates:
        parts = ds.split("_", 1)
        if len(parts) == 2:
            suf = parts[1]
            candidates += sorted(Path(root).glob(f"*_{suf}.npy"))
            candidates += sorted(Path(root).glob(f"*{suf}.npy"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"GT not found for ds='{ds}' under '{root}'")

def _load_fmri_flexible(root: Path, ds: str) -> np.ndarray:
    return np.load(resolve_fmri_file(root, ds))

# ---------------- dataset (windowed) ----------------
class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]): self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v): self.data[k] = v.to(device, non_blocking=True)
        return self

class WindowedDataset(Dataset):
    """
    每个样本 = episode 的一个窗口（N TR）：
      输入 video/text/audio: [G, D, N * frames_per_tr]  (G 为选层/分组数)
      目标 fmri: [1000, N]
    """
    def __init__(self,
                 ids: List[str],
                 video_root: Path, text_root: Path, audio_root: Path,
                 fmri_root: Path,
                 fractions: List[float],
                 layer_agg: str,
                 window_tr: int, stride_tr: int, frames_per_tr: int,
                 layers_arg: str = "", subject_id: int = 0):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_root = Path(fmri_root)
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.f = int(frames_per_tr)
        self.subject_id_fixed = int(subject_id)

        # 探测层数
        v0 = np.load(self.video_root / f"{ids[0]}.npy")  # [T,L,D]
        probe_L = v0.shape[1]
        self.layer_mode, payload = parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions":
            self.fracs, self.sel_indices = [float(x) for x in payload], None
        else:
            self.fracs, self.sel_indices = None, [int(i) for i in payload]
        self.layer_agg = layer_agg.lower()

        # 构建窗口索引
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            v = np.load(self.video_root / f"{ds}.npy")
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f
            arr = np.load(self.fmri_root / f"{ds}.npy") if (self.fmri_root / f"{ds}.npy").exists() else _load_fmri_flexible(self.fmri_root, ds)
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            O, T_tr_fmri = fmri.shape
            assert O == 1000
            T_tr = min(T_tr_feat, T_tr_fmri)
            self._episode_len_tr[ds] = T_tr
            for st in range(0, max(1, T_tr - self.N + 1), self.S):
                if st + self.N <= T_tr:
                    self._index.append((ds, st))

        # 记录维度
        first_ds, _ = self._index[0]
        self.G, self.Dv = self._maybe_pick_layers(self._load_feature_LDT(self.video_root / f"{first_ds}.npy")).shape[:2]
        self.Dt = self._maybe_pick_layers(self._load_feature_LDT(self.text_root  / f"{first_ds}.npy")).shape[1]
        self.Da = self._maybe_pick_layers(self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")).shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        arr = np.load(path_npy)   # [T,L,D]
        if arr.ndim != 3:
            raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))  # -> [L,D,T]

    def _maybe_pick_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        L = lat_LDT.shape[0]
        if self.layer_mode == "indices":
            sel = [i for i in self.sel_indices if 0 <= i < L] or [L - 1]
            return lat_LDT[sel]
        if self.layer_agg in ("group_mean", "groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)
        sel = sorted(set(int(round(f * (L - 1))) for f in self.fracs))
        sel = [min(L - 1, max(0, i)) for i in sel] or [L - 1]
        return lat_LDT[sel]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        win_frames = self.N * self.f
        s_frame = start_tr * self.f
        e_frame = s_frame + win_frames

        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_LDT = self._load_feature_LDT(root / f"{ds}.npy")    # [L,D,T]
            lat_GDT = self._maybe_pick_layers(lat_LDT)               # [G,D,T]
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]
                s_frame = e_frame - win_frames
            lat = lat_GDT[..., s_frame:e_frame]                      # [G,D,T2]
            feats[name] = torch.from_numpy(lat.astype(np.float32))

        # anchor GT（sub1）便于窗级参考
        fmri_path = self.fmri_root / f"{ds}.npy"
        arr = np.load(fmri_path) if fmri_path.exists() else _load_fmri_flexible(self.fmri_root, ds)
        if 1000 in arr.shape:
            fmri = arr if arr.shape[0] == 1000 else arr.T
        else:
            fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
        Y = fmri[:, start_tr:start_tr + self.N]                      # [1000,N]

        return {
            "video": feats["video"],   # [G,D,T2]
            "text":  feats["text"],
            "audio": feats["audio"],
            "fmri":  torch.from_numpy(Y.astype(np.float32)),  # [1000,N]
            "subject_id": torch.tensor(self.subject_id_fixed, dtype=torch.long),
            "ds": ds,
            "start_tr": int(start_tr),
        }

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    keys = ["video","text","audio","fmri","subject_id"]
    data: Dict[str, torch.Tensor] = {}
    for k in keys:
        data[k] = torch.stack([b[k] for b in batch], dim=0)
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)

# ---------------- episode recon/eval ----------------
def pick_friends_episode(cands: List[str]) -> str:
    fs = [ds for ds in cands if "friends" in ds.lower()]
    return fs[0] if fs else cands[0]

@torch.no_grad()
def reconstruct_one_episode_multi_subject(
    model: FmriEncoder_MMoE,
    ds: str,
    video_root: Path, text_root: Path, audio_root: Path,
    fmri_roots_by_subject: Dict[int, Path],
    layers_arg: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], int, List[int]]:
    """
    返回：
      preds_by_sub[s] : [T,1000]（仅对 available_subjects 返回）
      gts_by_sub[s]   : [T,1000]
      T_ds            : anchor 长度
      available_subjects: 实际有 GT 的 subject 列表
    """
    # anchor: sub1
    anchor_root = list(fmri_roots_by_subject.values())[0]
    dataset = WindowedDataset(
        ids=[ds],
        video_root=video_root, text_root=text_root, audio_root=audio_root,
        fmri_root=anchor_root,
        fractions=[1.0], layer_agg=layer_agg,
        window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr,
        layers_arg=layers_arg, subject_id=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=True)
    T_ds = dataset._episode_len_tr[ds]
    O = 1000
    S = getattr(model, "n_subjects", 4) or 4

    acc = {s: np.zeros((T_ds, O), dtype=np.float32) for s in range(S)}
    cnt = np.zeros((T_ds,), dtype=np.int32)

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        B = batch.data["video"].shape[0]
        N = batch.data["fmri"].shape[-1]

        # 对每个 subject 单独前向：利用 subject_id 差异 + 专属 router/readout
        y_all = []
        for s in range(S):
            batch.data["subject_id"] = torch.full((B,), s, dtype=torch.long, device=device)
            y_s = model(batch, pool_outputs=True)        # [B,O,N]
            y_all.append(y_s.unsqueeze(1))               # [B,1,O,N]
        y_all = torch.cat(y_all, dim=1).permute(0,1,3,2) # [B,S,N,O]

        st = int(batch.data["start_tr_list"][0])
        ed = min(st + N, T_ds)
        y_all = y_all[:, :, :ed-st, :].detach().cpu().numpy()
        for s in range(S):
            acc[s][st:ed] += y_all[0, s]
        cnt[st:ed] += 1

    cnt = np.maximum(cnt[:, None], 1)
    full_preds = {s: (acc[s] / cnt).astype(np.float32) for s in acc.keys()}

    # GT
    preds_by_sub, gts_by_sub, available_subjects = {}, {}, []
    for s, root in fmri_roots_by_subject.items():
        try:
            gt_all = _load_fmri_flexible(root, ds)
            if 1000 in gt_all.shape:
                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
            else:
                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
            gts_by_sub[s] = gt_all[:, :T_ds].T.astype(np.float32)  # [T,1000]
            preds_by_sub[s] = full_preds[s]
            available_subjects.append(s)
        except FileNotFoundError:
            continue

    return preds_by_sub, gts_by_sub, T_ds, available_subjects

@torch.no_grad()
def evaluate_episode_list(
    model: FmriEncoder_MMoE,
    episodes: List[str],
    roots_feat: Dict[str, Path],
    fmri_roots_by_subject: Dict[int, Path],
    layers_arg: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
    save_root: Path | None = None,
    save_split_name: str = "val",
):
    agg = {s: {"r": [], "rho": [], "r2": []} for s in range(4)}
    agg_isg = {s: [] for s in range(4)}
    used_counts = {s: 0 for s in range(4)}
    ep_mean_r: Dict[str, float] = {}

    for ds in episodes:
        preds_by_sub, gts_by_sub, _, available_subjects = reconstruct_one_episode_multi_subject(
            model=model, ds=ds,
            video_root=roots_feat["video"], text_root=roots_feat["text"], audio_root=roots_feat["audio"],
            fmri_roots_by_subject=fmri_roots_by_subject,
            layers_arg=layers_arg, layer_agg=layer_agg,
            window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr,
            device=device,
        )
        if not available_subjects:
            continue

        rs_this_ep = []
        for s in available_subjects:
            pred, gt = preds_by_sub[s], gts_by_sub[s]       # [T,1000]
            r   = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2  = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[s]["r"].append(r); agg[s]["rho"].append(rho); agg[s]["r2"].append(r2)
            used_counts[s] += 1
            rs_this_ep.append(r)

        ep_mean_r[ds] = float(np.mean(rs_this_ep)) if rs_this_ep else float("nan")

        # ISG：用“其他 subject 头”的预测与 s 的 GT 的 Pearson 平均
        for s in available_subjects:
            gt = gts_by_sub[s]
            r_list = []
            for t in available_subjects:
                if t == s: continue
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t], gt))))
            if r_list:
                agg_isg[s].append(float(np.mean(r_list)))

        # 保存 pred/gt
        if save_root is not None:
            subname = {0: "sub01", 1: "sub02", 2: "sub03", 3: "sub05"}
            for s in available_subjects:
                subdir = save_root / subname[s] / f"preds_{save_split_name}_episodes"
                subdir_gt = save_root / subname[s] / f"preds_{save_split_name}_episodes_gt"
                subdir.mkdir(parents=True, exist_ok=True)
                subdir_gt.mkdir(parents=True, exist_ok=True)
                np.save(subdir    / f"{ds}_pred.npy", preds_by_sub[s])  # [T,1000]
                np.save(subdir_gt / f"{ds}_gt.npy",   gts_by_sub[s])    # [T,1000]

    # 聚合
    per_sub_means, isg_means = {}, {}
    for s in range(4):
        if used_counts[s] > 0:
            per_sub_means[s] = {
                "r":   float(np.mean(agg[s]["r"])),
                "rho": float(np.mean(agg[s]["rho"])),
                "r2":  float(np.mean(agg[s]["r2"])),
            }
            isg_means[s] = float(np.mean(agg_isg[s])) if agg_isg[s] else float("nan")
        else:
            per_sub_means[s] = {"r": float("nan"), "rho": float("nan"), "r2": float("nan")}
            isg_means[s] = float("nan")

    # 找本轮 VAL 最好的 episode
    best_val_ep = None
    if ep_mean_r:
        best_val_ep = max(ep_mean_r.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -1e9))[0]
    return per_sub_means, isg_means, used_counts, best_val_ep

# ---------------- vis helper ----------------
def run_vis(gt_path: Path, pred_path: Path, atlas_path: Path, outdir: Path,
            subject_tag: str, modality_tag: str, threshold: float = 0.1):
    py = sys.executable
    script = PROJ / "vis" / "plot_encoding_map_plus.py"
    cmd = f'{shlex.quote(py)} {shlex.quote(str(script))} ' \
          f'--gt {shlex.quote(str(gt_path))} --pred {shlex.quote(str(pred_path))} ' \
          f'--atlas {shlex.quote(str(atlas_path))} --outdir {shlex.quote(str(outdir))} ' \
          f'--subject {shlex.quote(subject_tag)} --modality {shlex.quote(modality_tag)} ' \
          f'--threshold {threshold} --no-surf'  # 如需表面图可去掉 --no-surf
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[VIS] failed: {e}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # 列表
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="")
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--split_seed",  type=int,   default=33)

    # 特征根目录
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)

    # 多 subject fMRI 根目录
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # 层选择/聚合
    ap.add_argument("--layers", type=str, default="0.6,0.8,1.0")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean","none","None"])

    # 窗口
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)

    # MMoE / encoder
    ap.add_argument("--subject_embedding", action="store_true", help="为不同 subject 注入嵌入以区分路由/表示", default=True)
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_expert_layers", type=int, default=2)
    ap.add_argument("--moe_expert_hidden_mult", type=float, default=4.0)
    ap.add_argument("--moe_dropout", type=float, default=0.0)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_aux_weight", type=float, default=0.01)

    # 优化
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)
    ap.add_argument("--disable_swa", action="store_true")

    # 输出/日志/可视化
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out_dir", type=str, default="/home/lawrence/Desktop/algonauts-2025/algonauts2025/outputs/MoE_TRIBE")
    ap.add_argument("--log_dir", type=str, default="/home/lawrence/Desktop/algonauts-2025/algonauts2025/logs/MoE_TRIBE")
    ap.add_argument("--vis_enable", action="store_true")
    ap.add_argument("--vis_atlas", type=str, default="")
    ap.add_argument("--vis_threshold", type=float, default=0.1)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出 / 日志
    out_dir = Path(args.out_dir); (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # subject map
    sub_map = {0: "sub01", 1: "sub02", 2: "sub03", 3: "sub05"}
    fmri_roots = {
        0: Path(args.fmri_root_sub1),
        1: Path(args.fmri_root_sub2),
        2: Path(args.fmri_root_sub3),
        3: Path(args.fmri_root_sub5),
    }
    for s in sub_map.values():
        (out_dir / s / "preds_val_episodes").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "preds_val_episodes_gt").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "preds_trainprobe_episodes").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "preds_trainprobe_episodes_gt").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "vis_val").mkdir(parents=True, exist_ok=True)
        (out_dir / s / "vis_trainprobe").mkdir(parents=True, exist_ok=True)

    # 层聚合 flag
    agg = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    # 拆分
    if args.all_list:
        all_ids = read_ids(args.all_list)
        rnd = random.Random(args.split_seed); rnd.shuffle(all_ids)
        k = int(round(len(all_ids) * args.split_ratio)); k = max(1, min(len(all_ids)-1, k))
        train_ids, val_ids = all_ids[:k], all_ids[k:]
        print(f"[SPLIT] from --all_list: train={len(train_ids)}  val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise SystemExit("Provide both --train_list and --val_list, or use --all_list.")
        train_ids, val_ids = read_ids(args.train_list), read_ids(args.val_list)

    # 训练集 Friends 片段用于 train-probe
    train_probe_ds = pick_friends_episode(train_ids)
    print(f"[TRAIN-PROBE] Friends episode: {train_probe_ds}")

    # DataLoaders
    train_set = WindowedDataset(
        ids=train_ids,
        video_root=Path(args.video_root), text_root=Path(args.text_root), audio_root=Path(args.audio_root),
        fmri_root=fmri_roots[0],
        fractions=[1.0], layer_agg=agg,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
        layers_arg=args.layers, subject_id=0,
    )
    val_set_for_loss = WindowedDataset(
        ids=val_ids if len(val_ids) > 0 else train_ids[:1],
        video_root=Path(args.video_root), text_root=Path(args.text_root), audio_root=Path(args.audio_root),
        fmri_root=fmri_roots[0],
        fractions=[1.0], layer_agg=agg,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
        layers_arg=args.layers, subject_id=0,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader_for_loss = DataLoader(val_set_for_loss, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # —— 构建 MMoE 模型 —— #
    G, Dv, Dt, Da = train_set.G, train_set.Dv, train_set.Dt, train_set.Da
    feat_dims = {"video": (G, Dv), "text": (G, Dt), "audio": (G, Da)}
    model = FmriEncoder_MMoE(
        feature_dims=feat_dims,
        n_outputs=1000,
        n_output_timesteps=args.window_tr,
        n_subjects=4,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=args.subject_embedding,
        num_experts=args.moe_num_experts,
        expert_layers=args.moe_expert_layers,
        expert_hidden_mult=args.moe_expert_hidden_mult,
        expert_dropout=args.moe_dropout,
        gate_top_k=(args.moe_top_k if args.moe_top_k > 0 else None),
    ).to(device)

    # 优化
    criterion = nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8,
        fused=True  # ← 新增
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                           pct_start=args.warmup_pct, anneal_strategy="cos")

    # SWA
    use_swa = (not args.disable_swa) and (int(args.epochs * args.swa_start_ratio) < args.epochs)
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    swa_model = AveragedModel(model) if use_swa else None

    # 历史最优
    best_key_metric = float("-inf")
    best_by_sub = {s: {"r": (-np.inf, -1), "rho": (-np.inf, -1), "r2": (-np.inf, -1), "isg": (-np.inf, -1)} for s in range(4)}
    fmri_cache: Dict[Tuple[int, str], np.ndarray] = {}

    roots_feat = {"video": Path(args.video_root), "text": Path(args.text_root), "audio": Path(args.audio_root)}
    atlas_path = Path(args.vis_atlas) if args.vis_atlas else None

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # -------- Train --------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            B, N = batch.data["fmri"].shape[0], batch.data["fmri"].shape[-1]

            # 4 个 subject 分别前向，得到 [B,S,N,O]
            preds_s = []
            aux_list = []
            for s in range(4):
                batch.data["subject_id"] = torch.full((B,), s, dtype=torch.long, device=device)
                y_s = model(batch, pool_outputs=True)               # [B,O,N]
                preds_s.append(y_s.unsqueeze(1))                    # [B,1,O,N]
                if getattr(model, "last_aux_loss", None) is not None:
                    aux_list.append(model.last_aux_loss)
            y_all = torch.cat(preds_s, dim=1).permute(0,1,3,2)      # [B,S,N,O]

            # 对齐各 subject 的 GT
            ds_list = batch.data["ds_list"]; st_list = batch.data["start_tr_list"]
            loss_terms = []
            for i in range(B):
                ds = ds_list[i]; st = int(st_list[i]); ed = st + N
                for s, root in fmri_roots.items():
                    try:
                        key = (s, ds)
                        if key not in fmri_cache:
                            gt_all = _load_fmri_flexible(root, ds)
                            if 1000 in gt_all.shape:
                                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
                            else:
                                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
                            fmri_cache[key] = gt_all  # [1000,T]
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed:   # 不足则跳过该窗口/subject
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)  # [O,N]
                        pred_head = y_all[i, s].permute(1, 0)  # [O,N]
                        loss_terms.append(criterion(pred_head, gt_win))
                    except FileNotFoundError:
                        continue

            if not loss_terms:
                continue

            loss = torch.stack(loss_terms).mean()
            if aux_list and args.moe_aux_weight > 0:
                loss = loss + args.moe_aux_weight * torch.stack(aux_list).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # ↓ 释放临时对象的最后引用，避免它们在 step 前占着内存
            del loss_terms, preds_s, y_all
            # 可选：如果你还有别的临时 tensor 变量，一并 del 掉
            torch.cuda.empty_cache()
            optimizer.step()
            scheduler.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss = running / max(1, len(train_loader))
        writer.add_scalar("loss/train_epoch", float(train_loss), epoch)

        # -------- Val window-level loss（仅 anchor 粗估） --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_for_loss:
                batch = batch.to(device)
                B = batch.data["fmri"].shape[0]
                batch.data["subject_id"] = torch.zeros((B,), dtype=torch.long, device=device)
                y0 = model(batch, pool_outputs=True)         # [B,O,N]
                yt = batch.data["fmri"]                      # [B,O,N]
                val_loss += nn.functional.mse_loss(y0, yt).item()
        val_loss /= max(1, len(val_loader_for_loss))
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)

        # -------- Evaluate on validation episodes (save preds) --------
        per_sub_means, isg_means, used_counts, best_val_ep = evaluate_episode_list(
            model=model, episodes=val_ids, roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers_arg=args.layers, layer_agg=agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="val"
        )
        # Train-probe (Friends 1 集)
        probe_per_sub_means, probe_isg_means, probe_used_counts, _ = evaluate_episode_list(
            model=model, episodes=[train_probe_ds], roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers_arg=args.layers, layer_agg=agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="trainprobe"
        )

        # TB & log
        msg = [f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  |  VAL"]
        key_accumulate = []
        for s in range(4):
            r  = per_sub_means[s]["r"];   rho = per_sub_means[s]["rho"]; r2 = per_sub_means[s]["r2"]; isg = isg_means[s]
            n_used = used_counts[s]
            if not np.isnan(r): key_accumulate.append(r)
            writer.add_scalar(f"val/sub{s+1:02d}_pearson_mean",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_spearman_mean", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_r2_mean",       0.0 if np.isnan(r2) else r2,  epoch)
            if not np.isnan(isg):
                writer.add_scalar(f"val/sub{s+1:02d}_ISG_pearson", isg, epoch)
            msg.append(f" S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={n_used}")

            # best trackers
            if not np.isnan(r)   and r   > best_by_sub[s]["r"][0]:   best_by_sub[s]["r"]   = (r, epoch)
            if not np.isnan(rho) and rho > best_by_sub[s]["rho"][0]: best_by_sub[s]["rho"] = (rho, epoch)
            if not np.isnan(r2)  and r2  > best_by_sub[s]["r2"][0]:  best_by_sub[s]["r2"]  = (r2, epoch)
            if not np.isnan(isg) and isg > best_by_sub[s]["isg"][0]: best_by_sub[s]["isg"] = (isg, epoch)

        val_key_metric = float(np.mean(key_accumulate)) if key_accumulate else float("-inf")

        msg.append("  |  TRAIN-PROBE(Friends)")
        for s in range(4):
            r = probe_per_sub_means[s]["r"]; rho = probe_per_sub_means[s]["rho"]; r2 = probe_per_sub_means[s]["r2"]
            isg = probe_isg_means[s]; n_used = probe_used_counts[s]
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_pearson",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_spearman", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_r2",       0.0 if np.isnan(r2) else r2,  epoch)
            msg.append(f" S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={n_used}")
        print("  ".join(msg))

        # ---- 可视化：本轮最佳 VAL episode + Train-probe episode ----
        if args.vis_enable and atlas_path and atlas_path.exists():
            # 1) VAL 最佳 ep
            if best_val_ep is not None:
                for sid, sname in sub_map.items():
                    pred = out_dir / sname / "preds_val_episodes"    / f"{best_val_ep}_pred.npy"
                    gt   = out_dir / sname / "preds_val_episodes_gt" / f"{best_val_ep}_gt.npy"
                    if pred.exists() and gt.exists():
                        run_vis(gt, pred, atlas_path, out_dir / sname / "vis_val",
                                subject_tag=sname[-2:], modality_tag=f"VAL_{best_val_ep}",
                                threshold=args.vis_threshold)
            # 2) Train-probe
            for sid, sname in sub_map.items():
                pred = out_dir / sname / "preds_trainprobe_episodes"    / f"{train_probe_ds}_pred.npy"
                gt   = out_dir / sname / "preds_trainprobe_episodes_gt" / f"{train_probe_ds}_gt.npy"
                if pred.exists() and gt.exists():
                    run_vis(gt, pred, atlas_path, out_dir / sname / "vis_trainprobe",
                            subject_tag=sname[-2:], modality_tag=f"TRAIN_{train_probe_ds}",
                            threshold=args.vis_threshold)

        # 保存 best
        if val_key_metric > best_key_metric:
            best_key_metric = val_key_metric
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            torch.save(model, out_dir / "checkpoints" / "best_full.pt")

    # SWA
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best checkpoint by VAL Pearson(mean over subjects): {best_key_metric:.6f}")
    for s in range(4):
        br, er   = best_by_sub[s]['r']
        brh, erh = best_by_sub[s]['rho']
        br2, er2 = best_by_sub[s]['r2']
        bisg, eisg = best_by_sub[s]['isg']
        print(f"Subject S{s+1:02d} BEST: r={br:.6f}@{er}, ρ={brh:.6f}@{erh}, R²={br2:.6f}@{er2}, ISG={bisg:.6f}@{eisg}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    for sname in ["sub01","sub02","sub03","sub05"]:
        print(f"VAL preds dir: {out_dir / sname / 'preds_val_episodes'}")
        print(f"TRAIN-PROBE preds dir: {out_dir / sname / 'preds_trainprobe_episodes'}")
        print(f"VAL vis dir: {out_dir / sname / 'vis_val'}")
        print(f"TRAIN-PROBE vis dir: {out_dir / sname / 'vis_trainprobe'}")

if __name__ == "__main__":
    main()