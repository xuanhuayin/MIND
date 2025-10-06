# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/train_standalone_windows.py
# -*- coding: utf-8 -*-
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
import subprocess

# ---- 工程根路径 ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---- 最小模型（保持接口） ----
# from algonauts2025.standalone.fmri_model_min import FmriEncoder, FmriEncoderConfig
from .fmri_model_min import FmriEncoder, FmriEncoderConfig


# ---------------- 工具 ----------------
def set_seed(seed: int = 33):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    """
    输入: [L_full, D, T] → 输出: [G, D, T]，G=len(fractions)
    fractions 当作“层索引比例”的右边界，每段做均值。
    """
    L = lat_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L - 1))) for f in fractions))
    if not idxs:
        idxs = [L - 1]
    if idxs[-1] != L - 1:
        idxs[-1] = L - 1
    bounds = [i + 1 for i in idxs]  # 右开
    starts = [0] + bounds[:-1]
    ends = bounds
    groups = []
    for s, e in zip(starts, ends):
        s = max(0, min(s, L))
        e = max(0, min(e, L))
        if e <= s:
            s, e = L - 1, L
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))  # [D, T]
    return np.stack(groups, axis=0)  # [G, D, T]


# ---------------- 极简 Batch 容器 ----------------
class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device, non_blocking=True)
        return self


# ---------------- 数据集（滑动窗口） ----------------
class WindowedDataset(Dataset):
    """
    样本 = 某个视频的一个窗口（N 个 TR）：
      - 输入（video/text/audio）：[G, D, N * frames_per_tr]  —— 帧级
      - 目标（fmri）：             [1000, N]                 —— TR 级（固定 1000 维）
    """
    def __init__(
        self,
        ids: List[str],
        video_root: Path,
        text_root: Path,
        audio_root: Path,
        fmri_root: Path,
        fractions: List[float],
        layer_agg: str,
        window_tr: int,
        stride_tr: int,
        frames_per_tr: int,
    ):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_root = Path(fmri_root)
        self.fracs = fractions
        self.layer_agg = layer_agg.lower()
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.f = int(frames_per_tr)

        # 预建所有窗口的索引：(ds, start_tr)
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}  # 用于重建整段预测
        for ds in ids:
            key = ds  # feature 文件名就是 ds

            # 读取任一模态拿到帧长度 T_frames
            v_path = self.video_root / f"{key}.npy"
            if not v_path.exists():
                raise FileNotFoundError(f"Missing video feature: {v_path}")
            v = np.load(v_path)  # [T_frames, L, D]
            if v.ndim != 3:
                raise ValueError(f"Expect [T,L,D], got {v.shape} @ {v_path}")
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f  # 按帧映射到 TR 计数（向下取整）

            # fMRI 读取，固定 O=1000
            fmri_path = self.fmri_root / f"{ds}.npy"
            if not fmri_path.exists():
                raise FileNotFoundError(f"Missing fmri npy: {fmri_path}")
            arr = np.load(fmri_path)
            if arr.ndim != 2:
                raise ValueError(f"fmri must be 2D, got {arr.shape}: {fmri_path}")
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T  # (1000,T_tr_fmri)
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
                print(f"[WARN] fmri {arr.shape} no 1000-dim axis, used heuristic.")
            O, T_tr_fmri = fmri.shape
            assert O == 1000, f"Expect O=1000, got {O} for {fmri_path}"

            T_tr = min(T_tr_feat, T_tr_fmri)  # 对齐至双方最短
            self._episode_len_tr[ds] = T_tr

            # 按窗口切（严格固定 N 个 TR，丢掉尾部不完整窗）
            for start_tr in range(0, T_tr - self.N + 1, self.S):
                self._index.append((ds, start_tr))

        # 记录聚合后的“层数 G”和“每层维度 D*”
        first_ds, _ = self._index[0]
        v_LDT = self._load_feature_LDT(self.video_root / f"{first_ds}.npy")  # [L,D,Tf]
        self.G = self._maybe_aggregate_layers(v_LDT).shape[0]
        self.Dv = self._maybe_aggregate_layers(v_LDT).shape[1]
        t_LDT = self._load_feature_LDT(self.text_root / f"{first_ds}.npy")
        a_LDT = self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")
        self.Dt = self._maybe_aggregate_layers(t_LDT).shape[1]
        self.Da = self._maybe_aggregate_layers(a_LDT).shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        """把缓存的 [T,L,D] 转为 [L,D,T]"""
        arr = np.load(path_npy)
        if arr.ndim != 3:
            raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))  # [L,D,T]

    def _maybe_aggregate_layers(self, lat_LDT: np.ndarray) -> np.ndarray:
        if self.layer_agg in ("group_mean", "groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)   # [G,D,T]
        elif self.layer_agg in ("none", "null"):
            L = lat_LDT.shape[0]
            sel = sorted(set(int(round(f*(L-1))) for f in self.fracs))
            sel = [min(L-1, max(0, i)) for i in sel]
            if not sel: sel = [L-1]
            return lat_LDT[sel]  # [G,D,T]
        else:
            raise ValueError(f"Unsupported layer_aggregation: {self.layer_agg}")

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        key = ds

        # 高频特征（帧级）：取 N*frames_per_tr 帧
        win_frames = self.N * self.f
        s_frame = start_tr * self.f
        e_frame = s_frame + win_frames

        feats = {}
        for name, root in (("video", self.video_root), ("text", self.text_root), ("audio", self.audio_root)):
            lat_LDT = self._load_feature_LDT(root / f"{key}.npy")
            lat_GDT = self._maybe_aggregate_layers(lat_LDT)  # [G,D,Tf]
            # 裁帧
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]
                s_frame = e_frame - win_frames
            lat = lat_GDT[..., s_frame:e_frame]              # [G,D,win_frames]
            feats[name] = torch.from_numpy(lat.astype(np.float32))

        # fMRI（TR 级）：取 N 个 TR，O=1000 固定
        fmri_path = self.fmri_root / f"{ds}.npy"
        arr = np.load(fmri_path)
        if 1000 in arr.shape:
            fmri = arr if arr.shape[0] == 1000 else arr.T  # [1000, T_tr]
        else:
            fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            print(f"[WARN] fmri {arr.shape} no 1000-dim axis, used heuristic.")
        Y = fmri[:, start_tr:start_tr + self.N]            # [1000, N]

        sample = {
            "video": feats["video"],  # [G,D,win_frames]
            "text" : feats["text" ],  # [G,D,win_frames]
            "audio": feats["audio"],  # [G,D,win_frames]
            "fmri" : torch.from_numpy(Y.astype(np.float32)),  # [1000,N]
            "subject_id": torch.tensor(0, dtype=torch.long),   # 单被试
            "ds": ds,
            "start_tr": start_tr,
        }
        return sample


# ---------------- collate ----------------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    keys = ["video","text","audio","fmri","subject_id"]
    data: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k == "subject_id":
            data[k] = torch.stack([b[k] for b in batch], dim=0)  # [B]
        else:
            data[k] = torch.stack([b[k] for b in batch], dim=0)  # video/text/audio: [B,G,D,win_frames]; fmri: [B,1000,N]
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)


# ---------------- 指标工具 ----------------
@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    pred, true: [N_total, O]  把所有窗口（或重建后所有 TR）拼起来
    返回：每个 voxel 的皮尔逊相关 [O]
    """
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)

@torch.no_grad()
def voxelwise_spearman(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Spearman = Pearson(rank(pred), rank(true)) 按列（voxel）计算。
    输入形状同上 [N_total, O]
    """
    def rank_along_axis(x: np.ndarray) -> np.ndarray:
        # 稳定排序生成秩，处理并列（平均秩）
        n, m = x.shape
        ranks = np.empty_like(x, dtype=np.float32)
        for j in range(m):
            col = x[:, j]
            order = np.argsort(col, kind="mergesort")
            inv = np.empty_like(order)
            inv[order] = np.arange(n)
            # 平均并列秩
            sorted_col = col[order]
            diffs = np.diff(sorted_col)
            ties = np.where(np.isclose(diffs, 0.0))[0]
            if ties.size == 0:
                ranks[:, j] = inv.astype(np.float32)
            else:
                # 找到并列段
                start = 0
                r = np.zeros(n, dtype=np.float32)
                k = 0
                while k < n:
                    k2 = k
                    while k2 + 1 < n and math.isclose(sorted_col[k2+1], sorted_col[k2], rel_tol=0.0, abs_tol=0.0):
                        k2 += 1
                    # 平均秩
                    avg_rank = (k + k2) / 2.0
                    r[k:k2+1] = avg_rank
                    k = k2 + 1
                # 还原到原次序
                inv_order = np.empty_like(order)
                inv_order[order] = np.arange(n)
                ranks[:, j] = r[inv_order]
        return ranks

    rp = rank_along_axis(pred)
    rt = rank_along_axis(true)
    return voxelwise_pearson(rp, rt)

@torch.no_grad()
def voxelwise_r2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    每个 voxel 的 R^2：1 - SSE/SST，沿样本维（N_total）计算
    """
    y = true
    yhat = pred
    y_mean = y.mean(axis=0, keepdims=True)
    sse = ((y - yhat) ** 2).sum(axis=0)
    sst = ((y - y_mean) ** 2).sum(axis=0)
    r2 = 1.0 - (sse / (sst + 1e-8))
    return r2.astype(np.float32)

def load_noise_ceiling(noise_ceiling_path: Optional[str], n_voxels: int) -> Optional[np.ndarray]:
    """
    允许提供:
    - [1000] 每 voxel 一个值
    - [T,1000] 按时间的噪声上限（将对 T 取均值）
    """
    if not noise_ceiling_path:
        return None
    p = Path(noise_ceiling_path)
    if not p.exists():
        print(f"[NC][WARN] noise_ceiling_path not found: {p}")
        return None
    arr = np.load(p)
    if arr.ndim == 1 and arr.shape[0] == n_voxels:
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == n_voxels:
        return arr.mean(axis=0).astype(np.float32)
    print(f"[NC][WARN] Unexpected noise ceiling shape {arr.shape}, expect [1000] or [T,1000]. Ignored.")
    return None

@torch.no_grad()
def compute_metrics(preds_np: np.ndarray, trues_np: np.ndarray, noise_ceiling_vec: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    输入: preds_np, trues_np 形状 [N_total, O]
    输出: 多指标的“voxel 均值”
    """
    out: Dict[str, float] = {}
    if preds_np.shape[0] == 0:
        for k in ["pearson", "spearman", "r2", "norm_corr", "noise_ceiling"]:
            out[k] = float("nan")
        return out

    pear = voxelwise_pearson(preds_np, trues_np)
    spear = voxelwise_spearman(preds_np, trues_np)
    r2v = voxelwise_r2(preds_np, trues_np)

    out["pearson"] = float(np.nanmean(pear))
    out["spearman"] = float(np.nanmean(spear))
    out["r2"] = float(np.nanmean(r2v))

    if noise_ceiling_vec is not None:
        out["noise_ceiling"] = float(np.nanmean(noise_ceiling_vec))
        denom = (noise_ceiling_vec + 1e-8)
        norm_corr = pear / denom
        out["norm_corr"] = float(np.nanmean(norm_corr))
    else:
        out["noise_ceiling"] = float("nan")
        out["norm_corr"] = float("nan")
    return out


# ---------------- 评估一个 episode 的整段 ----------------
@torch.no_grad()
def eval_full_episode(
    model: nn.Module,
    ds: str,
    video_root: Path,
    text_root: Path,
    audio_root: Path,
    fmri_root: Path,
    fractions: List[float],
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (pred_full[T,1000], gt_full[T,1000])，都在 CPU numpy float32。
    """
    dataset = WindowedDataset(
        ids=[ds],
        video_root=video_root,
        text_root=text_root,
        audio_root=audio_root,
        fmri_root=fmri_root,
        fractions=fractions,
        layer_agg=layer_agg,
        window_tr=window_tr,
        stride_tr=stride_tr,
        frames_per_tr=frames_per_tr,
    )
    pin_mem_local = (device.type == "cuda")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=pin_mem_local)

    T_ds = dataset._episode_len_tr[ds]
    n_outputs = 1000
    acc = np.zeros((T_ds, n_outputs), dtype=np.float32)
    cnt = np.zeros((T_ds, ), dtype=np.int32)

    # 读整段 GT（与 ds 完全同名）
    fmri_path = Path(fmri_root) / f"{ds}.npy"
    gt_full = np.load(fmri_path)
    if 1000 in gt_full.shape:
        gt_full = gt_full if gt_full.shape[0] == 1000 else gt_full.T  # [1000,T]
    else:
        gt_full = gt_full.T if gt_full.shape[0] > gt_full.shape[1] else gt_full
    gt_full = gt_full[:, :T_ds].T.astype(np.float32)  # [T,1000]

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            y_pred = model(batch)   # [1,1000,N]
        start_tr = int(batch.data["start_tr_list"][0])
        N = y_pred.shape[-1]
        yp = y_pred[0].permute(1,0).detach().cpu().numpy()  # [N,1000]
        end_tr = min(start_tr + N, T_ds)
        acc[start_tr:end_tr] += yp[:end_tr-start_tr, :]
        cnt[start_tr:end_tr] += 1

    cnt = np.maximum(cnt[:,None], 1)
    pred_full = acc / cnt  # [T,1000]
    return pred_full.astype(np.float32), gt_full.astype(np.float32)


# ---------------- 运行可视化脚本 ----------------
def _run_vis_cli(
    script: str,
    gt: str,
    pred: str,
    atlas: str,
    outdir: str,
    subject: str,
    modality: str,
    align: str,
    delay: int,
):
    if (not script) or (not Path(script).exists()):
        print(f"[VIS][WARN] Visualization script not found: {script}")
        return
    cmd = [
        sys.executable, str(script),
        "--gt", str(gt),
        "--pred", str(pred),
        "--atlas", str(atlas),
        "--outdir", str(outdir),
        "--subject", str(subject),
        "--modality", str(modality),
        "--align", str(align),
        "--delay", str(int(delay)),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[VIS][WARN] Visualization script failed: {e}")


# ---------------- 解析 train-probe 片段码 -> 具体 train_list id ----------------
def resolve_train_probe_ds(train_ids: List[str], epcode: str) -> str:
    """
    epcode: 如 's01e02a'（不区分大小写）
    在 train_ids 中查找包含 task-<epcode> 的完整 ds，优先返回首个匹配。
    若找不到，回退到 train_ids[0] 并打印告警。
    """
    epi = epcode.lower()
    for ds in train_ids:
        m = re.search(r"task-(s\d{2}e\d{2}[a-d])$", ds, flags=re.IGNORECASE)
        if m and m.group(1).lower() == epi:
            return ds
    print(f"[WARN] train_probe_ep='{epcode}' 未在 train_list 中精确匹配，回退到第一条: {train_ids[0]}")
    return train_ids[0]


def split_ids(all_ids: List[str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    ids = list(all_ids)
    rng.shuffle(ids)
    n_train = max(1, int(round(len(ids) * train_ratio)))
    train_ids = ids[:n_train]
    val_ids = ids[n_train:] if n_train < len(ids) else ids[-max(1, len(ids)//10):]
    return train_ids, val_ids


# ---------------- 训练主函数 ----------------
def main():
    ap = argparse.ArgumentParser()
    # 数据与缓存
    ap.add_argument("--train_list", type=str, required=False, default="")
    ap.add_argument("--val_list",   type=str, required=False, default="")
    ap.add_argument("--all_list",   type=str, required=False, default="",
                    help="若提供，则从该全集列表中按 90/10 自动划分 train/val，忽略 --train_list/--val_list")
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--fmri_root",  type=str, required=True)

    # 层聚合
    ap.add_argument("--layers", type=str, default="0.5,0.75,1.0")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean","none","None"])

    # 窗口参数（与论文一致）
    ap.add_argument("--window_tr", type=int, default=100, help="N，窗口 TR 数")
    ap.add_argument("--stride_tr", type=int, default=50, help="滑动步幅（TR）")
    ap.add_argument("--frames_per_tr", type=int, default=3, help="特征帧/每TR（2Hz 对应 3）")

    # 优化 & 训练（按论文）
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1, help="OneCycleLR warmup 比例")
    ap.add_argument("--modality_dropout", type=float, default=0.2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # TensorBoard 日志与训练集整段评估
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "standalone_windows"),
                    help="TensorBoard base 目录，会自动加时间戳子目录")

    # 训练集子集评估
    ap.add_argument("--train_eval_subset_ratio", type=float, default=0.9,
                    help="每个 epoch 结束后，用训练集的该比例子集做评估（默认 0.9）")

    # ★ Train-probe
    ap.add_argument("--train_probe_ep", type=str, default="s01e02a",
                    help="例如 s01e02a；将从 train_list 中匹配 task-s01e02a 的 dataset")
    ap.add_argument("--train_probe_ds", type=str, default="",
                    help="（可选）直接给完整 id，如 ses-001_task-s01e02a；若提供则优先生效")

    # 可视化 & 额外
    ap.add_argument("--vis_script", type=str, default=str(PROJ / "vis" / "plot_pred_on_brain.py"),
                    help="可执行的可视化脚本路径（可为空，不存在则跳过）")
    ap.add_argument("--atlas_path", type=str,
                    default=str(PROJ / "download" / "algonauts_2025.competitors" / "fmri" / "sub-01" / "atlas" /
                                "sub-01_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz"))
    ap.add_argument("--subject_code", type=str, default="01")
    ap.add_argument("--vis_align", type=str, default="truncate", choices=["truncate","resample"])
    ap.add_argument("--vis_delay", type=int, default=0)
    ap.add_argument("--print_vis_shapes", action="store_true")

    # 噪声上限（可选）
    ap.add_argument("--noise_ceiling_path", type=str, default="",
                    help="可选 .npy 路径，形状 [1000] 或 [T,1000]；用于计算 normalized correlation")

    # 设备选择
    ap.add_argument("--cuda", type=int, default=1,
                    help="CUDA 设备索引，-1 表示使用 CPU。默认 1（第二块显卡）。")

    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "standalone_windows"))

    args = ap.parse_args()
    set_seed(args.seed)

    # 设备选择逻辑
    if args.cuda is not None and args.cuda >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.cuda)
        device = torch.device(f"cuda:{args.cuda}")
        print(f"[DEV] Using CUDA device: {device}")
        pin_mem = True
    else:
        device = torch.device("cpu")
        print("[DEV] Using CPU")
        pin_mem = False

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_windows").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val_episodes").mkdir(parents=True, exist_ok=True)
    # 可视化需要的目录
    (out_dir / "preds_val_episodes_gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_train_episode").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_train_episode_gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis_val_best").mkdir(parents=True, exist_ok=True)
    (out_dir / "vis_trainprobe_best").mkdir(parents=True, exist_ok=True)

    # TensorBoard：当前 run 建独立时间戳子目录
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_dir = Path(args.log_dir) / ts
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TB] Logging to: {tb_dir}")

    fractions = [float(x) for x in args.layers.split(",") if x.strip()]
    agg = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    # ---- 划分数据 ----
    if args.all_list.strip():
        all_ids = read_ids(args.all_list.strip())
        train_ids, val_ids = split_ids(all_ids, 0.9, args.seed)
        print(f"[SPLIT] Using --all_list, split to train={len(train_ids)}  val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise ValueError("请提供 --all_list 或同时提供 --train_list 与 --val_list")
        train_ids = read_ids(args.train_list)
        val_ids   = read_ids(args.val_list)
        print(f"[SPLIT] Using provided train/val lists: train={len(train_ids)}  val={len(val_ids)}")

    # 解析 train-probe 选择
    if args.train_probe_ds.strip():
        train_probe_ds = args.train_probe_ds.strip()
        if train_probe_ds not in set(train_ids):
            print(f"[WARN] --train_probe_ds='{train_probe_ds}' 不在 train_list 中（若特征存在仍可评估）。")
    else:
        train_probe_ds = resolve_train_probe_ds(train_ids, args.train_probe_ep)
    print(f"[INFO] Train-probe episode = {train_probe_ds}  (from ep='{args.train_probe_ep}')")

    # ---- 构建数据集/加载器 ----
    def build_loader(ids: List[str], shuffle: bool) -> DataLoader:
        ds = WindowedDataset(
            ids=ids,
            video_root=Path(args.video_root),
            text_root =Path(args.text_root),
            audio_root=Path(args.audio_root),
            fmri_root =Path(args.fmri_root),
            fractions=fractions,
            layer_agg=agg,
            window_tr=args.window_tr,
            stride_tr=args.stride_tr,
            frames_per_tr=args.frames_per_tr,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=pin_mem, drop_last=False
        )
        return ds, loader

    train_set, train_loader = build_loader(train_ids, shuffle=True)
    val_set,   val_loader   = build_loader(val_ids, shuffle=False)

    # —— 构建模型 —— #
    G, Dv, Dt, Da = train_set.G, train_set.Dv, train_set.Dt, train_set.Da
    feat_dims = {"video": (G, Dv), "text": (G, Dt), "audio": (G, Da)}
    n_outputs = 1000  # 固定

    window_frames = args.window_tr * args.frames_per_tr

    cfg = FmriEncoderConfig(
        n_subjects=1,
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

    # 时间位置编码长度对齐（若存在）
    with torch.no_grad():
        if hasattr(model, "time_pos_embed"):
            T_old = model.time_pos_embed.shape[1]
            if T_old != window_frames:
                pos = model.time_pos_embed  # [1,T_old,H]
                pos = torch.nn.functional.interpolate(
                    pos.transpose(1,2), size=window_frames, mode="linear", align_corners=False
                ).transpose(1,2)
                model.time_pos_embed = nn.Parameter(pos)

    # —— 损失与优化器 —— #
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = len(train_loader) if len(train_loader) > 0 else 1
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=args.warmup_pct, anneal_strategy="cos"
    )

    # —— SWA —— #
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = swa_start_epoch < args.epochs
    swa_model = AveragedModel(model) if use_swa else None

    # —— Best by Pearson on VAL —— #
    best_val = float("-inf")
    best_trainprobe = float("-inf")
    global_step = 0
    warned_nc_once = False

    # 预加载噪声上限（若提供）
    noise_ceiling_vec_val: Optional[np.ndarray] = None
    if args.noise_ceiling_path:
        noise_ceiling_vec_val = load_noise_ceiling(args.noise_ceiling_path, n_outputs)

    for epoch in range(1, args.epochs + 1):
        # ---------------- Train ----------------
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            y_pred = model(batch)            # [B,1000,N]
            y_true = batch.data["fmri"]      # [B,1000,N]
            loss = criterion(y_pred, y_true)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * y_pred.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            writer.add_scalar("loss/train_step", float(loss.item()), global_step)
            global_step += 1

            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)

        train_loss /= max(1, len(train_set))
        writer.add_scalar("loss/train_epoch", float(train_loss), epoch)

        # ---------------- Val ----------------
        model.eval()
        val_loss = 0.0
        preds_all = []
        trues_all = []

        # 重建整段：ds -> (accum[T_ds,1000], count[T_ds])
        recon_pred: Dict[str, np.ndarray] = {}
        recon_cnt : Dict[str, np.ndarray] = {}
        T_ds_map  : Dict[str, int]        = val_set._episode_len_tr.copy()
        best_ds = None
        best_len = -1

        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val  ", leave=False)
            for batch in pbar_v:
                batch = batch.to(device)
                y_pred = model(batch)            # [B,1000,N]
                y_true = batch.data["fmri"]      # [B,1000,N]
                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * y_pred.size(0)

                yp = y_pred.permute(0,2,1).reshape(-1, n_outputs).detach().cpu().numpy()  # [B*N,1000]
                yt = y_true.permute(0,2,1).reshape(-1, n_outputs).detach().cpu().numpy()  # [B*N,1000]
                preds_all.append(yp)
                trues_all.append(yt)

                # 窗口落盘 + 重建
                ds_list = batch.data["ds_list"]
                start_list = batch.data["start_tr_list"]
                B = y_pred.shape[0]
                for i in range(B):
                    ds = ds_list[i]
                    st = int(start_list[i])

                    npy_win = y_pred[i].permute(1,0).cpu().numpy()  # [N,1000]
                    np.save(out_dir / "preds_val_windows" / f"{ds}_start{st:05d}_pred.npy", npy_win)

                    T_ds = T_ds_map[ds]
                    if ds not in recon_pred:
                        recon_pred[ds] = np.zeros((T_ds, n_outputs), dtype=np.float32)
                        recon_cnt [ds] = np.zeros((T_ds, ), dtype=np.int32)
                    end_tr = min(st + args.window_tr, T_ds)
                    recon_pred[ds][st:end_tr] += npy_win[:end_tr-st, :]
                    recon_cnt [ds][st:end_tr] += 1

        val_loss /= max(1, len(val_set))
        preds_np = np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0, n_outputs), dtype=np.float32)
        trues_np = np.concatenate(trues_all, axis=0) if trues_all else np.zeros((0, n_outputs), dtype=np.float32)

        # 计算多指标（VAL）
        if noise_ceiling_vec_val is None and not warned_nc_once:
            print("[NC] 未提供 --noise_ceiling_path，Noise ceiling 与 Normalized correlation 将显示为 nan。")
            warned_nc_once = True
        metrics_val = compute_metrics(preds_np, trues_np, noise_ceiling_vec_val)
        val_pearson_mean = metrics_val["pearson"]

        # 保存重建后的整段预测 + GT，并记录最长 episode 作为默认可视化对象
        for ds, acc in recon_pred.items():
            cnt = np.maximum(recon_cnt[ds][:,None], 1)
            merged = acc / cnt  # [T,1000]
            np.save(out_dir / "preds_val_episodes" / f"{ds}_pred.npy", merged.astype(np.float32))

            gt_all = np.load(Path(args.fmri_root) / f"{ds}.npy")
            if 1000 in gt_all.shape:
                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
            else:
                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
            T = merged.shape[0]
            np.save(out_dir / "preds_val_episodes_gt" / f"{ds}_gt.npy", gt_all[:, :T].T.astype(np.float32))

            if T > best_len:
                best_len = T
                best_ds = ds

        # ---------------- 训练集子集评估（每个 epoch） ----------------
        # 采样 train_ids 的一个子集
        subset_ratio = max(0.0, min(1.0, args.train_eval_subset_ratio))
        k = max(1, int(round(len(train_ids) * subset_ratio)))
        rng = random.Random(args.seed + epoch)  # 每轮变化
        train_subset_ids = rng.sample(train_ids, k) if k < len(train_ids) else list(train_ids)
        train_subset_set, train_subset_loader = (None, None)
        # 构建一次轻量 loader
        train_subset_set, train_subset_loader = (None, None)
        train_subset_set, train_subset_loader = build_loader(train_subset_ids, shuffle=False)

        preds_trall, trues_trall = [], []
        with torch.no_grad():
            for batch in tqdm(train_subset_loader, desc=f"[Epoch {epoch}] Train-eval ({k}/{len(train_ids)})", leave=False):
                batch = batch.to(device)
                y_pred = model(batch)
                y_true = batch.data["fmri"]
                yp = y_pred.permute(0,2,1).reshape(-1, n_outputs).detach().cpu().numpy()
                yt = y_true.permute(0,2,1).reshape(-1, n_outputs).detach().cpu().numpy()
                preds_trall.append(yp)
                trues_trall.append(yt)
        preds_tr = np.concatenate(preds_trall, axis=0) if preds_trall else np.zeros((0, n_outputs), dtype=np.float32)
        trues_tr = np.concatenate(trues_trall, axis=0) if trues_trall else np.zeros((0, n_outputs), dtype=np.float32)
        metrics_train_eval = compute_metrics(preds_tr, trues_tr, noise_ceiling_vec_val)  # 同一 NC 向量

        # ---------------- Train-Probe 整段评估（一次） ----------------
        try:
            pred_full_tr, gt_full_tr = eval_full_episode(
                model=model,
                ds=train_probe_ds,
                video_root=Path(args.video_root),
                text_root =Path(args.text_root),
                audio_root=Path(args.audio_root),
                fmri_root =Path(args.fmri_root),
                fractions=fractions,
                layer_agg=agg,
                window_tr=args.window_tr,
                stride_tr=args.stride_tr,
                frames_per_tr=args.frames_per_tr,
                device=device,
            )
            tp_r = voxelwise_pearson(pred_full_tr, gt_full_tr)   # [1000]
            trainprobe_pearson_mean = float(np.nanmean(tp_r))
        except Exception as e:
            print(f"[WARN] Train-probe eval failed on '{train_probe_ds}': {e}")
            trainprobe_pearson_mean = float("nan")
            pred_full_tr = gt_full_tr = None  # 可视化时判空

        # ---------------- Logging & 打印 ----------------
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)

        # 记录 VAL 指标
        writer.add_scalar("metric/val_pearson_mean", float(metrics_val["pearson"]), epoch)
        writer.add_scalar("metric/val_spearman_mean", float(metrics_val["spearman"]), epoch)
        writer.add_scalar("metric/val_r2_mean", float(metrics_val["r2"]), epoch)
        if not math.isnan(metrics_val["norm_corr"]):
            writer.add_scalar("metric/val_normcorr_mean", float(metrics_val["norm_corr"]), epoch)

        # 记录 Train-eval 指标
        writer.add_scalar("metric/traineval_pearson_mean", float(metrics_train_eval["pearson"]), epoch)
        writer.add_scalar("metric/traineval_spearman_mean", float(metrics_train_eval["spearman"]), epoch)
        writer.add_scalar("metric/traineval_r2_mean", float(metrics_train_eval["r2"]), epoch)
        if not math.isnan(metrics_train_eval["norm_corr"]):
            writer.add_scalar("metric/traineval_normcorr_mean", float(metrics_train_eval["norm_corr"]), epoch)

        # Train-probe
        if not np.isnan(trainprobe_pearson_mean):
            writer.add_scalar("metric/trainprobe_pearson_mean", float(trainprobe_pearson_mean), epoch)
        writer.flush()

        # 控制台打印
        print(
            "Epoch {ep}: "
            "train_loss={tr:.6f} | "
            "VAL: pearson={vp:.6f}, spearman={vs:.6f}, R2={vr:.6f}, norm_corr={vnc} | "
            "TrainEval: pearson={tp:.6f}, spearman={ts:.6f}, R2={tr2:.6f}, norm_corr={tnc} | "
            "trainprobe_pearson={tpp:.6f}".format(
                ep=epoch,
                tr=train_loss,
                vp=metrics_val["pearson"], vs=metrics_val["spearman"], vr=metrics_val["r2"],
                vnc="nan" if math.isnan(metrics_val["norm_corr"]) else f"{metrics_val['norm_corr']:.6f}",
                tp=metrics_train_eval["pearson"], ts=metrics_train_eval["spearman"], tr2=metrics_train_eval["r2"],
                tnc="nan" if math.isnan(metrics_train_eval["norm_corr"]) else f"{metrics_train_eval['norm_corr']:.6f}",
                tpp=trainprobe_pearson_mean
            )
        )

        # ---------------- 按 VAL Pearson 更新最佳 ----------------
        if not np.isnan(val_pearson_mean) and val_pearson_mean > best_val:
            best_val = val_pearson_mean
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            torch.save(model, out_dir / "checkpoints" / "best_full.pt")

            if best_ds is not None:
                ep_gt_path   = out_dir / "preds_val_episodes_gt" / f"{best_ds}_gt.npy"
                ep_pred_path = out_dir / "preds_val_episodes"    / f"{best_ds}_pred.npy"

                if args.print_vis_shapes:
                    try:
                        _gt  = np.load(ep_gt_path)
                        _pr  = np.load(ep_pred_path)
                        print(f"[VIS] VAL FULL ds={best_ds} GT={_gt.shape}  PRED={_pr.shape}")
                    except Exception as e:
                        print(f"[VIS][WARN] 读取 VAL FULL 可视化数组失败: {e}")

                _run_vis_cli(
                    script=args.vis_script,
                    gt=str(ep_gt_path),
                    pred=str(ep_pred_path),
                    atlas=args.atlas_path,
                    outdir=str(out_dir / "vis_val_best"),
                    subject=args.subject_code,
                    modality=f"val_best_{best_ds}",
                    align=args.vis_align,
                    delay=args.vis_delay,
                )

        # ---------------- TRAIN-PROBE 可视化（提升时） ----------------
        if (not np.isnan(trainprobe_pearson_mean)) and (trainprobe_pearson_mean > best_trainprobe) and (pred_full_tr is not None):
            best_trainprobe = trainprobe_pearson_mean

            # 落盘 train-probe 的整段 pred/gt（可视化用）
            np.save(out_dir / "preds_train_episode"    / f"{train_probe_ds}_pred.npy", pred_full_tr.astype(np.float32))
            np.save(out_dir / "preds_train_episode_gt" / f"{train_probe_ds}_gt.npy",   gt_full_tr.astype(np.float32))

            tp_gt_path   = out_dir / "preds_train_episode_gt" / f"{train_probe_ds}_gt.npy"
            tp_pred_path = out_dir / "preds_train_episode"    / f"{train_probe_ds}_pred.npy"

            if args.print_vis_shapes:
                try:
                    _gtp = np.load(tp_gt_path); _prp = np.load(tp_pred_path)
                    print(f"[VIS] TRAIN-PROBE FULL ds={train_probe_ds} GT={_gtp.shape}  PRED={_prp.shape}")
                except Exception as e:
                    print(f"[VIS][WARN] 读取 TRAIN-PROBE FULL 可视化数组失败: {e}")

            _run_vis_cli(
                script=args.vis_script,
                gt=str(tp_gt_path),
                pred=str(tp_pred_path),
                atlas=args.atlas_path,
                outdir=str(out_dir / "vis_trainprobe_best"),
                subject=args.subject_code,
                modality=f"trainprobe_{train_probe_ds}",
                align=args.vis_align,
                delay=args.vis_delay,
            )

    # 训练结束：若用了 SWA，可以把 BN 更新并另存一个 SWA 权重
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best val pearson (mean over voxels): {best_val:.6f}")
    print(f"Best train-probe pearson (mean over voxels): {best_trainprobe:.6f}")
    print(f"Checkpoints dir: {out_dir / 'checkpoints'}")
    print(f"TensorBoard log dir: {tb_dir}")
    print(f"Window preds dir: {out_dir / 'preds_val_windows'}  (shape per file: [N,1000], N={args.window_tr})")
    print(f"Val episodes (pred/gt) dir: {out_dir / 'preds_val_episodes'} / {out_dir / 'preds_val_episodes_gt'}")
    print(f"Train-probe (pred/gt) dir: {out_dir / 'preds_train_episode'} / {out_dir / 'preds_train_episode_gt'}")
    print(f"VIS dirs: {out_dir / 'vis_val_best'} , {out_dir / 'vis_trainprobe_best'}")


if __name__ == "__main__":
    """
    使用示例：

    # 1) 从全集自动切分 90/10，默认用 GPU:1
    python -m algonauts2025.standalone.train_standalone_windows \
      --all_list /path/to/all_ids.txt \
      --video_root ... --text_root ... --audio_root ... --fmri_root ...

    # 2) 指定训练/验证列表（不自动切分）
    python -m algonauts2025.standalone.train_standalone_windows \
      --train_list ... --val_list ... \
      --video_root ... --text_root ... --audio_root ... --fmri_root ...

    # 3) 指定噪声上限（可选）用于 normalized correlation
    ... --noise_ceiling_path /path/to/noise_ceiling.npy

    # 4) 设备选择
    ... --cuda 1        # 第二块 GPU
    ... --cuda 0        # 第一块 GPU
    ... --cuda -1       # 强制 CPU
    """
    main()