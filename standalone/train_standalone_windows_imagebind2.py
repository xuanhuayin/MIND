# -*- coding: utf-8 -*-
"""
Standalone windowed training using ImageBind TRIBE-like features @ 2Hz.

要点：
- 训练时同时监督 4 个 subject 预测头；缺失数据的 subject/窗口自动跳过。
- 验证/Train-probe 在“整段 episode”上评测：Pearson / Spearman / R² / ISG
- 选择最佳模型：验证集 Pearson（可用 subject 的均值）最大
- 兼容 fMRI 文件名前缀 ses-xxx 在不同 subject 不一致：用 task-xxxx 匹配
"""

from __future__ import annotations
import argparse, os, sys, random, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---- repo root ----
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
PKG_PARENT = PROJ.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# ---- model (你提供的新实现) ----
from algonauts2025.standalone.fmri_model_min1 import FmriEncoder, FmriEncoderConfig


# ---------------- utils ----------------
def set_seed(seed: int = 33):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]


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


# ---------------- tiny batch container ----------------
class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v):
                self.data[k] = v.to(device, non_blocking=True)
        return self


# ---------------- dataset (windowed) ----------------
class WindowedDataset(Dataset):
    """
    每个样本 = episode 的一个窗口（N TR）：
      输入 video/text/audio: [G, D, N * frames_per_tr]
      目标 fmri: [1000, N]  （此处用于窗级 loss 的 anchor，来自 sub1/root）
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
        layers_arg: str = "",
        subject_id: int = 0,  # 用于 batch 携带
    ):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_root = Path(fmri_root)  # 仅用于 anchor 尺度
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.f = int(frames_per_tr)
        self.subject_id_fixed = int(subject_id)

        # probe for L
        v0 = np.load(self.video_root / f"{ids[0]}.npy")
        probe_L = v0.shape[1]
        self.layer_mode, payload = parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions":
            self.fracs, self.sel_indices = [float(x) for x in payload], None
        else:
            self.fracs, self.sel_indices = None, [int(i) for i in payload]
        self.layer_agg = layer_agg.lower()

        # build windows
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            v = np.load(self.video_root / f"{ds}.npy")
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f

            arr = np.load(self.fmri_root / f"{ds}.npy") if (self.fmri_root / f"{ds}.npy").exists() else None
            if arr is None:
                # 兼容 anchor 缺 exact：用 task-xxx 去估长
                arr = _load_fmri_flexible(self.fmri_root, ds)
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            O, T_tr_fmri = fmri.shape
            assert O == 1000, f"Expect O=1000, got {O}"
            T_tr = min(T_tr_feat, T_tr_fmri)
            self._episode_len_tr[ds] = T_tr
            for st in range(0, max(1, T_tr - self.N + 1), self.S):
                if st + self.N <= T_tr:
                    self._index.append((ds, st))

        first_ds, _ = self._index[0]
        v_LDT = self._load_feature_LDT(self.video_root / f"{first_ds}.npy")
        v_GDT = self._maybe_pick_layers(v_LDT)
        self.G, self.Dv = v_GDT.shape[0], v_GDT.shape[1]
        t_LDT = self._load_feature_LDT(self.text_root / f"{first_ds}.npy")
        a_LDT = self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")
        self.Dt = self._maybe_pick_layers(t_LDT).shape[1]
        self.Da = self._maybe_pick_layers(a_LDT).shape[1]

    def __len__(self): return len(self._index)

    @staticmethod
    def _load_feature_LDT(path_npy: Path) -> np.ndarray:
        arr = np.load(path_npy)
        if arr.ndim != 3:
            raise ValueError(f"Expect [T,L,D], got {arr.shape}: {path_npy}")
        return np.transpose(arr, (1, 2, 0))

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
            lat_LDT = self._load_feature_LDT(root / f"{ds}.npy")
            lat_GDT = self._maybe_pick_layers(lat_LDT)
            if e_frame > lat_GDT.shape[-1]:
                e_frame = lat_GDT.shape[-1]
                s_frame = e_frame - win_frames
            lat = lat_GDT[..., s_frame:e_frame]
            feats[name] = torch.from_numpy(lat.astype(np.float32))

        # anchor GT （sub1）只用于形状参考/可视化，训练真正的损失会按需读取各 subject 的 GT
        fmri_path = self.fmri_root / f"{ds}.npy"
        if fmri_path.exists():
            arr = np.load(fmri_path)
        else:
            arr = _load_fmri_flexible(self.fmri_root, ds)
        if 1000 in arr.shape:
            fmri = arr if arr.shape[0] == 1000 else arr.T
        else:
            fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
        Y = fmri[:, start_tr:start_tr + self.N]

        return {
            "video": feats["video"],
            "text": feats["text"],
            "audio": feats["audio"],
            "fmri": torch.from_numpy(Y.astype(np.float32)),
            "subject_id": torch.tensor(self.subject_id_fixed, dtype=torch.long),
            "ds": ds,
            "start_tr": start_tr,
        }


# ---------------- collate ----------------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    keys = ["video","text","audio","fmri","subject_id"]
    data: Dict[str, torch.Tensor] = {}
    for k in keys:
        if k == "subject_id":
            data[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            data[k] = torch.stack([b[k] for b in batch], dim=0)
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)


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
    """
    在 root 下解析 ds 的 GT 路径：
      1) exact: root/ds.npy
      2) 按 task-xxx 搜索：*_{task-xxx}.npy 或 *task-xxx.npy
      3) 退化：用 '_' 之后的后缀
    """
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
    p = resolve_fmri_file(root, ds)
    return np.load(p)


# ---------------- hidden & multi-head preds ----------------
def compute_hidden_tr(model: FmriEncoder, batch: Batch) -> torch.Tensor:
    x = model.aggregate_features(batch)                         # [B,T2,H]
    sid = batch.data.get("subject_id", None)
    x = model.transformer_forward(x, subject_id=sid)            # [B,T2,H]
    x = x.transpose(1, 2)                                       # [B,H,T2]
    return model.pooler(x).transpose(1, 2)                      # [B,N,H]

def predict_all_heads_from_hidden(model: FmriEncoder, x_tr: torch.Tensor) -> torch.Tensor:
    """
    输入:
      x_tr: [B, N, H]  —— 共享编码后的每 TR 隐表示
    输出:
      y_all: [B, S, N, O] —— 同时计算 S=4 个 subject 头的预测
    """
    ph = model.pred_head
    if not (hasattr(ph, "weight") and hasattr(ph, "bias")):
        # 单头退化（不会出现在你现在的配置中）
        y = ph(x_tr, None)                                     # [B,N,O]
        return y.unsqueeze(1)                                   # [B,1,N,O]
    W, BIAS = ph.weight, ph.bias                                # [S,O,H], [S,O] (nn.Parameter, require_grad=True)
    return torch.einsum("bnh,soh->bsno", x_tr, W) + BIAS[None, :, None, :]


# ---------------- episode pickers ----------------
def pick_friends_episode(cands: List[str]) -> str:
    fs = [ds for ds in cands if "friends" in ds.lower()]
    return fs[0] if fs else cands[0]


# ---------------- reconstruct one episode (multi-subject with skipping) ----------------
@torch.no_grad()
def reconstruct_one_episode_multi_subject(
    model: FmriEncoder,
    ds: str,
    video_root: Path,
    text_root: Path,
    audio_root: Path,
    fmri_roots_by_subject: Dict[int, Path],  # {0: sub1_root, 1: sub2_root, 2: sub3_root, 3: sub5_root}
    layers_arg: str,
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], int, List[int]]:
    """
    返回：
      preds_by_sub[s] : [T,1000]（仅对 available_subjects 返回）
      gts_by_sub[s]   : [T,1000]
      T_ds            : anchor 的长度
      available_subjects: 实际找到 GT 的 subject 列表
    """
    anchor_root = list(fmri_roots_by_subject.values())[0]
    dataset = WindowedDataset(
        ids=[ds],
        video_root=video_root,
        text_root=text_root,
        audio_root=audio_root,
        fmri_root=anchor_root,          # 仅用于确定窗口与时长
        fractions=[1.0],
        layer_agg=layer_agg,
        window_tr=window_tr,
        stride_tr=stride_tr,
        frames_per_tr=frames_per_tr,
        layers_arg=layers_arg,
        subject_id=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=True)

    T_ds = dataset._episode_len_tr[ds]
    O = 1000
    S = getattr(model.config, "n_subjects", 1) or 1

    # 先计算所有头的窗级预测
    model.eval()
    acc = {s: np.zeros((T_ds, O), dtype=np.float32) for s in range(S)}
    cnt = np.zeros((T_ds,), dtype=np.int32)
    for batch in loader:
        batch = batch.to(device)
        x_tr = compute_hidden_tr(model, batch)             # [1,N,H]
        y_all = predict_all_heads_from_hidden(model, x_tr) # [1,S,N,O]
        st = int(batch.data["start_tr_list"][0])
        N = y_all.shape[2]; ed = min(st + N, T_ds)
        for s in range(S):
            yp = y_all[0, s, :ed - st, :].detach().cpu().numpy()
            acc[s][st:ed] += yp
        cnt[st:ed] += 1
    cnt = np.maximum(cnt[:, None], 1)
    full_preds = {s: (acc[s] / cnt).astype(np.float32) for s in acc.keys()}

    # 读取各 subject 的 GT（允许缺失）
    preds_by_sub, gts_by_sub, available_subjects = {}, {}, []
    for s, root in fmri_roots_by_subject.items():
        try:
            gt_all = _load_fmri_flexible(root, ds)
            if 1000 in gt_all.shape:
                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
            else:
                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
            gts_by_sub[s] = gt_all[:, :T_ds].T.astype(np.float32)
            preds_by_sub[s] = full_preds[s]
            available_subjects.append(s)
        except FileNotFoundError:
            continue

    return preds_by_sub, gts_by_sub, T_ds, available_subjects


# ---------------- evaluate a list of episodes ----------------
@torch.no_grad()
def evaluate_episode_list(
    model: FmriEncoder,
    episodes: List[str],
    roots_feat: Dict[str, Path],
    fmri_roots_by_subject: Dict[int, Path],
    layers_arg: str,
    layer_agg: str,
    window_tr: int,
    stride_tr: int,
    frames_per_tr: int,
    device: torch.device,
    save_root: Path | None = None,   # 若提供则保存各 subject 的 pred/gt（仅对可用 subject）
    save_split_name: str = "val",
):
    """
    返回：
      per_sub_means: {s: {'r':mean,'rho':mean,'r2':mean}}
      isg_means:     {s: mean}
      used_counts:   {s: 实际参与评测的 episode 数}
    """
    agg = {s: {"r": [], "rho": [], "r2": []} for s in range(4)}
    agg_isg = {s: [] for s in range(4)}
    used_counts = {s: 0 for s in range(4)}

    for ds in episodes:
        preds_by_sub, gts_by_sub, _, available_subjects = reconstruct_one_episode_multi_subject(
            model=model,
            ds=ds,
            video_root=roots_feat["video"],
            text_root=roots_feat["text"],
            audio_root=roots_feat["audio"],
            fmri_roots_by_subject=fmri_roots_by_subject,
            layers_arg=layers_arg,
            layer_agg=layer_agg,
            window_tr=window_tr,
            stride_tr=stride_tr,
            frames_per_tr=frames_per_tr,
            device=device,
        )

        if not available_subjects:
            continue

        # per-subject metrics
        for s in available_subjects:
            pred, gt = preds_by_sub[s], gts_by_sub[s]    # [T,1000]
            r   = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2  = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[s]["r"].append(r); agg[s]["rho"].append(rho); agg[s]["r2"].append(r2)
            used_counts[s] += 1

        # ISG：对每个 available_subject，用其他 available_subject 的预测与其 GT 的 Pearson 平均
        for s in available_subjects:
            gt = gts_by_sub[s]
            r_list = []
            for t in available_subjects:
                if t == s: continue
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t], gt))))
            if r_list:
                agg_isg[s].append(float(np.mean(r_list)))

        # 保存
        if save_root is not None:
            subname = {0: "sub01", 1: "sub02", 2: "sub03", 3: "sub05"}
            for s in available_subjects:
                subdir = save_root / subname[s] / f"preds_{save_split_name}_episodes"
                subdir_gt = save_root / subname[s] / f"preds_{save_split_name}_episodes_gt"
                subdir.mkdir(parents=True, exist_ok=True)
                subdir_gt.mkdir(parents=True, exist_ok=True)
                np.save(subdir    / f"{ds}_pred.npy", preds_by_sub[s])
                np.save(subdir_gt / f"{ds}_gt.npy",   gts_by_sub[s])

    # means
    per_sub_means = {}
    isg_means = {}
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

    return per_sub_means, isg_means, used_counts


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # splits
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="")
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--split_seed",  type=int,   default=33)

    # feature roots
    ap.add_argument("--video_root", type=str,
                    default=str(PROJ / "pipeline_IMAGEBIND" / "features" / "video_2hz" / "sub-01"))
    ap.add_argument("--text_root",  type=str,
                    default=str(PROJ / "pipeline_IMAGEBIND" / "features" / "text_2hz" / "sub-01"))
    ap.add_argument("--audio_root", type=str,
                    default=str(PROJ / "pipeline_IMAGEBIND" / "features" / "audio_2hz" / "sub-01"))

    # 多 subject fMRI 根目录
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # layer selection / aggregation
    ap.add_argument("--layers", type=str, default="last41")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean","none","None"])

    # windows
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)
    ap.add_argument("--frames_per_tr", type=int, default=3)  # 2Hz → 3 frames per TR

    # optimization
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--modality_dropout", type=float, default=0.2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # misc
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "standalone_windows_imagebind"))
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "standalone_imagebind"))
    ap.add_argument("--print_vis_shapes", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出 / 日志
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
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

    # layer agg flag
    agg = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    # 数据集拆分
    if args.all_list:
        all_ids = read_ids(args.all_list)
        rnd = random.Random(args.split_seed); rnd.shuffle(all_ids)
        k = int(round(len(all_ids) * args.split_ratio))
        k = max(1, min(len(all_ids)-1, k))
        train_ids, val_ids = all_ids[:k], all_ids[k:]
        print(f"[SPLIT] from --all_list: train={len(train_ids)}, val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise SystemExit("Provide both --train_list and --val_list, or use --all_list.")
        train_ids, val_ids = read_ids(args.train_list), read_ids(args.val_list)

    # 训练集 Friends 片段（若无则回退）
    train_probe_ds = pick_friends_episode(train_ids)
    print(f"[TRAIN-PROBE] Friends episode: {train_probe_ds}")

    # —— window loaders（训练/窗级 val loss 用）—— #
    train_set = WindowedDataset(
        ids=train_ids,
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        fmri_root =fmri_roots[0],          # anchor for windows
        fractions=[1.0],
        layer_agg=agg,
        window_tr=args.window_tr,
        stride_tr=args.stride_tr,
        frames_per_tr=args.frames_per_tr,
        layers_arg=args.layers,
        subject_id=0,
    )
    val_set_for_loss = WindowedDataset(
        ids=val_ids if len(val_ids) > 0 else train_ids[:1],
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        fmri_root =fmri_roots[0],
        fractions=[1.0],
        layer_agg=agg,
        window_tr=args.window_tr,
        stride_tr=args.stride_tr,
        frames_per_tr=args.frames_per_tr,
        layers_arg=args.layers,
        subject_id=0,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader_for_loss = DataLoader(val_set_for_loss, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # —— build model —— #
    G, Dv, Dt, Da = train_set.G, train_set.Dv, train_set.Dt, train_set.Da
    feat_dims = {"video": (G, Dv), "text": (G, Dt), "audio": (G, Da)}
    cfg = FmriEncoderConfig(
        n_subjects=4, feature_aggregation="cat", layer_aggregation="cat",
        subject_embedding=False, modality_dropout=args.modality_dropout,
    )
    model = FmriEncoder(feature_dims=feat_dims, n_outputs=1000,
                        n_output_timesteps=args.window_tr, config=cfg).to(device)

    # 优化器/调度
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                           pct_start=args.warmup_pct, anneal_strategy="cos")

    # SWA
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = swa_start_epoch < args.epochs
    swa_model = AveragedModel(model) if use_swa else None

    # 历史最优（按验证集 Pearson 均值）
    best_key_metric = float("-inf")

    # 每 subject 的最佳指标追踪
    best_by_sub = {
        s: {"r": (-np.inf, -1), "rho": (-np.inf, -1), "r2": (-np.inf, -1), "isg": (-np.inf, -1)}
        for s in range(4)
    }

    # 简单的 GT 缓存：避免每个 batch 反复读取同一 (s, ds)
    fmri_cache: Dict[Tuple[int, str], np.ndarray] = {}

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # -------- Train (multi-head supervised) --------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            # 先做共享编码
            x_tr = compute_hidden_tr(model, batch)                  # [B,N,H]
            y_all = predict_all_heads_from_hidden(model, x_tr)      # [B,S,N,O]

            B, _, N, O = y_all.shape
            ds_list = batch.data["ds_list"]
            st_list = batch.data["start_tr_list"]

            # 准备各 subject 的 GT 窗口
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
                            fmri_cache[key] = gt_all  # [1000, T]
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed:
                            # 长度不足，跳过该 subject/窗口
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)  # [O,N]
                        pred_head = y_all[i, s].permute(1, 0)  # [O,N]
                        loss_terms.append(criterion(pred_head, gt_win))
                    except FileNotFoundError:
                        continue
            if not loss_terms:
                # 当前 batch 所有 subject 都缺或不足，跳过
                continue

            loss = torch.stack(loss_terms).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
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

        # -------- Val window-level loss（参考） --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_for_loss:
                batch = batch.to(device)
                # 仅用 anchor 头与 anchor GT 粗略估计窗级 loss（可选）
                y_anchor = model(batch)                      # [B,1000,N]（内部用 subject_id=0）
                yt = batch.data["fmri"]
                val_loss += criterion(y_anchor, yt).item()
        val_loss /= max(1, len(val_loader_for_loss))
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)

        # -------- Evaluate on ALL validation episodes --------
        roots_feat = {"video": Path(args.video_root), "text": Path(args.text_root), "audio": Path(args.audio_root)}
        per_sub_means, isg_means, used_counts = evaluate_episode_list(
            model=model, episodes=val_ids, roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers_arg=args.layers, layer_agg=agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="val"
        )

        # 训练集 Friends 单集评测（打印，不参与 best）
        probe_per_sub_means, probe_isg_means, probe_used_counts = evaluate_episode_list(
            model=model, episodes=[train_probe_ds], roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers_arg=args.layers, layer_agg=agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="trainprobe"
        )

        # 写 TB & 打印
        msg = [f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  |  VAL(all episodes)"]
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

            # 更新 per-metric 的历史最优（各自独立）
            if not np.isnan(r)   and r   > best_by_sub[s]["r"][0]:   best_by_sub[s]["r"]   = (r, epoch)
            if not np.isnan(rho) and rho > best_by_sub[s]["rho"][0]: best_by_sub[s]["rho"] = (rho, epoch)
            if not np.isnan(r2)  and r2  > best_by_sub[s]["r2"][0]:  best_by_sub[s]["r2"]  = (r2, epoch)
            if not np.isnan(isg) and isg > best_by_sub[s]["isg"][0]: best_by_sub[s]["isg"] = (isg, epoch)

        val_key_metric = float(np.mean(key_accumulate)) if key_accumulate else float("-inf")

        # Train-probe 打印
        msg.append("  |  TRAIN-PROBE(Friends)")
        for s in range(4):
            r = probe_per_sub_means[s]["r"]; rho = probe_per_sub_means[s]["rho"]; r2 = probe_per_sub_means[s]["r2"]
            isg = probe_isg_means[s]; n_used = probe_used_counts[s]
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_pearson",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_spearman", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_r2",       0.0 if np.isnan(r2) else r2,  epoch)
            msg.append(f" S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={n_used}")

        print("  ".join(msg))

        # -------- 选择 best --------
        if val_key_metric > best_key_metric:
            best_key_metric = val_key_metric
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            torch.save(model, out_dir / "checkpoints" / "best_full.pt")

    # SWA finalize
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best checkpoint selected by VAL Pearson(mean over available subjects): {best_key_metric:.6f}")
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


if __name__ == "__main__":
    main()