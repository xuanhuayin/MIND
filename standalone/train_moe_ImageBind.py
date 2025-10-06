# -*- coding: utf-8 -*-
"""
Windowed training using your FmriEncoder_MoE (token-level router + experts)

- 训练：同一份共享MoE主干，但通过 subject_id 进行“受试者条件化”路由。
  对每个 batch，我们对 4 个 subject 各跑一遍 forward，分别与对应 GT 计算 loss 后求平均。
  若设置 --moe_aux_weight>0，会把模型 forward 里的负载均衡辅助损失（model.last_aux_loss）加权并入主损失。

- 评测：在完整 episode 上重建 [T, 1000]，分别对 4 个 subject 计算 r/ρ/R²。
  同时计算 ISG：对每个 s，用其它 t!=s 的预测与 s 的 GT 做相关，然后求平均。
  （注意：我们用 subject_id 条件化，4 个 subject 的预测不同，因此 ISG ≠ 普通 VAL 平均。）

- 可视化（可选）：每个 epoch 结束后
    1) 在验证集上挑 **r 最高** 的 (subject, episode)，
    2) 以及固定的训练集 Friends 片段（train-probe），
  调用  /home/lawrence/Desktop/algonauts-2025/algonauts2025/vis/plot_encoding_map_plus.py
  生成图（阈值=0.1）。前提：提供 --vis_atlas（Schaefer-1000 labels NIfTI）。

用法示例见文件末尾注释。
"""

from __future__ import annotations
import argparse, os, sys, random, re, subprocess
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
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# ---- your model ----
# 确保你已在 algonauts2025/standalone/fmri_model_min1.py 里定义并导出了 FmriEncoder_MoE
from algonauts2025.standalone.moe_decoder import FmriEncoder_MoE

# CUDA matmul 优化（保持数值为 FP32）
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


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


# ---------------- small container ----------------
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
    一个样本 = episode 的一个窗口（N TR）：
      输入 video/text/audio: [G, D, N * frames_per_tr]
      输出字段里会带：ds/start_tr/subject_id(占位)；训练时真正的 GT 会按 subject 根目录动态加载。
    """
    def __init__(
        self,
        ids: List[str],
        video_root: Path,
        text_root: Path,
        audio_root: Path,
        anchor_fmri_root: Path,   # 仅用于推断 episode 长度/窗口索引
        layers_arg: str,
        layer_agg: str,
        window_tr: int,
        stride_tr: int,
        frames_per_tr: int,
    ):
        self.ids = ids
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.anchor_fmri_root = Path(anchor_fmri_root)
        self.N = int(window_tr); self.S = int(stride_tr); self.f = int(frames_per_tr)

        # probe L
        v0 = np.load(self.video_root / f"{ids[0]}.npy")
        probe_L = v0.shape[1]
        self.layer_mode, payload = parse_layers_arg(layers_arg, probe_L)
        if self.layer_mode == "fractions":
            self.fracs, self.sel_indices = [float(x) for x in payload], None
        else:
            self.fracs, self.sel_indices = None, [int(i) for i in payload]
        self.layer_agg = layer_agg.lower()

        # windows
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            v = np.load(self.video_root / f"{ds}.npy")
            T_frames = v.shape[0]
            T_tr_feat = T_frames // self.f

            arr = load_fmri_flexible(self.anchor_fmri_root, ds)
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            T_tr = min(T_tr_feat, fmri.shape[1])
            self._episode_len_tr[ds] = T_tr
            for st in range(0, max(1, T_tr - self.N + 1), self.S):
                if st + self.N <= T_tr:
                    self._index.append((ds, st))

        # dims
        first_ds, _ = self._index[0]
        v_LDT = self._load_feature_LDT(self.video_root / f"{first_ds}.npy")
        t_LDT = self._load_feature_LDT(self.text_root / f"{first_ds}.npy")
        a_LDT = self._load_feature_LDT(self.audio_root / f"{first_ds}.npy")

        v_GDT = self._maybe_pick_layers(v_LDT)
        t_GDT = self._maybe_pick_layers(t_LDT)
        a_GDT = self._maybe_pick_layers(a_LDT)

        self.G, self.Dv = v_GDT.shape[0], v_GDT.shape[1]
        self.Dt, self.Da = t_GDT.shape[1], a_GDT.shape[1]

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

        return {
            "video": feats["video"],
            "text": feats["text"],
            "audio": feats["audio"],
            "ds": ds,
            "start_tr": start_tr,
        }


# ---------------- collate ----------------
def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    data: Dict[str, torch.Tensor] = {}
    for k in ["video","text","audio"]:
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
    return (1.0 - (ss_res / ss_tot)).astype(np.float32)


# ---------------- file resolver ----------------
_task_rx = re.compile(r"(task-[A-Za-z0-9]+(?:_[^.]*)?)", re.IGNORECASE)

def fmri_canonical(root: Path, ds: str) -> Path:
    p = Path(root) / f"{ds}.npy"
    if p.exists(): return p
    m = _task_rx.search(ds)
    if m:
        key = m.group(1)
        cands = sorted(Path(root).glob(f"*_{key}.npy")) + sorted(Path(root).glob(f"*{key}.npy"))
        if cands: return cands[0]
    # fallback：截掉前缀
    parts = ds.split("_", 1)
    if len(parts) == 2:
        suf = parts[1]
        cands = sorted(Path(root).glob(f"*_{suf}.npy")) + sorted(Path(root).glob(f"*{suf}.npy"))
        if cands: return cands[0]
    raise FileNotFoundError(f"GT not found for '{ds}' under '{root}'")

def load_fmri_flexible(root: Path, ds: str) -> np.ndarray:
    p = fmri_canonical(root, ds)
    return np.load(p)


# ---------------- episode reconstruction (MoE, subject-conditioned) ----------------
@torch.no_grad()
def reconstruct_episode_subject_conditioned(
    model: FmriEncoder_MoE,
    ds: str,
    video_root: Path, text_root: Path, audio_root: Path,
    fmri_roots_by_subject: Dict[int, Path],  # {0: sub1_root, 1: sub2_root, 2: sub3_root, 3: sub5_root}
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[int]]:
    """
    返回：
      preds_by_sub[s]: [T,1000]（subject_id=s 条件化得到）
      gts_by_sub[s]:   [T,1000]
      available_subjects: 实际存在 GT 的 subject 列表
    """
    # 用 sub1 根目录只是为了确定窗口切分
    anchor_root = list(fmri_roots_by_subject.values())[0]
    ds_tmp = WindowedDataset(
        ids=[ds],
        video_root=video_root, text_root=text_root, audio_root=audio_root,
        anchor_fmri_root=anchor_root,
        layers_arg=layers, layer_agg=layer_agg,
        window_tr=window_tr, stride_tr=stride_tr, frames_per_tr=frames_per_tr
    )
    loader = DataLoader(ds_tmp, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type=='cuda'))

    T_ds = ds_tmp._episode_len_tr[ds]
    O = 1000
    preds_sum = {s: np.zeros((T_ds, O), dtype=np.float32) for s in range(4)}
    preds_cnt = np.zeros((T_ds,), dtype=np.int32)

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        st = int(batch.data["start_tr_list"][0])
        # 对每个 subject_id 跑一次（共享特征，但我们简单起见逐次 forward）
        outs = {}
        for s in range(4):
            # 拼个 “带 subject_id 的 batch”
            batch.data["subject_id"] = torch.full((1,), s, dtype=torch.long, device=device)
            y = model(batch, pool_outputs=True)    # [B, O, N]
            outs[s] = y[0].permute(1, 0).detach().cpu().numpy()   # [N,O]
        N = list(outs.values())[0].shape[0]
        ed = min(st + N, T_ds)
        span = ed - st
        for s in range(4):
            preds_sum[s][st:ed] += outs[s][:span]
        preds_cnt[st:ed] += 1

    cnt = np.maximum(preds_cnt[:, None], 1)
    preds_by_sub = {s: (preds_sum[s] / cnt).astype(np.float32) for s in range(4)}

    # GT
    gts_by_sub, available_subjects = {}, []
    for s, root in fmri_roots_by_subject.items():
        try:
            gt = load_fmri_flexible(root, ds)
            if 1000 in gt.shape:
                gt = gt if gt.shape[0] == 1000 else gt.T
            else:
                gt = gt.T if gt.shape[0] > gt.shape[1] else gt
            gts_by_sub[s] = gt[:, :T_ds].T.astype(np.float32)
            available_subjects.append(s)
        except FileNotFoundError:
            continue
    return preds_by_sub, gts_by_sub, available_subjects


# ---------------- evaluate a list of episodes ----------------
@torch.no_grad()
def evaluate_episodes(
    model: FmriEncoder_MoE,
    episodes: List[str],
    roots_feat: Dict[str, Path],
    fmri_roots_by_subject: Dict[int, Path],
    layers: str, layer_agg: str,
    window_tr: int, stride_tr: int, frames_per_tr: int,
    device: torch.device,
    save_root: Path | None = None, save_split_name: str = "val",
):
    """
    返回：
      per_sub_means: {s: {'r','rho','r2'}}
      isg_means:     {s: mean}
      used_counts:   {s: #episodes}
      per_episode_scores: {s: List[Tuple[ds, r_mean]]}
    """
    agg = {s: {"r": [], "rho": [], "r2": []} for s in range(4)}
    agg_isg = {s: [] for s in range(4)}
    used_counts = {s: 0 for s in range(4)}
    per_episode_scores = {s: [] for s in range(4)}

    for ds in episodes:
        preds_by_sub, gts_by_sub, available = reconstruct_episode_subject_conditioned(
            model, ds,
            roots_feat["video"], roots_feat["text"], roots_feat["audio"],
            fmri_roots_by_subject,
            layers, layer_agg, window_tr, stride_tr, frames_per_tr, device
        )
        if not available:
            continue

        # per-subject metrics
        for s in available:
            pred, gt = preds_by_sub[s], gts_by_sub[s]
            r   = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2  = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[s]["r"].append(r); agg[s]["rho"].append(rho); agg[s]["r2"].append(r2)
            used_counts[s] += 1
            per_episode_scores[s].append((ds, r))

        # ISG: 用其它头(t)的预测去拟合 s 的 GT
        for s in available:
            r_list = []
            for t in available:
                if t == s: continue
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t], gts_by_sub[s]))))
            if r_list:
                agg_isg[s].append(float(np.mean(r_list)))

        # 保存 pred/gt
        if save_root is not None:
            subname = {0:"sub01", 1:"sub02", 2:"sub03", 3:"sub05"}
            for s in available:
                pdir = save_root / subname[s] / f"preds_{save_split_name}_episodes"
                gdir = save_root / subname[s] / f"preds_{save_split_name}_episodes_gt"
                pdir.mkdir(parents=True, exist_ok=True); gdir.mkdir(parents=True, exist_ok=True)
                np.save(pdir / f"{ds}_pred.npy", preds_by_sub[s])
                np.save(gdir / f"{ds}_gt.npy",   gts_by_sub[s])

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
    return per_sub_means, isg_means, used_counts, per_episode_scores


# ---------------- helpers ----------------
def pick_friends_episode(ids: List[str]) -> str:
    fs = [ds for ds in ids if "friends" in ds.lower()]
    return fs[0] if fs else ids[0]


def top_val_episode_for_vis(per_episode_scores: Dict[int, List[Tuple[str,float]]]) -> Optional[Tuple[int,str,float]]:
    best = None
    for s, lst in per_episode_scores.items():
        for ds, r in lst:
            if (best is None) or (r > best[2]):
                best = (s, ds, r)
    return best


def call_visualizer(gt_path: Path, pred_path: Path, atlas_path: Path, outdir: Path,
                    subject_tag: str, modality: str = "all", threshold: str = "0.1"):
    script = PROJ / "vis" / "plot_encoding_map_plus.py"
    if (not script.exists()) or (not atlas_path.exists()):
        print(f"[VIS][SKIP] script or atlas missing: {script}, {atlas_path}")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--gt", str(gt_path),
        "--pred", str(pred_path),
        "--atlas", str(atlas_path),
        "--outdir", str(outdir),
        "--subject", subject_tag,
        "--modality", modality,
        "--threshold", threshold,
        "--align", "truncate",
        "--delay", "0",
        "--no-surf"  # 可按需关/开；保留玻璃脑+体素图，跑得更快些
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"[VIS] saved under {outdir}")
    except subprocess.CalledProcessError as e:
        print(f"[VIS][WARN] failed: {e}")


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # splits
    ap.add_argument("--train_list", type=str, default="")
    ap.add_argument("--val_list",   type=str, default="")
    ap.add_argument("--all_list",   type=str, default="")
    ap.add_argument("--split_ratio", type=float, default=0.9)
    ap.add_argument("--split_seed",  type=int,   default=33)
    ap.add_argument("--moe_dropout", type=float, default=0.1,
                    help="Dropout prob before each expert (default 0.1)")

    # feature roots
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)

    # fMRI roots (multi-subject)
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # layer selection
    ap.add_argument("--layers", type=str, default="last41")
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

    # model (your MoE)
    ap.add_argument("--subject_embedding", action="store_true",
                    help="让 MoE 路由在隐藏表征上加 subject embedding 偏置")
    ap.add_argument("--moe_num_experts", type=int, default=4)
    ap.add_argument("--moe_top_k", type=int, default=2)
    ap.add_argument("--moe_aux_weight", type=float, default=0.0,
                    help="负载均衡辅助损失的权重（建议 0.001~0.02）")

    # grad ckpt / SWA
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--disable_swa", action="store_true")

    # output & logs (你要的 arg)
    ap.add_argument("--out_dir", type=str, default=str(PROJ / "outputs" / "MoE_IMAGEBIND"))
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "MoE_IMAGEBIND"))

    # visualization
    ap.add_argument("--vis_enable", action="store_true",
                    help="每个 epoch 结束后做一次可视化；需要提供 --vis_atlas")
    ap.add_argument("--vis_atlas",  type=str, default="",
                    help="Schaefer-1000 等 label NIfTI 路径")
    ap.add_argument("--vis_threshold", type=str, default="0.1")

    # misc
    ap.add_argument("--seed", type=int, default=33)

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEV] Using {device}")

    # 路径
    out_dir = Path(args.out_dir); (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir); log_root.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    tb_dir = log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[TB] Logging to: {tb_dir}")

    # subject roots
    sub_map = {0:"sub01", 1:"sub02", 2:"sub03", 3:"sub05"}
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

    # 数据集拆分
    if args.all_list:
        all_ids = read_ids(args.all_list)
        rnd = random.Random(args.seed); rnd.shuffle(all_ids)
        k = max(1, min(len(all_ids)-1, int(round(len(all_ids) * 0.9))))
        train_ids, val_ids = all_ids[:k], all_ids[k:]
        print(f"[SPLIT] Using --all_list, split to train={len(train_ids)}  val={len(val_ids)}")
    else:
        if not args.train_list or not args.val_list:
            raise SystemExit("Provide --all_list or both --train_list/--val_list")
        train_ids = read_ids(args.train_list); val_ids = read_ids(args.val_list)
        print(f"[SPLIT] Using provided lists: train={len(train_ids)} val={len(val_ids)}")

    # Friends 片段（train probe）
    train_probe_ds = pick_friends_episode(train_ids)
    print(f"[TRAIN-PROBE] Friends episode: {train_probe_ds}")

    # layer agg flag
    layer_agg = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    # window datasets
    train_set = WindowedDataset(
        ids=train_ids,
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        anchor_fmri_root=fmri_roots[0],
        layers_arg=args.layers, layer_agg=layer_agg,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr
    )
    val_set_for_loss = WindowedDataset(
        ids=val_ids if len(val_ids)>0 else train_ids[:1],
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        anchor_fmri_root=fmri_roots[0],
        layers_arg=args.layers, layer_agg=layer_agg,
        window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader_for_loss = DataLoader(val_set_for_loss, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # 构建模型（你的 FmriEncoder_MoE）
    feat_dims = {"video": (train_set.G, train_set.Dv), "text": (train_set.G, train_set.Dt), "audio": (train_set.G, train_set.Da)}
    model = FmriEncoder_MoE(
        feature_dims=feat_dims,
        n_outputs=1000,
        n_output_timesteps=args.window_tr,
        n_subjects=4,
        num_experts=args.moe_num_experts,
        top_k=args.moe_top_k,
        feature_aggregation="cat",
        layer_aggregation="cat",
        subject_embedding=args.subject_embedding,
        moe_dropout=args.moe_dropout,  # ← 这里用 argparse 的值
    ).to(device)
    print(f"[MODEL] FmriEncoder_MoE | experts={args.moe_num_experts} topk={args.moe_top_k} subject_embed={args.subject_embedding}")

    # grad ckpt（若可用）
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

    # 优化器 / 调度
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_pct, anneal_strategy="cos")

    # SWA
    swa_start_epoch = int(args.epochs * args.swa_start_ratio)
    use_swa = (not args.disable_swa) and (swa_start_epoch < args.epochs)
    swa_model = AveragedModel(model) if use_swa else None

    # 记录最优
    best_key = float("-inf")

    # 简单 GT 缓存
    fmri_cache: Dict[Tuple[int,str], np.ndarray] = {}

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # -------- Train --------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            loss_terms = []
            aux_terms = []

            # 针对 4 个 subject，分别 forward 并计算窗口级 MSE
            for s in range(4):
                batch.data["subject_id"] = torch.full((batch.data["video"].size(0),), s, dtype=torch.long, device=device)
                y = model(batch, pool_outputs=True)         # [B, O, N]
                B, O, N = y.shape
                ds_list = batch.data["ds_list"]
                st_list = batch.data["start_tr_list"]

                for i in range(B):
                    ds = ds_list[i]; st = int(st_list[i]); ed = st + N
                    try:
                        key = (s, ds)
                        if key not in fmri_cache:
                            gt_all = load_fmri_flexible(fmri_roots[s], ds)
                            if 1000 in gt_all.shape:
                                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
                            else:
                                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
                            fmri_cache[key] = gt_all
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed:
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)  # [O,N]
                        loss_terms.append(criterion(y[i], gt_win))
                    except FileNotFoundError:
                        continue

                # MoE 负载均衡辅助损失（若启用权重）
                if args.moe_aux_weight > 0 and getattr(model, "last_aux_loss", None) is not None:
                    aux_terms.append(model.last_aux_loss)

            if not loss_terms:
                continue

            loss = torch.stack(loss_terms).mean()
            if aux_terms:
                loss = loss + float(args.moe_aux_weight) * torch.stack(aux_terms).mean()

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

        # -------- Val (window loss, anchor=subject 0) --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader_for_loss:
                batch = batch.to(device)
                batch.data["subject_id"] = torch.zeros((batch.data["video"].size(0),), dtype=torch.long, device=device)
                y = model(batch, pool_outputs=True)  # [B,O,N]
                # 用 sub1 的 GT 粗略评估窗级 loss
                B, O, N = y.shape
                for i in range(B):
                    ds = batch.data["ds_list"][i]; st = int(batch.data["start_tr_list"][i]); ed = st + N
                    try:
                        key = (0, ds)
                        if key not in fmri_cache:
                            gt_all = load_fmri_flexible(fmri_roots[0], ds)
                            if 1000 in gt_all.shape:
                                gt_all = gt_all if gt_all.shape[0] == 1000 else gt_all.T
                            else:
                                gt_all = gt_all.T if gt_all.shape[0] > gt_all.shape[1] else gt_all
                            fmri_cache[key] = gt_all
                        gt = fmri_cache[key]
                        if gt.shape[1] < ed:
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)
                        val_loss += criterion(y[i], gt_win).item()
                    except FileNotFoundError:
                        continue
        val_loss /= max(1, len(val_loader_for_loss))
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)

        # -------- Evaluate on full validation episodes --------
        roots_feat = {"video": Path(args.video_root), "text": Path(args.text_root), "audio": Path(args.audio_root)}
        per_sub_means, isg_means, used_counts, per_episode_scores = evaluate_episodes(
            model=model, episodes=val_ids, roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers=args.layers, layer_agg=layer_agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="val"
        )

        # Train-probe（Friends）
        probe_means, probe_isg, probe_used, _ = evaluate_episodes(
            model=model, episodes=[train_probe_ds], roots_feat=roots_feat,
            fmri_roots_by_subject=fmri_roots, layers=args.layers, layer_agg=layer_agg,
            window_tr=args.window_tr, stride_tr=args.stride_tr, frames_per_tr=args.frames_per_tr,
            device=device, save_root=out_dir, save_split_name="trainprobe"
        )

        # TB & 打印
        acc_key = []
        log_line = [f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  |  VAL(all episodes)"]
        for s in range(4):
            r = per_sub_means[s]["r"]; rho = per_sub_means[s]["rho"]; r2 = per_sub_means[s]["r2"]; isg = isg_means[s]
            n_used = used_counts[s]
            if not np.isnan(r): acc_key.append(r)
            writer.add_scalar(f"val/sub{s+1:02d}_pearson_mean",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_spearman_mean", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"val/sub{s+1:02d}_r2_mean",       0.0 if np.isnan(r2) else r2,  epoch)
            if not np.isnan(isg): writer.add_scalar(f"val/sub{s+1:02d}_ISG_pearson", isg, epoch)
            log_line.append(f"S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}, used_ep={n_used}")
        val_key = float(np.mean(acc_key)) if acc_key else float("-inf")

        log_line.append(" | TRAIN-PROBE(Friends)")
        for s in range(4):
            r = probe_means[s]["r"]; rho = probe_means[s]["rho"]; r2 = probe_means[s]["r2"]; isg = probe_isg[s]
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_pearson",  0.0 if np.isnan(r) else r,   epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_spearman", 0.0 if np.isnan(rho) else rho, epoch)
            writer.add_scalar(f"trainprobe/sub{s+1:02d}_r2",       0.0 if np.isnan(r2) else r2,  epoch)
            log_line.append(f"S{s+1:02d}: r={r:.6f}, ρ={rho:.6f}, R²={r2:.6f}, ISG={isg:.6f}")
        print("  ".join(log_line))
        writer.add_scalar("val/mean_pearson", 0.0 if np.isnan(val_key) else val_key, epoch)

        # 保存 best
        if val_key > best_key:
            best_key = val_key
            torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
            try:
                torch.save(model, out_dir / "checkpoints" / "best_full.pt")
            except Exception as e:
                print(f"[SAVE][WARN] torch.save(model) failed: {e}")

        # -------- 可视化（可选） --------
        if args.vis_enable and args.vis_atlas:
            atlas_path = Path(args.vis_atlas)
            subname = {0:"sub01",1:"sub02",2:"sub03",3:"sub05"}

            # 1) 取验证集上 r 最高的 (subject, episode)
            best = top_val_episode_for_vis(per_episode_scores)
            if best is not None:
                s_best, ds_best, r_best = best
                pred_path = out_dir / subname[s_best] / "preds_val_episodes" / f"{ds_best}_pred.npy"
                gt_path   = out_dir / subname[s_best] / "preds_val_episodes_gt" / f"{ds_best}_gt.npy"
                vis_out   = out_dir / subname[s_best] / "vis_val" / ds_best
                if pred_path.exists() and gt_path.exists():
                    call_visualizer(gt_path, pred_path, atlas_path, vis_out, subject_tag=subname[s_best][-2:], threshold=args.vis_threshold)

            # 2) Friends train-probe 可视化（对已存在 GT 的 subject 逐个做）
            for s in range(4):
                pred_path = out_dir / subname[s] / "preds_trainprobe_episodes" / f"{train_probe_ds}_pred.npy"
                gt_path   = out_dir / subname[s] / "preds_trainprobe_episodes_gt" / f"{train_probe_ds}_gt.npy"
                vis_out   = out_dir / subname[s] / "vis_trainprobe" / train_probe_ds
                if pred_path.exists() and gt_path.exists():
                    call_visualizer(gt_path, pred_path, atlas_path, vis_out, subject_tag=subname[s][-2:], threshold=args.vis_threshold)

    # SWA finalize
    if use_swa:
        print("Updating BN statistics for SWA model...")
        update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")

    writer.close()
    print("\n[Done]")
    print(f"Best VAL mean Pearson: {best_key:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    for sname in ["sub01","sub02","sub03","sub05"]:
        print(f"VAL preds dir: {out_dir / sname / 'preds_val_episodes'}")
        print(f"TRAIN-PROBE preds dir: {out_dir / sname / 'preds_trainprobe_episodes'}")


if __name__ == "__main__":
    """
    例子（与之前一致，注意这里不再有 --decoder，因为直接用 FmriEncoder_MoE）：

    CUDA_VISIBLE_DEVICES=0 python -m algonauts2025.standalone.train_moe \
      --all_list /home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/TRIBE_8features/all_list.txt \
      --video_root  /home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/TRIBE_8features/video_2hz/sub-01 \
      --text_root   /home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/TRIBE_8features/text_2hz/sub-01 \
      --audio_root  /home/lawrence/Desktop/algonauts-2025/algonauts2025/pipeline_TRIBE/TRIBE_8features/audio_2hz/sub-01 \
      --fmri_root_sub1 /home/lawrence/Desktop/algonauts-2025/algonauts2025/fmri_data/sub1 \
      --fmri_root_sub2 /home/lawrence/Desktop/algonauts-2025/algonauts2025/fmri_data/sub2 \
      --fmri_root_sub3 /home/lawrence/Desktop/algonauts-2025/algonauts2025/fmri_data/sub3 \
      --fmri_root_sub5 /home/lawrence/Desktop/algonauts-2025/algonauts2025/fmri_data/sub5 \
      --layers 0.6,0.8,1.0 --layer_aggregation group_mean \
      --window_tr 100 --stride_tr 50 --frames_per_tr 3 \
      --epochs 25 --batch_size 1 --num_workers 0 \
      --grad_ckpt --disable_swa \
      --subject_embedding \
      --moe_num_experts 4 --moe_top_k 2 --moe_aux_weight 0.01 \
      --out_dir /home/lawrence/Desktop/algonauts-2025/algonauts2025/outputs/MoE_TRIBE \
      --log_dir /home/lawrence/Desktop/algonauts-2025/algonauts2025/logs/MoE_TRIBE \
      --vis_enable --vis_atlas /path/to/Schaefer_1000_labels.nii.gz --vis_threshold 0.1
    """
    main()