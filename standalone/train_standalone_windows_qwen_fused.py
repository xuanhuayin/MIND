# -*- coding: utf-8 -*-
"""
Standalone windowed training with fused 2Hz features (Qwen2.5-Omni), no encoder.
- 输入特征: 每个 episode 一个 .npy, 形状 [T, 2048] (T=TR数)
- 模型: 仅训练 4 个 subject 的线性预测头 (共享输入、独立输出)，一次性多头监督
- 训练/验证: 窗口滑动训练；整段 episode 评测 Pearson / Spearman / R² / ISG
- best 选择: 验证集 Pearson（对可用 subject 取均值）最大
- 兼容 fMRI 文件名差异: 通过 task-xxxx 灵活匹配
"""

from __future__ import annotations
import argparse, os, sys, random, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ---------------- repo root ----------------
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
PKG_PARENT = PROJ.parent
if str(PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(PKG_PARENT))

# ---------------- utils ----------------
def set_seed(seed: int = 33):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

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
    if p.exists(): return p
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

# ---------------- dataset ----------------
class FusedWindowedDataset(Dataset):
    """
    每个样本 = episode 的一个窗口（N TR）：
      输入 fused: [N, H] (H=2048)
      目标 fmri: [1000, N]  （anchor 仅用于窗级参考/可视化）
    """
    def __init__(
        self,
        ids: List[str],
        fused_root: Path,
        fmri_root_anchor: Path,   # anchor for length/windows (通常用 sub1)
        window_tr: int,
        stride_tr: int,
        subject_id: int = 0,      # 仅携带到 batch
    ):
        self.ids = ids
        self.fused_root = Path(fused_root)
        self.fmri_root_anchor = Path(fmri_root_anchor)
        self.N = int(window_tr)
        self.S = int(stride_tr)
        self.subject_id_fixed = int(subject_id)

        # 读第一条以探测维度
        f0 = np.load(self.fused_root / f"{ids[0]}.npy")
        assert f0.ndim == 2 and f0.shape[1] == 2048, f"Expect [T,2048], got {f0.shape}"
        self.H = int(f0.shape[1])

        # 构建窗口索引
        self._index: List[Tuple[str, int]] = []
        self._episode_len_tr: Dict[str, int] = {}
        for ds in ids:
            fx = np.load(self.fused_root / f"{ds}.npy")        # [T,2048]
            T_feat = fx.shape[0]
            arr = np.load(self.fmri_root_anchor / f"{ds}.npy") if (self.fmri_root_anchor / f"{ds}.npy").exists() else _load_fmri_flexible(self.fmri_root_anchor, ds)
            if 1000 in arr.shape:
                fmri = arr if arr.shape[0] == 1000 else arr.T
            else:
                fmri = arr.T if arr.shape[0] > arr.shape[1] else arr
            O, T_tr_fmri = fmri.shape
            assert O == 1000, f"Expect O=1000, got {O}"
            T_tr = min(T_feat, T_tr_fmri)
            self._episode_len_tr[ds] = T_tr
            for st in range(0, max(1, T_tr - self.N + 1), self.S):
                if st + self.N <= T_tr:
                    self._index.append((ds, st))

    def __len__(self): return len(self._index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        ds, start_tr = self._index[i]
        fused = np.load(self.fused_root / f"{ds}.npy")  # [T,2048]
        # 读 anchor GT 仅用于返回（训练真正的 GT 在循环内按 subject 读取）
        fmri_path = self.fmri_root_anchor / f"{ds}.npy"
        if fmri_path.exists():
            arr = np.load(fmri_path)
        else:
            arr = _load_fmri_flexible(self.fmri_root_anchor, ds)
        if 1000 in arr.shape:
            fmri = arr if arr.shape[0] == 1000 else arr.T
        else:
            fmri = arr.T if arr.shape[0] > arr.shape[1] else arr

        N = self.N
        x = fused[start_tr:start_tr+N, :]                     # [N,2048]
        Y = fmri[:, start_tr:start_tr + N]                    # [1000,N]

        return {
            "fused": torch.from_numpy(x.astype(np.float32)),  # [N,2048]
            "fmri": torch.from_numpy(Y.astype(np.float32)),   # [1000,N]
            "subject_id": torch.tensor(self.subject_id_fixed, dtype=torch.long),
            "ds": ds,
            "start_tr": int(start_tr),
        }

# ---------------- collate ----------------
class Batch:
    def __init__(self, data: Dict[str, torch.Tensor]): self.data = data
    def to(self, device):
        for k, v in self.data.items():
            if torch.is_tensor(v): self.data[k] = v.to(device, non_blocking=True)
        return self

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Batch:
    keys = ["fused","fmri","subject_id"]
    data: Dict[str, torch.Tensor] = {}
    for k in keys:
        data[k] = torch.stack([b[k] for b in batch], dim=0)
    data["ds_list"] = [b["ds"] for b in batch]
    data["start_tr_list"] = [int(b["start_tr"]) for b in batch]
    return Batch(data)

# ---------------- simple multi-head linear model ----------------
class LinearMultiHead(nn.Module):
    """
    多 subject 线性头:
      输入 x_tr: [B,N,H]  (H=2048)
      输出 y_all: [B,S,N,O]  (S=4, O=1000)
    """
    def __init__(self, in_dim: int = 2048, out_voxels: int = 1000, n_subjects: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim; self.out_voxels = out_voxels; self.n_subjects = n_subjects
        self.drop = nn.Dropout(dropout)
        # 参数形状与之前脚本一致: weight[S,O,H], bias[S,O]
        self.weight = nn.Parameter(torch.empty(n_subjects, out_voxels, in_dim))
        self.bias   = nn.Parameter(torch.zeros(n_subjects, out_voxels))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x_tr: torch.Tensor) -> torch.Tensor:
        # x_tr: [B,N,H] -> dropout(按特征)
        x = self.drop(x_tr)
        # y: [B,S,N,O] = einsum over H 与加偏置
        y = torch.einsum("bnh,soh->bsno", x, self.weight) + self.bias[None, :, None, :]
        return y

# ---------------- episode recon/eval ----------------
def pick_friends_episode(cands: List[str]) -> str:
    fs = [ds for ds in cands if "friends" in ds.lower()]
    return fs[0] if fs else cands[0]

@torch.no_grad()
def reconstruct_one_episode_multi_subject(
    model: LinearMultiHead,
    ds: str,
    fused_root: Path,
    fmri_roots_by_subject: Dict[int, Path],
    window_tr: int,
    stride_tr: int,
    device: torch.device,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], int, List[int]]:
    """
    返回：
      preds_by_sub[s] : [T,1000]（仅对 available_subjects 返回）
      gts_by_sub[s]   : [T,1000]
      T_ds            : anchor 的长度（取 sub1 GT 与特征的 min）
      available_subjects: 实际找到 GT 的 subject 列表
    """
    # anchor 用 sub1 root
    anchor_root = list(fmri_roots_by_subject.values())[0]
    dataset = FusedWindowedDataset(
        ids=[ds],
        fused_root=fused_root,
        fmri_root_anchor=anchor_root,
        window_tr=window_tr,
        stride_tr=stride_tr,
        subject_id=0,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=True)
    T_ds = dataset._episode_len_tr[ds]
    O = 1000
    S = 4

    # 滑窗重建：对重叠处求平均
    model.eval()
    acc = {s: np.zeros((T_ds, O), dtype=np.float32) for s in range(S)}
    cnt = np.zeros((T_ds,), dtype=np.int32)
    for batch in loader:
        batch = batch.to(device)
        x_tr = batch.data["fused"]  # [1,N,H]
        y_all = model(x_tr)         # [1,S,N,O]
        st = int(batch.data["start_tr_list"][0])
        N = y_all.shape[2]; ed = min(st + N, T_ds)
        y_all = y_all[:, :, :ed-st, :].detach().cpu().numpy()
        for s in range(S):
            acc[s][st:ed] += y_all[0, s]
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
            gts_by_sub[s] = gt_all[:, :T_ds].T.astype(np.float32)  # [T,1000]
            preds_by_sub[s] = full_preds[s]
            available_subjects.append(s)
        except FileNotFoundError:
            continue

    return preds_by_sub, gts_by_sub, T_ds, available_subjects

@torch.no_grad()
def evaluate_episode_list(
    model: LinearMultiHead,
    episodes: List[str],
    fused_root: Path,
    fmri_roots_by_subject: Dict[int, Path],
    window_tr: int,
    stride_tr: int,
    device: torch.device,
    save_root: Path | None = None,
    save_split_name: str = "val",
):
    agg = {s: {"r": [], "rho": [], "r2": []} for s in range(4)}
    agg_isg = {s: [] for s in range(4)}
    used_counts = {s: 0 for s in range(4)}

    for ds in episodes:
        preds_by_sub, gts_by_sub, _, available_subjects = reconstruct_one_episode_multi_subject(
            model=model, ds=ds, fused_root=fused_root,
            fmri_roots_by_subject=fmri_roots_by_subject,
            window_tr=window_tr, stride_tr=stride_tr, device=device
        )
        if not available_subjects:
            continue

        for s in available_subjects:
            pred, gt = preds_by_sub[s], gts_by_sub[s]    # [T,1000]
            r   = float(np.nanmean(voxelwise_pearson(pred, gt)))
            rho = float(np.nanmean(voxelwise_spearman(pred, gt)))
            r2  = float(np.nanmean(voxelwise_r2(pred, gt)))
            agg[s]["r"].append(r); agg[s]["rho"].append(rho); agg[s]["r2"].append(r2)
            used_counts[s] += 1

        # ISG
        for s in available_subjects:
            gt = gts_by_sub[s]
            r_list = []
            for t in available_subjects:
                if t == s: continue
                r_list.append(float(np.nanmean(voxelwise_pearson(preds_by_sub[t], gt))))
            if r_list:
                agg_isg[s].append(float(np.mean(r_list)))

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
    return per_sub_means, isg_means, used_counts

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    # 列表：固定用 datasets_8.txt 的前7作训练、第8作验证
    ap.add_argument("--all_list", type=str,
        default=str(PROJ / "pipeline_QWEN" / "timelines" / "datasets_8.txt"))

    # 特征根目录（fused）
    ap.add_argument("--fused_root", type=str,
        default=str(PROJ / "pipeline_QWEN" / "Qwen_8features" / "qwen2p5_fused_2hz" / "sub-01"))

    # 多 subject fMRI 根目录
    ap.add_argument("--fmri_root_sub1", type=str, required=True)
    ap.add_argument("--fmri_root_sub2", type=str, required=True)
    ap.add_argument("--fmri_root_sub3", type=str, required=True)
    ap.add_argument("--fmri_root_sub5", type=str, required=True)

    # 窗口
    ap.add_argument("--window_tr", type=int, default=100)
    ap.add_argument("--stride_tr", type=int, default=50)

    # 优化
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup_pct", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6)

    # 其它
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--out_dir", type=str,
                    default="/home/lawrence/Desktop/algonauts-2025/algonauts2025/outputs/standalone_windows_qwen")
    ap.add_argument("--log_dir", type=str, default=str(PROJ / "logs" / "standalone_qwen_fused"))

    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出 / 日志
    out_dir = Path(args.out_dir); (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

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

    # 读取 8 个 episode，前 7 训练/第 8 验证（不 shuffle，按文件顺序）
    all_ids = read_ids(args.all_list)
    assert len(all_ids) >= 8, f"--all_list 需至少 8 条，当前 {len(all_ids)}"
    train_ids, val_ids = all_ids[:7], [all_ids[7]]
    print(f"[SPLIT] train={len(train_ids)} val={len(val_ids)} ; val ep = {val_ids[0]}")

    # 训练集 Friends 单集用于 train-probe（若无 Friends 则取第一条）
    train_probe_ds = pick_friends_episode(train_ids)
    print(f"[TRAIN-PROBE] Friends episode: {train_probe_ds}")

    # 数据加载器
    train_set = FusedWindowedDataset(
        ids=train_ids,
        fused_root=Path(args.fused_root),
        fmri_root_anchor=fmri_roots[0],
        window_tr=args.window_tr,
        stride_tr=args.stride_tr,
        subject_id=0,
    )
    val_set_for_loss = FusedWindowedDataset(
        ids=val_ids,
        fused_root=Path(args.fused_root),
        fmri_root_anchor=fmri_roots[0],
        window_tr=args.window_tr,
        stride_tr=args.stride_tr,
        subject_id=0,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader_for_loss = DataLoader(val_set_for_loss, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # 模型/优化器
    model = LinearMultiHead(in_dim=2048, out_voxels=1000, n_subjects=4, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
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
    best_by_sub = {s: {"r": (-np.inf, -1), "rho": (-np.inf, -1), "r2": (-np.inf, -1), "isg": (-np.inf, -1)} for s in range(4)}
    fmri_cache: Dict[Tuple[int, str], np.ndarray] = {}

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        # -------- Train --------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            x_tr = batch.data["fused"]             # [B,N,2048]
            B, N, _ = x_tr.shape
            y_all = model(x_tr)                    # [B,S,N,O]

            ds_list = batch.data["ds_list"]
            st_list = batch.data["start_tr_list"]
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
                        if gt.shape[1] < ed:
                            continue
                        gt_win = torch.from_numpy(gt[:, st:ed].astype(np.float32)).to(device)  # [O,N]
                        pred_head = y_all[i, s].permute(1, 0)  # [O,N]
                        loss_terms.append(criterion(pred_head, gt_win))
                    except FileNotFoundError:
                        continue
            if not loss_terms:
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
                # 仅 anchor 头与 anchor GT 粗估
                y_all = model(batch.data["fused"])          # [B,S,N,O]
                y_anchor = y_all[:, 0].permute(0,2,1)       # [B,O,N]
                yt = batch.data["fmri"]                     # [B,O,N]
                val_loss += criterion(y_anchor, yt).item()
        val_loss /= max(1, len(val_loader_for_loss))
        writer.add_scalar("loss/val_epoch", float(val_loss), epoch)

        # -------- Evaluate on validation episode(s) --------
        fused_root = Path(args.fused_root)
        per_sub_means, isg_means, used_counts = evaluate_episode_list(
            model=model, episodes=val_ids, fused_root=fused_root,
            fmri_roots_by_subject=fmri_roots,
            window_tr=args.window_tr, stride_tr=args.stride_tr,
            device=device, save_root=out_dir, save_split_name="val"
        )

        # Train-probe on one Friends ep from train list
        probe_per_sub_means, probe_isg_means, probe_used_counts = evaluate_episode_list(
            model=model, episodes=[train_probe_ds], fused_root=fused_root,
            fmri_roots_by_subject=fmri_roots,
            window_tr=args.window_tr, stride_tr=args.stride_tr,
            device=device, save_root=out_dir, save_split_name="trainprobe"
        )

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