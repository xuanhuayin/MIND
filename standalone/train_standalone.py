# /home/lawrence/Desktop/algonauts-2025/algonauts2025/standalone/train_standalone.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm
from einops import rearrange

# ============== 工具 & 映射 ==============

def read_ids(txt: str) -> List[str]:
    return [ln.strip() for ln in open(txt, "r", encoding="utf-8") if ln.strip()]

def ds2friends(ds: str) -> str:
    """
    把 'ses-002_task-s01e04b' 映射成 'friends_s01e04b'
    """
    m = re.search(r"task-(s\d{2}e\d{2}[a-d])", ds, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse episode-part from dataset id: {ds}")
    return f"friends_{m.group(1).lower()}"

def group_mean_layers(lat_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    """
    输入: [L_full, D, T]；输出: [G, D, T]
    fractions 作为分段右边界(比例)，各段内做均值（与官方语义一致）
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
        groups.append(lat_LDT[s:e].mean(axis=0, keepdims=False))  # [D,T]
    return np.stack(groups, axis=0)

# ============== 极简 SegmentData ==============

class SegmentData:
    def __init__(self, data: Dict[str, torch.Tensor], subject_id: Optional[torch.Tensor]=None):
        self.data = data
        if subject_id is None:
            # 默认单被试
            B = data[next(iter(data))].shape[0]
            self.data["subject_id"] = torch.zeros((B,), dtype=torch.long)
        else:
            self.data["subject_id"] = subject_id

    def to(self, device: torch.device | str):
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                self.data[k] = v.to(device, non_blocking=True)
        return self

# ============== 数据集（缓存特征 + fMRI） ==============

class CachedMultimodalDataset(Dataset):
    """
    每个样本:
      video/text/audio: [G, D?, T]  （G 由 layer fractions/group-mean 决定）
      fmri: [O, T_fmri]  （和 2Hz 特征 T 可能不同，无需强制一致）
    """
    def __init__(
        self,
        id_list: List[str],
        video_root: Path,
        text_root: Path,
        audio_root: Path,
        fmri_root: Path,
        layer_fracs: List[float],
        layer_aggregation: str = "group_mean",
    ):
        self.ids = id_list
        self.video_root = Path(video_root)
        self.text_root = Path(text_root)
        self.audio_root = Path(audio_root)
        self.fmri_root = Path(fmri_root)
        self.fracs = layer_fracs
        self.layer_aggregation = layer_aggregation.lower()

    def __len__(self): return len(self.ids)

    def _load_feature_LDT(self, root: Path, key: str) -> np.ndarray:
        p = root / f"{key}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing feature npy: {p}")
        arr = np.load(p)  # 期望 [T, L, D]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape for {p}, expect [T,L,D], got {arr.shape}")
        return np.transpose(arr, (1, 2, 0))  # -> [L,D,T]

    def _maybe_aggregate(self, lat_LDT: np.ndarray) -> np.ndarray:
        if self.layer_aggregation in ("group_mean", "groupmean"):
            return group_mean_layers(lat_LDT, self.fracs)  # [G,D,T]
        elif self.layer_aggregation in ("none", "null"):
            L = lat_LDT.shape[0]
            sel = sorted(set(int(round(f*(L-1))) for f in self.fracs))
            return lat_LDT[sel]
        else:
            raise ValueError(f"Unsupported layer_aggregation: {self.layer_aggregation}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ds = self.ids[idx]
        key = ds  # 三模态缓存直接就是 ses-xxx_task-... 命名

        v_LDT = self._load_feature_LDT(self.video_root, key)
        t_LDT = self._load_feature_LDT(self.text_root , key)
        a_LDT = self._load_feature_LDT(self.audio_root, key)

        v = self._maybe_aggregate(v_LDT)  # [G,Dv,T2hz]
        t = self._maybe_aggregate(t_LDT)
        a = self._maybe_aggregate(a_LDT)

        # fMRI 用 friends_* 映射
        friends_key = ds2friends(ds)
        fmri_path = self.fmri_root / f"{friends_key}_fmri.npy"
        if not fmri_path.exists():
            raise FileNotFoundError(f"Missing fmri npy: {fmri_path}")
        fmri = np.load(fmri_path)  # (T_fmri,O) 或 (O,T_fmri)
        if fmri.ndim != 2:
            raise ValueError(f"fMRI shape should be 2D, got {fmri.shape}")
        if fmri.shape[0] < fmri.shape[1]:
            fmri = fmri.T  # -> (O, T_fmri)

        sample = {
            "video": torch.from_numpy(v.astype(np.float32)),
            "text" : torch.from_numpy(t.astype(np.float32)),
            "audio": torch.from_numpy(a.astype(np.float32)),
            "fmri" : torch.from_numpy(fmri.astype(np.float32)),  # [O, T_fmri]
            "ids"  : ds,
            "T2hz" : v.shape[-1],     # 保存 2Hz 长度，方便外部做任意对齐策略
            "Ttr"  : fmri.shape[1],   # 保存 TR 长度
        }
        return sample

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> SegmentData:
    """
    简单堆叠（默认 batch_size=1，避免不同 T 的 padding；若你调大 batch_size，请自己加 pad 对齐）
    """
    if len(batch) == 1:
        b = batch[0]
        data = {
            "video": b["video"].unsqueeze(0),
            "text" : b["text"].unsqueeze(0),
            "audio": b["audio"].unsqueeze(0),
            "fmri" : b["fmri"].unsqueeze(0),
            "ids"  : [b["ids"]],
            "T2hz" : torch.tensor([b["T2hz"]], dtype=torch.long),
            "Ttr"  : torch.tensor([b["Ttr"]], dtype=torch.long),
        }
        return SegmentData(data)
    else:
        raise NotImplementedError("Batching with variable T not implemented; use batch_size=1.")

# ============== 指标（逐 voxel Pearson） ==============

@torch.no_grad()
def voxelwise_pearson(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    pred,true: [N,O] （把 batch 和 time 展平）
    """
    pred = pred - pred.mean(axis=0, keepdims=True)
    true = true - true.mean(axis=0, keepdims=True)
    num = (pred * true).sum(axis=0)
    den = np.sqrt((pred**2).sum(axis=0) * (true**2).sum(axis=0)) + 1e-8
    return (num / den).astype(np.float32)

# ============== 模型（与你的 fmri_model_min 一致） ==============

# 保留与原仓库一致的模块依赖
from modeling_utils.models.common import MlpConfig, SubjectLayers
from modeling_utils.models.transformer import TransformerEncoderConfig
import typing as tp
import pydantic

class FmriEncoderConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriEncoder"] = "FmriEncoder"
    n_subjects: int | None = None
    feature_aggregation: tp.Literal["sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    subject_embedding: bool = False
    modality_dropout: float = 0.2  # 论文默认 0.2

    def build(self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int) -> nn.Module:
        return FmriEncoder(feature_dims, n_outputs, n_output_timesteps, config=self)

class FmriEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoderConfig,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.projectors = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)
        hidden = 3072
        for modality, tup in feature_dims.items():
            if tup is None:
                continue
            num_layers, feature_dim = tup
            input_dim = (feature_dim * num_layers if config.layer_aggregation == "cat" else feature_dim)
            output_dim = (hidden // len(feature_dims) if config.feature_aggregation == "cat" else hidden)
            self.projectors[modality] = MlpConfig(
                norm_layer="layer", activation_layer="gelu", dropout=0.0
            ).build(input_dim, output_dim)
        self.combiner = nn.Identity()
        self.predictor = SubjectLayers(
            in_channels=hidden,
            out_channels=n_outputs,
            n_subjects=config.n_subjects,
            average_subjects=False,
            bias=True,
        )
        # 先占位，稍后会用推断出的 T_max 替换
        self.time_pos_embed = nn.Parameter(torch.randn(1, 1024, hidden))
        if config.subject_embedding and config.n_subjects is not None:
            self.subject_embed = nn.Embedding(config.n_subjects, hidden)
        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

    def forward(self, batch: SegmentData, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)  # B,T,H
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)      # B,H,T
        x = self.predictor(x, subject_id)  # B,O,T
        if pool_outputs:
            out = self.pooler(x)   # B,O,T'
        else:
            out = x
        return out

    def aggregate_features(self, batch: SegmentData):
        tensors = []
        # 取任意存在的模态确定 B,T
        first = None
        for m in ("video","text","audio"):
            if m in batch.data:
                first = m
                break
        x0 = batch.data[first]  # [B,L,D,T]
        B, T = x0.shape[0], x0.shape[-1]

        # 随机丢模态（论文 0.2）
        modalities_to_dropout = []
        for m in self.feature_dims.keys():
            if torch.rand(1).item() < self.config.modality_dropout and self.training:
                modalities_to_dropout.append(m)
        if len(modalities_to_dropout) == len(self.feature_dims):
            modalities_to_dropout = modalities_to_dropout[:-1]

        for modality in self.feature_dims.keys():
            if modality not in self.projectors:
                data = torch.zeros(B, T, 3072 // len(self.feature_dims), device=x0.device)
            else:
                data = batch.data[modality]  # [B,L,D,T] 或 [B,D,T]
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)          # B,1,D,T
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)           # B,D,T
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)           # B,T,D
                data = self.projectors[modality](data)  # B,T,H
                if modality in modalities_to_dropout:
                    data = torch.zeros_like(data)
            tensors.append(data)

        if self.config.feature_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)  # B,T,H
        else:
            out = sum(tensors)
        return out

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            T = x.size(1)
            x = x + self.time_pos_embed[:, :T]  # 只切片，不再改形状
        if hasattr(self, "subject_embed") and subject_id is not None:
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x

# ============== 位置编码长度：推断 T_max 并一次性插值 ==============

@torch.no_grad()
def infer_T_max_2hz(all_ids: List[str], video_root: Path) -> int:
    T_max = 0
    for ds in all_ids:
        p = video_root / f"{ds}.npy"
        if not p.exists():  # 如果个别缺失，直接跳过
            continue
        arr = np.load(p, mmap_mode="r")
        if arr.ndim != 3:
            continue
        T = arr.shape[0]  # [T,L,D]
        if T > T_max:
            T_max = T
    return int(T_max)

def resize_time_pos_embed_(model: FmriEncoder, new_len: int):
    with torch.no_grad():
        pe = model.time_pos_embed  # [1, L_old, H]
        L_old, H = pe.shape[1], pe.shape[2]
        if new_len == L_old:
            return
        # 线性插值到 new_len
        pe_new = torch.nn.functional.interpolate(
            pe.transpose(1, 2), size=new_len, mode="linear", align_corners=False
        ).transpose(1, 2).contiguous()
        model.time_pos_embed = nn.Parameter(pe_new)

# ============== 训练主逻辑 ==============

def main():
    ap = argparse.ArgumentParser()
    # 数据与缓存
    ap.add_argument("--train_list", type=str, required=True)
    ap.add_argument("--val_list",   type=str, required=True)
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--fmri_root",  type=str, required=True)

    ap.add_argument("--layers", type=str, default="0.5,0.75,1.0",
                    help="fractions for group boundaries, comma-separated")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean", "none", "None"])

    # 训练超参
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1)   # 不同 T，建议 batch=1
    ap.add_argument("--num_workers", type=int, default=0)

    # 优化器（论文 AdamW + OneCycleLR cosine）
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--onecycle_warmup_pct", type=float, default=0.10)

    # SWA
    ap.add_argument("--use_swa", action="store_true", default=True)
    ap.add_argument("--swa_start_ratio", type=float, default=0.6, help="开始 SWA 的 epoch 比例")
    ap.add_argument("--swa_lr", type=float, default=1e-5)
    ap.add_argument("--swa_anneal_epochs", type=int, default=5)

    # 输出长度（窗口 N），默认 100，符合论文窗口训练示例
    ap.add_argument("--n_output_timesteps", type=int, default=100)

    # 输出目录
    default_out = Path(__file__).resolve().parents[1] / "outputs" / "standalone"
    ap.add_argument("--out_dir", type=str, default=str(default_out))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "preds_val").mkdir(parents=True, exist_ok=True)

    # 读取清单
    train_ids = read_ids(args.train_list)
    val_ids   = read_ids(args.val_list)

    # 过滤缺失 fMRI 的样本（避免中途再报错）
    fmri_root_path = Path(args.fmri_root)
    def has_fmri(ds: str) -> bool:
        return (fmri_root_path / f"{ds2friends(ds)}_fmri.npy").exists()
    miss_train = [ds for ds in train_ids if not has_fmri(ds)]
    miss_val   = [ds for ds in val_ids   if not has_fmri(ds)]
    if miss_train or miss_val:
        print("[WARN] missing fMRI files for:")
        for x in (miss_train + miss_val):
            print("  -", x)
        train_ids = [ds for ds in train_ids if ds not in miss_train]
        val_ids   = [ds for ds in val_ids   if ds not in miss_val]
        if not train_ids or not val_ids:
            raise RuntimeError("All samples filtered out due to missing fMRI files.")

    # 推断全数据的 T_max（2Hz 步数），一次性把位置编码插值到该长度
    T_max = infer_T_max_2hz(train_ids + val_ids, Path(args.video_root))
    print(f"[INFO] Inferred T_max (2Hz steps) = {T_max}")

    # 数据集/加载器
    fractions = [float(x) for x in args.layers.split(",") if x.strip()]
    agg_mode = "group_mean" if args.layer_aggregation.lower() not in ("none","null") else "none"

    train_set = CachedMultimodalDataset(
        id_list=train_ids,
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        fmri_root =Path(args.fmri_root),
        layer_fracs=fractions,
        layer_aggregation=agg_mode,
    )
    val_set = CachedMultimodalDataset(
        id_list=val_ids,
        video_root=Path(args.video_root),
        text_root =Path(args.text_root),
        audio_root=Path(args.audio_root),
        fmri_root =Path(args.fmri_root),
        layer_fracs=fractions,
        layer_aggregation=agg_mode,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              collate_fn=collate_fn, pin_memory=True)

    # 取一个 batch 推断 feature_dims、输出维度
    sample = next(iter(train_loader))
    feat_dims = {}
    for m in ("text","audio","video"):
        x = sample.data[m]  # [B,L,D,T]
        L, D = x.shape[1], x.shape[2]
        feat_dims[m] = (L, D)
    n_outputs = sample.data["fmri"].shape[1]  # O
    n_output_timesteps = args.n_output_timesteps  # N

    # 构建模型
    cfg = FmriEncoderConfig(
        n_subjects=1,
        feature_aggregation="cat",
        layer_aggregation="cat",  # 数据侧已把层聚成 G 组，这里按层 concat
        subject_embedding=False,
        modality_dropout=0.2,
    )
    model = FmriEncoder(
        feature_dims=feat_dims,
        n_outputs=n_outputs,
        n_output_timesteps=n_output_timesteps,
        config=cfg,
    ).to(device)

    # 位置编码长度固定为 T_max，避免后续 SWA 期间形状变化
    resize_time_pos_embed_(model, T_max)

    # 优化器 + OneCycleLR (cosine)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.onecycle_warmup_pct,
        anneal_strategy="cos",
        cycle_momentum=False,   # AdamW 通常关掉 momentum cycle
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # SWA
    use_swa = bool(args.use_swa)
    swa_start_epoch = int(args.swa_start_ratio * args.epochs)
    swa_model = AveragedModel(model) if use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr,
                          anneal_epochs=args.swa_anneal_epochs,
                          anneal_strategy="cos") if use_swa else None

    # 目标侧池化（把 B,O,T_fmri 自适应到 N）
    target_pooler = nn.AdaptiveAvgPool1d(n_output_timesteps).to(device)

    best_val = float("-inf")

    global_step = 0
    for epoch in range(1, args.epochs+1):

        # ========== Train ==========
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}] Train"):
            batch = batch.to(device)
            y_pred = model(batch)          # [B,O,N]，已池到 N
            y_true = batch.data["fmri"]    # [B,O,T_fmri]
            y_true_ds = target_pooler(y_true)  # [B,O,N]

            loss = nn.functional.mse_loss(y_pred, y_true_ds)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Scheduler / SWA
            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            running_loss += loss.item() * y_pred.size(0)
            global_step += 1

        train_loss = running_loss / len(train_set)

        # ========== Val ==========
        current_model = swa_model if (use_swa and epoch >= swa_start_epoch) else model
        current_model.eval()
        val_loss = 0.0
        all_preds, all_trues = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Epoch {epoch}/{args.epochs}] Val  "):
                ids = batch.data["ids"]
                batch = batch.to(device)
                # forward（AveragedModel 也可直接调用）
                y_pred = current_model(batch)           # [B,O,N]
                y_true = batch.data["fmri"]             # [B,O,T_fmri]
                y_true_ds = target_pooler(y_true)       # [B,O,N]
                loss = nn.functional.mse_loss(y_pred, y_true_ds)
                val_loss += loss.item() * y_pred.size(0)

                # 评估：展平时间 -> [N_total,O]
                yp = y_pred.detach().cpu().transpose(1,2).reshape(-1, y_pred.shape[1]).numpy()
                yt = y_true_ds.detach().cpu().transpose(1,2).reshape(-1, y_true_ds.shape[1]).numpy()
                all_preds.append(yp)
                all_trues.append(yt)

                # 保存单条验证预测：preds_val/{id}_pred.npy 形状 [N,O]
                for i, key in enumerate(ids):
                    npy = y_pred[i].detach().cpu().numpy().T  # -> [N,O]
                    np.save(out_dir / "preds_val" / f"{key}_pred.npy", npy)

        val_loss = val_loss / len(val_set)
        preds_np = np.concatenate(all_preds, axis=0)
        trues_np = np.concatenate(all_trues, axis=0)
        pearson = voxelwise_pearson(preds_np, trues_np)
        pearson_mean = float(np.nanmean(pearson))

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  val_pearson_mean={pearson_mean:.6f}")

        # 保存当前评估
        np.save(out_dir / f"val_voxel_pearson_epoch{epoch}.npy", pearson.astype(np.float32))

        # 最优权重（按 val_pearson_mean）
        if pearson_mean > best_val:
            best_val = pearson_mean
            if use_swa and epoch >= swa_start_epoch:
                torch.save(swa_model.state_dict(), out_dir / "checkpoints" / "best_swa.pt")
                torch.save(swa_model,              out_dir / "checkpoints" / "best_swa_full.pt")
            else:
                torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pt")
                torch.save(model,              out_dir / "checkpoints" / "best_full.pt")

    # 结束后若用了 SWA，可（可选）再用训练集跑一遍 BN 更新；本模型主要是 LN，不强制需要
    # if use_swa:
    #     update_bn(train_loader, swa_model, device=device)

    # 合并验证集预测，方便后续可视化
    val_dict = {}
    for key in val_ids:
        p = out_dir / "preds_val" / f"{key}_pred.npy"
        if p.exists():
            val_dict[key] = np.load(p)
    np.save(out_dir / "val_predictions_dict.npy", val_dict, allow_pickle=True)

    print(f"\n[Done] best val pearson (mean over voxels): {best_val:.6f}")
    print(f"Checkpoints: {out_dir / 'checkpoints'}")
    print(f"Val predictions (.npy per id): {out_dir / 'preds_val'}")
    print(f"Merged val predictions dict: {out_dir / 'val_predictions_dict.npy'}")


if __name__ == "__main__":
    main()