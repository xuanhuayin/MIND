# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import ClassVar, Optional, List

import numpy as np
import pandas as pd
import pydantic
import torch

# --------- 保证能 import 到官方包 ---------
PROJ = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025").resolve()
if str(PROJ.parent) not in sys.path:
    sys.path.insert(0, str(PROJ.parent))

# 官方训练入口与模型配置
from algonauts2025.main import Experiment, Data
from algonauts2025.model import FmriEncoderConfig

# 官方数据与特征基类
from data_utils.data import StudyLoader
from data_utils.events import EventTypesHelper
from data_utils.base import TimedArray
from data_utils.features.video import VJEPA2
from data_utils.features.text import LLAMA3p2
from data_utils.features.audio import Wav2VecBert
from data_utils.features.neuro import Fmri

# ---------- 关键：禁用 events 的触盘初始化，避免 moviepy/ffmpeg ----------
from data_utils import events as ev
def _noop(self, __context=None):
    # 不做任何 I/O / 解析，完全跳过
    return None
for _cls in ("Video", "Sound", "Word", "Fmri"):
    if hasattr(ev, _cls):
        setattr(getattr(ev, _cls), "model_post_init", _noop)

# ----------------- 工具函数 -----------------
def read_list(path: str) -> list[str]:
    return [l.strip() for l in open(path, "r", encoding="utf-8") if l.strip()]

def parse_key_from_dataset(ds: str) -> str:
    """
    从 'ses-xxx_task-sXXeYYp' 抽取 'friends_sXXeYYp'
    例如: ses-001_task-s01e02a -> friends_s01e02a
    """
    m = re.search(r"task-(s\d{2}e\d{2}[a-d])$", ds, flags=re.IGNORECASE)
    assert m, f"Cannot parse episode-part from dataset: {ds}"
    return f"friends_{m.group(1).lower()}"

def group_mean_layers(latents_LDT: np.ndarray, fractions: List[float]) -> np.ndarray:
    """
    输入: latents [L_full, D, T]
    输出: group_mean 后 [G, D, T]，G=len(fractions)
    官方语义：fractions 是层索引分段的右边界“比例”；各分段做均值。
    """
    L_full = latents_LDT.shape[0]
    idxs = sorted(set(int(round(f * (L_full - 1))) for f in fractions))
    if not idxs:
        idxs = [L_full - 1]
    if idxs[-1] != L_full - 1:
        idxs[-1] = L_full - 1
    bounds = [i + 1 for i in idxs]  # 右开
    starts = [0] + bounds[:-1]
    ends = bounds
    groups = []
    for s, e in zip(starts, ends):
        s = max(0, min(s, L_full))
        e = max(0, min(e, L_full))
        if e <= s:
            s, e = L_full - 1, L_full
        groups.append(latents_LDT[s:e].mean(axis=0, keepdims=False))  # -> [D, T]
    return np.stack(groups, axis=0)  # [G, D, T]


# ----------------- 四个“缓存读取”特征类：继承官方基类 -----------------
class VJEPA2Cached(VJEPA2):
    EVENT_TYPE: ClassVar[str] = "Video"  # 必须与官方类型匹配
    root: Path
    layers: List[float] = [0.5, 0.75, 1.0]
    layer_aggregation: Optional[str] = "group_mean"

    _eth: EventTypesHelper = pydantic.PrivateAttr()

    model_config = pydantic.ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._eth = EventTypesHelper(self.EVENT_TYPE)

    def prepare(self, obj):  # 不重算，只打通接口
        pass

    def __call__(self, events, start: float, duration: float, trigger=None) -> torch.Tensor:
        ds = None
        if isinstance(trigger, dict) and "dataset" in trigger:
            ds = trigger["dataset"]
        if ds is None:
            d = events if isinstance(events, dict) else getattr(events, "to_dict", lambda: {})()
            ds = d.get("dataset", None)
        assert ds is not None, "Missing dataset in trigger/events."

        key = parse_key_from_dataset(ds)
        npy_path = self.root / f"{key}.npy"  # [T, L_full, D]
        assert npy_path.exists(), f"Missing cached video npy: {npy_path}"
        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape for {npy_path}, expect 3D, got {arr.shape}")
        lat_LDT = np.transpose(arr, (1, 2, 0))      # T,L,D -> L,D,T

        if (self.layer_aggregation or "").lower() == "group_mean":
            lat = group_mean_layers(lat_LDT, self.layers)   # [G, D, T]
        else:
            L_full = lat_LDT.shape[0]
            sel = sorted(set(int(round(f*(L_full-1))) for f in self.layers))
            lat = lat_LDT[sel]                              # [G, D, T]

        out = TimedArray(data=lat.astype(np.float32, copy=False), start=float(start), frequency=2.0)
        sub = out.overlap(start=start, duration=duration) or out.overlap(start=out.start, duration=0)
        return torch.from_numpy(sub.data)


class LLAMA3p2Cached(LLAMA3p2):
    EVENT_TYPE: ClassVar[str] = "Word"
    root: Path
    layers: List[float] = [0.5, 0.75, 1.0]
    layer_aggregation: Optional[str] = "group_mean"

    _eth: EventTypesHelper = pydantic.PrivateAttr()

    model_config = pydantic.ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._eth = EventTypesHelper(self.EVENT_TYPE)

    def prepare(self, obj):  # 不重算，只打通接口
        pass

    def __call__(self, events, start: float, duration: float, trigger=None) -> torch.Tensor:
        ds = None
        if isinstance(trigger, dict) and "dataset" in trigger:
            ds = trigger["dataset"]
        if ds is None:
            d = events if isinstance(events, dict) else getattr(events, "to_dict", lambda: {})()
            ds = d.get("dataset", None)
        assert ds is not None, "Missing dataset in trigger/events."

        key = parse_key_from_dataset(ds)
        npy_path = self.root / f"{key}.npy"  # [T, L_full, D]
        assert npy_path.exists(), f"Missing cached text npy: {npy_path}"
        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape for {npy_path}, expect 3D, got {arr.shape}")
        lat_LDT = np.transpose(arr, (1, 2, 0))

        if (self.layer_aggregation or "").lower() == "group_mean":
            lat = group_mean_layers(lat_LDT, self.layers)
        else:
            L_full = lat_LDT.shape[0]
            sel = sorted(set(int(round(f*(L_full-1))) for f in self.layers))
            lat = lat_LDT[sel]

        out = TimedArray(data=lat.astype(np.float32, copy=False), start=float(start), frequency=2.0)
        sub = out.overlap(start=start, duration=duration) or out.overlap(start=out.start, duration=0)
        return torch.from_numpy(sub.data)


class Wav2VecBertCached(Wav2VecBert):
    EVENT_TYPE: ClassVar[str] = "Sound"
    root: Path
    layers: List[float] = [0.5, 0.75, 1.0]
    layer_aggregation: Optional[str] = "group_mean"

    _eth: EventTypesHelper = pydantic.PrivateAttr()

    model_config = pydantic.ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._eth = EventTypesHelper(self.EVENT_TYPE)

    def prepare(self, obj):  # 不重算，只打通接口
        pass

    def __call__(self, events, start: float, duration: float, trigger=None) -> torch.Tensor:
        ds = None
        if isinstance(trigger, dict) and "dataset" in trigger:
            ds = trigger["dataset"]
        if ds is None:
            d = events if isinstance(events, dict) else getattr(events, "to_dict", lambda: {})()
            ds = d.get("dataset", None)
        assert ds is not None, "Missing dataset in trigger/events."

        key = parse_key_from_dataset(ds)
        npy_path = self.root / f"{key}.npy"  # [T, L_full, D]
        assert npy_path.exists(), f"Missing cached audio npy: {npy_path}"
        arr = np.load(npy_path)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected shape for {npy_path}, expect 3D, got {arr.shape}")
        lat_LDT = np.transpose(arr, (1, 2, 0))

        if (self.layer_aggregation or "").lower() == "group_mean":
            lat = group_mean_layers(lat_LDT, self.layers)
        else:
            L_full = lat_LDT.shape[0]
            sel = sorted(set(int(round(f*(L_full-1))) for f in self.layers))
            lat = lat_LDT[sel]

        out = TimedArray(data=lat.astype(np.float32, copy=False), start=float(start), frequency=2.0)
        sub = out.overlap(start=start, duration=duration) or out.overlap(start=out.start, duration=0)
        return torch.from_numpy(sub.data)


class FmriCached(Fmri):
    root: Path
    _eth: EventTypesHelper = pydantic.PrivateAttr()

    model_config = pydantic.ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self._eth = EventTypesHelper("Fmri")

    def prepare(self, obj):  # 不读磁盘，这里只打通接口
        pass

    def __call__(self, events, start: float, duration: float, trigger=None) -> torch.Tensor:
        ds = None
        if isinstance(trigger, dict) and "dataset" in trigger:
            ds = trigger["dataset"]
        if ds is None:
            d = events if isinstance(events, dict) else getattr(events, "to_dict", lambda: {})()
            ds = d.get("dataset", None)
        assert ds is not None, "Missing dataset in trigger/events."

        key = parse_key_from_dataset(ds)
        npy_path = self.root / f"{key}_fmri.npy"   # (T,n_vox) 或 (n_vox,T)
        assert npy_path.exists(), f"Missing fmri npy: {npy_path}"
        fmri = np.load(npy_path)
        if fmri.ndim != 2:
            raise ValueError(f"Unexpected fmri shape: {fmri.shape}")
        # 统一为 (n_vox, T)
        if fmri.shape[0] < fmri.shape[1]:
            fmri = fmri.T  # -> (T, n_vox)
        data = fmri.T.astype(np.float32, copy=False)  # (n_vox, T)

        out = TimedArray(data=data, start=float(start), frequency=2.0)
        sub = out.overlap(start=start, duration=duration) or out.overlap(start=out.start, duration=0)
        return torch.from_numpy(sub.data)


# ----------------- 自定义 Study：直接返回我们构造的 events -----------------
class PatchedStudy(StudyLoader):
    def __init__(self, path: str, events_df: pd.DataFrame):
        super().__init__(path=path)
        self._events_df = events_df

    def build(self) -> pd.DataFrame:
        return self._events_df


# ----------------- 主程序 -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", type=str, required=True)
    ap.add_argument("--val_list",   type=str, required=True)
    ap.add_argument("--video_root", type=str, required=True)
    ap.add_argument("--text_root",  type=str, required=True)
    ap.add_argument("--audio_root", type=str, required=True)
    ap.add_argument("--fmri_root",  type=str, required=True)
    ap.add_argument("--layers", type=str, default="0.5,0.75,1.0")
    ap.add_argument("--layer_aggregation", type=str, default="group_mean",
                    choices=["group_mean", "none", "None"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    train_ids = read_list(args.train_list)
    val_ids   = read_list(args.val_list)

    # 构造 events（每个 ds 4 条：Video/Sound/Word/Fmri）
    rows = []
    def add_rows(ds: str, split: str):
        for typ in ("Video", "Sound", "Word", "Fmri"):
            rows.append(dict(
                index=0,
                subject="sub-01",
                timeline="dummy",
                type=typ,
                dataset=ds,
                split=split,
                chunk=ds,            # 供 DeterministicSplitter 使用
                start=0.0,
                duration=1e9,       # 大窗口，具体对齐由 TimedArray.overlap 处理
                filepath="",         # 放空即可；我们已禁用 model_post_init
            ))
    for ds in train_ids: add_rows(ds, "train")
    for ds in val_ids:   add_rows(ds, "val")
    events = pd.DataFrame(rows)

    # Study：把我们的 events 交给官方 Data
    study = PatchedStudy(path=str(PROJ), events_df=events)

    # 缓存特征（严格保持官方的 layer 分段语义）
    fractions = [float(x) for x in args.layers.split(",") if x.strip()]
    agg = None if args.layer_aggregation.lower() in ("none", "null") else "group_mean"

    video_feat = VJEPA2Cached(root=Path(args.video_root), layers=fractions, layer_aggregation=agg)
    text_feat  = LLAMA3p2Cached(root=Path(args.text_root), layers=fractions, layer_aggregation=agg)
    audio_feat = Wav2VecBertCached(root=Path(args.audio_root), layers=fractions, layer_aggregation=agg)
    fmri_feat  = FmriCached(root=Path(args.fmri_root))

    # 官方 Data
    data = Data(
        study=study,
        neuro=fmri_feat,
        text_feature=text_feat,
        audio_feature=audio_feat,
        video_feature=video_feat,
        layers=fractions,
        layer_aggregation=agg,
        num_workers=args.num_workers,
    )

    # 官方模型配置
    brain_model_cfg = FmriEncoderConfig()

    # ===== Official-style configs: 用 dict 让 Pydantic 解析到正确类 =====
    # Loss
    loss_cfg = {"name": "MSELoss"}

    # Optimizer + Scheduler —— 严格套用 LightningOptimizerConfig schema
    optim_cfg = {
        "name": "LightningOptimizer",
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-3,
            "kwargs": {
                "weight_decay": 1e-2,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
            },
        },
        "scheduler": {
            "name": "CosineAnnealingLR",
            "kwargs": {
                "T_max": args.epochs,
                "eta_min": 1e-5,
            }
        },
    }

    metrics_cfg = [
        {"name": "PearsonCorrCoef", "log_name": "pearson"},
    ]

    # 输出目录 & 单卡本地运行
    from exca import TaskInfra
    run_dir = Path("/home/lawrence/Desktop/algonauts-2025/algonauts2025/outputs/cached_official")
    run_dir.mkdir(parents=True, exist_ok=True)

    xp = Experiment(
        data=data,
        brain_model_config=brain_model_cfg,
        loss=loss_cfg,
        optim=optim_cfg,
        metrics=metrics_cfg,
        monitor="val/pearson",
        accelerator="gpu",
        n_epochs=args.epochs,
        patience=None,
        enable_progress_bar=True,
        save_checkpoints=True,
        infra=TaskInfra(version="1", folder=str(run_dir), gpus_per_node=1),
    )

    xp.run()


if __name__ == "__main__":
    main()