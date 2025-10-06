# -*- coding: utf-8 -*-
from __future__ import annotations
import typing as tp
import pydantic
import torch
from torch import nn
from einops import rearrange

# 首选：仓库根级的 modeling_utils
from modeling_utils.modeling_utils.models.common import MlpConfig
from modeling_utils.modeling_utils.models.transformer import TransformerEncoderConfig  # 包内相对导入


class FmriEncoderConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriEncoder"] = "FmriEncoder"
    n_subjects: int | None = None
    feature_aggregation: tp.Literal["sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    subject_embedding: bool = False
    modality_dropout: float = 0.0

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int
    ) -> nn.Module:
        return FmriEncoder(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            config=self,
        )


class SubjectConditionalLinear(nn.Module):
    """
    与 TRIBE 描述一致的 subject-conditional linear：
    - 若 n_subjects>1：为每个 subject 维护一套 (W_s, b_s)
      W: [S, O, H], b: [S, O]
      输入 x: [B, N, H], subject_id: [B] -> 输出 [B, N, O]
    - 若 n_subjects is None 或 1：退化为单一 Linear，作用在时间维上
    """
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int | None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_subjects = n_subjects if (n_subjects is not None and n_subjects > 1) else None

        if self.n_subjects is None:
            self.lin = nn.Linear(in_channels, out_channels, bias=True)
        else:
            S = self.n_subjects
            O, H = out_channels, in_channels
            # Kaiming/ Xavier 可按需改，这里用 pytorch 默认均匀初始化
            self.weight = nn.Parameter(torch.empty(S, O, H))
            self.bias = nn.Parameter(torch.empty(S, O))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            fan_in = H
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, N, H]; subject_id: [B] or None
        return: [B, N, O]
        """
        if self.n_subjects is None:
            # 单头：逐时间步线性
            return self.lin(x)  # [B, N, O]
        else:
            if subject_id is None:
                raise ValueError("SubjectConditionalLinear: subject_id is required when n_subjects>1.")
            # 取每个样本对应的 (W_s, b_s)
            # W: [B, O, H], b: [B, O]
            W = self.weight[subject_id]  # gather by subject
            b = self.bias[subject_id]
            # y[b, n, o] = sum_h x[b, n, h] * W[b, o, h] + b[b, o]
            y = torch.einsum("bnh,boh->bno", x, W) + b.unsqueeze(1)
            return y


class FmriEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoderConfig,
        n_input_timesteps_2hz: int | None = None,   # 固定位置编码长度（对应 2Hz 序列长度）
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs

        hidden = 3072  # 与 TRIBE 超参一致（见其附录表）
        self.hidden = hidden

        # === 每模态 MLP 投影（层聚合后 -> 投到统一隐藏维） ===
        self.projectors = nn.ModuleDict()
        for modality, tup in feature_dims.items():
            if tup is None:
                print(f"Warning: {modality} has no feature dimensions. Skipping projector.")
                continue
            num_layers, feature_dim = tup
            input_dim = (feature_dim * num_layers
                         if config.layer_aggregation == "cat" else feature_dim)
            output_dim = (hidden // len(feature_dims)
                          if config.feature_aggregation == "cat" else hidden)
            self.projectors[modality] = MlpConfig(
                norm_layer="layer", activation_layer="gelu", dropout=0.0
            ).build(input_dim, output_dim)

        # === 模态融合后时序编码 ===
        self.combiner = nn.Identity()

        # 时间位置编码长度：若提供窗口内 2Hz 步数，用其构建；否则给个上限（如 1024）
        pe_len = n_input_timesteps_2hz if n_input_timesteps_2hz is not None else 1024
        self.time_pos_embed = nn.Parameter(torch.randn(1, pe_len, hidden))

        # 可选：subject 嵌入（加法注入到每个时间步）
        if config.subject_embedding and (config.n_subjects is not None):
            self.subject_embed = nn.Embedding(config.n_subjects, hidden)

        # Transformer（与 TRIBE 一致：8 层）
        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

        # === 与 TRIBE 对齐的下游头：先池化到每 TR，再做 subject-conditional 线性投影 ===
        # 自适应平均池化：把 2Hz 长度（T2）压到 N = n_output_timesteps（每 TR 一步）
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        # subject-conditional 线性层：从 hidden -> n_outputs（1000 parcels）
        self.pred_head = SubjectConditionalLinear(
            in_channels=hidden, out_channels=n_outputs, n_subjects=config.n_subjects
        )

    # -------- 公共前向 --------
    def forward(self, batch, pool_outputs: bool = True) -> torch.Tensor:
        """
        返回形状：
        - 默认（pool_outputs=True）：[B, O, N_TR]   —— 与 TRIBE：每 TR 一步的 fMRI 预测
        - 若 False：返回未池化的每 2Hz 步预测（与下游兼容时可按需改，这里保持与设计一致）
        """
        # 1) 融合三模态，得到 [B, T2, H]
        x = self.aggregate_features(batch)

        # 2) 时序编码（加时间/subject 位置嵌入 + Transformer）
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id=subject_id)   # [B, T2, H]

        # 3) TRIBE 顺序：先池化到每 TR（在通道前，故先转 [B,H,T2]）
        x = x.transpose(1, 2)                       # [B, H, T2]
        if pool_outputs:
            x_tr = self.pooler(x)                   # [B, H, N_TR]
            x_tr = x_tr.transpose(1, 2)            # [B, N_TR, H]
            # 4) subject-conditional 线性：逐 TR 投到 parcels
            y = self.pred_head(x_tr, subject_id)   # [B, N_TR, O]
            return y.transpose(1, 2)               # [B, O, N_TR]
        else:
            # 不池化的分支（可选）：直接逐 2Hz 步做 subject-conditional 线性
            x2 = x.transpose(1, 2)                 # [B, T2, H]
            y2 = self.pred_head(x2, subject_id)    # [B, T2, O]
            return y2.transpose(1, 2)              # [B, O, T2]

    # -------- 三模态聚合（与原逻辑一致）--------
    def aggregate_features(self, batch):
        tensors = []
        # 拿到 B,T2：从任一已有模态获取
        x0 = next(v for k, v in batch.data.items() if k in self.feature_dims)
        B, T2 = x0.shape[0], x0.shape[-1]

        # 随机丢模态（训练中）
        drop = []
        for m in self.feature_dims.keys():
            if torch.rand(1).item() < self.config.modality_dropout and self.training:
                drop.append(m)
        if len(drop) == len(self.feature_dims):
            drop = drop[:-1]

        for modality in self.feature_dims.keys():
            if modality not in self.projectors:
                data = torch.zeros(B, T2, self.hidden // len(self.feature_dims), device=x0.device)
            else:
                data = batch.data[modality]  # [B,L,D,T2] 或 [B,D,T2]
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)           # [B,1,D,T2]
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)            # [B,D,T2]
                else:
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)            # [B,T2,D]
                data = self.projectors[modality](data) # [B,T2,H_m]
                if modality in drop:
                    data = torch.zeros_like(data)
            tensors.append(data)

        if self.config.feature_aggregation == "cat":
            x = torch.cat(tensors, dim=-1)             # [B,T2,hidden]
        else:
            x = sum(tensors)                           # [B,T2,hidden]
        return x

    # -------- Transformer 前向（与原逻辑一致，但返回 [B,T2,H]）--------
    def transformer_forward(self, x, subject_id=None):
        # x: [B, T2, H]
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed") and (subject_id is not None):
            # subject_embed: [S,H] -> 广播到 [B,T2,H]
            x = x + self.subject_embed(subject_id).unsqueeze(1)
        x = self.encoder(x)                            # [B, T2, H]
        return x