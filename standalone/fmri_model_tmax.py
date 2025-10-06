# -*- coding: utf-8 -*-
from __future__ import annotations
import typing as tp
import pydantic
import torch
from torch import nn
from einops import rearrange

# 与官方仓库一致的依赖
from modeling_utils.models.common import MlpConfig, SubjectLayers
from modeling_utils.models.transformer import TransformerEncoderConfig


class FmriEncoderConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    name: tp.Literal["FmriEncoder"] = "FmriEncoder"
    n_subjects: int | None = None
    feature_aggregation: tp.Literal["sum", "cat"] = "cat"
    layer_aggregation: tp.Literal["mean", "cat"] = "cat"
    subject_embedding: bool = False
    modality_dropout: float = 0.0

    def build(
        self, feature_dims: dict[int], n_outputs: int, n_output_timesteps: int, pos_length: int
    ) -> nn.Module:
        return FmriEncoder(
            feature_dims,
            n_outputs,
            n_output_timesteps,
            pos_length=pos_length,
            config=self,
        )


class FmriEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        pos_length: int,  # 关键：固定时间位置编码长度
        config: FmriEncoderConfig,
    ):
        super().__init__()
        self.config = config
        self.feature_dims = feature_dims
        self.n_outputs = n_outputs
        self.projectors = nn.ModuleDict()
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        hidden = 3072

        # 每个模态一个投影头
        for modality, tup in feature_dims.items():
            if tup is None:
                print(f"Warning: {modality} has no feature dimensions. Skipping projector.")
                continue
            num_layers, feature_dim = tup
            in_dim = (feature_dim * num_layers) if config.layer_aggregation == "cat" else feature_dim
            out_dim = (hidden // len(feature_dims)) if config.feature_aggregation == "cat" else hidden
            self.projectors[modality] = MlpConfig(
                norm_layer="layer", activation_layer="gelu", dropout=0.0
            ).build(in_dim, out_dim)

        # 汇合后的通道维度
        _in_dim = ((hidden // len(feature_dims)) * len(feature_dims)) if config.feature_aggregation == "cat" else hidden
        self.combiner = nn.Identity()

        # 预测头（被试条件线性层）
        self.predictor = SubjectLayers(
            in_channels=hidden,
            out_channels=n_outputs,
            n_subjects=config.n_subjects,
            average_subjects=False,
            bias=True,
        )

        # 固定长度的位置编码参数（不再在训练中改形状）
        self.time_pos_embed = nn.Parameter(torch.randn(1, pos_length, hidden) * 0.02)

        if config.subject_embedding and config.n_subjects is not None:
            self.subject_embed = nn.Embedding(config.n_subjects, hidden)

        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

    def forward(self, batch, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)          # [B, T, H]
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id) # [B, T, H]
        x = x.transpose(1, 2)                       # [B, H, T]
        x = self.predictor(x, subject_id)           # [B, O, T]
        if pool_outputs:
            out = self.pooler(x)                    # [B, O, T']
        else:
            out = x
        return out

    def aggregate_features(self, batch):
        tensors = []
        # 取到 B,T
        for modality in batch.data.keys():
            if modality in self.feature_dims:
                ref = batch.data[modality]
                break
        B, T = ref.shape[0], ref.shape[-1]

        # 模态丢弃（论文里用过 p=0.2）
        modalities_to_dropout = []
        for modality in self.feature_dims.keys():
            if torch.rand(1).item() < self.config.modality_dropout and self.training:
                modalities_to_dropout.append(modality)
        if len(modalities_to_dropout) == len(self.feature_dims):
            modalities_to_dropout = modalities_to_dropout[:-1]

        for modality in self.feature_dims.keys():
            if modality not in self.projectors:
                data = torch.zeros(B, T, 3072 // len(self.feature_dims), device=ref.device)
            else:
                data = batch.data[modality]  # [B, L, D, T] 或 [B, D, T]
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)         # [B,1,D,T]
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)          # [B,D,T]
                elif self.config.layer_aggregation == "cat":
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)          # [B,T,D]
                data = self.projectors[modality](data)  # [B,T,H]
                if modality in modalities_to_dropout:
                    data = torch.zeros_like(data)
            tensors.append(data)

        if self.config.feature_aggregation == "cat":
            out = torch.cat(tensors, dim=-1)         # [B,T,H]
        else:
            out = sum(tensors)                       # [B,T,H]
        return out

    def transformer_forward(self, x, subject_id=None):
        # x: [B,T,H]，位置编码只做切片，不改变参数形状
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            T = x.size(1)
            x = x + self.time_pos_embed[:, :T]       # 仅切片
        if hasattr(self, "subject_embed") and subject_id is not None:
            x = x + self.subject_embed(subject_id)
        x = self.encoder(x)
        return x