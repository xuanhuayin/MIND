# -*- coding: utf-8 -*-
from __future__ import annotations
import typing as tp
import pydantic
import torch
from torch import nn
from einops import rearrange
# 首选：仓库根级的 modeling_utils
from modeling_utils.modeling_utils.models.common import MlpConfig, SubjectLayers

from modeling_utils.modeling_utils.models.transformer import TransformerEncoderConfig  # 包内相对导入

# from modeling_utils.models.transformer import TransformerEncoderConfig

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

class FmriEncoder(nn.Module):
    def __init__(
        self,
        feature_dims: dict[str, tuple[int, int]],
        n_outputs: int,
        n_output_timesteps: int,
        config: FmriEncoderConfig,
        n_input_timesteps_2hz: int | None = None,   # 新增：固定位置编码长度
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

        self.combiner = nn.Identity()
        self.predictor = SubjectLayers(
            in_channels=hidden,
            out_channels=n_outputs,
            n_subjects=config.n_subjects,
            average_subjects=False,
            bias=True,
        )

        # 位置编码长度：如果给了窗口内 2Hz 步数（固定不变），就按它建
        pe_len = n_input_timesteps_2hz if n_input_timesteps_2hz is not None else 1024
        self.time_pos_embed = nn.Parameter(torch.randn(1, pe_len, hidden))

        if config.subject_embedding and (config.n_subjects is not None):
            self.subject_embed = nn.Embedding(config.n_subjects, hidden)

        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

    def forward(self, batch, pool_outputs: bool = True) -> torch.Tensor:
        x = self.aggregate_features(batch)  # [B, T2, H]
        subject_id = batch.data.get("subject_id", None)
        x = self.transformer_forward(x, subject_id)
        x = x.transpose(1, 2)              # [B, H, T2]
        x = self.predictor(x, subject_id)  # [B, O, T2]
        return self.pooler(x) if pool_outputs else x

    def aggregate_features(self, batch):
        tensors = []
        # 拿到 B,T2
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
                data = torch.zeros(B, T2, 3072 // len(self.feature_dims), device=x0.device)
            else:
                data = batch.data[modality]  # [B, L, D, T2] 或 [B, D, T2]
                data = data.to(torch.float32)
                if data.ndim == 3:
                    data = data.unsqueeze(1)           # [B,1,D,T2]
                if self.config.layer_aggregation == "mean":
                    data = data.mean(dim=1)            # [B,D,T2]
                else:
                    data = rearrange(data, "b l d t -> b (l d) t")
                data = data.transpose(1, 2)            # [B,T2,D]
                # print(modality)
                # print(data.shape)
                data = self.projectors[modality](data) # [B,T2,H]
                if modality in drop:
                    data = torch.zeros_like(data)
            tensors.append(data)
        return torch.cat(tensors, dim=-1) if self.config.feature_aggregation == "cat" else sum(tensors)

    def transformer_forward(self, x, subject_id=None):
        x = self.combiner(x)
        if hasattr(self, "time_pos_embed"):
            # 位置编码按实际 T2 截断（我们已把 pe 长度设为固定窗口 T2，不会溢出）
            x = x + self.time_pos_embed[:, : x.size(1)]
        if hasattr(self, "subject_embed") and (subject_id is not None):
            x = x + self.subject_embed(subject_id)
        return self.encoder(x)