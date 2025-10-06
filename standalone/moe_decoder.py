# -*- coding: utf-8 -*-
# algonauts2025/standalone/moe_decoder.py

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# 这两个来自你的仓库
from modeling_utils.modeling_utils.models.common import MlpConfig
from modeling_utils.modeling_utils.models.transformer import TransformerEncoderConfig


class _ExpertMLP(nn.Module):
    """
    单个专家的解码 MLP：
      layers=1:  Linear(H -> O)
      layers>=2: Linear(H -> H*mult) + (L-2 个隐藏层 H*mult) + Linear(H*mult -> O)
    """
    def __init__(self, in_dim: int, out_dim: int, layers: int = 1,
                 hidden_mult: float = 4.0, dropout: float = 0.1):
        super().__init__()
        layers = max(1, int(layers))
        if layers == 1:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, out_dim))
        else:
            h = int(round(in_dim * hidden_mult))
            blocks = [nn.Dropout(dropout), nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            for _ in range(layers - 2):
                blocks += [nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout)]
            blocks += [nn.Linear(h, out_dim)]
            self.net = nn.Sequential(*blocks)

    def forward(self, x):  # x: [..., H]
        return self.net(x)  # [..., O]


class FmriEncoder_MoE(nn.Module):
    """
    多模态 -> Transformer 编码 -> (池化到TR) -> MoE 解码
    - 路由：Linear(H->E)，支持 top_k 稀疏门控；top-k 概率会重新归一化
    - 专家：每个专家是可调层数的 MLP，将 H -> O
    - subject_embedding: 若提供 subject_id，会在路由前将其加到隐藏态上（实现“被试条件化”的路由差异）
    - last_aux_loss: 负载均衡辅助损失（switch风格近似）

    返回：
      forward(..., pool_outputs=True) -> [B, O, N_TR]
      forward(..., pool_outputs=False) -> [B, O, T2]
    """
    def __init__(self,
                 feature_dims: dict[str, tuple[int, int]],
                 n_outputs: int,
                 n_output_timesteps: int,
                 n_subjects: int | None = None,
                 num_experts: int = 4,
                 top_k: int = 1,
                 feature_aggregation: str = "cat",
                 layer_aggregation: str = "cat",
                 subject_embedding: bool = False,
                 moe_dropout: float = 0.1,
                 # 新增：专家 MLP 深度与宽度
                 expert_layers: int = 1,
                 expert_hidden_mult: float = 4.0):
        super().__init__()
        assert feature_aggregation in ("cat", "sum")
        assert layer_aggregation in ("cat", "mean")
        self.feature_dims = feature_dims
        self.feature_aggregation = feature_aggregation
        self.layer_aggregation = layer_aggregation

        self.n_subjects = n_subjects if (n_subjects is not None and n_subjects > 1) else None
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_layers = int(max(1, expert_layers))
        self.expert_hidden_mult = float(expert_hidden_mult)

        hidden = 3072
        self.hidden = hidden
        self.n_outputs = int(n_outputs)

        # ---------- 各模态投影 ----------
        self.projectors = nn.ModuleDict()
        num_modalities = len(feature_dims)
        for modality, (num_layers, feat_dim) in feature_dims.items():
            in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
            out_dim = hidden if feature_aggregation == "sum" else (hidden // num_modalities)
            self.projectors[modality] = MlpConfig(
                norm_layer="layer", activation_layer="gelu", dropout=0.0
            ).build(in_dim, out_dim)
        # 若 "cat" 下最后一模态补齐 hidden
        if feature_aggregation == "cat":
            used = (hidden // num_modalities) * num_modalities
            if used != hidden:
                last = list(self.projectors.keys())[-1]
                # 重新构建最后一个投影到补齐维度
                want = hidden - (hidden // num_modalities) * (num_modalities - 1)
                num_layers, feat_dim = feature_dims[last]
                in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
                self.projectors[last] = MlpConfig(
                    norm_layer="layer", activation_layer="gelu", dropout=0.0
                ).build(in_dim, want)

        # ---------- 位置 / 受试者嵌入 ----------
        max_T2 = max(n_output_timesteps * 2, 1024)
        self.time_pos_embed = nn.Parameter(torch.randn(1, max_T2, hidden))
        if subject_embedding and self.n_subjects:
            self.subject_embed = nn.Embedding(self.n_subjects, hidden)

        # ---------- 主干 Transformer ----------
        self.encoder = TransformerEncoderConfig(
            attn_dropout=0.0, ff_dropout=0.0, layer_dropout=0.0, depth=8
        ).build(dim=hidden)

        # ---------- 2Hz -> TR 池化 ----------
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        # ---------- MoE：路由 + 专家 ----------
        self.router = nn.Linear(hidden, self.num_experts)
        self.experts = nn.ModuleList([
            _ExpertMLP(hidden, self.n_outputs,
                       layers=self.expert_layers,
                       hidden_mult=self.expert_hidden_mult,
                       dropout=moe_dropout)
            for _ in range(self.num_experts)
        ])

        self.last_aux_loss: torch.Tensor | None = None  # 训练循环里会读

    # ---------- 多模态特征聚合 ----------
    def _aggregate_features(self, batch) -> torch.Tensor:
        tensors = []
        for modality, (num_layers, feat_dim) in self.feature_dims.items():
            data = batch.data[modality].to(torch.float32)   # [B, L, D, T] 或 [B, D, T]
            if data.ndim == 3:
                data = data.unsqueeze(1)                    # [B,1,D,T]
            if self.layer_aggregation == "mean":
                data = data.mean(dim=1)                     # [B,D,T]
            else:
                data = rearrange(data, "b l d t -> b (l d) t")
            data = data.transpose(1, 2)                     # [B,T,*]
            proj = self.projectors[modality](data)          # [B,T,H_mod]
            tensors.append(proj)
        x = torch.cat(tensors, dim=-1) if self.feature_aggregation == "cat" else sum(tensors)
        return x  # [B, T2, H]

    # ---------- 核心：从隐藏态路由 & 解码 ----------
    def _route_and_decode(self, x_tr: torch.Tensor, subject_id: torch.LongTensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_tr: [B, N, H]（已经池化到 TR）
        subject_id: [B] 或 None；若提供且有 subject_embed，会在路由前加偏置
        返回:
          y: [B, N, O]
          probs: [B, N, E]（用于辅助损失/调试）
        """
        B, N, H = x_tr.shape

        if hasattr(self, "subject_embed") and (subject_id is not None):
            x_routed = x_tr + self.subject_embed(subject_id).unsqueeze(1)  # [B,N,H]
        else:
            x_routed = x_tr

        logits = self.router(x_routed)                 # [B,N,E]
        probs_full = F.softmax(logits, dim=-1)         # [B,N,E]

        if self.top_k is None or self.top_k >= self.num_experts:
            probs = probs_full
            topk_idx = None
        else:
            topk_probs, topk_idx = torch.topk(probs_full, self.top_k, dim=-1)     # [B,N,K]
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
            # 把稀疏权重放回 E 维度（便于后续辅助损失同一口径）
            probs = torch.zeros_like(probs_full)
            probs.scatter_(dim=-1, index=topk_idx, src=topk_probs)

        # 计算所有专家的输出，再按 probs 加权求和
        x_flat = x_tr.reshape(-1, H)                   # [B*N, H]
        outs = []
        for expert in self.experts:
            y_e = expert(x_flat).reshape(B, N, self.n_outputs)  # [B,N,O]
            outs.append(y_e)
        experts_out = torch.stack(outs, dim=2)         # [B,N,E,O]
        y = torch.einsum("bneo,bne->bno", experts_out, probs)   # [B,N,O]

        # 负载均衡辅助损失（近似 Switch）
        all_probs = probs_full.reshape(-1, self.num_experts)    # 使用软概率的“重要性”
        importance = all_probs.mean(dim=0)                      # p_e
        if self.top_k == 1:
            top1 = torch.argmax(probs_full, dim=-1)             # [B,N]
            load = F.one_hot(top1.view(-1), num_classes=self.num_experts).float().mean(dim=0)  # f_e
        else:
            # 选中并归一化后的权重作为 load 近似
            load = probs.reshape(-1, self.num_experts).mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(load * importance)

        return y, probs_full

    # ---------- 前向 ----------
    def forward(self, batch, pool_outputs: bool = True):
        # 1) 聚合 -> 2) 时间位置嵌入 -> 3) Transformer
        x = self._aggregate_features(batch)                    # [B,T2,H]
        T2 = x.size(1)
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} 超出 time_pos_embed 长度 {self.time_pos_embed.size(1)}")
        x = x + self.time_pos_embed[:, :T2]
        x = self.encoder(x)                                    # [B,T2,H]

        if pool_outputs:
            # 4) 池化到每个 TR
            x_t = x.transpose(1, 2)                            # [B,H,T2]
            x_tr = self.pooler(x_t).transpose(1, 2)            # [B,N,H]
            subj = batch.data.get("subject_id", None) if isinstance(batch, object) else None
            y_bn, _ = self._route_and_decode(x_tr, subj)       # [B,N,O]
            return y_bn.transpose(1, 2)                        # [B,O,N]
        else:
            # 直接在 2Hz 序列上 MoE（不常用）
            subj = batch.data.get("subject_id", None) if isinstance(batch, object) else None
            y_bt, _ = self._route_and_decode(x, subj)          # [B,T2,O]
            return y_bt.transpose(1, 2)                        # [B,O,T2]