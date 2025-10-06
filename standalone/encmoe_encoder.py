# -*- coding: utf-8 -*-
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from modeling_utils.modeling_utils.models.common import MlpConfig
from modeling_utils.modeling_utils.models.transformer import TransformerEncoderConfig


class MoEFFNExpert(nn.Module):
    """
    Expert MLP: H -> (mult*H) -> ... -> H
    """
    def __init__(self, dim: int, hidden_mult: float = 2.0, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        layers = []
        if num_layers <= 1:
            layers += [nn.Dropout(dropout), nn.Linear(dim, dim)]
        else:
            h = int(dim * hidden_mult)
            for i in range(num_layers - 1):
                layers += [nn.Linear(dim if i == 0 else h, h), nn.GELU(), nn.Dropout(dropout)]
            layers += [nn.Linear(h, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B,T,H]
        return self.net(x)


class MoEAdapterLayer(nn.Module):
    """
    放在 Transformer 编码器后面的 MoE-FFN 适配层（不改动原 encoder 的内部结构）
    预归一化 + 残差: y = x + Mix(Experts(LN(x)))
    """
    def __init__(self, dim: int, num_experts: int = 4, top_k: int = 2,
                 expert_layers: int = 1, expert_hidden_mult: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.ln = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([
            MoEFFNExpert(dim=dim, hidden_mult=expert_hidden_mult,
                         num_layers=expert_layers, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.dropout = nn.Dropout(dropout)
        self.last_aux_loss = None  # per-layer

    def _mix_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        logits: [B,T,E] -> weights [B,T,E]，若 top_k<E 则对 top_k 重归一
        """
        probs = F.softmax(logits, dim=-1)           # [B,T,E]
        if self.top_k >= self.num_experts:
            return probs
        topv, topi = torch.topk(probs, self.top_k, dim=-1)  # [B,T,K]
        mask = torch.zeros_like(probs)                       # [B,T,E]
        mask.scatter_(-1, topi, topv)
        # 只用选中的 top-k，再做一次归一化
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return mask / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,H]
        """
        z = self.ln(x)
        logits = self.router(z)                       # [B,T,E]
        w = self._mix_weights(logits)                # [B,T,E]

        # 所有专家一次性前向: stack -> [B,T,E,H]
        outs = torch.stack([e(z) for e in self.experts], dim=2)
        mixed = torch.sum(outs * w.unsqueeze(-1), dim=2)  # [B,T,H]
        y = x + self.dropout(mixed)

        # 负载均衡辅助损失（Switch 风格近似）：E * sum(mean_prob^2)
        mean_p = w.reshape(-1, self.num_experts).mean(dim=0)     # [E]
        self.last_aux_loss = self.num_experts * torch.sum(mean_p * mean_p)
        return y


class FmriEncoder_EncMoE(nn.Module):
    """
    将 MoE-FFN 作为“适配层”追加在 Transformer 编码器后（不改动原 encoder）。
    输出：按 subject 的独立读出头 (4 个) -> [B, 4, N_TR, O]；默认 forward 返回单个 subject 的 [B,O,N_TR]
    """
    def __init__(self,
                 feature_dims: dict[str, tuple[int, int]],
                 n_outputs: int,
                 n_output_timesteps: int,
                 n_subjects: int = 4,
                 # feature fusion
                 feature_aggregation: str = "cat",
                 layer_aggregation: str = "cat",
                 subject_embedding: bool = True,
                 hidden: int = 3072,
                 # base encoder
                 encoder_depth: int = 8,
                 attn_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 layer_dropout: float = 0.0,
                 # MoE adapters
                 moe_n_layers: int = 2,
                 moe_num_experts: int = 4,
                 moe_top_k: int = 2,
                 moe_expert_layers: int = 1,
                 moe_hidden_mult: float = 2.0,
                 moe_dropout: float = 0.1,
                 pos_len_cap: int = 2048):
        super().__init__()
        assert feature_aggregation in ("cat", "sum")
        assert layer_aggregation in ("cat", "mean")
        self.feature_dims = feature_dims
        self.feature_aggregation = feature_aggregation
        self.layer_aggregation = layer_aggregation
        self.hidden = hidden
        self.n_outputs = n_outputs
        self.n_subjects = n_subjects
        self.moe_n_layers = moe_n_layers

        # -------- Projectors --------
        self.projectors = nn.ModuleDict()
        num_modalities = len(feature_dims)
        for modality, (num_layers, feat_dim) in feature_dims.items():
            in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
            out_dim = hidden if feature_aggregation == "sum" else (hidden // num_modalities)
            self.projectors[modality] = MlpConfig(norm_layer="layer", activation_layer="gelu",
                                                  dropout=0.0).build(in_dim, out_dim)
        if feature_aggregation == "cat":
            used = (hidden // num_modalities) * num_modalities
            if used != hidden:
                # 最后一个模态补齐
                last = list(self.projectors.keys())[-1]
                num_layers, feat_dim = feature_dims[last]
                in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
                delta = hidden - (hidden // num_modalities) * (num_modalities - 1)
                self.projectors[last] = MlpConfig(norm_layer="layer", activation_layer="gelu",
                                                  dropout=0.0).build(in_dim, delta)

        # -------- Pos / Subject --------
        self.time_pos_embed = nn.Parameter(torch.randn(1, max(n_output_timesteps * 2, pos_len_cap), hidden))
        if subject_embedding and n_subjects and n_subjects > 1:
            self.subject_embed = nn.Embedding(n_subjects, hidden)

        # -------- Base Transformer Encoder (unchanged) --------
        self.encoder = TransformerEncoderConfig(
            attn_dropout=attn_dropout, ff_dropout=ff_dropout, layer_dropout=layer_dropout, depth=encoder_depth
        ).build(dim=hidden)

        # -------- MoE Adapter blocks (after encoder) --------
        self.moe_adapters = nn.ModuleList([
            MoEAdapterLayer(dim=hidden,
                            num_experts=moe_num_experts,
                            top_k=moe_top_k,
                            expert_layers=moe_expert_layers,
                            expert_hidden_mult=moe_hidden_mult,
                            dropout=moe_dropout)
            for _ in range(moe_n_layers)
        ])

        # -------- Pooler (2Hz -> TR) --------
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        # -------- Subject readouts (vectorized) --------
        self.readout_W = nn.Parameter(torch.randn(n_subjects, n_outputs, hidden))
        self.readout_b = nn.Parameter(torch.zeros(n_subjects, n_outputs))
        nn.init.kaiming_uniform_(self.readout_W, a=5 ** 0.5)
        bound = 1.0 / (hidden ** 0.5)
        nn.init.uniform_(self.readout_b, -bound, bound)

        self.last_aux_loss = None  # aggregated over MoE layers

    # ---------- features ----------
    def _aggregate_features(self, batch) -> torch.Tensor:
        tensors = []
        for modality, _ in self.feature_dims.items():
            data = batch.data[modality].to(torch.float32)
            if data.ndim == 3:
                data = data.unsqueeze(1)  # [B,1,D,T]
            if self.layer_aggregation == "mean":
                data = data.mean(dim=1)   # [B,D,T]
            else:
                data = rearrange(data, "b l d t -> b (l d) t")  # [B, L*D, T]
            data = data.transpose(1, 2)  # [B, T, *]
            tensors.append(self.projectors[modality](data))  # -> [B,T,H_mod]
        x = torch.cat(tensors, dim=-1) if self.feature_aggregation == "cat" else sum(tensors)
        return x  # [B,T2,H]

    # ---------- forward (single subject) ----------
    def forward(self, batch, pool_outputs: bool = True):
        """
        返回单个 subject 的输出：[B, O, N_TR] 或 [B, O, T2]
        需要 batch.data["subject_id"] (LongTensor, [B])
        """
        x = self._aggregate_features(batch)      # [B,T2,H]
        T2 = x.size(1)
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} > pos_len={self.time_pos_embed.size(1)}")

        x = x + self.time_pos_embed[:, :T2]
        if hasattr(self, "subject_embed") and "subject_id" in batch.data:
            sid = batch.data["subject_id"]
            x = x + self.subject_embed(sid).unsqueeze(1)

        x = self.encoder(x)                      # [B,T2,H]
        # MoE adapters
        aux_losses = []
        for layer in self.moe_adapters:
            x = layer(x)
            if layer.last_aux_loss is not None:
                aux_losses.append(layer.last_aux_loss)
        self.last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None

        if pool_outputs:
            x_t = x.transpose(1, 2)                           # [B,H,T2]
            x_tr = self.pooler(x_t).transpose(1, 2)           # [B,N,H]
        else:
            x_tr = x                                          # [B,T2,H]
        # readout of selected subject
        sid = batch.data["subject_id"]                        # [B]
        W = self.readout_W[sid]                               # [B,O,H]
        b = self.readout_b[sid]                               # [B,O]
        y = torch.einsum("bnh,boh->bno", x_tr, W) + b.unsqueeze(1)  # [B,N,O]
        return y.transpose(1, 2) if pool_outputs else y.transpose(1, 2)  # [B,O,N or T2]

    # ---------- helper: hidden -> all heads ----------
    @torch.no_grad()
    def readout_all_heads(self, x_tr: torch.Tensor) -> torch.Tensor:
        """
        x_tr: [B,N,H] -> [B,4,N,O]
        """
        B, N, H = x_tr.shape
        W = self.readout_W.unsqueeze(0).expand(B, -1, -1, -1)     # [B,4,O,H]
        b = self.readout_b.unsqueeze(0).expand(B, -1, -1)         # [B,4,O]
        y = torch.einsum("bnh,bsoh->bsno", x_tr, W) + b.unsqueeze(2)  # [B,4,N,O]
        return y

    @torch.no_grad()
    def encode_to_tr(self, batch) -> torch.Tensor:
        """
        仅编码+MoE，返回 [B,N,H] （不使用 subject_id）
        注意：MoE 使用的是“共享”路由（未加 subject 偏置）；若你希望 subject 条件化，请分别 forward。
        """
        x = self._aggregate_features(batch)
        T2 = x.size(1)
        x = x + self.time_pos_embed[:, :T2]
        x = self.encoder(x)
        for layer in self.moe_adapters:
            x = layer(x)
        x_t = x.transpose(1, 2)
        return self.pooler(x_t).transpose(1, 2)  # [B,N,H]