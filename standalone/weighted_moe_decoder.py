# -*- coding: utf-8 -*-
# algonauts2025/standalone/weighted_moe_decoder.py

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

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
    多模态 -> Transformer 编码 -> (池化到TR) -> MoE 解码（专家输出的 weighted 加权相加）

    combine_mode:
      - "router":           仅用路由器 softmax 概率加权（随 token/时间步变化）
      - "learned":          仅用可学习的全局专家权重（可按 subject 加偏置）加权
      - "router_x_learned": 路由器概率 × 可学习权重，再归一化后加权（推荐）
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
                 expert_layers: int = 1,
                 expert_hidden_mult: float = 4.0,
                 combine_mode: str = "router",
                 subject_expert_bias: bool = False):
        super().__init__()
        assert feature_aggregation in ("cat", "sum")
        assert layer_aggregation in ("cat", "mean")
        assert combine_mode in ("router", "learned", "router_x_learned")

        self.feature_dims = feature_dims
        self.feature_aggregation = feature_aggregation
        self.layer_aggregation = layer_aggregation
        self.combine_mode = combine_mode

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
        # "cat" 模式下补齐 hidden
        if feature_aggregation == "cat":
            used = (hidden // num_modalities) * num_modalities
            if used != hidden:
                last = list(self.projectors.keys())[-1]
                num_layers, feat_dim = feature_dims[last]
                in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
                want = hidden - (hidden // num_modalities) * (num_modalities - 1)
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
        self.router = nn.Linear(hidden, self.num_experts)  # 仅在使用 router 的两种模式下生效
        self.experts = nn.ModuleList([
            _ExpertMLP(hidden, self.n_outputs,
                       layers=self.expert_layers,
                       hidden_mult=self.expert_hidden_mult,
                       dropout=moe_dropout)
            for _ in range(self.num_experts)
        ])

        # 可学习全局专家权重（logit）
        if self.combine_mode in ("learned", "router_x_learned"):
            self.expert_logit = nn.Parameter(torch.zeros(self.num_experts))
        if subject_expert_bias and self.n_subjects:
            self.subject_expert_bias = nn.Embedding(self.n_subjects, self.num_experts)
        else:
            self.subject_expert_bias = None

        self.last_aux_loss: torch.Tensor | None = None
        self._debug_last_weights_avg: torch.Tensor | None = None       # [E]  (Top-K 之后)
        self._debug_last_weights_pre_avg: torch.Tensor | None = None   # [E]  (Top-K 之前)

    # ---------- 多模态特征聚合 ----------
    def _aggregate_features(self, batch) -> torch.Tensor:
        tensors = []
        for modality, (num_layers, feat_dim) in self.feature_dims.items():
            data = batch.data[modality].to(torch.float32)   # [B,L,D,T] 或 [B,D,T]
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

    def _compute_router_probs(self, x_routed: torch.Tensor) -> torch.Tensor:
        logits = self.router(x_routed)                  # [B,N,E]
        return F.softmax(logits, dim=-1)                # [B,N,E]

    def _get_learned_weights(self, B: int, N: int, subject_id: torch.LongTensor | None) -> torch.Tensor:
        base = self.expert_logit
        if self.subject_expert_bias is not None and (subject_id is not None):
            bias = self.subject_expert_bias(subject_id)                     # [B,E]
            w_be = F.softmax(base.unsqueeze(0) + bias, dim=-1)              # [B,E]
            w = w_be.unsqueeze(1).expand(B, N, self.num_experts)            # [B,N,E]
        else:
            w_e = F.softmax(base, dim=-1)                                   # [E]
            w = w_e.view(1, 1, -1).expand(B, N, self.num_experts)           # [B,N,E]
        return w

    # ---------- 路由 & 解码（返回 experts_out、weights_final、weights_pre） ----------
    def _route_and_decode_with_experts(self, x_tr: torch.Tensor, subject_id: torch.LongTensor | None):
        """
        x_tr: [B, N, H]  返回:
          y: [B, N, O],
          weights_final: [B, N, E]   （真正参与加权、含 Top-K 稀疏化后的）
          experts_out:   [B, N, E, O]
          weights_pre:   [B, N, E]   （Top-K 之前的权重；router: probs；learned: learned；router_x_learned: mix归一化）
        """
        B, N, H = x_tr.shape

        if hasattr(self, "subject_embed") and (subject_id is not None):
            x_routed = x_tr + self.subject_embed(subject_id).unsqueeze(1)  # [B,N,H]
        else:
            x_routed = x_tr

        # 所有专家输出
        x_flat = x_tr.reshape(-1, H)                        # [B*N, H]
        experts_out = torch.stack(
            [expert(x_flat).reshape(B, N, self.n_outputs) for expert in self.experts],
            dim=2
        )  # [B, N, E, O]

        self.last_aux_loss = None

        if self.combine_mode == "router":
            probs_full = self._compute_router_probs(x_routed)         # [B,N,E]
            weights_pre = probs_full                                  # 这里 pre = probs

            if self.top_k is not None and self.top_k < self.num_experts:
                topk_probs, topk_idx = torch.topk(probs_full, self.top_k, dim=-1)
                topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
                weights_final = torch.zeros_like(probs_full)
                weights_final.scatter_(dim=-1, index=topk_idx, src=topk_probs)
            else:
                weights_final = probs_full

            # 负载均衡
            all_probs = probs_full.reshape(-1, self.num_experts)
            importance = all_probs.mean(dim=0)
            if self.top_k == 1:
                top1 = torch.argmax(probs_full, dim=-1)
                load = F.one_hot(top1.view(-1), num_classes=self.num_experts).float().mean(dim=0)
            else:
                load = probs_full.reshape(-1, self.num_experts).mean(dim=0)
            self.last_aux_loss = self.num_experts * torch.sum(load * importance)

        elif self.combine_mode == "learned":
            weights_pre = self._get_learned_weights(B, N, subject_id)  # learned 本身
            weights_final = weights_pre                                # 无路由 → 无 Top-K 稀疏化

        else:  # router_x_learned
            probs_full = self._compute_router_probs(x_routed)          # [B,N,E]
            learned_w  = self._get_learned_weights(B, N, subject_id)   # [B,N,E]
            mix = probs_full * learned_w                                # [B,N,E]
            weights_pre = mix / (mix.sum(dim=-1, keepdim=True) + 1e-8) # Top-K 之前的归一化混合权重

            if (self.top_k is not None) and (self.top_k < self.num_experts):
                topk_vals, topk_idx = torch.topk(mix, self.top_k, dim=-1)
                topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)
                weights_final = torch.zeros_like(mix)
                weights_final.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            else:
                weights_final = weights_pre  # 无稀疏化

            # 负载均衡项基于 probs_full
            all_probs = probs_full.reshape(-1, self.num_experts)
            importance = all_probs.mean(dim=0)
            if self.top_k == 1:
                top1 = torch.argmax(probs_full, dim=-1)
                load = F.one_hot(top1.view(-1), num_classes=self.num_experts).float().mean(dim=0)
            else:
                load = probs_full.reshape(-1, self.num_experts).mean(dim=0)
            self.last_aux_loss = self.num_experts * torch.sum(load * importance)

        # weighted 加权相加
        y = torch.einsum("bneo,bne->bno", experts_out, weights_final)  # [B,N,O]

        # 记录最近一次平均权重
        with torch.no_grad():
            # Top-K 之后（用于兼容旧接口）
            self._debug_last_weights_avg = weights_final.mean(dim=(0, 1)).detach().cpu()      # [E]
            # Top-K 之前（本次新增）
            self._debug_last_weights_pre_avg = weights_pre.mean(dim=(0, 1)).detach().cpu()    # [E]

        return y, weights_final, experts_out, weights_pre

    # ----- 兼容旧接口 -----
    def _route_and_decode(self, x_tr: torch.Tensor, subject_id: torch.LongTensor | None):
        y, w_final, _, _ = self._route_and_decode_with_experts(x_tr, subject_id)
        return y, w_final

    def forward(self, batch, pool_outputs: bool = True):
        x = self._aggregate_features(batch)                    # [B,T2,H]
        T2 = x.size(1)
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} 超出 time_pos_embed 长度 {self.time_pos_embed.size(1)}")
        x = x + self.time_pos_embed[:, :T2]
        x = self.encoder(x)                                    # [B,T2,H]

        subj = batch.data.get("subject_id", None) if hasattr(batch, "data") else None

        if pool_outputs:
            x_t = x.transpose(1, 2)                            # [B,H,T2]
            x_tr = self.pooler(x_t).transpose(1, 2)            # [B,N,H]
            y_bn, _ = self._route_and_decode(x_tr, subj)       # [B,N,O]
            return y_bn.transpose(1, 2)                        # [B,O,N]
        else:
            y_bt, _ = self._route_and_decode(x, subj)          # [B,T2,O]
            return y_bt.transpose(1, 2)                        # [B,O,T2]

    def forward_with_details(self, batch, pool_outputs: bool = True):
        x = self._aggregate_features(batch)
        T2 = x.size(1)
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} 超出 time_pos_embed 长度 {self.time_pos_embed.size(1)}")
        x = x + self.time_pos_embed[:, :T2]
        x = self.encoder(x)

        subj = batch.data.get("subject_id", None) if hasattr(batch, "data") else None

        if pool_outputs:
            x_t  = x.transpose(1, 2)
            x_tr = self.pooler(x_t).transpose(1, 2)            # [B,N,H]
            y_bn, w_final, experts_out, w_pre = self._route_and_decode_with_experts(x_tr, subj)
            return y_bn.transpose(1, 2), w_final, experts_out, w_pre  # [B,O,N], [B,N,E], [B,N,E,O], [B,N,E]
        else:
            y_bt, w_final, experts_out, w_pre = self._route_and_decode_with_experts(x, subj)
            return y_bt.transpose(1, 2), w_final, experts_out, w_pre

    # ----- 调试/统计接口 -----
    @torch.no_grad()
    def get_expert_weights(self, subject_id: torch.LongTensor | None = None):
        """
        返回静态权重快照：
          - combine_mode="router"：返回 None（权重随 token 变化，非静态）
          - "learned"/"router_x_learned"：返回 [E] 或 [B,E]（若传入 subject_id）
        """
        if getattr(self, "combine_mode", "router") == "router":
            return None
        if subject_id is None:
            return F.softmax(self.expert_logit, dim=-1).cpu()
        if getattr(self, "subject_expert_bias", None) is None:
            w = F.softmax(self.expert_logit, dim=-1).expand(subject_id.shape[0], -1)
            return w.cpu()
        bias = self.subject_expert_bias(subject_id)
        w = F.softmax(self.expert_logit.unsqueeze(0) + bias, dim=-1)
        return w.cpu()

    @torch.no_grad()
    def get_last_weight_avg(self):
        """返回最近一次 forward 的专家平均权重 [E]（按 B 与 N 平均，Top-K 之后）。若还未 forward，返回 None。"""
        return getattr(self, "_debug_last_weights_avg", None)

    @torch.no_grad()
    def get_last_weight_pre_avg(self):
        """返回最近一次 forward 的专家平均权重（Top-K 之前，weights_pre）[E]。若还未 forward，返回 None。"""
        return getattr(self, "_debug_last_weights_pre_avg", None)
