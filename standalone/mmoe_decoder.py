# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from modeling_utils.modeling_utils.models.common import MlpConfig
from modeling_utils.modeling_utils.models.transformer import TransformerEncoderConfig


class ExpertBlock(nn.Module):
    """
    共享专家：H -> H 的小 MLP（可设层数/dropout），默认两层
    """
    def __init__(self, dim: int, hidden_mult: float = 4.0, dropout: float = 0.0, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = dim
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, int(dim * hidden_mult)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(dim * hidden_mult), dim),
                nn.Dropout(dropout),
            ]
            in_dim = dim
        # 若只一层则为恒等映射；上面已经构建 num_layers-1 个“残块”，再接一层 LN 稳定
        self.net = nn.Sequential(*layers) if layers else nn.Identity()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, H]
        if isinstance(self.net, nn.Identity):
            return x
        return self.ln(self.net(x) + x)


class FmriEncoder_MMoE(nn.Module):
    """
    共享专家 + 多门门控（MMoE）的 fMRI 编码器。
    - 专家：共享的 H->H 专家网络（不直接到 O）
    - 门控：每个 head（或每个 subject）一套 router，给出 E 个专家权重
    - 读出：每个 head（或每个 subject）各自 Linear(H->O)

    两种 head 模式：
      1) head_mode='per_subject' & n_subjects>1：每位被试一套 gate + 一套读出
      2) head_mode='multi'：手动指定 n_heads（如多任务）

    返回：
      - per_subject：y ∈ [B, O, N_TR]（按 batch 中的 subject_id 选对应 head）
      - multi：y ∈ [B, Hds, O, N_TR]（Hds = n_heads）
    """
    def __init__(self,
                 feature_dims: dict[str, tuple[int, int]],
                 n_outputs: int,
                 n_output_timesteps: int,
                 n_subjects: int | None = None,
                 feature_aggregation: str = "cat",
                 layer_aggregation: str = "cat",
                 subject_embedding: bool = False,
                 # --- MMoE 关键参数 ---
                 num_experts: int = 4,
                 expert_layers: int = 2,
                 expert_hidden_mult: float = 4.0,
                 expert_dropout: float = 0.0,
                 gate_top_k: int | None = None,   # None=用全部专家；>0=稀疏 top-k
                 # --- 多头模式 ---
                 head_mode: str = "per_subject",  # 'per_subject' | 'multi'
                 n_heads: int | None = None,      # head_mode='multi' 时必填
                 # --- 其他 ---
                 hidden: int = 3072,
                 attn_dropout: float = 0.0,
                 ff_dropout: float = 0.0,
                 layer_dropout: float = 0.0,
                 pos_len_cap: int = 1024):
        super().__init__()
        assert feature_aggregation in ("cat", "sum")
        assert layer_aggregation in ("cat", "mean")
        assert head_mode in ("per_subject", "multi")

        self.feature_dims = feature_dims
        self.feature_aggregation = feature_aggregation
        self.layer_aggregation = layer_aggregation
        self.hidden = hidden
        self.n_outputs = n_outputs
        self.n_subjects = n_subjects if (n_subjects is not None and n_subjects > 1) else None
        self.num_experts = num_experts
        self.gate_top_k = gate_top_k

        # ---------- 投影层（各模态） ----------
        self.projectors = nn.ModuleDict()
        num_modalities = len(feature_dims)
        for modality, (num_layers, feat_dim) in feature_dims.items():
            in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
            out_dim = hidden if feature_aggregation == "sum" else (hidden // num_modalities)
            self.projectors[modality] = MlpConfig(norm_layer="layer",
                                                  activation_layer="gelu",
                                                  dropout=0.0).build(in_dim, out_dim)
        # 若 "cat" 且不能整除，给最后一个模态补齐
        if feature_aggregation == "cat":
            used = (hidden // num_modalities) * num_modalities
            if used != hidden:
                last = list(self.projectors.keys())[-1]
                delta = hidden - (hidden // num_modalities) * (num_modalities - 1)
                # 重新建最后一个
                modality = last
                num_layers, feat_dim = feature_dims[modality]
                in_dim = feat_dim * num_layers if layer_aggregation == "cat" else feat_dim
                self.projectors[modality] = MlpConfig(norm_layer="layer",
                                                      activation_layer="gelu",
                                                      dropout=0.0).build(in_dim, delta)

        # ---------- 位置 & 受试者嵌入 ----------
        max_T2 = max(n_output_timesteps * 2, pos_len_cap)
        self.time_pos_embed = nn.Parameter(torch.randn(1, max_T2, hidden))
        if subject_embedding and self.n_subjects:
            self.subject_embed = nn.Embedding(self.n_subjects, hidden)

        # ---------- 主干 Transformer ----------
        self.encoder = TransformerEncoderConfig(
            attn_dropout=attn_dropout, ff_dropout=ff_dropout, layer_dropout=layer_dropout, depth=8
        ).build(dim=hidden)

        # ---------- 2Hz -> TR 池化 ----------
        self.pooler = nn.AdaptiveAvgPool1d(n_output_timesteps)

        # ---------- 共享专家 ----------
        self.experts = nn.ModuleList([
            ExpertBlock(dim=hidden,
                        hidden_mult=expert_hidden_mult,
                        dropout=expert_dropout,
                        num_layers=expert_layers)
            for _ in range(num_experts)
        ])

        # ---------- 多门 & 头部 ----------
        if head_mode == "per_subject":
            assert self.n_subjects is not None, "per_subject 模式需要 n_subjects>1"
            self.head_count = self.n_subjects
        else:
            assert n_heads is not None and n_heads > 0, "multi 模式需要指定 n_heads>0"
            self.head_count = n_heads

        # 每个 head 一个 router(H->E) 和 一个读出头(H->O)
        self.routers = nn.ModuleList([nn.Linear(hidden, num_experts) for _ in range(self.head_count)])
        self.readouts = nn.ModuleList([nn.Linear(hidden, n_outputs) for _ in range(self.head_count)])

        # 记录辅助损失（每个 head 一份，再求和）
        self.last_aux_loss = None

    # ---------- 公共：聚合多模态 ----------
    def _aggregate_features(self, batch):
        tensors = []
        any_mod = next(iter(self.feature_dims))
        data_any = batch.data[any_mod]
        # 允许 [B,L,D,T2] 或 [B,D,T2]
        for modality, (num_layers, feat_dim) in self.feature_dims.items():
            data = batch.data[modality].to(torch.float32)
            if data.ndim == 3:
                data = data.unsqueeze(1)  # [B,1,D,T2]
            if self.layer_aggregation == "mean":
                data = data.mean(dim=1)   # [B,D,T2]
            else:  # cat
                data = rearrange(data, "b l d t -> b (l d) t")
            data = data.transpose(1, 2)  # [B,T2,*]
            proj = self.projectors[modality](data)  # [B,T2,H_mod]
            tensors.append(proj)
        x = torch.cat(tensors, dim=-1) if self.feature_aggregation == "cat" else sum(tensors)
        return x  # [B, T2, H]

    # ---------- 核心：一次性跑专家，再按各自 gate 混合 ----------
    def _experts_forward(self, x):
        """
        x: [B, S, H]  (S 为时间步：T2 或 N_TR)
        return: experts_out: [B, S, E, H]
        """
        B, S, H = x.shape
        # 堆叠后一次性并行
        outs = []
        for expert in self.experts:
            outs.append(expert(x))  # [B,S,H]
        experts_out = torch.stack(outs, dim=2)  # [B,S,E,H]
        return experts_out

    def _mix_by_gate(self, experts_out, router, x, top_k=None):
        """
        experts_out: [B,S,E,H]
        router: nn.Linear(H->E)
        x: [B,S,H] 用于产生 gate logits
        return: mixed: [B,S,H], probs: [B,S,E]
        """
        logits = router(x)                  # [B,S,E]
        probs = F.softmax(logits, dim=-1)   # [B,S,E]
        if top_k is not None and top_k < self.num_experts:
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)  # [B,S,K]
            # 稀疏加权：只保留 top-k
            B, S, K = topk_idx.shape
            # 取出对应专家输出
            # gather: [B,S,K,H]
            gathered = experts_out.gather(dim=2, index=topk_idx[..., None].expand(B, S, K, experts_out.size(-1)))
            # 归一化（可选）：top-k 概率本身已是 softmax 的前 k 项，无需再归一；若想严格稀疏，可再 renorm
            mixed = torch.sum(gathered * topk_probs[..., None], dim=2)  # [B,S,H]
        else:
            # 全量混合：einsum 更高效
            mixed = torch.einsum("bseh, bse -> bsh", experts_out, probs)  # [B,S,H]
        return mixed, probs

    def forward(self, batch, pool_outputs: bool = True):
        """
        per_subject:
          输入需包含 batch.data["subject_id"] (LongTensor, [B])
          返回 y ∈ [B, O, N_TR] 或 [B, O, T2]
        multi:
          返回 y ∈ [B, Hds, O, N_TR] 或 [B, Hds, O, T2]
        """
        # 1) 特征聚合
        x = self._aggregate_features(batch)        # [B,T2,H]
        T2 = x.size(1)

        # 2) 位置/受试者嵌入
        if T2 > self.time_pos_embed.size(1):
            raise RuntimeError(f"T2={T2} 超出 time_pos_embed 长度 {self.time_pos_embed.size(1)}，"
                               f"请增大 pos_len_cap 或改用可扩展位置编码。")
        x = x + self.time_pos_embed[:, :T2]
        if hasattr(self, "subject_embed") and "subject_id" in batch.data:
            subj_ids = batch.data["subject_id"]  # [B]
            x = x + self.subject_embed(subj_ids).unsqueeze(1)

        # 3) 主干编码
        x = self.encoder(x)  # [B,T2,H]

        # 4) 是否池化到 TR
        if pool_outputs:
            x_t = x.transpose(1, 2)           # [B,H,T2]
            x_p = self.pooler(x_t).transpose(1, 2)  # [B,N_TR,H]
            seq = x_p
        else:
            seq = x                            # [B,T2,H]

        B, S, H = seq.shape

        # 5) 一次性跑共享专家
        experts_out = self._experts_forward(seq)  # [B,S,E,H]

        aux_losses = []
        if self.head_count == 1 and self.n_subjects is None and hasattr(self, "routers"):
            # 理论上不会走到这（head_count 至少是 1）；保留占位
            pass

        if self.n_subjects is not None and self.head_count == self.n_subjects and hasattr(self, "routers"):
            # ---- per_subject：按 batch 中 subject_id 为每个样本选择对应 head ----
            subj_ids = batch.data.get("subject_id", None)
            if subj_ids is None:
                raise ValueError("per_subject 模式需要 batch.data['subject_id']")
            # 为每个样本选择其 router 和 readout
            y = torch.zeros(B, S, self.n_outputs, device=seq.device, dtype=seq.dtype)

            # 逐 unique subject 向量化计算，避免逐样本 Python 循环
            uniq_subj = torch.unique(subj_ids)
            for sid in uniq_subj.tolist():
                mask = (subj_ids == sid)                  # [B]
                if not mask.any():
                    continue
                x_s = seq[mask]                           # [B_s, S, H]
                e_s = experts_out[mask]                   # [B_s, S, E, H]
                router = self.routers[sid]
                head = self.readouts[sid]
                mixed, probs = self._mix_by_gate(e_s, router, x_s, self.gate_top_k)  # [B_s,S,H], [B_s,S,E]
                y_s = head(mixed)                         # [B_s,S,O]
                y[mask] = y_s

                # 辅助损失（importance 近似）：E * sum_e (mean_prob_e^2)
                p = probs.reshape(-1, self.num_experts).mean(dim=0)  # [E]
                aux_losses.append(self.num_experts * torch.sum(p * p))

            y = y.transpose(1, 2)  # [B,O,S]
            self.last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
            return y  # [B,O,N_TR] 或 [B,O,T2]

        # ---- multi：所有 head 都要输出（堆叠在一维）----
        all_y = []
        for h in range(self.head_count):
            mixed, probs = self._mix_by_gate(experts_out, self.routers[h], seq, self.gate_top_k)  # [B,S,H]
            y_h = self.readouts[h](mixed)  # [B,S,O]
            all_y.append(y_h.transpose(1, 2))  # [B,O,S]

            p = probs.reshape(-1, self.num_experts).mean(dim=0)
            aux_losses.append(self.num_experts * torch.sum(p * p))

        self.last_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        y = torch.stack(all_y, dim=1)  # [B,Hds,O,S]
        return y