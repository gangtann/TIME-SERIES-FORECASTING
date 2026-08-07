"""DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting (KDD 2025).

官方仓库: https://github.com/decisionintelligence/DUET
本体: ts_benchmark/baselines/duet/models/duet_model.py DUETModel

移植改动:
1. forward 适配本框架四参签名 (x_enc, x_mark_enc, x_dec, x_mark_dec)，仅使用 x_enc
   （官方 DUETModel.forward(input) 只吃单输入，其余参数忽略）。
2. 辅助损失通路: 官方 wrapper（duet.py _process + deep_forecasting_model_base.py
   L342-349）训练时把 MoE 均衡损失 L_importance 直接加进 criterion（无额外系数，
   cluster.forward 的 loss_coef 默认 1）。此处训练态返回 (output, L_importance) 元组
   由本框架 exp 解包相加，eval 态只返回 output。
3. 模型专属超参加 duet_ 前缀（duet_num_experts / duet_k / duet_hidden_size /
   duet_fc_dropout / duet_noisy_gating / duet_CI / duet_moving_avg，见
   configs/default.yaml duet 节）；通用超参（d_model/d_ff/n_heads/e_layers/factor/
   dropout/activation）复用 model 节。
4. output_attention 固定 False（官方配方 output_attention=0，本框架不消费注意力图）。
5. 全部专属依赖层并入 layers/DUET_layers.py（官方自带 RevIN 等与本框架同名文件
   不复用，类名加 DUET 前缀防撞）；device 修正见该文件头注释。
"""

import torch
import torch.nn as nn
from einops import rearrange

from layers.DUET_layers import (
    DUETLinearExtractorCluster,
    DUETMahalanobisMask,
    DUETEncoder,
    DUETEncoderLayer,
    DUETAttentionLayer,
    DUETFullAttention,
)


class Model(nn.Module):
    """DUET: 时间维 MoE 聚类（噪声 top-k 门控的 DLinear 专家簇）+
    通道维频域马氏距离掩码的稀疏 channel transformer。

    数据流（[B, L, C] 输入）:
      CI 展开 (b n) l 1 → MoE 专家簇（内含 RevIN norm）→ [B, d_model, C]
      → 频域马氏 mask 生成通道 0/1 掩码 → masked channel transformer（通道为 token）
      → 线性头 [B, C, pred_len] → 转置 → RevIN denorm → [B, pred_len, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.cluster = DUETLinearExtractorCluster(configs)
        self.CI = configs.duet_CI
        self.n_vars = configs.enc_in
        self.mask_generator = DUETMahalanobisMask(configs.seq_len)
        self.Channel_transformer = DUETEncoder(
            [
                DUETEncoderLayer(
                    DUETAttentionLayer(
                        DUETFullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.linear_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.duet_fc_dropout),
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        # x_enc: [batch_size, seq_len, n_vars]
        if self.CI:
            channel_independent_input = rearrange(x_enc, "b l n -> (b n) l 1")
            reshaped_output, L_importance = self.cluster(channel_independent_input)
            temporal_feature = rearrange(
                reshaped_output, "(b n) l 1 -> b l n", b=x_enc.shape[0]
            )
        else:
            temporal_feature, L_importance = self.cluster(x_enc)

        # B x d_model x n_vars -> B x n_vars x d_model
        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")
        if self.n_vars > 1:
            changed_input = rearrange(x_enc, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)
            channel_group_feature, _ = self.Channel_transformer(
                x=temporal_feature, attn_mask=channel_mask
            )
            output = self.linear_head(channel_group_feature)
        else:
            output = self.linear_head(temporal_feature)

        output = rearrange(output, "b n d -> b d n")  # [B, pred_len, n_vars]
        output = self.cluster.revin(output, "denorm")

        if self.training:
            return output, L_importance
        return output
