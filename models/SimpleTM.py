"""SimpleTM: A Simple Baseline for Multivariate Time Series Forecasting (ICLR 2025).

官方仓库: https://github.com/vsingh-group/SimpleTM (commit 3c77d82)
论文: https://openreview.net/pdf?id=oANkBaVci5

倒置 embedding（变量为 token）+ 平稳小波变换（SWT）多尺度分解 + 几何（楔积）注意力。
官方训练损失 = criterion + l1_weight * attn[0]（experiments/exp_long_term_forecasting.py:173），
其中 attn[0] 是第 1 层 GeomAttention 返回的标量 scores.abs().mean()（仅首层，非全部层）。

移植改动（相对官方 model/SimpleTM.py）：
  1. forward 签名对齐本框架四参契约，返回 [B, pred_len, c_out]；
     官方 forward 恒以 x_mark_enc=None 调 forecast（时间特征不进 embedding），此处保持一致。
  2. 稀疏正则移入模型内：训练态返回 (dec_out, aux_loss)，
     aux_loss = stm_l1_weight * attns[0]（标量，已含权重系数），由本框架 exp 自动加进训练损失；
     eval 态（self.training=False）或 stm_l1_weight == 0 时只返回 dec_out。
  3. 专属超参改读 stm_ 前缀键（stm_alpha/stm_l1_weight/stm_wv/stm_m/
     stm_geomattn_dropout/stm_requires_grad/stm_kernel_size）；use_norm 读 model 节共享键。
  4. output_attention 恒为 False（本框架不输出注意力图；False 路径才返回正则标量）。
  5. 依赖层全部自带于 layers/SimpleTM_layers.py；GeomAttention 因果掩码分支的
     torch.tril(...).to(scores.device) 改为直接在 scores.device 上创建（该分支实际不会走到）。
依赖：pywt（PyWavelets）。
"""
import torch
import torch.nn as nn

from layers.SimpleTM_layers import (
    GeomAttention,
    GeomAttentionLayer,
    STM_DataEmbedding_inverted,
    STM_Encoder,
    STM_EncoderLayer,
)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.alpha = configs.stm_alpha
        self.l1_weight = configs.stm_l1_weight
        self.kernel_size = configs.stm_kernel_size

        self.enc_embedding = STM_DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.encoder = STM_Encoder(
            [
                STM_EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False, configs.factor, attention_dropout=configs.dropout,
                            output_attention=False, alpha=self.alpha
                        ),
                        configs.d_model,
                        requires_grad=configs.stm_requires_grad,
                        wv=configs.stm_wv,
                        m=configs.stm_m,
                        d_channel=configs.dec_in,
                        kernel_size=self.kernel_size,
                        geomattn_dropout=configs.stm_geomattn_dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # Linear Projection             B L N -> B N D (变量 token)
        # 官方 forward 恒传 x_mark=None，时间特征不参与 embedding
        enc_out = self.enc_embedding(x_enc, None)

        # SimpleTM Layers               B N D -> B N D
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output Projection             B N D -> B H N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out, attns = self.forecast(x_enc)
        # 官方训练损失: criterion + l1_weight * attn[0]（仅第 1 层的标量 |scores|.mean()）
        if self.training and self.l1_weight > 0:
            aux_loss = self.l1_weight * attns[0]
            return dec_out, aux_loss
        return dec_out  # [B, pred_len, D]
