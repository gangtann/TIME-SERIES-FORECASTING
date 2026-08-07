"""iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024).

论文: https://arxiv.org/abs/2310.06625
官方仓库: https://github.com/thuml/iTransformer (model/iTransformer.py)

移植改动:
  - forward 恒返回 [B, pred_len, c_out]；官方 output_attention 分支移除（本框架 exp 不消费
    attns，且 configs.output_attention 键不存在），FullAttention 恒 output_attention=False。
  - 官方 configs.class_strategy 只存不用（projection 头硬编码），未引入该键。
  - use_norm 读本框架 model 节共享键 configs.use_norm（官方 run.py 默认 True）。
  - 全部依赖层（DataEmbedding_inverted/Encoder/EncoderLayer/FullAttention/
    AttentionLayer）自包含于 layers/iTransformer_layers.py（与官方逐行等价，
    差异见该文件头注释）。
  - x_dec/x_mark_dec 官方即不使用，签名保留；x_mark_enc 照官方作为协变量 token 拼入嵌入，
    投影后 [:, :, :N] 截掉协变量通道。
  - 无内部辅助损失，无 device 硬编码（官方模型文件本就干净）。
"""
import torch
from torch import nn

from layers.iTransformer_layers import (AttentionLayer, DataEmbedding_inverted,
                                        Encoder, EncoderLayer, FullAttention)


class Model(nn.Module):
    """iTransformer: 变量作 token 的倒置 Transformer（encoder-only）。"""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        # Embedding: [B, L, N] -> [B, N(+n_mark), d_model]
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model,
                                                    configs.embed, configs.freq, configs.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape  # B L N

        # B L N -> B N E（时间戳特征作为额外协变量 token 一并嵌入）
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N，截掉协变量 token
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
