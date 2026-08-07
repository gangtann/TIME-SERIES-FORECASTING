"""iTransformer 专属依赖层。

来源：官方仓库 thuml/iTransformer layers/Embed.py 的 DataEmbedding_inverted（逐行照搬）。
本框架 layers/Embed.py 无此类，故独立放在本文件；其余依赖
（Encoder/EncoderLayer/FullAttention/AttentionLayer）与官方逐行等价，直接复用框架内文件。
"""
import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """倒置嵌入：把每个变量的整段序列 [B, L, N] -> [B, N, d_model] 线性映射为变量 token。

    x_mark 非 None 时，时间戳特征也作为额外协变量 token 拼入（官方原始行为）。
    embed_type/freq 参数官方保留但未使用，此处照抄以保持签名一致。
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)
