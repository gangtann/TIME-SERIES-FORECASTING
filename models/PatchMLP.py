"""PatchMLP (AAAI 2025).

论文: PatchMLP — Unlocking the Potential of MLPs for Time Series Forecasting
      (Tang & Zhang, AAAI 2025)
官方仓库: https://github.com/TangPeiwang/PatchMLP (commit b36bbc9, 2025-02-19)

移植改动（相对官方 model/PatchMLP.py，其余逐行等价）：
1. 依赖层（moving_avg/series_decomp/Emb/EmbLayer/Encoder）合并进
   layers/PatchMLP_layers.py，类名加 PML 前缀。
2. 删除对 configs.output_attention 的读取（官方读了但从未使用，本框架无此键）。
3. 官方硬编码的 series_decomp(13) 与 patch_len=[48,24,12,6] 参数化为
   configs.pml_decomp_kernel / configs.pml_patch_lens（默认值与官方一致）。
   注：官方 run.py 的 --moving_avg 参数模型并不读取，实际生效的是硬编码 13。
4. forward 保留官方 mask=None 兜底参数；本框架按四参调用，行为一致。
5. 无辅助损失、无设备硬编码，直接返回 [B, pred_len, c_out]。
"""

import torch
import torch.nn as nn

from layers.PatchMLP_layers import PMLEmb, PMLEncoder, PMLSeriesDecomp


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm

        self.decompsition = PMLSeriesDecomp(configs.pml_decomp_kernel)
        # 多尺度 patch 嵌入：[B, V, seq_len] -> [B, V, d_model]
        self.emb = PMLEmb(configs.seq_len, configs.d_model,
                          patch_len=configs.pml_patch_lens)
        self.seasonal_layers = nn.ModuleList([
            PMLEncoder(configs.d_model, configs.enc_in)
            for _ in range(configs.e_layers)
        ])
        self.trend_layers = nn.ModuleList([
            PMLEncoder(configs.d_model, configs.enc_in)
            for _ in range(configs.e_layers)
        ])

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)

        # 注意：官方在 embedding 维（最后一维）上做分解，非时间轴——照搬官方行为
        seasonal_init, trend_init = self.decompsition(x)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
