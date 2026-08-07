"""xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend
Decomposition (AAAI 2025)。

官方仓库: https://github.com/stitsyuk/xPatch（官方模型文件 models/xPatch.py）。

移植改动（相对官方源码）:
- forward 适配本框架四参签名 forward(x_enc, x_mark_enc, x_dec, x_mark_dec)，
  官方 forward 只吃 x_enc，其余三个参数忽略；无辅助损失，恒返回单值 [B, pred_len, c_out]。
- 专属依赖层（RevIN/EMA/DEMA/DECOMP/Network）全部合并进 layers/xPatch_layers.py，
  类名加 XPatch 前缀（本框架已有的 layers/RevIN.py、layers/Decomp.py 为 PRISM 专用，
  语义不同，不复用）；其中 EMA/DEMA 的硬编码 cuda 已设备无关化（详见该文件头）。
- 模型专属超参读扁平命名空间的 xp_ 前缀键：xp_ma_type / xp_alpha / xp_beta /
  xp_revin / xp_padding_patch；patch_len、stride 复用 model 节通用键。
- ma_type='reg'（无分解消融）时不构造 DECOMP（官方无条件构造但留空 self.ma）；
  官方消融用的 NetworkMLP/NetworkCNN（注释状态）未搬运。
其余结构与计算逻辑与官方逐行等价。
"""

from torch import nn

from layers.xPatch_layers import XPatchDECOMP, XPatchNetwork, XPatchRevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len    # lookback window L
        pred_len = configs.pred_len  # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in        # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.xp_padding_patch

        # Normalization
        self.revin = configs.xp_revin
        self.revin_layer = XPatchRevIN(c_in, affine=True, subtract_last=False)

        # Moving Average
        self.ma_type = configs.xp_ma_type
        alpha = configs.xp_alpha     # smoothing factor for EMA
        beta = configs.xp_beta       # smoothing factor for DEMA

        if self.ma_type != 'reg':
            self.decomp = XPatchDECOMP(self.ma_type, alpha, beta)
        self.net = XPatchNetwork(seq_len, pred_len, patch_len, stride, padding_patch)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [Batch, Input, Channel]；mark/dec 参数为框架签名占位，官方模型不使用
        x = x_enc

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':  # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x  # [Batch, pred_len, Channel]
