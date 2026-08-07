"""PatchMLP 专属依赖层（AAAI 2025）。

来源（官方仓库 https://github.com/TangPeiwang/PatchMLP，commit b36bbc9）：
- PMLMovingAvg / PMLSeriesDecomp  <- model/PatchMLP.py 的 moving_avg / series_decomp
  （注意：官方此版本作用在 [B, V, d_model] 的最后一维即 embedding 维上，与常见
  TSLib 时间轴分解不同——这是官方行为，逐行照搬，勿"修正"）
- PMLEmbLayer / PMLEmb            <- layers/Embed.py 的 EmbLayer / Emb（多尺度 patch 嵌入）
- PMLEncoder                       <- model/PatchMLP.py 的 Encoder（MLP 混合块）

类名统一加 PML 前缀防撞；除下列两点外与官方逐行等价：
1. PMLEmb 把官方硬编码的 patch_len=[48,24,12,6]、d_model//4、4 个手写 EmbLayer_i
   改为按 patch_lens 列表构造 ModuleList（默认值与官方完全一致，等价泛化）。
2. dropout 保持官方硬编码 0.1（官方 Encoder 内 nn.Dropout(0.1)，不读 configs.dropout）。
"""

import torch
import torch.nn as nn


class PMLMovingAvg(nn.Module):
    """Moving average block to highlight the trend of time series.

    官方 model/PatchMLP.py::moving_avg，作用于输入张量最后一维。
    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class PMLSeriesDecomp(nn.Module):
    """Series decomposition block（官方 model/PatchMLP.py::series_decomp）。

    返回 (res, moving_mean)，即（季节项，趋势项）。
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = PMLMovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class PMLEmbLayer(nn.Module):
    """单尺度 patch 嵌入（官方 layers/Embed.py::EmbLayer，逐行等价）。

    x: [B, V, L] -> unfold 成 patch -> 线性到 d_model//patch_num -> flatten -> 线性到 d_model。
    """

    def __init__(self, patch_len, patch_step, seq_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.patch_step = patch_step

        patch_num = int((seq_len - patch_len) / patch_step + 1)
        self.d_model = d_model // patch_num
        self.ff = nn.Sequential(
            nn.Linear(patch_len, self.d_model),
        )
        self.flatten = nn.Flatten(start_dim=-2)

        self.ff_1 = nn.Sequential(
            nn.Linear(self.d_model * patch_num, d_model),
        )

    def forward(self, x):
        B, V, L = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)
        x = self.ff_1(x)
        return x


class PMLEmb(nn.Module):
    """多尺度 patch 嵌入（官方 layers/Embed.py::Emb）。

    官方硬编码 patch_len=[48,24,12,6]、步长为各 patch_len 的一半、每尺度分得
    d_model//4，4 路输出 concat 回 d_model。此处泛化为对 patch_lens 列表循环
    （len=4 且取默认值时与官方逐参数等价）。
    """

    def __init__(self, seq_len, d_model, patch_len=(48, 24, 12, 6)):
        super().__init__()
        patch_len = list(patch_len)
        d_model_branch = d_model // len(patch_len)
        self.emb_layers = nn.ModuleList([
            PMLEmbLayer(p, p // 2, seq_len, d_model_branch) for p in patch_len
        ])

    def forward(self, x):
        s_out = torch.cat([layer(x) for layer in self.emb_layers], -1)
        return s_out


class PMLEncoder(nn.Module):
    """MLP 混合编码块（官方 model/PatchMLP.py::Encoder，逐行等价）。

    ff1 沿 embedding 维混合，ff2 沿变量维混合；dropout 官方硬编码 0.1。
    官方定义了 norm2 但从未使用（两处都用 norm1），为逐行对齐原样保留。
    """

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)
        return y_2
