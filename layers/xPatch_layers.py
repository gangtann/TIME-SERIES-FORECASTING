"""xPatch 专属依赖层（合并自官方仓库 layers/{revin,ema,dema,decomp,network}.py）。

论文: xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend
      Decomposition (AAAI 2025)
官方仓库: https://github.com/stitsyuk/xPatch

移植改动（相对官方源码）:
- 全部类名加 XPatch 前缀防撞：PRISM 的 RevIN/分解在 layers/PRISM_layers.py 内且
  语义不同，均不复用、不覆盖。
- 设备无关化：官方 ema.py 中 `weights = torch.pow(...).to('cuda')` 改为按输入张量
  device 创建；官方 dema.py 中 `alpha.to(device='cuda')` 对 python float 本身会报错
  （官方遗留 bug，源自可学习参数版本的注释代码），此处按语义直接保存 float。
- XPatchDECOMP 对未知 ma_type 显式抛 ValueError（官方静默不设 self.ma，forward 时才
  AttributeError）；'reg'（无分解）由模型侧分支处理，不会构造 DECOMP。
- 官方 network_mlp.py / network_cnn.py 仅供消融（官方模型主文件中为注释状态），未搬运。
其余计算逻辑与官方逐行等价。
"""

import torch
from torch import nn


class XPatchRevIN(nn.Module):
    """官方 layers/revin.py 的 RevIN（逐行等价）。"""

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(XPatchRevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class XPatchEMA(nn.Module):
    """官方 layers/ema.py 的 EMA。

    Exponential Moving Average (EMA) block to highlight the trend of time series.
    移植改动：权重在输入张量 device 上创建（官方硬编码 .to('cuda')），计算仍按官方
    在 double 精度下进行、最后转回 float32。
    """

    def __init__(self, alpha):
        super(XPatchEMA, self).__init__()
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        _, t, _ = x.shape
        powers = torch.flip(
            torch.arange(t, dtype=torch.double, device=x.device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class XPatchDEMA(nn.Module):
    """官方 layers/dema.py 的 DEMA。

    Double Exponential Moving Average (DEMA) block to highlight the trend of time series.
    移植改动：官方对 float 调 .to(device='cuda')（会报错），此处直接保存 python float，
    递推逻辑逐行等价。
    """

    def __init__(self, alpha, beta):
        super(XPatchDEMA, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
            b = self.beta * (s - s_prev) + (1 - self.beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)


class XPatchDECOMP(nn.Module):
    """官方 layers/decomp.py 的 DECOMP：EMA/DEMA 序列分解。

    返回 (seasonal, trend)。ma_type='reg'（无分解）由模型侧分支处理，不会走到这里。
    """

    def __init__(self, ma_type, alpha, beta):
        super(XPatchDECOMP, self).__init__()
        if ma_type == 'ema':
            self.ma = XPatchEMA(alpha)
        elif ma_type == 'dema':
            self.ma = XPatchDEMA(alpha, beta)
        else:
            raise ValueError(f'unknown ma_type for xPatch DECOMP: {ma_type}')

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class XPatchNetwork(nn.Module):
    """官方 layers/network.py 的 Network（逐行等价）：双流主干。

    非线性流（patch + 深度可分离 CNN）吃季节项，线性流（MLP）吃趋势项，末端拼接融合。
    """

    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(XPatchNetwork, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]

        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B * C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B * C, I))  # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]

        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]

        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x
