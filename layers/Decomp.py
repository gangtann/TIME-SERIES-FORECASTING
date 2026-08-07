import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgDecomp(nn.Module):
    """滑动平均趋势-季节分解，作用于 [B, C, L] 的时间轴（最后一维）。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        # 两端复制填充，保持长度不变
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :, :1].repeat(1, 1, pad_left)
        end = x[:, :, -1:].repeat(1, 1, pad_right)
        trend = F.avg_pool1d(torch.cat([front, x, end], dim=-1),
                             kernel_size=self.kernel_size, stride=1)
        return trend, x - trend


class EMADecomp(nn.Module):
    """指数滑动平均分解（xPatch 风格），作用于 [B, C, L] 的时间轴。

    trend[t] = alpha * x[t] + (1 - alpha) * trend[t-1]，用幂权重的并行形式计算。
    alpha 越小趋势越平滑。
    """

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        B, C, L = x.shape
        device, dtype = x.device, x.dtype
        t = torch.arange(L, device=device, dtype=dtype)
        decay = (1.0 - self.alpha) ** t                       # (1-a)^t, t=0..L-1
        # 递推展开：trend[t] = sum_{j=0..t} a*(1-a)^j * x[t-j] + (1-a)^{t+1} * x[0]
        kernel = torch.flip(self.alpha * decay, dims=[0]).view(1, 1, L)
        x_pad = F.pad(x.reshape(B * C, 1, L), (L - 1, 0))
        conv = F.conv1d(x_pad, kernel).reshape(B, C, L)
        trend = conv + (1.0 - self.alpha) * decay.view(1, 1, L) * x[:, :, :1]
        return trend, x - trend
