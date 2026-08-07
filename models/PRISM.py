import torch
from torch import nn

from layers.Decomp import EMADecomp, MovingAvgDecomp
from layers.PRISM_blocks import SpectralAffinityRouter, HeteroGraphFilter, InfoClockBackbone
from layers.RevIN import RevIN


class Model(nn.Module):
    """PRISM: Frequency-Guided Adaptive Patching and Spectral Relational Modeling.

    数据流（[B, L, C] 输入）：
      RevIN → 时间轴分解(trend, season)
      trend  → 线性头 y_trend
      season → 幅度谱 → FAP 路由门控 G → SRM 异构图谱滤波
             → 谱驱动信息时钟自适应 patch 主干 → y_season
      y = y_season + y_trend → RevIN 反归一化

    自适应 patch = 信息时钟：短时谱与谱模式字典的局部亲和给出逐位置信息密度，
    密度积分发放 patch 边界（谱共振决定 patch 划分，与全局路由共享同一字典）。
    消融开关：clock_mode(边界自适应) / clock_density(频域驱动) / use_graph /
    use_srm / use_fusion（见 configs/default.yaml prism 节）。
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.use_srm = configs.use_srm
        self.use_fusion = configs.use_fusion
        self.route_reg = configs.route_reg
        self.clock_reg = configs.clock_reg
        self.clock_detach_dict = configs.clock_detach_dict

        C, L = configs.enc_in, configs.seq_len
        n_freq = L // 2 + 1

        self.revin = RevIN(C, affine=True)
        if configs.decomp_type == 'ema':
            self.decomp = EMADecomp(alpha=configs.decomp_alpha)
        else:
            self.decomp = MovingAvgDecomp(configs.moving_avg)
        self.trend_head = nn.Linear(L, self.pred_len)

        self.router = SpectralAffinityRouter(
            configs.n_patterns, n_freq, configs.freq_topk,
            configs.smooth_sigma, configs.route_temp_init)
        self.srm = HeteroGraphFilter(
            C, configs.n_patterns, n_freq, configs.d_graph, configs.n_heads,
            configs.graph_layers, configs.dropout, use_graph=configs.use_graph)
        self.backbone = InfoClockBackbone(configs, n_freq)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = self.revin(x_enc, 'norm')            # [B, L, C]
        x = x.permute(0, 2, 1)                   # [B, C, L]
        trend, season = self.decomp(x)
        y = self.trend_head(trend)               # [B, C, H]

        gate, aux = None, None
        if self.use_fusion:
            x_freq = torch.fft.rfft(season, dim=-1)
            mag = x_freq.abs()
            gate = self.router(mag)              # [B, C, K]
            if self.use_srm:
                season = self.srm(season, x_freq, mag, gate)
            d = self.router.dictionary
            if self.clock_detach_dict:
                d = d.detach()
            y_season, kl_reg = self.backbone(season, d)
            if self.training and self.clock_reg > 0:
                aux = self.clock_reg * kl_reg
            y = y + y_season                     # [B, C, H]

        y = self.revin(y.permute(0, 2, 1), 'denorm')   # [B, H, C]

        if self.training and self.route_reg > 0 and gate is not None:
            entropy = -(gate * (gate + 1e-8).log()).sum(-1).mean()
            aux = (0 if aux is None else aux) + self.route_reg * entropy
        if self.training and aux is not None:
            return y, aux
        return y
