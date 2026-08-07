import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer


def topk_sparse(x, k):
    """保留最后一维 top-k 分量，其余置零（频域能量集中假设下的硬阈值去噪）。"""
    top_val, top_idx = torch.topk(x, min(k, x.shape[-1]), dim=-1)
    sparse_x = torch.zeros_like(x)
    sparse_x.scatter_(dim=-1, index=top_idx, src=top_val)
    return sparse_x


def gaussian_smooth(x, kernel_size=5, sigma=1.0):
    """最后一维一维高斯平滑，容忍相邻频点的微小偏移（picket-fence 效应）。"""
    radius = kernel_size // 2
    offsets = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size)
    shape = x.shape
    y = F.conv1d(x.reshape(-1, 1, shape[-1]), kernel, padding='same')
    return y.view(shape)


class SpectralAffinityRouter(nn.Module):
    """FAP 路由器：可学习谱模式字典 + 稀疏谱余弦亲和 + 温度锐化 softmax 门控。

    逐样本逐通道计算（无 batch 维聚合），训练/推理路径完全一致（无采样）。
    """

    def __init__(self, n_patterns, n_freq, freq_topk, smooth_sigma, temp_init):
        super().__init__()
        self.freq_topk = freq_topk
        self.smooth_sigma = smooth_sigma
        self.dictionary = nn.Parameter(torch.rand(n_patterns, n_freq))
        self.log_temp = nn.Parameter(torch.tensor(float(temp_init)).log())

    def _sharpen(self, spec):
        s = topk_sparse(spec, self.freq_topk)
        if self.smooth_sigma > 0:
            s = gaussian_smooth(s, sigma=self.smooth_sigma)
        return F.normalize(s, dim=-1)

    def forward(self, mag):
        # mag: [B, C, F] 幅度谱 -> G: [B, C, K]
        q = self._sharpen(mag)
        d = self._sharpen(F.relu(self.dictionary))          # 字典约束为非负“幅度谱”
        affinity = torch.einsum('bcf,kf->bck', q, d)        # 余弦（双方已 L2 归一）
        return F.softmax(self.log_temp.exp() * affinity, dim=-1)


class AxialAttention(nn.Module):
    """在 [B, K, C, D] 的指定网格轴上做多头自注意力（另一轴并入 batch）。"""

    def __init__(self, d_model, n_heads, dropout, axis):
        super().__init__()
        assert axis in ('C', 'K')
        self.axis = axis
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, h):
        B, K, C, D = h.shape
        if self.axis == 'C':
            x = h.reshape(B * K, C, D)
        else:
            x = h.permute(0, 2, 1, 3).reshape(B * C, K, D)
        out, _ = self.attn(x, x, x, need_weights=False)
        if self.axis == 'C':
            return out.reshape(B, K, C, D)
        return out.reshape(B, C, K, D).permute(0, 2, 1, 3)


class HeteroGraphFilter(nn.Module):
    """SRM：(C, K) 专家节点网格上的异构图（两类边 = 两个轴向注意力），输出门控谱滤波器。

    节点特征混入输入幅度谱投影（输入条件化）；滤波经零初始化残差门写回季节分量，
    初始时恒等映射，图模块只能带来增益不会破坏底线。
    """

    def __init__(self, n_vars, n_patterns, n_freq, d_graph, n_heads, n_layers, dropout, use_graph):
        super().__init__()
        self.use_graph = use_graph
        self.pattern_emb = nn.Parameter(torch.randn(n_patterns, d_graph) * 0.02)
        self.channel_emb = nn.Parameter(torch.randn(n_vars, d_graph) * 0.02)
        self.spec_proj = nn.Linear(n_freq, d_graph)
        if use_graph:
            self.intra_layers = nn.ModuleList(
                [AxialAttention(d_graph, n_heads, dropout, axis='C') for _ in range(n_layers)])
            self.inter_layers = nn.ModuleList(
                [AxialAttention(d_graph, n_heads, dropout, axis='K') for _ in range(n_layers)])
            # 两类边的可学习边型缩放（异构性来源）；小值初始化——外层 res_gate 已零初始化，
            # 这里若也取零会形成双重零门，res_gate 打开前内部得不到任何梯度
            self.relation_pri = nn.Parameter(torch.full((n_layers, 2), 0.1))
            self.ffn = nn.ModuleList([
                nn.Sequential(nn.Linear(d_graph, d_graph * 2), nn.GELU(),
                              nn.Dropout(dropout), nn.Linear(d_graph * 2, d_graph))
                for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(d_graph) for _ in range(n_layers)])
        self.filter_head = nn.Linear(d_graph, n_freq)
        self.res_gate = nn.Parameter(torch.zeros(n_vars))    # tanh(0)=0：初始不干预

    def forward(self, season, x_freq, mag, gate):
        # season [B,C,L]; x_freq [B,C,F] complex; mag [B,C,F]; gate [B,C,K]
        L = season.shape[-1]
        u = self.spec_proj(mag)                                            # [B,C,Dg]
        h = (self.pattern_emb[None, :, None, :]
             + self.channel_emb[None, None, :, :]
             + u[:, None, :, :])                                           # [B,K,C,Dg]
        if self.use_graph:
            for i in range(len(self.intra_layers)):
                h = h + self.relation_pri[i, 0].tanh() * self.intra_layers[i](h)
                h = h + self.relation_pri[i, 1].tanh() * self.inter_layers[i](h)
                h = self.norms[i](h + self.ffn[i](h))
        phi = self.filter_head(h)                                          # [B,K,C,F]
        eff = torch.einsum('bck,bkcf->bcf', gate, phi)                     # 门控聚合
        gain = F.softplus(eff)                                             # 正实增益，保相位
        filtered = torch.fft.irfft(x_freq * gain, n=L, dim=-1)
        return season + self.res_gate.tanh().view(1, -1, 1) * filtered


def _build_encoder(configs):
    """PRISM backbone 的 Transformer encoder（PatchTST 同款机制：FullAttention + BatchNorm）。"""
    return Encoder(
        [EncoderLayer(
            AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                              output_attention=False), configs.d_model, configs.n_heads),
            configs.d_model, configs.d_ff, dropout=configs.dropout,
            activation=configs.activation)
         for _ in range(configs.e_layers)],
        norm_layer=nn.Sequential(_Transpose(1, 2), nn.BatchNorm1d(configs.d_model),
                                 _Transpose(1, 2)))


class _Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class InfoClockPatchEmbedding(nn.Module):
    """谱驱动信息时钟 patch 化（真·逐位置自适应 patch，与论文 FAP 的频域划分思想同构）。

    短时谱与谱模式字典的局部亲和给出逐位置信息密度 α（谱共振强 → 密度大 → patch 细），
    α 沿时间积分，累积量每跨过一个整数阈值发放一个 patch 边界（软分箱，分段线性可微）。
    数学上等价于学习单调时间扭曲：信息时间轴上均匀分 patch ⟺ 物理时间轴上自适应分 patch。

    关键性质：
    - β 零初始化 ⇒ 初始 α 严格为常数 ⇒ 从"诚实均匀 patch"出发学习偏离；
    - softmax 重参数化天然 Σα=N（无归一化除法的缩放零方向），floor 混合封顶单箱物理宽度；
    - 反向积分锚定使最近时间步获得最强边界梯度（与预测任务对齐）；
    - 全程无采样/取整，eval 确定，逐样本逐通道独立，token 数恒为 N。

    消融：clock_mode ∈ {uniform(严格等分), reweight_only(边界钉死仅箱内重加权), adaptive}
    拆分"边界移动"与"箱内软注意力"的贡献；clock_density ∈ {spectral, mlp} 拆分频域信号的贡献。
    """

    def __init__(self, configs, n_freq):
        super().__init__()
        self.N = configs.clock_n_patch
        self.R = configs.clock_basis
        self.floor = configs.clock_floor
        self.reverse = configs.clock_reverse
        self.win = configs.clock_win
        self.density = configs.clock_density
        self.mode = configs.clock_mode
        self.kl_lo, self.kl_hi = configs.clock_kl_lo, configs.clock_kl_hi
        L, d_v = configs.seq_len, configs.clock_d_v
        assert L % self.N == 0, 'clock_n_patch 须整除 seq_len（reweight_only 消融依赖等分箱）'

        # 内容前端：保序局部特征（RF=13），与密度路径分离
        self.frontend = nn.Sequential(
            nn.Conv1d(1, d_v, 5, padding=2), nn.GELU(),
            nn.Conv1d(d_v, d_v, 5, padding=4, dilation=2), nn.GELU())
        self.norm = nn.LayerNorm(d_v)

        # 谱密度路径：滑窗短时谱 + 频带聚合矩阵（全局字典 n_freq bin → 局部 f_loc bin）
        f_loc = self.win // 2 + 1
        self.register_buffer('hann', torch.hann_window(self.win))
        band = torch.zeros(f_loc, n_freq)
        for fg in range(n_freq):
            band[min(round(fg * self.win / L), f_loc - 1), fg] = 1.0
        self.register_buffer('band', band / band.sum(-1, keepdim=True).clamp(min=1.0))
        # 逐谱模式粒度系数：模式 k 的共振区该切多细；零初始化 ⇒ 第 0 步严格均匀
        self.beta = nn.Parameter(torch.zeros(configs.n_patterns))
        self.log_temp = nn.Parameter(torch.tensor(float(configs.clock_temp_init)).log())
        if self.density == 'mlp':                     # 自由密度对照（消融频域信号的价值）
            self.density_mlp = nn.Sequential(
                nn.Linear(d_v, d_v), nn.GELU(), nn.Linear(d_v, 1))
            nn.init.zeros_(self.density_mlp[-1].weight)
            nn.init.zeros_(self.density_mlp[-1].bias)

        # token 化：余弦基调制池化（变宽 patch 上的软 R 抽头保序投影，R=1 退化为纯均值）
        self.proj = nn.Linear(self.R * d_v, configs.d_model)
        self.pos = nn.Linear(2, configs.d_model)      # 物理中心/宽度 → 编码器可见的物理位置
        self.dropout = nn.Dropout(configs.dropout)
        self.register_buffer('r_idx', torch.arange(self.R, dtype=torch.float))

    def _soft_bins(self, a):
        # a [BC,L]（Σ=N）-> m [BC,N,L] 软分箱（每箱质量 1）、u [BC,N,L] 箱内信息坐标
        A = a.cumsum(-1)
        A0 = A - a
        i = torch.arange(1, self.N + 1, device=a.device, dtype=a.dtype).view(1, self.N, 1)
        hi = torch.minimum(A.unsqueeze(1), i)
        lo = torch.maximum(A0.unsqueeze(1), i - 1)
        m = (hi - lo).clamp(min=0)
        u = ((hi + lo) / 2 - (i - 1)).clamp(0, 1)
        return m, u

    def _density(self, x, z, dictionary):
        # x [BC,1,L]; z [BC,L,d_v] -> a_raw [BC,L]
        if self.mode == 'uniform':
            return x.new_zeros(x.shape[0], x.shape[-1])
        if self.density == 'mlp':
            return self.density_mlp(z).squeeze(-1)
        pad = (self.win // 2, self.win - 1 - self.win // 2)
        frames = F.pad(x, pad, mode='reflect').squeeze(1).unfold(-1, self.win, 1)  # [BC,L,win]
        s = F.normalize(torch.fft.rfft(frames * self.hann, dim=-1).abs(), dim=-1)  # [BC,L,f_loc]
        d = F.normalize(F.relu(dictionary) @ self.band.t(), dim=-1)                # [K,f_loc]
        g = torch.einsum('blf,kf->blk', s, d)                                      # 局部谱亲和
        return F.softmax(self.log_temp.exp() * g, dim=-1) @ self.beta

    def forward(self, season, dictionary):
        # season [B,C,L]; dictionary [K,n_freq]（可 detach）-> tok [B*C,N,D], kl_reg 标量
        B, C, L = season.shape
        x = season.reshape(B * C, 1, L)
        z = self.norm(self.frontend(x).transpose(1, 2))            # [BC,L,d_v]

        a_raw = self._density(x, z, dictionary)
        p = (1 - self.floor) * F.softmax(a_raw, dim=-1) + self.floor / L
        kl = (p * (p * L).clamp(min=1e-8).log()).sum(-1)           # KL(p‖uniform) ≥ 0
        reg = (F.relu(kl - self.kl_hi) + F.relu(self.kl_lo - kl)).mean()

        if self.reverse:                                           # 反向锚定：近期步梯度最强
            p, z = p.flip(-1), z.flip(1)
        a = self.N * p
        if self.mode == 'reweight_only':                           # 边界钉死在均匀位置
            w = L // self.N
            a = a.view(-1, self.N, w)
            a = (a / a.sum(-1, keepdim=True)).reshape(-1, L)
        m, u = self._soft_bins(a)

        phi = torch.cos(torch.pi * self.r_idx.view(1, 1, 1, -1) * u.unsqueeze(-1))
        pooled = torch.einsum('bnl,bnlr,bld->bnrd', m, phi, z).flatten(-2)   # [BC,N,R*d_v]

        t_axis = torch.linspace(0, 1, L, device=a.device)
        c = (m * t_axis).sum(-1)                                   # 物理中心（当前坐标系）
        sig = ((m * (t_axis - c.unsqueeze(-1)) ** 2).sum(-1) + 1e-8).sqrt()
        if self.reverse:
            c = 1 - c
        tok = self.proj(pooled) + self.pos(torch.stack([c, sig], dim=-1))
        if self.reverse:
            tok = tok.flip(1)                                      # token 顺序恢复为物理先后
        return self.dropout(tok), reg


class InfoClockBackbone(nn.Module):
    """信息时钟 patch 化 → 共享 Transformer encoder → 展平预测头。"""

    def __init__(self, configs, n_freq):
        super().__init__()
        self.embed = InfoClockPatchEmbedding(configs, n_freq)
        self.encoder = _build_encoder(configs)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(configs.clock_n_patch * configs.d_model, configs.d_head), nn.GELU(),
            nn.Dropout(configs.head_dropout), nn.Linear(configs.d_head, configs.pred_len))

    def forward(self, season, dictionary):
        B, C, _ = season.shape
        tok, reg = self.embed(season, dictionary)
        enc, _ = self.encoder(tok)
        return self.head(enc).reshape(B, C, -1), reg
