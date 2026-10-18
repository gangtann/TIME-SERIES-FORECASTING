"""PRISM 专属依赖层（合并自 FRAME 原版实现的 layers/ 目录）。

论文: FRAME: Frequency-aware Relational Adaptive Mixture-of-Experts for
      Multivariate Time Series Forecasting
原版实现: ~/Projects/Time-Temp/Frame（layers/{RevIN,Autoformer_EncDec,
      PatternEmbedding,SpectralResonanceAlignment,ParallelDynamicPatching,
      HGNN,SpectralMoE}.py，七文件合并至此）

移植改动（相对原版源码）:
- 推理确定化：原版 ParallelDynamicPatching 无条件调用 gumbel_softmax，eval 期
  仍随机采样 patch 长度，导致同一输入两次前向结果不同。此处 training 时保持
  Gumbel 采样（温度提为 gumbel_tau 配置键，原版硬编码 0.5），eval 时改为
  argmax one-hot。
- 删死参数：原版 HeteroGraphAttention 初始化 relation_att / relation_msg
  两个 [2, H, d_k, d_k] 参数但 forward 从未使用（xavier 初始化后即悬空），
  一并删除其 reset_parameters；纯透传的 HeteroGNNModel 包装层去掉。
- 删不可达分支：原版 SpectralMoE._forward_static 定义三参、调用处只传两参，
  sel_moe=None 时必然 TypeError；而 Model 恒传 sel_moe，该分支不可达，删除。
  同时删除原版 `from layers.GCN import GCN` 死导入（GCN 仅在注释中构造）。
- 删无用返回：原版 ParallelDynamicPatching 收集并返回 pat_ids，模型侧从不消费。
- 高斯平滑带宽提为 sra_sigma 配置键（原版 PDP 调用处硬编码 sigma=0.3，而 SRA
  自身默认 1.0，此处统一由配置给出）。
- 设备无关：原版已按 x.device 建张量，无 .to('cuda')，保持原样。

保留的原版行为（非 bug，勿当缺陷修改；作为第二阶段改动的候选点已备案）:
- patch 边界为 batch 级全局决策：SpectralResonanceAlignment 对相似度矩阵在
  batch 维取均值，ParallelDynamicPatching 维护单份 [C] 游标，故同一 batch 内
  所有样本共享分块方案，样本间存在耦合；测试末批 batch 变小时该批分块会不同
  （本框架 drop_last=False）。
- SpectralMoE 内 series_decomp 沿 patch 轴（dim=1）而非时间轴做滑动平均，
  趋势项实为"跨 patch 平滑"。
- patch 内容零填充到 max_patch_len 后再 rFFT，短 patch 带零尾谱泄漏（原版以
  n=max_len 显式指定，属设计选择）。

自适应分块 v2（patch_mode=greedy_v2，本框架新增，原版无）：保留"谱内容决定分段 +
游标贪心 + 可达性表"的思路，换掉三处让自适应无法兑现的接口——
- 长度归一化（patch_norm=resample）：候选段与选中段都面积重采样到 adp_resample_len
  再做谱运算，不再零填充到 max_len（零填充下短段谱被矩形窗泄漏主导，相似度是长度的
  函数而非内容的函数；同一个 emb_in 看到的短段/长段落在不同子空间）。等分是它的特例。
- 可训练路由：共振分数降级为特征，与 log(ℓ/T0) 一起进两层小 MLP 出 logit，由预测损失
  经直通估计训练（原版直接拿"与某个自由原型最像"当决策，与预测无关）。直通混合作用
  在重采样后的候选段上，梯度按候选逐个区分（标量直通给所有候选同一梯度，等于没有）。
- 重栅格化（adp_regrid，在 models/PRISM.py）：token 按时间重叠投影回 N 个等分槽位，
  展平头读到的槽位语义不再随边界漂移（原版 13.5% 代价所在）。

oracle-EM 分块（patch_mode=oracle，本框架新增）：上面三版自适应（余弦贪心 / 可学路由贪心 / 换周期池）
在 ETTh1 上全部塌缩成与输入无关的固定划分且输给等分，根因是选划分的信号来自模型内部代理（余弦、
直通估计梯度）：直通梯度是"候选段线性混合"的梯度、与换划分后的真实损失无关；无探索的 argmax 让初始
偏好被下游共适应自我强化；贪心的逐步信用分配看不到首步逼出的后续碎段；共享槽位头偏好平稳输入、
把逐样本变划分当噪声惩罚。oracle 挡把判据换成真实预测损失本身：构造期枚举全部合法划分，训练时
E 步逐划分前向取每条样本的 argmin（models/PRISM.py 的 oracle_step），M 步在该划分上训练并用交叉熵
让路由器模仿；暖启动期恒用等分。固定最优 ⇒ E 步恒选等分 ⇒ 路由器学成常数 ⇒ 数值等于固定 patch。

本目录约定：每个模型一个自包含 <X>_layers.py，无跨模型公共层文件。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def resolve_token_len(configs):
    """SpectralMoE / PatternEmbedding 看到的 patch 宽度（PDP 与 Model 共用同一规则）。

    uniform：段长 L//N；重采样管线（greedy_v2 或 patch_norm=resample）：adp_resample_len；
    其余（零填充）：池内最大长度。
    """
    if configs.patch_mode == 'uniform':
        return configs.seq_len // configs.n_tokens
    if configs.patch_mode in ('greedy_v2', 'oracle') or configs.patch_norm == 'resample':
        return configs.adp_resample_len
    return configs.min_patch_len + max(configs.patch_pool)


class RevIN(nn.Module):
    """原版 layers/RevIN.py（逐行等价）：可逆实例归一化。"""

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
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


class MovingAvg(nn.Module):
    """原版 layers/Autoformer_EncDec.py 的 moving_avg（逐行等价）。"""

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """原版 layers/Autoformer_EncDec.py 的 series_decomp（逐行等价）。"""

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class PatternEmbedding(nn.Module):
    """原版 layers/PatternEmbedding.py（逐行等价）：可学习频域模式原型。

    输出 [C, K, F]，F = max_patch_len // 2 + 1。
    strategy: 0 混合（半随机 + 半数据驱动）| 1 共享随机 | 2 逐通道随机
              | 3 数据驱动共享 | 4 数据驱动逐通道
    """

    def __init__(self, configs, max_patch_len: int, strategy: int = 1, init_std: float = 0.02):
        super().__init__()
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pattern_nums = configs.pattern_nums
        self.F = max_patch_len // 2 + 1
        self.strategy = strategy
        self.init_std = init_std

        rand_nums = self.pattern_nums // 2
        data_nums = self.pattern_nums - rand_nums

        if self.strategy == 0:
            self.pattern_emb = nn.Parameter(self.init_std * torch.randn(rand_nums, self.F))
            self.pattern_linears = nn.ModuleList(
                [nn.Linear(self.seq_len, self.F) for _ in range(data_nums)])
        elif self.strategy == 1:
            self.pattern_shared = nn.Parameter(
                self.init_std * torch.randn(self.pattern_nums, self.F))
        elif self.strategy == 2:
            self.pattern_c = nn.Parameter(
                self.init_std * torch.randn(self.enc_in, self.pattern_nums, self.F))
        elif self.strategy in (3, 4):
            self.pattern_linears = nn.ModuleList(
                [nn.Linear(self.seq_len, self.F) for _ in range(self.pattern_nums)])
        else:
            raise ValueError(f'unknown pattern_strategy: {self.strategy}')

    def forward(self, x):
        """x: [B, C, L] -> pattern_emb: [C, K, F]"""
        B, C, L = x.shape

        if self.strategy == 0:
            rand_emb = self.pattern_emb.unsqueeze(0).expand(C, -1, -1)
            x_mean = x.mean(dim=(0, 1), keepdim=False).unsqueeze(0)  # [1, L]
            data_emb = torch.stack([lin(x_mean) for lin in self.pattern_linears], dim=1)
            data_emb = data_emb.expand(C, -1, -1)
            return torch.concat([rand_emb, data_emb], dim=1)
        if self.strategy == 1:
            return self.pattern_shared.unsqueeze(0).expand(C, -1, -1)
        if self.strategy == 2:
            return self.pattern_c
        if self.strategy == 3:
            x_mean = x.mean(dim=(0, 1), keepdim=False).unsqueeze(0)  # [1, L]
            pat = torch.stack([lin(x_mean) for lin in self.pattern_linears], dim=1)
            return pat.expand(C, -1, -1)
        # strategy == 4
        x_b = x.mean(dim=0)  # [C, L]
        return torch.stack([lin(x_b) for lin in self.pattern_linears], dim=1)


class SpectralResonanceAlignment(nn.Module):
    """原版 layers/SpectralResonanceAlignment.py（逐行等价）：稀疏谱相似度。

    top-k 能量集中稀疏化 → 高斯平滑（容忍 DFT 栅栏效应的频率微移）
    → L2 归一化 → 余弦相似度 → batch 维取均值。
    """

    def __init__(self, configs):
        super(SpectralResonanceAlignment, self).__init__()
        self.top_k = configs.freq_topk
        self.sparse = configs.sra_sparse  # 消融: false = 跳过稀疏化与平滑，直接算原始谱余弦

    def get_sparse_vector(self, x, k):
        top_val, top_idx = torch.topk(x, k, dim=-1)
        sparse_x = torch.zeros_like(x)
        sparse_x.scatter_(dim=-1, index=top_idx, src=top_val)
        return sparse_x

    def apply_gaussian_smoothing(self, x, kernel_size=3, sigma=1.0):
        radius = kernel_size // 2
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=x.device)
        kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size).to(x.dtype)

        x_shape = x.shape
        flat_x = x.flatten(0, -2).unsqueeze(1)
        y = F.conv1d(flat_x, kernel, padding='same')
        return y.view(x_shape)

    def forward(self, y_real, patterns, sigma=1.0, kernel_size=3, reduce_batch=True):
        """y_real: [C, P, B, F]; patterns: [C, K, B, F]
        -> [C, P, K]（reduce_batch=True，原版）或 [C, B, P, K]（保留逐样本维）"""
        if self.sparse:
            sparse_y = self.get_sparse_vector(y_real, self.top_k)
            sparse_p = self.get_sparse_vector(patterns, self.top_k)
        else:
            sparse_y, sparse_p = y_real, patterns

        if self.sparse and sigma > 0:
            sparse_y = self.apply_gaussian_smoothing(sparse_y, kernel_size=kernel_size, sigma=sigma)
            sparse_p = self.apply_gaussian_smoothing(sparse_p, kernel_size=kernel_size, sigma=sigma)

        sparse_y_norm = F.normalize(sparse_y, p=2, dim=-1, eps=1e-8)
        sparse_p_norm = F.normalize(sparse_p, p=2, dim=-1, eps=1e-8)

        scores_per_batch = torch.einsum('cpbf, ckbf -> cbpk', sparse_y_norm, sparse_p_norm)
        if not reduce_batch:
            return scores_per_batch
        # batch 维聚合：分块方案对整个 batch 共享（见文件头"保留的原版行为"）
        return scores_per_batch.mean(dim=1)


class ParallelDynamicPatching(nn.Module):
    """原版 layers/ParallelDynamicPatching.py：谱共振驱动的并行动态分块。

    逐通道维护游标，循环提出下一段 patch 长度（候选池与谱原型做相似度打分，
    Gumbel-Softmax 硬选择），直到所有通道扫完整条序列。
    """

    def __init__(self, configs):
        super(ParallelDynamicPatching, self).__init__()
        self.seq_len = configs.seq_len
        self.pool = list(configs.patch_pool)
        self.min_patch_len = configs.min_patch_len
        self.max_patch_len = self.min_patch_len + max(self.pool)
        self.steps_list = [self.min_patch_len + p for p in self.pool]
        self.gumbel_tau = configs.gumbel_tau
        self.sra_sigma = configs.sra_sigma
        self.patch_select = configs.patch_select

        # patch_mode: greedy(原版，段长与段数都由游标扫描涌现，段数随 batch 变化)
        #           | uniform(等分 N 段，段数固定 → token 可展平直喂预测头)
        #           | greedy_v2(自适应 v2：重采样 + 可训练路由 + 恰好 N 段，见文件头)
        self.patch_mode = configs.patch_mode
        self.n_tokens = configs.n_tokens
        if self.patch_mode not in ('greedy', 'uniform', 'greedy_fixed', 'fixed_random',
                                   'greedy_v2', 'oracle'):
            raise ValueError(f'unknown patch_mode: {self.patch_mode}')
        if self.patch_mode == 'uniform' and self.seq_len % self.n_tokens != 0:
            raise ValueError(
                f'patch_mode=uniform requires seq_len % n_tokens == 0, '
                f'got seq_len={self.seq_len}, n_tokens={self.n_tokens}')

        # patch_norm: pad(原版，段零填充到 max_len) | resample(段面积重采样到 adp_resample_len)
        # uniform 挡两者恒等（段长本就相同，不填充也不重采样）。
        self.patch_norm = configs.patch_norm
        if self.patch_norm not in ('pad', 'resample'):
            raise ValueError(f'unknown patch_norm: {self.patch_norm}')
        if self.patch_mode in ('greedy_v2', 'oracle') and self.patch_norm != 'resample':
            raise ValueError(f'patch_mode={self.patch_mode} 需要 patch_norm=resample（重采样是其组成部分）')
        if self.patch_mode in ('greedy', 'greedy_fixed') and self.patch_norm == 'resample':
            raise ValueError(f'patch_mode={self.patch_mode} 只支持 patch_norm=pad；'
                             f'重采样版自适应请用 greedy_v2')
        self.resample_len = configs.adp_resample_len
        self.token_len = resolve_token_len(configs)

        if self.patch_mode in ('greedy_fixed', 'fixed_random', 'greedy_v2'):
            self._build_reach_table()
        if self.patch_mode == 'fixed_random':
            self._build_fixed_random(configs.enc_in)
        if self.patch_norm == 'resample':
            self._build_resample_table()
        if self.patch_mode == 'oracle':
            self._build_partitions()

        # v2 路由器：输入 [K 个共振分数, log(ℓ/T0)]，输出 1 个 logit；由预测损失经直通估计训练。
        # adp_lambda：选段分布的熵正则（对整批所有决策的软概率取均值再算熵，鼓励不塌缩到单一长度），
        # 经 aux_loss 进训练损失。len_hist：eval 期累计的 [N, P] 选段直方图（按槽位），供诊断读出。
        self.aux_loss = None
        if self.patch_mode == 'greedy_v2':
            if configs.boundary_scope != 'sample':
                raise ValueError('patch_mode=greedy_v2 只支持 boundary_scope=sample')
            n_pool = len(self.steps_list)
            self.router = nn.Sequential(
                nn.Linear(configs.pattern_nums + 1, configs.adp_router_hidden), nn.GELU(),
                nn.Linear(configs.adp_router_hidden, 1))
            self.adp_lambda = configs.adp_lambda
            self.register_buffer('len_hist', torch.zeros(self.n_tokens, n_pool),
                                 persistent=False)

        self.sra = SpectralResonanceAlignment(configs)

        # MoE 门控逆温度（原版缺失，实测缺陷）：raw_scores 是余弦相似度、值域钉死在
        # [-1,1]，直接过 softmax 时 K=5 的门控最多只能在 0.09~0.42 浮动；实测训练后
        # 门控熵达最大熵的 99.7%（值全落在 0.16~0.23），"混合专家"退化为等权平均，
        # 故 use_moe_gate 消融效应精确为零。可学习逆温度让门控具备区分度。
        # gate_temp_init=1.0 与原版逐位等价。
        self.log_gate_temp = nn.Parameter(
            torch.tensor(float(math.log(configs.gate_temp_init))))

        # gate_scope（原版只有 channel）：原版把逐样本、逐段的共振分数 mean 掉，门控形状
        # 恒为 [C, K]——不随样本变也不随 patch 变。此时 eff_w=Σ_k w_k·g_k 在数学上等价于
        # "每通道一个自由滤波器"（w 本就是自由参数），门控只是重参数化，可表达集合不变，
        # 故 use_moe_gate 消融与门控温度扫描效应均精确为零（已实测）。
        # token 挡保留逐样本逐段维度 [B, P, K]，路由才真正随输入变化。
        self.gate_scope = configs.gate_scope
        if self.gate_scope not in ('channel', 'token'):
            raise ValueError(f'unknown gate_scope: {self.gate_scope}')

        # boundary_scope（原版只有 batch）：原版 SRA 对 batch 维取均值后再选长度，
        # 分块方案由整批投票决定——实测同一条样本换批次同伴，分段就从 [16,36,20,8,8,8]
        # 变成 [24,32,16,8,8,8]。等分挡边界固定，此缺陷不显形；一开自适应即致命：
        # 训练时"第 i 个 token 覆盖哪段时间"每批都在跳，展平预测头学不到稳定的槽位权重，
        # 推理时预测还依赖同批的无关样本（实测 greedy_fixed 掉 5.5%、种子方差涨到 8.5%）。
        # sample 挡让每条样本按自己的谱决定自己的分段。
        self.boundary_scope = configs.boundary_scope
        if self.boundary_scope not in ('batch', 'sample'):
            raise ValueError(f'unknown boundary_scope: {self.boundary_scope}')

    def train(self, mode=True):
        # 进入 eval 即清零选段直方图：训练循环里每轮验证都走 eval，测试前再进一次 eval，
        # 故测试末尾读到的 len_hist 只含测试集
        if not mode and hasattr(self, 'len_hist'):
            self.len_hist.zero_()
        if not mode and hasattr(self, 'part_hist'):
            self.part_hist.zero_()
        return super().train(mode)

    def _build_partitions(self):
        """oracle 挡：枚举"恰好 N 段、和为 L、段长取自池"的全部划分（有序）。

        part_pidx [P, N] 为各段在候选池中的下标（查 resample_w），part_lens [P, N] 为段长；
        uniform_part 为等分划分的下标——它必须存在，否则"固定最优时退回固定"无从谈起。
        划分数随 N 与池大小组合爆炸（96/6/{8..48} 达上千），E 步逐划分前向承受不起，故设上限。
        """
        N, L = self.n_tokens, self.seq_len
        parts = []

        def rec(rem, n, acc):
            if n == 0:
                if rem == 0:
                    parts.append(list(acc))
                return
            for p, st in enumerate(self.steps_list):
                if st <= rem:
                    rec(rem - st, n - 1, acc + [p])

        rec(L, N, [])
        if not parts:
            raise ValueError(f'patch_mode=oracle 无解：长度 {L} 无法由恰好 {N} 个候选长度 {self.steps_list} 拼出')
        if len(parts) > 512:
            raise ValueError(f'patch_mode=oracle 划分数 {len(parts)} 过多（上限 512），请缩小 patch_pool 或 n_tokens')
        pidx = torch.tensor(parts, dtype=torch.long)
        lens = torch.tensor(self.steps_list, dtype=torch.long)[pidx]
        uni = [i for i, row in enumerate(lens.tolist()) if all(v == L // N for v in row)]
        if not uni:
            raise ValueError(f'patch_mode=oracle 要求等分长度 {L // N} 在候选池 {self.steps_list} 内')
        self.uniform_part = uni[0]
        self.n_parts = len(parts)
        self.register_buffer('part_pidx', pidx, persistent=False)
        self.register_buffer('part_lens', lens, persistent=False)
        self.register_buffer('part_hist', torch.zeros(self.n_parts), persistent=False)

    def _partition_forward(self, x, pattern_emb, part_idx):
        """按给定划分下标取块（逐样本、跨通道共享）：part_idx [B] -> patches [B, C*N, T0]。

        _fixed_random_forward 的逐样本泛化：段长查 part_lens，重采样权重查 resample_w[part_pidx]。
        """
        B, C, L = x.shape
        N, M, T0 = self.n_tokens, self.max_patch_len, self.token_len
        lens = self.part_lens[part_idx]                                  # [B, N]
        starts = lens.cumsum(-1) - lens
        grid = torch.arange(M, device=x.device)
        idx = (starts.unsqueeze(-1) + grid).clamp(max=L - 1)            # [B, N, M]
        mask = (grid < lens.unsqueeze(-1)).to(x.dtype)
        gathered = x.gather(2, idx.reshape(B, 1, -1).expand(B, C, -1))
        gathered = gathered.reshape(B, C, N, M) * mask.unsqueeze(1)
        w = self.resample_w[self.part_pidx[part_idx]]                    # [B, N, T0, M]
        patches_bcn = torch.einsum('bntm, bcnm -> bcnt', w, gathered)    # [B, C, N, T0]

        spec = torch.fft.rfft(patches_bcn, n=T0, dim=-1).real.permute(1, 2, 0, 3)  # [C, N, B, F]
        patterns = pattern_emb.unsqueeze(2).expand(-1, -1, B, -1)
        temp = self.log_gate_temp.exp()
        if self.gate_scope == 'token':
            sc = self.sra(spec, patterns, sigma=self.sra_sigma, reduce_batch=False)  # [C, B, N, K]
            sel_moe = F.softmax(sc * temp, dim=-1).permute(1, 0, 2, 3).reshape(B, C * N, -1)
        else:
            sc = self.sra(spec, patterns, sigma=self.sra_sigma)                     # [C, N, K]
            sel_moe = F.softmax(sc.mean(dim=1) * temp, dim=-1)

        if not self.training:
            self.part_hist += torch.bincount(part_idx, minlength=self.n_parts).to(self.part_hist.dtype)
        patches = patches_bcn.reshape(B, C * N, T0)
        lengths = lens.unsqueeze(1).expand(B, C, N).reshape(B, C * N)
        var_ids = torch.arange(C, device=x.device).repeat_interleave(N)
        return patches, lengths, var_ids, sel_moe

    def _build_reach_table(self):
        """精确可达性表 reach[m][r]：剩余长度 r 能否被恰好 m 个池内长度凑出。

        greedy_fixed 要求游标恰好走 N 步且恰好铺满 L。候选长度是离散池，"区间可行"
        （min_len·M ≤ R ≤ max_len·M）不等于真正可达（例：池 {8,16}、R=20、M=2 落在
        区间内却无解），故必须用精确表。表规模 (N+1)×(L+1)，构造期一次性算好。
        """
        N, L = self.n_tokens, self.seq_len
        reach = torch.zeros(N + 1, L + 1, dtype=torch.bool)
        reach[0, 0] = True
        for m in range(1, N + 1):
            for step in self.steps_list:
                if step <= L:
                    reach[m, step:] |= reach[m - 1, :L + 1 - step]

        if not bool(reach[N, L]):
            raise ValueError(
                f'patch_mode=greedy_fixed 无解：长度 {L} 无法由恰好 {N} 个候选长度 '
                f'{self.steps_list} 拼出。请调整 n_tokens / min_patch_len / patch_pool')

        # 首步可选长度数 = 1 时段长被唯一确定，"自适应"名存实亡，提前报错而非静默退化
        n_first = sum(1 for st in self.steps_list
                      if st <= L and bool(reach[N - 1, L - st]))
        if n_first < 2:
            raise ValueError(
                f'patch_mode=greedy_fixed 无自适应空间：L={L}、N={N} 下首步仅 {n_first} '
                f'种合法长度（候选 {self.steps_list}）。请调小 n_tokens 或 min_patch_len')

        self.register_buffer('reach', reach, persistent=False)

    def _build_resample_table(self):
        """面积重采样权重 resample_w[p, t, τ]：池内第 p 种长度 ℓ 的段（局部位置 τ∈[0,max_len)）
        重采样到 T0 个点，输出格 t 覆盖局部区间 [t·ℓ/T0, (t+1)·ℓ/T0)，权重 = 与输入格 [τ,τ+1)
        的重叠 / (ℓ/T0)。ℓ=T0 时恒等；ℓ>T0 为平均池化（不丢点、无混叠）；ℓ<T0 为分段常数上采样。
        每行权重和为 1，对任意 ℓ 保持段的均值。
        """
        T0, M = self.resample_len, self.max_patch_len
        t = torch.arange(T0, dtype=torch.float64)
        tau = torch.arange(M, dtype=torch.float64)
        w = torch.zeros(len(self.steps_list), T0, M, dtype=torch.float64)
        for p, length in enumerate(self.steps_list):
            cell = length / T0
            lo, hi = (t * cell).unsqueeze(1), ((t + 1) * cell).unsqueeze(1)          # [T0, 1]
            ov = torch.minimum(hi, tau + 1) - torch.maximum(lo, tau)                 # [T0, M]
            w[p] = ov.clamp_min(0) / cell
        self.register_buffer('resample_w', w.float(), persistent=False)

    def _calc_spectral_scores_ps(self, x, curr_pos, lens_eff, patterns):
        """逐样本版打分。curr_pos: [C, B]; lens_eff: [C, B, P] -> [C, B, P, K]"""
        xt = x.transpose(0, 1)  # [C, B, L]
        C, B, L = xt.shape
        max_len = self.max_patch_len

        grid_t = torch.arange(max_len, device=x.device)
        idx_abs = curr_pos.unsqueeze(-1) + grid_t.view(1, 1, -1)      # [C, B, max_len]
        raw = xt.gather(2, idx_abs.clamp(max=L - 1))                  # [C, B, max_len]

        mask_seq = (idx_abs < L).to(x.dtype)                          # [C, B, max_len]
        mask_len = (grid_t.view(1, 1, 1, -1) < lens_eff.unsqueeze(-1)).to(x.dtype)
        masked = (raw * mask_seq).unsqueeze(2) * mask_len             # [C, B, P, max_len]

        spec = torch.fft.rfft(masked, n=max_len, dim=-1).real         # [C, B, P, F]
        # SRA 约定 y_real 为 [C, P, B, F]
        return self.sra(spec.transpose(1, 2), patterns, sigma=self.sra_sigma,
                        reduce_batch=False)                           # [C, B, P, K]

    def _propose_next_patch_ps(self, x, curr_pos, cand_pool, patterns, rem_steps):
        """逐样本提议。curr_pos: [C, B] -> sel_moe([C,B,K] 或 [C,K]), sel_len [C, B]"""
        C, B = curr_pos.shape
        L = self.seq_len

        lens_eff = cand_pool.view(1, 1, -1).expand(C, B, -1).contiguous()
        rest = (L - curr_pos).unsqueeze(-1) - lens_eff.long()          # [C, B, P]
        mask_valid = (rest >= 0) & self.reach[rem_steps - 1][rest.clamp_min(0)]

        scores_all = self._calc_spectral_scores_ps(x, curr_pos, lens_eff, patterns)
        scores = scores_all.amax(dim=-1).clone()                       # [C, B, P]
        scores[~mask_valid] = -1e9

        if not self.training:
            probs_one_hot = F.one_hot(scores.argmax(-1), scores.size(-1)).to(scores.dtype)
        elif self.patch_select == 'gumbel':
            probs_one_hot = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=True, dim=-1)
        elif self.patch_select == 'st':
            probs = F.softmax(scores / self.gumbel_tau, dim=-1)
            hard = F.one_hot(probs.argmax(-1), probs.size(-1)).to(probs.dtype)
            probs_one_hot = hard + probs - probs.detach()
        else:
            raise ValueError(f'unknown patch_select: {self.patch_select}')

        sel_len = (lens_eff.to(scores.dtype) * probs_one_hot).sum(-1).long()  # [C, B]

        raw_scores = (scores_all * probs_one_hot.unsqueeze(-1)).sum(dim=2)    # [C, B, K]
        sel_moe = F.softmax(raw_scores * self.log_gate_temp.exp(), dim=-1)
        if self.gate_scope != 'token':
            sel_moe = sel_moe.mean(dim=1)                                     # [C, K]
        return sel_moe, sel_len

    def _gather_patches_ps(self, x, curr_pos, sel_len):
        """逐样本取块 -> [C, B, max_len]，右侧零填充。"""
        xt = x.transpose(0, 1)  # [C, B, L]
        L = xt.size(-1)
        grid_t = torch.arange(self.max_patch_len, device=x.device)
        idx_abs = curr_pos.unsqueeze(-1) + grid_t.view(1, 1, -1)
        patch = xt.gather(2, idx_abs.clamp(max=L - 1))
        mask = (grid_t.view(1, 1, -1) < sel_len.unsqueeze(-1)) & (idx_abs < L)
        return patch * mask.type_as(patch)

    def _build_fixed_random(self, n_vars):
        """构造期一次性抽定一组可行分段，此后恒定不变（逐通道不同，但不随样本/批次变）。

        这是把"非等分"与"边界会变"两个因素拆开的对照挡：与 uniform 比只多了"段长不等"，
        与 greedy_fixed 比只少了"边界随输入变"。实测 greedy_fixed（无论整批还是逐样本
        定界）都比 uniform 差约 16%，本挡用于判定该惩罚来自哪一项。
        """
        gen = torch.Generator().manual_seed(0)  # 与训练随机种子无关，保证配方可复现
        rows = []
        for _ in range(n_vars):
            rem, lens = self.seq_len, []
            for m in range(self.n_tokens, 0, -1):
                cand = [st for st in self.steps_list
                        if st <= rem and bool(self.reach[m - 1, rem - st])]
                st = cand[torch.randint(len(cand), (1,), generator=gen).item()]
                lens.append(st)
                rem -= st
            rows.append(lens)
        self.register_buffer('fixed_lens', torch.tensor(rows, dtype=torch.long),
                             persistent=False)  # [C, N]
        pidx = [[self.steps_list.index(st) for st in lens] for lens in rows]
        self.register_buffer('fixed_pidx', torch.tensor(pidx, dtype=torch.long),
                             persistent=False)  # [C, N] 每段在候选池中的下标（重采样查表用）

    def _fixed_random_forward(self, x, pattern_emb):
        """固定非等分分块：段长表恒定，与 uniform 挡的唯一差别就是"段长不等"。"""
        B, C, L = x.shape
        N, max_len = self.n_tokens, self.max_patch_len

        lens = self.fixed_lens                              # [C, N]
        starts = lens.cumsum(-1) - lens
        grid = torch.arange(max_len, device=x.device)
        idx = (starts.unsqueeze(-1) + grid).clamp(max=L - 1)  # [C, N, max_len]
        mask = (grid < lens.unsqueeze(-1)).to(x.dtype)

        xt = x.transpose(0, 1)                               # [C, B, L]
        gathered = xt.gather(2, idx.reshape(C, -1).unsqueeze(1).expand(C, B, -1))
        patches_cbn = gathered.reshape(C, B, N, max_len) * mask.unsqueeze(1)

        if self.patch_norm == 'resample':
            # 每段按自己的长度查表重采样到 T0：[C, N, T0, max_len] x [C, B, N, max_len]
            w = self.resample_w[self.fixed_pidx]
            patches_cbn = torch.einsum('cntm, cbnm -> cbnt', w, patches_cbn)
        spec = torch.fft.rfft(patches_cbn.permute(0, 2, 1, 3), n=self.token_len, dim=-1).real
        patterns = pattern_emb.unsqueeze(2).expand(-1, -1, B, -1)
        temp = self.log_gate_temp.exp()
        if self.gate_scope == 'token':
            sc = self.sra(spec, patterns, sigma=self.sra_sigma, reduce_batch=False)
            sel_moe = F.softmax(sc * temp, dim=-1).permute(1, 0, 2, 3).reshape(B, C * N, -1)
        else:
            sc = self.sra(spec, patterns, sigma=self.sra_sigma)   # [C, N, K]
            sel_moe = F.softmax(sc.mean(dim=1) * temp, dim=-1)

        patches = patches_cbn.permute(1, 0, 2, 3).reshape(B, C * N, self.token_len)
        var_ids = torch.arange(C, device=x.device).repeat_interleave(N)
        return patches, lens.reshape(-1), var_ids, sel_moe

    def _greedy_v2_forward(self, x, pattern_emb):
        """自适应 v2：游标贪心 + 恰好 N 段（可达性表），每步——
          候选段 ──面积重采样到 T0──▶ rFFT ──▶ 与原型的共振分数 s(ℓ) [K]
                                              └─▶ 路由器 MLP([s(ℓ), log ℓ/T0]) → logit(ℓ)
          直通硬选 ℓ*：前向取候选 ℓ* 的重采样段，反向对每个候选给出各自的梯度
          （混合作用在候选段上，而不是标量权重上）；门控仍用选中段的共振分数。
        返回 patches [B, C*N, T0]、lengths [B, C*N]（逐样本）、var_ids、sel_moe。
        """
        B, C, L = x.shape
        N, T0, M = self.n_tokens, self.resample_len, self.max_patch_len
        xt = x.transpose(0, 1)                                            # [C, B, L]
        patterns = pattern_emb.unsqueeze(2).expand(-1, -1, B, -1)         # [C, K, B, F]
        cand = x.new_tensor(self.steps_list)                              # [P]
        P = cand.numel()
        log_len = torch.log(cand / T0).view(1, 1, P, 1).expand(C, B, P, 1)
        grid = torch.arange(M, device=x.device)
        temp = self.log_gate_temp.exp()

        curr = x.new_zeros((C, B), dtype=torch.long)
        segs, lens, moes, soft = [], [], [], []
        for i in range(N):
            idx = curr.unsqueeze(-1) + grid                                # [C, B, M]
            raw = xt.gather(2, idx.clamp(max=L - 1)) * (idx < L).to(x.dtype)
            cands = torch.einsum('ptm, cbm -> cbpt', self.resample_w, raw)  # [C, B, P, T0]
            spec = torch.fft.rfft(cands, n=T0, dim=-1).real                 # [C, B, P, F]
            scores_all = self.sra(spec.transpose(1, 2), patterns, sigma=self.sra_sigma,
                                  reduce_batch=False)                      # [C, B, P, K]
            logit = self.router(torch.cat((scores_all, log_len), dim=-1)).squeeze(-1)

            rest = (L - curr).unsqueeze(-1) - cand.long()                  # [C, B, P]
            valid = (rest >= 0) & self.reach[N - i - 1][rest.clamp_min(0)]
            logit = logit.masked_fill(~valid, -1e9)

            probs = F.softmax(logit / self.gumbel_tau, dim=-1)
            hard = F.one_hot(probs.argmax(-1), P).to(probs.dtype)
            if not self.training:
                p_st = hard
            elif self.patch_select == 'st':
                p_st = hard + probs - probs.detach()
            elif self.patch_select == 'gumbel':
                p_st = F.gumbel_softmax(logit, tau=self.gumbel_tau, hard=True, dim=-1)
                hard = p_st.detach()
            else:
                raise ValueError(f'unknown patch_select: {self.patch_select}')

            sel_len = (cand * hard).sum(-1).long()                         # [C, B]
            segs.append((cands * p_st.unsqueeze(-1)).sum(2))               # [C, B, T0]
            lens.append(sel_len)
            raw_scores = (scores_all * p_st.unsqueeze(-1)).sum(2)          # [C, B, K]
            sel_moe = F.softmax(raw_scores * temp, dim=-1)
            moes.append(sel_moe if self.gate_scope == 'token' else sel_moe.mean(dim=1))
            soft.append(probs)
            if not self.training:
                self.len_hist[i] += hard.sum(dim=(0, 1))
            curr = curr + sel_len

        # 步主序 [N, C, B, ·] -> [B, C, N, ·] -> [B, C*N, ·]（通道主序，与 var_ids 一致）
        patches = torch.stack(segs, dim=0).permute(2, 1, 0, 3).reshape(B, C * N, T0)
        lengths = torch.stack(lens, dim=0).permute(2, 1, 0).reshape(B, C * N)
        var_ids = torch.arange(C, device=x.device).repeat_interleave(N)
        if self.gate_scope == 'token':
            sel_moe = torch.stack(moes, dim=0).permute(2, 1, 0, 3).reshape(B, C * N, -1)
        else:
            sel_moe = moes[-1]

        if self.training and self.adp_lambda > 0:
            # 负载/熵正则：整批所有决策的平均软选择分布 p̄ 应远离单点（log P − H(p̄) ≥ 0，等于 0 即均匀）
            p_bar = torch.stack(soft, dim=0).mean(dim=(0, 1, 2))            # [P]
            entropy = -(p_bar * (p_bar + 1e-9).log()).sum()
            self.aux_loss = self.adp_lambda * (math.log(P) - entropy)
        else:
            self.aux_loss = None
        return patches, lengths, var_ids, sel_moe

    def _greedy_fixed_forward(self, x, pattern_emb):
        """贪心迭代 + 固定段数：游标从左往右扫，每步由谱共振决定下一段多长，
        但恰好走 N 步、恰好铺满 L（候选长度过可达性掩码）。

        与 uniform 挡的唯一差别就是"边界是否自适应"，二者构成 Adaptive 主张的干净
        A/B 对照。段数固定 → token 可展平，故 tokenize=direct 也能用本挡。
        """
        B, C, L = x.shape
        N = self.n_tokens
        patterns = pattern_emb.unsqueeze(2).expand(-1, -1, B, -1)  # [C, K, B, F]

        cand_pool = x.new_tensor(self.steps_list)
        per_sample = self.boundary_scope == 'sample'

        curr_pos = x.new_zeros((C, B) if per_sample else (C,), dtype=torch.long)
        list_patches, list_lens, list_vars, list_moe = [], [], [], []
        for i in range(N):
            if per_sample:
                sel_moe, sel_len = self._propose_next_patch_ps(
                    x, curr_pos, cand_pool, patterns, rem_steps=N - i)
                list_patches.append(
                    self._gather_patches_ps(x, curr_pos, sel_len).transpose(0, 1))
                list_lens.append(sel_len)  # [C, B]
            else:
                sel_moe, sel_len = self._propose_next_patch(
                    x, curr_pos, L - curr_pos, cand_pool, patterns, rem_steps=N - i)
                self._gather_patches(x, curr_pos, sel_len,
                                     list_patches, list_lens, list_vars)
            list_moe.append(sel_moe)
            curr_pos = curr_pos + sel_len.long()

        # 可达性掩码保证每通道每步都取到一段，故各步张量形状一致，可直接 stack。
        # 步主序 -> 通道主序（tokenize=direct 的展平按通道分组，顺序不能错）
        patches = torch.stack(list_patches, dim=1)                  # [B, N, C, max_len]
        patches = patches.transpose(1, 2).reshape(B, C * N, -1)
        if per_sample:
            # [N, C, B] -> [B, C, N] -> [B, C*N]；逐样本段长，仅供诊断与断言
            lengths = torch.stack(list_lens, dim=0).permute(2, 1, 0).reshape(B, C * N)
        else:
            lengths = torch.stack(list_lens, dim=0).transpose(0, 1).reshape(-1)  # [C*N]
        var_ids = torch.arange(C, device=x.device).repeat_interleave(N)

        if self.gate_scope == 'token':
            # 各步 [C, B, K] -> [B, C, N, K] -> [B, C*N, K]
            sel_moe = torch.stack(list_moe, dim=0).permute(2, 1, 0, 3).reshape(B, C * N, -1)
        else:
            sel_moe = list_moe[-1]  # [C, K]，原版行为：末轮门控逐通道复用

        return patches, lengths, var_ids, sel_moe

    def _calc_spectral_scores(self, x, curr_pos, lens_eff, patterns, reduce_batch=True):
        """x: [B, C, L]; curr_pos: [C]; lens_eff: [C, P]; patterns: [C, K, B, F]
        -> [C, P, K]（reduce_batch=True）或 [C, B, P, K]
        """
        x = x.transpose(0, 1)  # [C, B, L]
        C, B, L = x.shape
        max_len = self.max_patch_len

        grid_t = torch.arange(max_len, device=x.device)
        idx_abs = curr_pos.unsqueeze(-1) + grid_t.unsqueeze(0)  # [C, max_len]
        idx_gather = idx_abs.clamp(max=L - 1).unsqueeze(1).expand(-1, B, -1)
        patches_raw = x.gather(2, idx_gather)  # [C, B, max_len]

        # 双重掩码：序列边界 + 候选长度
        mask_seq = (idx_abs < L).to(x.dtype).unsqueeze(1)            # [C, 1, max_len]
        mask_len = (grid_t.view(1, 1, -1) < lens_eff.unsqueeze(-1)).to(x.dtype)  # [C, P, max_len]
        patches_masked = (patches_raw * mask_seq).unsqueeze(1) * mask_len.unsqueeze(2)

        spec_complex = torch.fft.rfft(patches_masked, n=max_len, dim=-1)
        spec_rep = spec_complex.real  # [C, P, B, F]

        return self.sra(spec_rep, patterns, sigma=self.sra_sigma, reduce_batch=reduce_batch)

    def _propose_next_patch(self, x, curr_pos, rem_len, cand_pool, patterns,
                            rem_steps=None):
        """返回 sel_moe（[C, K] 或 gate_scope=token 时 [C, B, K]）与 sel_len [C]。

        rem_steps=None 为原版贪心：候选长度按剩余长度截断，段数随数据涌现。
        rem_steps=M 为 greedy_fixed：候选长度不截断，且必须留下能被恰好 M-1 段铺满
        的余量（查精确可达性表），从而保证总步数恰为 N、总长恰为 L。
        """
        B, C, L = x.shape

        if rem_steps is None:
            lens_eff = cand_pool.view(1, -1).minimum(rem_len.unsqueeze(-1))  # [C, P]
            mask_valid = lens_eff >= self.min_patch_len
        else:
            lens_eff = cand_pool.view(1, -1).expand(C, -1).contiguous()
            rest = rem_len.unsqueeze(-1) - lens_eff.long()  # [C, P] 选完后的剩余长度
            mask_valid = (rest >= 0) & self.reach[rem_steps - 1][rest.clamp_min(0)]

        scores_b = self._calc_spectral_scores(
            x, curr_pos, lens_eff, patterns,
            reduce_batch=(self.gate_scope != 'token'))
        # 分块方案对整个 batch 共享（原版行为），故打分恒用 batch 均值
        scores_all = scores_b if self.gate_scope != 'token' else scores_b.mean(dim=1)
        scores = scores_all.amax(dim=-1)  # [C, P]

        scores = scores.clone()
        scores[~mask_valid] = -1e9

        # 推理期恒取 argmax（确定性）；训练期按 patch_select 选择随机或确定性硬选择。
        # gumbel（原版）每步重采样分块方案，是本模型 run 间方差的主要来源（ETTh1 实测
        # 相对标准差 2.7~5.1%）；st 用无噪声直通估计，前向确定、梯度仍经 softmax 回传。
        if not self.training:
            probs_one_hot = F.one_hot(scores.argmax(dim=-1), scores.size(-1)).to(scores.dtype)
        elif self.patch_select == 'gumbel':
            probs_one_hot = F.gumbel_softmax(scores, tau=self.gumbel_tau, hard=True, dim=-1)
        elif self.patch_select == 'st':
            probs = F.softmax(scores / self.gumbel_tau, dim=-1)
            hard = F.one_hot(probs.argmax(dim=-1), probs.size(-1)).to(probs.dtype)
            probs_one_hot = hard + probs - probs.detach()
        else:
            raise ValueError(f'unknown patch_select: {self.patch_select}')

        sel_len = (lens_eff.float() * probs_one_hot).sum(dim=-1).long()

        if self.gate_scope == 'token':
            # 逐样本门控：选中段的每样本共振分数 [C, B, K]（边界仍 batch 共享）
            raw_scores = (scores_b * probs_one_hot[:, None, :, None]).sum(dim=2)
        else:
            raw_scores = (scores_all * probs_one_hot.unsqueeze(-1)).sum(dim=1)  # [C, K]
        sel_moe = F.softmax(raw_scores * self.log_gate_temp.exp(), dim=-1)

        return sel_moe, sel_len

    def _gather_patches(self, x, curr_pos, sel_len, list_patches, list_lens, list_vars):
        """按各通道游标与选中长度抽取 patch，右侧零填充到 max_patch_len。"""
        B, C, L = x.shape
        max_len = self.max_patch_len

        curr_pos = curr_pos.long()
        rem = (L - curr_pos).clamp_min(0)
        lens_eff = sel_len.long().minimum(rem)  # [C]

        mask_valid = lens_eff > 0
        if not mask_valid.any():
            return

        ids = torch.nonzero(mask_valid, as_tuple=True)[0]  # [Nv]
        Nv = ids.numel()
        pos_valid = curr_pos[ids]
        len_valid = lens_eff[ids]

        grid_t = torch.arange(max_len, device=x.device)
        idx_abs = pos_valid.unsqueeze(1) + grid_t.unsqueeze(0)  # [Nv, max_len]
        idx_clamped = idx_abs.clamp(max=L - 1)
        gather_ids = idx_clamped.unsqueeze(1).expand(Nv, B, max_len)

        x_trans = x.permute(1, 0, 2)  # [C, B, L]
        x_valid = x_trans[ids]        # [Nv, B, L]
        patches = x_valid.gather(dim=2, index=gather_ids)  # [Nv, B, max_len]

        mask_pad = grid_t.view(1, 1, -1) < len_valid.view(Nv, 1, 1)
        patches = patches * mask_pad.type_as(patches)

        list_patches.append(patches.permute(1, 0, 2).contiguous())  # [B, Nv, max_len]
        list_lens.append(len_valid)
        list_vars.append(ids)

    def reconstruct_from_patches(self, patches, patch_lens, var_ids, n_vars, seq_len):
        """逆操作：把 [B, P, max_len] 的 patch 散射回 [B, C, L]。"""
        device = patches.device
        dtype = patches.dtype
        B, P, max_len = patches.shape

        patch_lens = patch_lens.to(device, dtype=torch.long)
        var_ids = var_ids.to(device, dtype=torch.long)

        # 分段前缀和求每个 patch 在其通道内的起点
        mask_v = F.one_hot(var_ids, num_classes=n_vars).T  # [C, P]
        lens_masked = mask_v * patch_lens.unsqueeze(0)
        lens_cum = lens_masked.cumsum(dim=1)
        starts_map = lens_cum - lens_masked
        idx_p = torch.arange(P, device=device)
        start_pos = starts_map[var_ids, idx_p]  # [P]

        grid_t = torch.arange(max_len, device=device)
        idx_t_abs = start_pos.unsqueeze(1) + grid_t.unsqueeze(0)          # [P, max_len]
        mask_valid = grid_t.unsqueeze(0) < patch_lens.unsqueeze(1)        # [P, max_len]

        x_rec = torch.zeros(B, n_vars, seq_len, device=device, dtype=dtype)

        idx_b_grid = torch.arange(B, device=device).view(B, 1, 1).expand(B, P, max_len)
        idx_v_grid = var_ids.view(1, P, 1).expand(B, P, max_len)
        idx_t_grid = idx_t_abs.view(1, P, max_len).expand(B, P, max_len)
        mask_valid_exp = mask_valid.view(1, P, max_len).expand(B, -1, -1)

        b_flat = idx_b_grid[mask_valid_exp]
        v_flat = idx_v_grid[mask_valid_exp]
        t_flat = idx_t_grid[mask_valid_exp].clamp(max=seq_len - 1)
        val_flat = patches[mask_valid_exp]

        x_rec.index_put_((b_flat, v_flat, t_flat), val_flat, accumulate=False)
        return x_rec

    def _uniform_forward(self, x, pattern_emb):
        """等分 N 段（段数固定，不自适应）。段长 = L // N，故无需零填充。

        谱共振打分仍照常进行——它产出 MoE 门控 sel_moe，使谱原型、SRA 稀疏匹配、
        专家门控全部留在通路上；被移除的只有"边界自适应"这一项，因此本挡与
        patch_mode=greedy_fixed 构成 Adaptive 主张的干净 A/B 对照。

        Returns 与 forward 同约定；lengths 恒为 token_len，var_ids 按通道主序排列。
        """
        B, C, L = x.shape
        N, seg = self.n_tokens, self.token_len

        # [B, C, L] -> [B, C, N, seg] -> [B, C*N, seg]（通道主序，与 var_ids 一致）
        patches = x.reshape(B, C, N, seg).reshape(B, C * N, seg)

        # 谱共振打分：段谱 [C, N, B, F] vs 原型 [C, K, B, F] -> [C, N, K]
        spec = torch.fft.rfft(x.reshape(B, C, N, seg).permute(1, 2, 0, 3), n=seg, dim=-1).real
        patterns = pattern_emb.unsqueeze(2).expand(-1, -1, B, -1)
        temp = self.log_gate_temp.exp()
        if self.gate_scope == 'token':
            # [C, B, N, K] -> [B, C, N, K] -> [B, C*N, K]（通道主序，与 var_ids 一致）
            scores = self.sra(spec, patterns, sigma=self.sra_sigma, reduce_batch=False)
            sel_moe = F.softmax(scores * temp, dim=-1).permute(1, 0, 2, 3).reshape(B, C * N, -1)
        else:
            scores_all = self.sra(spec, patterns, sigma=self.sra_sigma)   # [C, N, K]
            sel_moe = F.softmax(scores_all.mean(dim=1) * temp, dim=-1)    # [C, K]

        lengths = x.new_full((C * N,), seg, dtype=torch.long)
        var_ids = torch.arange(C, device=x.device).repeat_interleave(N)
        return patches, lengths, var_ids, sel_moe

    def forward(self, x, pattern_emb, part_idx=None):
        """x: [B, C, L]; pattern_emb: [C, K, F]; part_idx: [B] 划分下标（仅 oracle 挡，由模型给出）

        Returns:
            patches: [B, P_total, max_len]
            lengths: [P_total]
            var_ids: [P_total]
            sel_moe: [C, K]（末轮门控，逐通道复用于该通道全部 patch，原版行为）
        """
        if self.patch_mode == 'uniform':
            return self._uniform_forward(x, pattern_emb)
        if self.patch_mode == 'greedy_fixed':
            return self._greedy_fixed_forward(x, pattern_emb)
        if self.patch_mode == 'fixed_random':
            return self._fixed_random_forward(x, pattern_emb)
        if self.patch_mode == 'greedy_v2':
            return self._greedy_v2_forward(x, pattern_emb)
        if self.patch_mode == 'oracle':
            if part_idx is None:
                raise ValueError('patch_mode=oracle 需要模型给出 part_idx')
            if bool((part_idx == self.uniform_part).all()):
                # 整批都是等分：走等分挡原生路径（reshape，不经 gather/重采样矩阵乘）。数值上两条路相同，
                # 但 GPU 上 gather 的反向用原子加、累加顺序不定，30 轮后与固定 patch 漂开；走同一路径才逐位相同
                if not self.training:
                    self.part_hist[self.uniform_part] += x.size(0)
                return self._uniform_forward(x, pattern_emb)
            return self._partition_forward(x, pattern_emb, part_idx)

        B, C, L = x.shape
        pattern_emb = pattern_emb.unsqueeze(2).expand(-1, -1, B, -1)  # [C, K, B, F]

        mask_active = x.new_ones(C, dtype=torch.bool)
        curr_pos = x.new_zeros(C, dtype=torch.long)
        cand_pool = x.new_tensor(self.steps_list)

        list_patches, list_lens, list_vars, list_moe = [], [], [], []

        while mask_active.any():
            rem_len = L - curr_pos
            sel_moe, sel_len = self._propose_next_patch(
                x, curr_pos, rem_len, cand_pool, pattern_emb)

            n_before = len(list_vars)
            self._gather_patches(x, curr_pos, sel_len, list_patches, list_lens, list_vars)
            if self.gate_scope == 'token' and len(list_vars) > n_before:
                # 本步取到 patch 的通道为 list_vars[-1]；门控按同一顺序取行，
                # 拼接后与 patches 的 [B, P_total, ...] 逐 patch 对齐
                list_moe.append(sel_moe[list_vars[-1]].permute(1, 0, 2))  # [B, Nv, K]

            curr_pos = curr_pos + sel_len.long()
            mask_active = curr_pos < L

        patches = torch.cat(list_patches, dim=1)  # [B, P_total, max_len]
        lengths = torch.cat(list_lens, dim=0)
        var_ids = torch.cat(list_vars, dim=0)
        if self.gate_scope == 'token':
            sel_moe = torch.cat(list_moe, dim=1)  # [B, P_total, K]

        return patches, lengths, var_ids, sel_moe


class PartitionRouter(nn.Module):
    """oracle 挡的划分路由器：全窗逐通道幅度谱 log1p(|rFFT|) → 逐通道 MLP → 通道均值 → P 个划分的 logit。

    训练时用交叉熵模仿 E 步（真实损失 argmin）的选择，推理时 argmax；它是有监督分类器，不经直通估计。
    """

    def __init__(self, seq_len, n_parts, hidden):
        super().__init__()
        n_freq = seq_len // 2 + 1
        self.net = nn.Sequential(nn.Linear(n_freq, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, n_parts)

    def forward(self, x):
        """x: [B, C, L] -> logits [B, P]"""
        feat = torch.log1p(torch.fft.rfft(x, dim=-1).abs())
        return self.out(self.net(feat).mean(dim=1))


class HeteroGraphAttention(nn.Module):
    """原版 layers/HGNN.py 的 HeteroGraphAttention（删 dead 参数后等价）。

    借鉴 HGT (WWW 2020) 与多重网络：把异构性分解为两步以省显存——
    1. 模式内聚合（同一模式下变量↔变量）
    2. 模式间聚合（同一变量下模式↔模式）
    输入输出均为 [K, N, D_p]（K 个专家滤波器 × N 个变量）。
    """

    def __init__(self, configs, split_gate=False):
        super(HeteroGraphAttention, self).__init__()
        self.p_model = configs.p_model
        self.n_heads = configs.n_heads
        self.d_k = self.p_model // self.n_heads
        self.inner_dim = self.n_heads * self.d_k

        # split_gate（原版无，仅 token_mix 用）：两步聚合各配一个标量门再相加。
        # 用在 token 序列上时 Step 1 = 同通道跨时间、Step 2 = 同时刻跨通道；共用一个门
        # 只能同开同关，而 ETTh1 跨通道可提取信息≈0、关掉整个混合反好 1.8%。初值 1 与
        # 不分门逐位等价；训练后门值直接读出该数据集用了哪种关系。
        self.split_gate = split_gate
        if split_gate:
            self.gate_intra = nn.Parameter(torch.ones(1))
            self.gate_inter = nn.Parameter(torch.ones(1))

        # 共享投影矩阵 + Pattern Embedding 作为上下文（避免逐关系对参数爆炸）
        self.w_q = nn.Linear(self.p_model, self.inner_dim)
        self.w_k = nn.Linear(self.p_model, self.inner_dim)
        self.w_v = nn.Linear(self.p_model, self.inner_dim)
        self.w_o = nn.Linear(self.inner_dim, self.p_model)

        # 关系先验：0 = 模式内边（变量交互），1 = 模式间边（模式交互）
        self.relation_pri = nn.Parameter(torch.ones(2, self.n_heads))

        self.norm = nn.LayerNorm(self.p_model)
        self.dropout = nn.Dropout(configs.dropout)

    def forward(self, x, pattern_emb):
        """x / pattern_emb: [K, N, D_p]（原版）或 [B, K, N, D_p]（逐样本上下文）。

        前导 batch 维用于 hgnn_ctx=spectral：节点上下文携带逐样本谱画像时，编码出的
        专家滤波器随样本变化，才不再是自由参数的重参数化。
        """
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        if pattern_emb.dim() == 3:
            pattern_emb = pattern_emb.unsqueeze(0)
        B, K, N, D = x.shape
        H, dk = self.n_heads, self.d_k
        residual = x

        h = x + pattern_emb

        q = self.w_q(h).view(B, K, N, H, dk)
        k = self.w_k(h).view(B, K, N, H, dk)
        v = self.w_v(h).view(B, K, N, H, dk)

        # Step 1: 模式内（每个 K 内部 N 个节点互相作用）
        q_intra = q.permute(0, 1, 3, 2, 4)  # [B, K, H, N, d_k]
        k_intra = k.permute(0, 1, 3, 2, 4)
        v_intra = v.permute(0, 1, 3, 2, 4)

        scores_intra = torch.matmul(q_intra, k_intra.transpose(-1, -2)) / math.sqrt(dk)
        scores_intra = scores_intra * self.relation_pri[0].view(1, 1, -1, 1, 1)
        attn_intra = self.dropout(torch.softmax(scores_intra, dim=-1))
        out_intra = torch.matmul(attn_intra, v_intra).permute(0, 1, 3, 2, 4).reshape(B, K, N, -1)

        # Step 2: 模式间（每个 N 上 K 个模式互相作用）
        q_inter = q.permute(0, 2, 3, 1, 4)  # [B, N, H, K, d_k]
        k_inter = k.permute(0, 2, 3, 1, 4)
        v_inter = v.permute(0, 2, 3, 1, 4)

        scores_inter = torch.matmul(q_inter, k_inter.transpose(-1, -2)) / math.sqrt(dk)
        scores_inter = scores_inter * self.relation_pri[1].view(1, 1, -1, 1, 1)
        attn_inter = self.dropout(torch.softmax(scores_inter, dim=-1))
        out_inter = torch.matmul(attn_inter, v_inter).permute(0, 3, 1, 2, 4).reshape(B, K, N, -1)

        if self.split_gate:
            out = self.w_o(self.gate_intra * out_intra + self.gate_inter * out_inter)
        else:
            out = self.w_o(out_intra + out_inter)
        out = self.norm(residual + out)
        return out.squeeze(0) if squeeze else out


class SpectralMoE(nn.Module):
    """原版 layers/SpectralMoE.py（删静态分支后等价）：频域混合专家滤波。

    依据卷积定理（时域卷积 ⇔ 频域 Hadamard 积），K 个时域滤波器 [K, C, D_p]
    经异构图注意力编码后做 rFFT，按门控 sel_moe 融合成逐通道有效滤波器，
    与 patch 频谱逐点相乘；季节谱分支与滑动平均趋势分支按 seasonal_alpha 混合。
    """

    def __init__(self, configs, max_len):
        super().__init__()
        self.n_vars = configs.enc_in
        self.n_patterns = configs.pattern_nums
        self.p_model = configs.p_model
        self.F = max_len // 2 + 1

        # tokenize=recon（原版）：滤波结果经 emb_out 回到 max_len 维，再散射回 [B,C,L]；
        # tokenize=direct：直接把 D_p 维特征当 token 交给预测头，emb_out 不构造
        # （散射回原形状会让预测头拿到与原始输入等价的信息，从而绕开整个主干——
        #  实测主干旁路对照四格差异 ≤0.4%，即主干贡献不可区分于零）。
        self.tokenize = configs.tokenize
        self.emb_in = nn.Linear(max_len, self.p_model)
        if self.tokenize == 'recon':
            self.emb_out = nn.Linear(self.p_model, max_len)
        elif self.tokenize != 'direct':
            raise ValueError(f'unknown tokenize: {self.tokenize}')

        self.alpha = configs.seasonal_alpha
        # 注意：x 为 [B, P, D_p]，此处沿 patch 轴（dim=1）平滑，见文件头备案
        self.decompsition = SeriesDecomp(configs.moving_avg)

        # 消融: use_hgnn=false 时专家滤波器不经异构图上下文化（emb_pat 一并不构造，避免死参数）
        self.use_hgnn = configs.use_hgnn
        # hgnn_ctx（原版只有 static）：原版图节点上下文是 pattern_emb，而默认
        # pattern_strategy=1 下它是纯自由参数——于是 HGNN 输出只是自由参数的重参数化，
        # 不引入任何新信息，use_hgnn 消融效应必为零（与门控 gate_scope=channel 同病）。
        # spectral 挡把"逐样本、跨通道的谱画像"注入节点上下文：专家滤波器随样本变化，
        # 且第一次让本模型具备真正的跨通道通路（通道独立线性基线最缺的机制）。
        self.hgnn_ctx = configs.hgnn_ctx
        if self.hgnn_ctx not in ('static', 'spectral'):
            raise ValueError(f'unknown hgnn_ctx: {self.hgnn_ctx}')
        self.f2 = self.p_model // 2 + 1
        if self.use_hgnn:
            self.emb_pat = nn.Linear(self.F, self.p_model)
            self.hgnn = HeteroGraphAttention(configs)
            if self.hgnn_ctx == 'spectral':
                self.ctx_norm = nn.LayerNorm(self.f2)
                self.ctx_proj = nn.Linear(self.f2, self.p_model)
                # 零初始化：初值下与 static 挡逐位等价，谱上下文作为从零学起的修正量
                nn.init.zeros_(self.ctx_proj.weight)
                nn.init.zeros_(self.ctx_proj.bias)
        # 消融: use_moe_gate=false 时门控退化为均匀 1/K（K 个专家等权平均）
        self.use_moe_gate = configs.use_moe_gate

        self.Linear_Trend = nn.Linear(self.p_model, self.p_model)

        # 时域存储专家滤波器，前向时做 rFFT 以自然满足 Hermitian 对称
        self.weights = nn.Parameter(torch.empty(self.n_patterns, self.n_vars, self.p_model))

        self.moe_init = configs.moe_init
        self.dropout = nn.Dropout(configs.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        # small（原版）：std=0.02，使 eff_w≈0 → 频域相乘把输入几乎清零（实测保留 5.6% 能量）
        # allpass：时域置单位冲激 w[...,0]=sqrt(D_p)，其 ortho-rFFT 恒为 1 → 起点是全通
        #   滤波器（恒等），主干不再摧毁信号，谱滤波从"不改变输入"开始学习偏离
        nn.init.normal_(self.weights, std=0.02)
        if self.moe_init == 'allpass':
            with torch.no_grad():
                self.weights[..., 0] += math.sqrt(self.p_model)
        elif self.moe_init != 'small':
            raise ValueError(f'unknown moe_init: {self.moe_init}')

    def forward(self, x, var_ids, pattern_emb, sel_moe):
        """x: [B, P, max_len]; var_ids: [P]; pattern_emb: [C, K, F]; sel_moe: [C, K]"""
        x = self.emb_in(x)                                        # [B, P, D_p]

        _, trend_init = self.decompsition(x)
        trend_output = self.Linear_Trend(trend_init)              # [B, P, D_p]

        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')          # [B, P, F2]

        if self.use_hgnn:
            pat = self.emb_pat(pattern_emb).permute(1, 0, 2)      # [K, C, D_p]
            if self.hgnn_ctx == 'spectral':
                # 逐通道谱画像：该通道所有 patch 的幅度谱均值 -> [B, C, F2]
                onehot = F.one_hot(var_ids, self.n_vars).type_as(x)          # [P, C]
                ctx = torch.einsum('bpf, pc -> bcf', x_freq.abs(), onehot)
                ctx = ctx / onehot.sum(0).clamp_min(1).view(1, -1, 1)
                ctx = self.ctx_proj(self.ctx_norm(ctx))                      # [B, C, D_p]
                # 图节点上下文 = 静态原型 + 逐样本谱画像（后者零初始化，起点与 static 等价）
                node_ctx = pat.unsqueeze(0) + ctx.unsqueeze(1)               # [B, K, C, D_p]
                w_enc = self.hgnn(
                    self.weights.unsqueeze(0).expand(node_ctx.size(0), -1, -1, -1), node_ctx)
            else:
                w_enc = self.hgnn(self.weights, pat)              # [K, C, D_p]
        else:
            w_enc = self.weights

        w_freq = torch.fft.rfft(w_enc, dim=-1, norm='ortho')      # [K,C,F2] 或 [B,K,C,F2]

        # 融合 MoE：y = x * Σ_k(w_k · g_k)，先算有效滤波器避免展开成 [K, B, C, F]
        if not self.use_moe_gate:
            sel_moe = torch.full_like(sel_moe, 1.0 / self.n_patterns)
        sel_moe = sel_moe.type_as(w_freq)
        per_sample_w = w_freq.dim() == 4                             # hgnn_ctx=spectral
        if sel_moe.dim() == 3:
            # 逐 token 门控 [B, P, K]：有效滤波器随样本与 patch 变化，无法被静态参数吸收
            if per_sample_w:
                eff_w = torch.einsum('bkpf, bpk -> bpf',
                                     w_freq[:, :, var_ids, :], sel_moe)
            else:
                eff_w = torch.einsum('kpf, bpk -> bpf',
                                     w_freq[:, var_ids, :], sel_moe)  # [B, P, F2]
            y_freq = x_freq * eff_w
        elif per_sample_w:
            eff_w = torch.einsum('bkcf, ck -> bcf', w_freq, sel_moe)  # [B, C, F2]
            y_freq = x_freq * eff_w[:, var_ids, :]
        else:
            eff_w = torch.einsum('kcf, ck -> cf', w_freq, sel_moe)   # [C, F2]
            y_freq = x_freq * eff_w[var_ids]                         # [B, P, F2]

        seasonal_output = torch.fft.irfft(y_freq, n=self.p_model, dim=-1, norm='ortho')
        y = self.alpha * seasonal_output + (1 - self.alpha) * trend_output
        if self.tokenize == 'recon':
            y = self.emb_out(y)                                 # [B, P, max_len]
        # direct 挡下 y 保持 [B, P, D_p]，即 token 本身

        return self.dropout(y)
