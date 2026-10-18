"""PRISM（主干移植自 FRAME）：Frequency-aware Relational Adaptive
Mixture-of-Experts for Multivariate Time Series Forecasting。

出处: 本项目自己的模型，原始实现在 ~/Projects/Time-Temp/Frame（models/Frame.py），
迁移进本仓库时改名为 PRISM。**不是第三方论文的复现**——本文件及
layers/PRISM_layers.py 中的「原版」一律指那份早先的 Frame 实现，其超参与行为
只是当时的自选值，没有需要恪守的外部权威；本仓库其他模型文件里的「官方」才是
指第三方 baseline 作者的仓库，两者语义不同。

因此凡经实验判定为缺陷的原版行为，默认值已直接改成修复版，原行为降级为可选
对照挡（供消融使用）：tokenize=direct、patch_mode=uniform、gate_scope=token、
boundary_scope=sample、patch_select=st、use_hgnn=false、head_type=linear、
n_tokens=6，各项理由见 configs/default.yaml 的 prism 节逐行注释。

迁移改动（相对原版源码）:
- 命名：模型注册名沿用本仓库既有的 PRISM，原版依赖层合并进
  layers/PRISM_layers.py（该文件头逐条列出层级改动：推理确定化、删 HGNN 死
  参数 relation_att/relation_msg、删不可达的 SpectralMoE 静态分支与 GCN 死
  导入、删无人消费的 pat_ids、高斯平滑带宽提为配置键）。
- 签名适配：原版 forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) 只用
  x_enc，此处对齐本框架四参签名，mark/dec 参数为占位；无辅助损失，恒返回单值
  [B, pred_len, c_out]。
- 超参改读扁平命名空间的 prism 节（configs/default.yaml），沿用原版键名：
  patch_pool / min_patch_len / pattern_nums / pattern_strategy / freq_topk /
  p_model / seasonal_alpha / moving_avg，另加原版硬编码提出的 gumbel_tau、
  sra_sigma；d_model（预测头宽度）、dropout、n_heads、enc_in、seq_len、
  pred_len 复用通用键。
- 增加构造期形状断言（p_model % n_heads、freq_topk ≤ F、min_patch_len ≤ seq_len），
  原版无检查、错配时在前向深处报错。
- 原版 run.py 的 --ablation / --submodule 两个 flag 模型代码从不读取，未移植。
- 训练配方（rec_lambda 时域 + auxi_lambda 频域的 FreDF 式复合损失）由本框架
  training 节提供；原版用全 fft，本框架用 rfft（实值信号下二者仅差 DC/Nyquist
  两 bin 的计数权重）。

其余结构与计算逻辑与原版逐行等价。
"""

import torch
import torch.nn.functional as F
from torch import nn

from layers.PRISM_layers import (HeteroGraphAttention, ParallelDynamicPatching,
                                 PartitionRouter, PatternEmbedding, RevIN, SeriesDecomp,
                                 SpectralMoE, resolve_token_len)


class Model(nn.Module):
    """数据流（[B, L, C] 输入）：

      RevIN → 谱模式原型 PatternEmbedding [C, K, F]
            → 并行动态分块 PDP（候选长度与原型谱共振打分，Gumbel 硬选择）
            → 谱混合专家 SpectralMoE（异构图注意力编码专家滤波器 → 频域相乘
              + 趋势分支按 seasonal_alpha 混合）
            → 逐 patch 散射回 [B, C, L]（recon 挡）/ token 展平后直喂头（direct 挡；
              非等分挡先经重栅格化 adp_regrid 投影回等分槽位）
            → 预测头（linear / mlp）
            (+) 近端流 aux_stream：tanh(γ)·Linear(x_aux → H)，与主干并联、零初始化门
            → RevIN 反归一化

    消融开关（configs/default.yaml 的 prism 节）：pattern_strategy（原型参数化
    五挡）/ seasonal_alpha（1.0 = 纯谱分支，0.0 = 纯趋势分支）/ patch_pool
    （单元素 = 退化为定长 patch）/ freq_topk（谱稀疏度）。
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.min_patch_len = configs.min_patch_len
        self.max_patch_len = self.min_patch_len + max(configs.patch_pool)
        self.pattern_strategy = configs.pattern_strategy

        # tokenize=direct：patch token 展平后是预测头的唯一输入，头再也读不到原始序列。
        # 原版 recon 挡把 patch 结果散射回 [B,C,L]（与输入同形状同量级），实测预测头
        # 可完全绕开主干——主干旁路对照四格差异 ≤0.4%、残差门自学到仅 0.06。
        self.tokenize = configs.tokenize
        self.patch_mode = configs.patch_mode
        self.n_tokens = configs.n_tokens
        if self.patch_mode in ('greedy_v2', 'oracle') and self.tokenize != 'direct':
            raise ValueError(f'patch_mode={self.patch_mode} 需要 tokenize=direct（逐样本段长无法走 recon 散射）')
        if self.tokenize == 'direct':
            if self.patch_mode == 'greedy':
                raise ValueError(
                    'tokenize=direct 需要固定段数，不能配 patch_mode=greedy'
                    '（原版贪心的段数随 batch 变化，无法展平）；'
                    '请用 uniform（等分）或 greedy_fixed（自适应边界 + 固定段数）')
            if configs.residual_mode != 'none':
                raise ValueError(
                    'tokenize=direct 必须配 residual_mode=none：残差会把原始序列'
                    '重新送进预测头，等于抵消本改动')
        elif self.tokenize != 'recon':
            raise ValueError(f'unknown tokenize: {self.tokenize}')

        # boundary_scope 只被 patch_mode=greedy_fixed 消费；仅在真正冲突时报错，
        # 不要因为一个用不到的键去拦截合法配置。
        # 冲突点：recon 的 reconstruct_from_patches 按 [P] 形状的段长散射，
        # 而逐样本边界下段长是 [B, P]，无法复用。
        if (configs.boundary_scope == 'sample' and self.patch_mode == 'greedy_fixed'
                and self.tokenize != 'direct'):
            raise ValueError(
                'boundary_scope=sample + patch_mode=greedy_fixed 需要 tokenize=direct：'
                'recon 挡的散射按整批共享的段长实现，无法处理逐样本边界')

        # SpectralMoE 的 patch 宽度：等分挡为段长，重采样管线为 adp_resample_len，零填充挡为池内最大长度
        self.token_len = resolve_token_len(configs)

        # 重栅格化（自适应 v2 的第三处接口，见 layers 文件头）：N 个自适应 token 按时间重叠权重
        # 投影回 N 个等分槽位（可微、无参数），展平头读到的第 j 个槽位永远覆盖 [j·L/N, (j+1)·L/N)。
        # 边界只改变"槽位内容从哪来"，不再改变槽位语义。uniform 挡下是恒等，故只对非等分挡生效。
        self.adp_regrid = configs.adp_regrid and self.patch_mode != 'uniform'

        assert configs.p_model % configs.n_heads == 0, (
            f'PRISM requires p_model % n_heads == 0, got p_model={configs.p_model}, '
            f'n_heads={configs.n_heads}')
        n_freq = self.token_len // 2 + 1
        assert configs.freq_topk <= n_freq, (
            f'PRISM requires freq_topk <= token_len//2+1 ({n_freq}), '
            f'got freq_topk={configs.freq_topk}, token_len={self.token_len}')
        assert self.min_patch_len <= self.seq_len, (
            f'PRISM requires min_patch_len <= seq_len, got min_patch_len='
            f'{self.min_patch_len}, seq_len={self.seq_len}')

        # 原版主干把 patch 输出散射回 [B, C, L] 后直接送 MLP 头，输入到头部无残差通路。
        # 实测初始化时主干输出与输入相关系数仅 0.045、能量仅 5.6%（谱权重 std=0.02 使
        # 频域相乘 y=x*≈0，主干近似零映射），头部因此从近噪声信号起步。
        #   gate_input     : enc = 主干(x) + tanh(g)·x（主干恒通，x 门控接入）
        #   gate_backbone  : enc = x + tanh(g)·主干(x)（x 恒通，主干作为零初始化修正）
        # 后者是标准零初始化残差分支：训练从"头部直接读原始序列"这一良态起点出发。
        self.residual_mode = configs.residual_mode
        assert self.residual_mode in ('none', 'gate_input', 'gate_backbone'), \
            f'unknown residual_mode: {self.residual_mode}'

        # 近端流 aux_stream（原版无）：与主干并联的一条线性通路，零初始化 tanh 门相加。
        # 动机：direct 挡下最后几个原始观测只能经"末段谱滤波→token"进头，谱滤波是频带选择，
        # 抹掉相位精确的末尾值。ETTh1 96 格按预测步拆误差：第 1 步落后赢家 24.6%、1–4 步
        # 15.6%、64–96 步仅 1.2%（且胜 xPatch）；旧版 MLP 头直读原始序列时 96 格好 2% 但
        # 336 格差 3%；96 格两个赢家（TimeMosaic/xPatch）都有原始序列→输出的线性通路。
        #   full  : x_aux = 全窗归一化序列 [B,C,L]（DLinear 式）
        #   trend : 时间轴滑动平均分解，趋势走本流、季节项进主干（xPatch/DLinear 分解式；
        #           原版唯一的"趋势分支"在 SpectralMoE 内沿 patch 轴平滑，无法外推趋势）
        #   last  : x_aux = 最后 L/N 个原始点（纯近端记忆，参数最少）
        # 门 γ 零初始化：full/last 初值与 none 逐位等价；trend 挡主干输入改为季节项。
        # 与 residual_mode 的区别：那是把原始序列喂回预测头（主干可被完全绕开），
        # 这里是能力上限为一层线性（≈DLinear）的独立门控流，主干必须优于它才能拿到权重。
        self.aux_stream = configs.aux_stream
        if self.aux_stream not in ('none', 'full', 'trend', 'last'):
            raise ValueError(f'unknown aux_stream: {self.aux_stream}')
        if self.aux_stream != 'none':
            self.aux_len = (self.seq_len // self.n_tokens if self.aux_stream == 'last'
                            else self.seq_len)
            self.aux_head = nn.Linear(self.aux_len, self.pred_len)
            self.aux_gate = nn.Parameter(torch.zeros(1))
            if self.aux_stream == 'trend':
                self.aux_decomp = SeriesDecomp(configs.aux_kernel)

        # 决定性对照：use_backbone=false 时完全旁路 FRAME 主干（分块+谱 MoE 都不构造），
        # 模型退化为 RevIN + 预测头。若该对照不劣于完整模型，则主干为死重。
        self.use_backbone = configs.use_backbone

        self.rin = RevIN(configs.enc_in, affine=True, subtract_last=False)
        if self.use_backbone:
            self.pat_emb = PatternEmbedding(configs, self.token_len, self.pattern_strategy)
            self.spec_moe = SpectralMoE(configs, self.token_len)
            self.pdp = ParallelDynamicPatching(configs)

        # oracle-EM 分块（见 layers 文件头）：E 步 oracle_step 用真实预测损失为每条样本选划分，
        # M 步在该划分上训练并用交叉熵训练路由器；推理走路由器 argmax。needs_oracle 让 exp 在每个
        # 训练 step 前调用 oracle_step、在测试末尾额外报告 oracle 上界与路由器一致率。
        self.needs_oracle = self.use_backbone and self.patch_mode == 'oracle'
        if self.needs_oracle:
            # 路由器用分叉的随机数流初始化：不消耗全局随机数，其后构造的 token 混合层、预测头与同种子的
            # 固定 patch 模型初始化逐位相同。于是路由器恒选等分时整个训练轨迹与固定 patch 逐位一致
            # （训练循环无全局梯度裁剪，路由器的交叉熵梯度只落在路由器自身参数上）。首轮实测未隔离时
            # 路由器 100% 选等分、336 格却因初始化不同比同种子固定 4×24 差 0.0067。
            with torch.random.fork_rng(devices=[]):
                self.router = PartitionRouter(self.seq_len, self.pdp.n_parts, configs.adp_router_hidden)
            self.oracle_warmup = configs.oracle_warmup
            self.oracle_ce = configs.oracle_ce
            self.oracle_chunk = configs.oracle_chunk
            # oracle_margin：E 步只在 argmin 划分的损失 < 等分损失 × (1 − margin) 时才离开等分，滤掉 31 选 1 的选择噪声
            # oracle_train_on：M 步用哪个划分训练。router = 路由器 argmax（训练即测试的用法，路由器恒为等分时训练
            #   与固定 patch 逐位相同——首轮实测 M 步按 oracle 标签训练、测试按路由器（100% 等分）用，训练/测试划分分布
            #   不一致，反而比固定差 1.3%）| oracle = E 步标签
            self.oracle_margin = configs.oracle_margin
            self.oracle_train_on = configs.oracle_train_on
            if self.oracle_train_on not in ('router', 'oracle'):
                raise ValueError(f'unknown oracle_train_on: {self.oracle_train_on}')
            self._oracle_part = None       # 当前训练 batch 的 E 步标签 [B]
            self._warm = True              # 暖启动期标志（由 oracle_step 维护）
            # 训练期标签统计（最近一个非暖启动 epoch）：[样本数, 非等分标签数, 路由器与标签一致数]
            self._tr, self._tr_epoch = [0, 0, 0], None
            self.register_buffer('router_hist', torch.zeros(self.pdp.n_parts), persistent=False)
            self.register_buffer('oracle_hist', torch.zeros(self.pdp.n_parts), persistent=False)
            self._agree = [0, 0]

        if self.residual_mode != 'none' and self.use_backbone:
            # 零初始化标量门；门值大小可直接读出主干对预测的实际贡献
            self.res_gate = nn.Parameter(torch.zeros(1))

        # 原版头为 Linear(L,d_model)-ReLU-Linear(d_model,H)，在 ETTh1 上独占 90~96% 参数
        # （主干恒 32.5k），既是长视野过拟合主嫌，也让消融效应被头部吸收。
        # head_type=linear 换成单层 Linear(L,H)，把容量让回主干；head_drop=0 时 mlp 挡与原版等价。
        # recon 挡头部读 [B,C,L]（in=L）；direct 挡读展平 token [B,C,N·D_p]
        # token_pos_enc：边界可变时，展平预测头对"第 i 个槽位"的固定权重会失配——槽位
        # 覆盖的时间范围逐样本漂移。实测 greedy_fixed（整批/逐样本定界均然）比等分差 16%。
        # 打开后给每个 token 拼上归一化的 (起点, 段长) 两维，让头至少能读到槽位的真实位置。
        # token_mix（原版无）：原版 token 之间从不交互——每个 patch 独立谱滤波后展平交给
        # 线性头，头虽混合 token 但权重是静态的、逐槽位固定的，且通道之间完全不通信。
        # 两个胜出 baseline 的共同点恰是"一层即全局感受野的数据依赖 token 混合"
        # （TimeFilter 是 C×T 联合图，TimeMosaic 是逐通道 attention），且各叠两层。
        # hgnn 挡把已有的 HeteroGraphAttention 从"给自由参数做重参数化"（消融恒为零，
        # 见 layers 文件头）移到 token 序列上：节点 = (通道, patch 位置)，其两步结构
        # intra(同组内跨节点)/inter(同节点跨组) 正好是 同通道跨时间 / 同时刻跨通道，
        # 即 TimeFilter 的 S/T 分解。零初始化门保证初值与 none 挡逐位等价。
        self.token_mix = configs.token_mix
        if self.token_mix not in ('none', 'hgnn'):
            raise ValueError(f'unknown token_mix: {self.token_mix}')
        if self.token_mix != 'none':
            if self.tokenize != 'direct':
                raise ValueError('token_mix 需要 tokenize=direct（recon 挡没有 token 序列）')
            # 两个胜出 baseline 均为 e_layers=2；每层各带零初始化门，加深不改变初值行为
            self.n_mix_layers = configs.token_mix_layers
            if self.n_mix_layers < 1:
                raise ValueError(f'token_mix_layers 须 >= 1，得到 {self.n_mix_layers}')
            # token_mix_gate=split：intra(跨时间)/inter(跨通道) 两步各配标量门（见 layers）
            if configs.token_mix_gate not in ('shared', 'split'):
                raise ValueError(f'unknown token_mix_gate: {configs.token_mix_gate}')
            self.token_hgnn = nn.ModuleList(
                HeteroGraphAttention(configs, split_gate=configs.token_mix_gate == 'split')
                for _ in range(self.n_mix_layers))
            # 位置/通道嵌入：图节点需要知道自己是"哪个通道的第几段"，否则两步注意力
            # 对同组节点完全对称，退化为置换不变的聚合
            self.token_pos = nn.Parameter(
                torch.zeros(configs.enc_in, self.n_tokens, configs.p_model))
            nn.init.normal_(self.token_pos, std=0.02)
            self.mix_gate = nn.Parameter(torch.zeros(self.n_mix_layers))

        self.token_pos_enc = configs.token_pos_enc
        tok_dim = configs.p_model + (2 if self.token_pos_enc else 0)
        head_in = (self.n_tokens * tok_dim if self.tokenize == 'direct' else self.seq_len)
        if configs.head_type == 'linear':
            head = [nn.Dropout(configs.head_drop)] if configs.head_drop > 0 else []
            head.append(nn.Linear(head_in, self.pred_len))
        elif configs.head_type == 'mlp':
            head = [nn.Linear(head_in, self.d_model), nn.ReLU()]
            if configs.head_drop > 0:
                head.append(nn.Dropout(configs.head_drop))
            head.append(nn.Linear(self.d_model, self.pred_len))
        else:
            raise ValueError(f'unknown head_type: {configs.head_type}')
        self.model = nn.Sequential(*head)

    def train(self, mode=True):
        # 进入 eval 即清零 oracle 挡的诊断计数（同 PDP.len_hist 的约定：测试末尾读到的只含测试集）
        if not mode and getattr(self, 'needs_oracle', False):
            self.router_hist.zero_()
            self.oracle_hist.zero_()
            self._agree = [0, 0]
        return super().train(mode)

    def _regrid(self, tokens, lengths, B, C, L):
        """token [B, C*N, D] 按时间重叠投影到等分槽位。lengths: [C*N] 或 [B, C*N]。

        W[b,c,j,i] = |[s_i,e_i) ∩ [j·seg,(j+1)·seg)| / seg；各段铺满 [0,L) 故每行和恒为 1。
        """
        n = self.n_tokens
        if bool((lengths == L // n).all()):
            return tokens      # 全为等分：重叠矩阵是单位阵，直接返回，GPU 上与等分挡逐位同路径
        lens = lengths.to(tokens.dtype)
        if lens.dim() == 1:
            lens = lens.unsqueeze(0).expand(B, -1)
        lens = lens.reshape(B, C, n)
        e_i = lens.cumsum(-1)
        s_i = e_i - lens                                             # [B, C, N_i]
        seg = L / n
        s_j = (torch.arange(n, device=tokens.device, dtype=tokens.dtype) * seg).view(1, 1, n, 1)
        ov = torch.minimum(e_i.unsqueeze(2), s_j + seg) - torch.maximum(s_i.unsqueeze(2), s_j)
        w = ov.clamp_min(0) / seg                                    # [B, C, N_j, N_i]
        return (w @ tokens.reshape(B, C, n, -1)).reshape(B, C * n, -1)

    def diagnostics(self):
        """训练后可读出的机制量（供 exp 在测试末尾打印）。"""
        out = {}
        if self.aux_stream != 'none':
            out['aux_gate(tanh)'] = f'{torch.tanh(self.aux_gate).item():+.4f}'
        if self.token_mix != 'none':
            out['mix_gate(tanh)'] = [f'{v:+.4f}' for v in torch.tanh(self.mix_gate).tolist()]
        if self.needs_oracle:
            lens = self.pdp.part_lens.tolist()
            for name, h in (('router', self.router_hist), ('oracle', self.oracle_hist)):
                tot = h.sum().item()
                if tot > 0:
                    top = torch.argsort(h, descending=True)[:6].tolist()
                    out[f'{name}_part_hist(top, %)'] = [
                        (lens[i], round(100 * h[i].item() / tot, 1)) for i in top]
                    out[f'{name}_uniform_share(%)'] = round(
                        100 * h[self.pdp.uniform_part].item() / tot, 1)
            if self._agree[1]:
                out['router_oracle_agree(%)'] = round(100 * self._agree[0] / self._agree[1], 1)
            if self._tr[0]:
                # 训练集上：E 步有多少样本选了非等分、路由器能否拟合这些标签（在样本内都拟合不了 → 特征/容量不足）
                out['train_label_nonuniform(%)'] = round(100 * self._tr[1] / self._tr[0], 1)
                out['train_router_agree(%)'] = round(100 * self._tr[2] / self._tr[0], 1)
        if self.use_backbone and self.patch_mode == 'greedy_v2':
            h = self.pdp.len_hist
            tot = h.sum(-1, keepdim=True).clamp_min(1)
            out['adp_len_pool'] = self.pdp.steps_list
            out['adp_len_hist(slot x pool, %)'] = [
                [round(v, 1) for v in row] for row in (100 * h / tot).tolist()]
        return out

    def _append_pos(self, tokens, lengths, B, C, L):
        """给每个 token 拼上归一化的 (起点, 段长)。lengths: [C*N] 或 [B, C*N]。"""
        n = self.n_tokens
        lens = lengths.to(tokens.dtype) / L
        if lens.dim() == 1:
            lens = lens.unsqueeze(0).expand(B, -1)
        lens = lens.reshape(B, C, n)
        starts = lens.cumsum(-1) - lens
        pos = torch.stack((starts, lens), dim=-1)              # [B, C, N, 2]
        return torch.cat((tokens.reshape(B, C, n, -1), pos), dim=-1)

    @torch.no_grad()
    def oracle_step(self, x_enc, y, epoch, select=None):
        """E 步：对全部 P 种划分逐一前向，用真实预测损失（逐样本 MSE）取 argmin 作为本 batch 的划分标签。

        暖启动期（epoch < oracle_warmup）恒选等分，模型从固定 patch 的解出发。E 步在 eval 模式下算
        （去掉 dropout 噪声），算完恢复训练模式。返回逐样本损失矩阵 [B, P]（暖启动期为 None）。
        select：只用这些预测步（slice）算损失——测试期交叉验证 oracle 用前半段挑、后半段打分。
        """
        B = x_enc.size(0)
        self._warm = epoch < self.oracle_warmup
        if self._warm:
            self._oracle_part = torch.full((B,), self.pdp.uniform_part, dtype=torch.long, device=x_enc.device)
            return None
        # 只在训练态下切 eval（去 dropout 噪声）；测试态本就是 eval，不再切换，否则 train(False)
        # 的进入钩子会把测试期累计的诊断计数清零
        was_training = self.training
        if was_training:
            self.eval()
        y_true = y[:, -self.pred_len:, :]
        sel = select if select is not None else slice(None)
        P = self.pdp.n_parts
        losses = []
        for start in range(0, P, self.oracle_chunk):
            parts = torch.arange(start, min(start + self.oracle_chunk, P), device=x_enc.device)
            k = parts.numel()
            x_rep = x_enc.repeat(k, 1, 1)                                   # [k*B, L, C]
            part_idx = parts.repeat_interleave(B)                            # [k*B]
            out = self._core(x_rep, part_idx)[0][:, -self.pred_len:, :]
            err = ((out[:, sel] - y_true.repeat(k, 1, 1)[:, sel]) ** 2).mean(dim=(1, 2))  # [k*B]
            losses.append(err.reshape(k, B).t())
        losses = torch.cat(losses, dim=1)                                    # [B, P]
        best = losses.argmin(dim=1)
        if self.oracle_margin > 0:
            # 只有明显优于等分才离开等分
            keep = losses.gather(1, best.unsqueeze(1)).squeeze(1) < losses[:, self.pdp.uniform_part] * (1 - self.oracle_margin)
            best = torch.where(keep, best, torch.full_like(best, self.pdp.uniform_part))
        self._oracle_part = best
        if was_training:
            if epoch != self._tr_epoch:
                self._tr, self._tr_epoch = [0, 0, 0], epoch
            pred = self.router(self._normalize(x_enc)[0]).argmax(-1)
            self._tr[0] += B
            self._tr[1] += (best != self.pdp.uniform_part).sum().item()
            self._tr[2] += (pred == best).sum().item()
            self.train(True)
        return losses

    @torch.no_grad()
    def oracle_forward(self, x_enc, y, select=None, record=True):
        """测试期上界：用 y 选划分后前向；record 时记录路由器与 oracle 的一致率与两者的划分直方图。
        select 见 oracle_step（交叉验证 oracle 传前半段、record=False，不污染直方图）。"""
        self.oracle_step(x_enc, y, epoch=float('inf'), select=select)
        part = self._oracle_part
        self._oracle_part = None
        if record:
            logits = self.router(self._normalize(x_enc)[0])
            self._agree[0] += (logits.argmax(-1) == part).sum().item()
            self._agree[1] += part.numel()
            self.oracle_hist += torch.bincount(part, minlength=self.pdp.n_parts).to(self.oracle_hist.dtype)
        return self._core(x_enc, part)[0]

    def _normalize(self, x_enc):
        """RevIN + 近端流分解，返回主干输入 [B, C, L] 与近端流输入（或 None）。"""
        x_enc = self.rin(x_enc, 'norm')
        x_aux = None
        if self.aux_stream == 'trend':
            # SeriesDecomp 对 [B, L, C] 沿时间轴平滑：季节项进主干，趋势走近端流
            x_enc, trend = self.aux_decomp(x_enc)
            x_aux = trend.permute(0, 2, 1)  # [B, C, L]
        x_enc = x_enc.permute(0, 2, 1)  # [B, C, L]
        if self.aux_stream == 'full':
            x_aux = x_enc
        elif self.aux_stream == 'last':
            x_aux = x_enc[:, :, -self.aux_len:]
        return x_enc, x_aux

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, L, C]；mark/dec 参数为框架签名占位，本模型不使用
        part_idx = target = None
        if self.needs_oracle and self.training:
            target = self._oracle_part
            self._oracle_part = None
            if target is None or target.size(0) != x_enc.size(0):
                # 未经 oracle_step（如直接调用）：退回等分
                target = torch.full((x_enc.size(0),), self.pdp.uniform_part,
                                    dtype=torch.long, device=x_enc.device)
            # M 步划分：暖启动期恒等分；之后按 oracle_train_on——router 挡传 None 让 _core 取路由器 argmax
            if self._warm or self.oracle_train_on == 'oracle':
                part_idx = target
        dec_out, logits = self._core(x_enc, part_idx)

        aux = self.pdp.aux_loss if self.use_backbone else None
        if self.needs_oracle and self.training:
            # 路由器模仿 E 步标签（有监督分类，不经直通估计）
            ce = self.oracle_ce * F.cross_entropy(logits, target)
            aux = ce if aux is None else aux + ce
        if aux is not None and self.training:
            return dec_out, aux
        return dec_out

    def _core(self, x_enc, part_idx=None):
        """完整前向（RevIN → 主干 → 头 → 反归一化）。oracle 挡下 part_idx 为 None 时走路由器 argmax。
        返回 (dec_out [B, H, C], router logits 或 None)。"""
        B, L, C = x_enc.shape
        x_enc, x_aux = self._normalize(x_enc)
        logits = None
        if self.needs_oracle:
            # 路由器输入 detach：否则交叉熵梯度经归一化输入回传到共享的 RevIN 仿射参数，路由器即便恒选
            # 等分也在改动主干训练（实测 336 格 router 100% 等分却比同种子固定 4×24 差 0.0024~0.0123）
            logits = self.router(x_enc.detach())
            if part_idx is None:
                part_idx = logits.argmax(dim=-1)
                if not self.training:
                    self.router_hist += torch.bincount(part_idx, minlength=self.pdp.n_parts).to(self.router_hist.dtype)

        if self.use_backbone:
            pattern_emb = self.pat_emb(x_enc)  # [C, K, F]

            # patches: [B, P_total, max_len]；sel_moe: [C, K]
            patches, lengths, var_ids, sel_moe = self.pdp(x_enc, pattern_emb, part_idx=part_idx)

            enc_out = self.spec_moe(patches, var_ids, pattern_emb, sel_moe)

            if self.tokenize == 'direct':
                if self.adp_regrid:
                    enc_out = self._regrid(enc_out, lengths, B, C, L)
                if self.token_mix == 'hgnn':
                    tok = enc_out.reshape(B, C, self.n_tokens, -1)
                    pos = self.token_pos.unsqueeze(0)
                    for i, layer in enumerate(self.token_hgnn):
                        # 零初始化门做线性插值：初值恒等，混合作为从零学起的修正
                        tok = tok + torch.tanh(self.mix_gate[i]) * (layer(tok, pos) - tok)
                    enc_out = tok.reshape(B, C * self.n_tokens, -1)
                # [B, C*N, D_p] -> [B, C, N*D_p]：token 是预测头的唯一输入
                if self.token_pos_enc:
                    enc_out = self._append_pos(enc_out, lengths, B, C, L)
                enc_out = enc_out.reshape(B, C, -1)
            else:
                enc_out = self.pdp.reconstruct_from_patches(
                    enc_out, lengths, var_ids, C, L)  # [B, C, L]

                if self.residual_mode == 'gate_input':
                    enc_out = enc_out + torch.tanh(self.res_gate) * x_enc
                elif self.residual_mode == 'gate_backbone':
                    enc_out = x_enc + torch.tanh(self.res_gate) * enc_out
        else:
            enc_out = x_enc

        dec_out = self.model(enc_out)  # [B, C, H]
        if x_aux is not None:
            # 零初始化门：近端流从零学起，训练后 tanh(γ) 可直接读出该数据集对近端原始值的依赖
            dec_out = dec_out + torch.tanh(self.aux_gate) * self.aux_head(x_aux)

        dec_out = dec_out.permute(0, 2, 1)  # [B, H, C]
        return self.rin(dec_out, 'denorm'), logits
