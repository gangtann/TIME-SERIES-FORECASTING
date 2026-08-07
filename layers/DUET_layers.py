"""DUET (KDD 2025) 专属依赖层，合并自官方仓库 decisionintelligence/DUET
ts_benchmark/baselines/duet/ 下的:
  layers/RevIN.py                       -> DUETRevIN
  layers/Autoformer_EncDec.py           -> DUETMovingAvg / DUETSeriesDecomp（只搬用到的两个类）
  layers/distributional_router_encoder.py -> DUETDistributionalRouter（官方类名 encoder）
  layers/linear_pattern_extractor.py    -> DUETLinearExtractor（官方类名 Linear_extractor）
  layers/linear_extractor_cluster.py    -> SparseDispatcher / DUETLinearExtractorCluster
  utils/masked_attention.py             -> DUETFullAttention / DUETAttentionLayer /
                                           DUETEncoderLayer / DUETEncoder / DUETMahalanobisMask

官方自带的 RevIN/Autoformer_EncDec/注意力层与本框架 layers/ 同名文件不保证逐行等价，
故全部自带并加 DUET 前缀防撞，不复用本框架已有文件。

与官方源码的差异（均为设备无关性/健壮性修正，不改数值语义）：
  1. DUETMahalanobisMask 中 torch.eye 直接以输入 device 创建（官方在 CPU 创建后 .to(device)）。
  2. DUETLinearExtractor 空 batch 分支 torch.empty 直接以输入 device 创建（官方 .to(device)）。
  3. DUETFullAttention 掩码 torch.where 的第三参写 0.0（官方为 int 0，靠隐式类型提升）。
  4. 模型专属超参从本框架扁平命名空间按 duet_ 前缀读取
     （duet_num_experts/duet_k/duet_hidden_size/duet_noisy_gating/duet_CI/duet_moving_avg）。
"""

import math
from math import sqrt

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.functional import gumbel_softmax
from einops import rearrange


# ---------------------------------------------------------------- RevIN
class DUETRevIN(nn.Module):
    """官方 duet/layers/RevIN.py 逐行等价。"""

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
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
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

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


# ------------------------------------------------- series decomposition
class DUETMovingAvg(nn.Module):
    """官方 duet/layers/Autoformer_EncDec.py moving_avg 逐行等价。"""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class DUETSeriesDecomp(nn.Module):
    """官方 duet/layers/Autoformer_EncDec.py series_decomp 逐行等价。"""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = DUETMovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# ------------------------------------------------- distributional router
class DUETDistributionalRouter(nn.Module):
    """官方 duet/layers/distributional_router_encoder.py encoder 逐行等价。"""

    def __init__(self, configs):
        super().__init__()
        input_size = configs.seq_len
        num_experts = configs.duet_num_experts
        encoder_hidden_size = configs.duet_hidden_size

        self.distribution_fit = nn.Sequential(
            nn.Linear(input_size, encoder_hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(encoder_hidden_size, num_experts, bias=False),
        )

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(mean)
        return out


# ---------------------------------------------- linear pattern extractor
class DUETLinearExtractor(nn.Module):
    """官方 duet/layers/linear_pattern_extractor.py Linear_extractor（DLinear 式专家）。

    注意官方即如此：pred_len 取 d_model（专家输出的是 d_model 维时间特征，非最终预测）。
    """

    def __init__(self, configs, individual=False):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.decompsition = DUETSeriesDecomp(configs.duet_moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.enc_in = 1 if configs.duet_CI else configs.enc_in
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = (
            seasonal_init.permute(0, 2, 1),
            trend_init.permute(0, 2, 1),
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return torch.empty(
                (0, self.pred_len, self.enc_in), device=x_enc.device
            )
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# ----------------------------------------------------- sparse dispatcher
class SparseDispatcher(object):
    """官方 duet/layers/linear_extractor_cluster.py SparseDispatcher 逐行等价。

    MoE 辅助类：按 gates 把 batch 派发给各专家并按门控权重合并输出。
    """

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = torch.einsum("i...,ij->i...", stitched, self._nonzero_gates)
        shape = list(expert_out[-1].shape)
        shape[0] = self._gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


# ----------------------------------------------- linear extractor cluster
class DUETLinearExtractorCluster(nn.Module):
    """官方 duet/layers/linear_extractor_cluster.py Linear_extractor_cluster。

    noisy top-k 门控的稀疏 MoE（专家为 DUETLinearExtractor），返回
    (特征, importance/load 均衡损失)。损失系数 loss_coef 官方默认 1，
    官方 wrapper 直接把该损失加进训练 criterion。
    """

    def __init__(self, configs):
        super().__init__()
        self.noisy_gating = configs.duet_noisy_gating
        self.num_experts = configs.duet_num_experts
        self.input_size = configs.seq_len
        self.k = configs.duet_k
        self.experts = nn.ModuleList(
            [DUETLinearExtractor(configs) for _ in range(self.num_experts)]
        )
        self.W_h = nn.Parameter(torch.eye(self.num_experts))
        self.gate = DUETDistributionalRouter(configs)
        self.noise = DUETDistributionalRouter(configs)

        self.n_vars = configs.enc_in
        self.revin = DUETRevIN(self.n_vars)

        self.CI = configs.duet_CI
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = self.gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise * noise_stddev)
            logits = noisy_logits @ self.W_h
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        if self.CI:
            x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)
            x_norm = self.revin(x_norm, "norm")
            x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)
        else:
            x_norm = self.revin(x, "norm")

        expert_inputs = dispatcher.dispatch(x_norm)

        gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)

        return y, loss


# ------------------------------------------- masked channel transformer
class DUETEncoderLayer(nn.Module):
    """官方 duet/utils/masked_attention.py EncoderLayer 逐行等价。"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class DUETEncoder(nn.Module):
    """官方 duet/utils/masked_attention.py Encoder 逐行等价。"""

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DUETFullAttention(nn.Module):
    """官方 duet/utils/masked_attention.py FullAttention。

    mask_flag=True 时 attn_mask 为通道相关性 0/1 掩码：
    scores * mask 后再给掩掉的位置加大负数偏置（官方实现原样保留）。
    """

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            large_negative = -math.log(1e10)
            attention_mask = torch.where(attn_mask == 0, large_negative, 0.0)
            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class DUETAttentionLayer(nn.Module):
    """官方 duet/utils/masked_attention.py AttentionLayer 逐行等价。"""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class DUETMahalanobisMask(nn.Module):
    """官方 duet/utils/masked_attention.py Mahalanobis_mask。

    频域马氏距离 -> 通道相关概率 -> Gumbel 伯努利重参数采样得到 0/1 通道掩码。
    device 修正：torch.eye 直接以输入 device 创建（官方在 CPU 创建后 .to(device)）。
    """

    def __init__(self, input_size):
        super().__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(
            torch.randn(frequency_size, frequency_size), requires_grad=True
        )

    def calculate_prob_distance(self, X):
        XF = torch.abs(torch.fft.rfft(X, dim=-1))
        X1 = XF.unsqueeze(2)
        X2 = XF.unsqueeze(1)

        # B x C x C x D
        diff = X1 - X2
        temp = torch.einsum("dk,bxck->bxcd", self.A, diff)
        dist = torch.einsum("bxcd,bxcd->bxc", temp, temp)

        exp_dist = 1 / (dist + 1e-10)
        # 对角线置零
        identity_matrices = 1 - torch.eye(exp_dist.shape[-1], device=exp_dist.device)
        mask = identity_matrices.repeat(exp_dist.shape[0], 1, 1)
        exp_dist = torch.einsum("bxc,bxc->bxc", exp_dist, mask)
        exp_max, _ = torch.max(exp_dist, dim=-1, keepdim=True)
        exp_max = exp_max.detach()

        # B x C x C
        p = exp_dist / exp_max

        identity_matrices = torch.eye(p.shape[-1], device=p.device)
        p1 = torch.einsum("bxc,bxc->bxc", p, mask)
        diag = identity_matrices.repeat(p.shape[0], 1, 1)
        p = (p1 + diag) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten_matrix = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        r_flatten_matrix = 1 - flatten_matrix

        log_flatten_matrix = torch.log(flatten_matrix / r_flatten_matrix)
        log_r_flatten_matrix = torch.log(r_flatten_matrix / flatten_matrix)

        new_matrix = torch.concat([log_flatten_matrix, log_r_flatten_matrix], dim=-1)
        resample_matrix = gumbel_softmax(new_matrix, hard=True)

        resample_matrix = rearrange(
            resample_matrix[..., 0], "(b c d) -> b c d", b=b, c=c, d=d
        )
        return resample_matrix

    def forward(self, X):
        p = self.calculate_prob_distance(X)
        # bernoulli 中两个通道有关系的概率
        sample = self.bernoulli_gumbel_rsample(p)
        mask = sample.unsqueeze(1)
        return mask
