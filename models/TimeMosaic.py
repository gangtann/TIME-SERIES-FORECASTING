"""TimeMosaic: Information-Density Guided Time Series Forecasting via Adaptive
Granularity Patch and Segment-wise Decoding (AAAI 2026).

官方仓库: https://github.com/BenchCouncil/TimeMosaic （arXiv:2509.19406，
镜像 https://github.com/Day333/TimeMosaic ），对应 models/TimeMosaic.py +
exp/exp_TimeMosaic.py。

移植改动（相对官方源码）：
1. forward 签名适配本框架契约 forward(x_enc, x_mark_enc, x_dec, x_mark_dec)：
   去掉官方第 5 参 mask（仅 mask_ratio>0 的重构分支使用，官方默认 0，该分支未移植，
   tms_mask_ratio>0 会在 __init__ 直接报错）。
2. 辅助损失内置：官方 Exp_TimeMosaic 在 exp 里算粒度分类比例正则
   loss_reg = L1(latest_cls_soft.mean(0), uniform)，lambda_cls=0.001（非 AMP 分支）。
   本移植在模型 forward 内算好，训练态返回 (dec_out, aux_loss) 元组（aux 已乘
   tms_lambda_cls），eval 态只返回 dec_out。官方训练主损失为 nn.L1Loss（exp 硬编码，
   无视 --loss 默认 MSE），对应 yaml 里 training.loss: MAE。
3. 官方 Model.__init__ 里 self.training = configs.is_training 会覆盖 nn.Module 的
   train/eval 状态位（随后仍被 model.train()/eval() 重置）；移植后直接用标准
   self.training，行为等价。Gumbel-softmax：训练态 F.gumbel_softmax(tau=0.5,
   hard=True)（随机、straight-through），eval 态 argmax one-hot（确定性）。
4. 依赖层（Encoder/FullAttention/AttentionLayer/DataEmbedding_inverted/RevIN）
   全部并入 layers/TimeMosaic_layers.py（TMS 前缀），不复用本框架同名文件。
5. 模型专属超参一律读 tms_ 前缀键（configs/default.yaml timemosaic 节）。
6. 无设备硬编码（官方模型本身即无 .to('cuda')）。

注意：官方协议 seq_len=320；本框架按统一协议 seq_len=96 跑（见各 yaml 注释）。
"""

import ast

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.TimeMosaic_layers import (
    TMSAttentionLayer,
    TMSDataEmbeddingInverted,
    TMSEncoder,
    TMSFullAttention,
    TMSRevIN,
)


class EncoderLayer(nn.Module):
    """官方 models/TimeMosaic.py 内的 EncoderLayer 变体（带 mask_last_tokens）。

    官方实例化时未传 num_latent_token（该行被注释），默认 0，掩码为 no-op，
    此处照搬保留。
    """

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", num_latent_token=0):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.num_latent_token = num_latent_token

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        q = self.mask_last_tokens(x)
        new_x, attn = self.attention(
            q, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

    def mask_last_tokens(self, x):
        x_masked = x.clone()
        if self.num_latent_token > 0:
            x_masked[:, :self.num_latent_token, :] = 0
        return x_masked


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # x: [B*C, num_patch, d_model]
        return self.pe[:, :x.size(1)]


class AdaptivePatchEmbedding(nn.Module):
    """自适应粒度 patch 嵌入（官方实现，去掉 training 构造参数，改用 nn.Module.training）。

    训练态 Gumbel-softmax(tau=0.5, hard=True) 随机选粒度（straight-through 可导）；
    eval 态 argmax one-hot，完全确定。latest_cls_soft 暴露给 Model.forward 算比例正则。
    """

    def __init__(self, d_model, patch_len_list, mode='fixed', dropout=0.0, seq_len=96, in_channels=1):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.mode = mode
        self.max_patch_len = max(patch_len_list)
        self.min_patch_len = min(patch_len_list)
        self.region_num = seq_len // self.max_patch_len
        self.d_model = d_model
        self.in_channels = in_channels

        self.register_buffer('target_ratio', torch.ones(len(patch_len_list)) / len(patch_len_list))

        self.region_cls = nn.Sequential(
            nn.Linear(self.max_patch_len, 64),
            nn.ReLU(),
            nn.Linear(64, len(patch_len_list))
        )

        self.embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model, bias=False) for patch_len in patch_len_list
        ])

        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, L]
        B, C, L = x.shape
        assert L == self.region_num * self.max_patch_len, \
            f"Expected seq_len={self.region_num * self.max_patch_len}, but got {L}"

        x = x.reshape(B * C, self.region_num, self.max_patch_len)  # [B*C, R, max_patch_len]

        all_patches = []
        cls_soft_list = []

        for region_idx in range(self.region_num):
            region = x[:, region_idx, :]  # [B*C, max_patch_len]

            cls_logits = self.region_cls(region)  # [B*C, num_classes]

            if self.training:
                cls_soft = F.gumbel_softmax(cls_logits, tau=0.5, hard=True, dim=-1)  # [B*C, num_classes]
            else:
                cls_pred = torch.argmax(cls_logits, dim=-1)
                cls_soft = F.one_hot(cls_pred, num_classes=len(self.patch_len_list)).float()

            cls_soft_list.append(cls_soft)

            patch_emb_list = []
            for idx, patch_len in enumerate(self.patch_len_list):
                patches = region.unfold(-1, patch_len, patch_len)  # [B*C, num_patch, patch_len]
                if self.mode == 'fixed':
                    target_patch_num = self.max_patch_len // self.min_patch_len
                    repeat = target_patch_num - patches.size(1)
                    if repeat > 0:
                        patches = patches.repeat_interleave(repeat + 1, dim=1)[:, :target_patch_num, :]
                patches_emb = self.embeddings[idx](patches)  # [B*C, num_patch, d_model]
                patch_emb_list.append(patches_emb)

            # [num_classes, B*C, num_patch, d_model]
            patch_emb_stack = torch.stack(patch_emb_list, dim=0)

            # cls_soft: [B*C, num_classes] -> [num_classes, B*C, 1, 1]
            cls_soft_trans = cls_soft.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)

            region_patches_sorted = (patch_emb_stack * cls_soft_trans).sum(dim=0)  # [B*C, num_patch, d_model]

            all_patches.append(region_patches_sorted)

        x_patch = torch.cat(all_patches, dim=1)  # [B*C, total_num_patch, d_model]
        x_patch += self.position_embedding(x_patch)
        x_patch = self.dropout(x_patch)

        self.latest_cls_soft = torch.cat(cls_soft_list, dim=0)  # [B*C*R, num_classes]

        return x_patch, C


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """TimeMosaic：自适应粒度 patch + prompt 引导分段解码。

    数据流（[B, L, C] 输入）：
      去均值/除标准差归一化 → 逐 region 粒度分类(Gumbel) → 混合粒度 patch 嵌入
      → 按 tms_channel 拼接通道/日历 extra token → 逐段 [prompt; patches; extra]
      过共享 Encoder → 各段 FlattenHead 出 seg_len → 拼接为 pred_len → 反归一化。
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.lambda_cls = configs.tms_lambda_cls

        pll = configs.tms_patch_len_list
        if isinstance(pll, str):
            pll = ast.literal_eval(pll)  # 官方 --patch_len_list '[8,16,32]' 字符串形式
        self.patch_len_list = list(pll)

        seg_len_map = {96: configs.tms_pre96, 192: configs.tms_pre192,
                       336: configs.tms_pre336, 720: configs.tms_pre720}
        self.seg_len = seg_len_map.get(configs.pred_len, configs.tms_pre12)
        if self.pred_len % self.seg_len != 0:
            raise ValueError(f'pred_len={self.pred_len} 必须能被段长 seg_len={self.seg_len} 整除')
        self.num_segs = self.pred_len // self.seg_len
        self.channel = configs.tms_channel

        if getattr(configs, 'tms_mask_ratio', 0):
            raise NotImplementedError(
                'TimeMosaic 的 mask 重构分支未移植（官方默认 mask_ratio=0）；请保持 tms_mask_ratio=0')

        self.patch_embedding = AdaptivePatchEmbedding(
            d_model=configs.d_model,
            patch_len_list=self.patch_len_list,
            mode='fixed',
            dropout=configs.dropout,
            seq_len=configs.seq_len,
            in_channels=configs.enc_in,
        )

        self.num_latent_token = configs.tms_num_latent_token
        self.prompt_embeddings = nn.Embedding(self.num_latent_token * self.num_segs, self.d_model)
        nn.init.xavier_uniform_(self.prompt_embeddings.weight)

        self.encoder = TMSEncoder([
            EncoderLayer(
                TMSAttentionLayer(
                    TMSFullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                     output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                # 官方注释掉了 num_latent_token=configs.num_latent_token（掩码关闭），照搬
            ) for l in range(configs.e_layers)
        ], norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2)))

        self.enc_embedding = TMSDataEmbeddingInverted(configs.seq_len, configs.d_model, configs.embed,
                                                      configs.freq, configs.dropout)

        # 官方 head 尺寸沿用 PatchTST 公式（构造默认 patch_len=16/stride=8），
        # 对 patch_len_list=[8,16,32] & seq_len=96 恰好等于真实 patch 数 3*4=12；
        # 这里显式校验这一巧合，防止改配方后 silent 切错 token。
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        actual_patch_num = (configs.seq_len // max(self.patch_len_list)) \
            * (max(self.patch_len_list) // min(self.patch_len_list))
        if self.patch_num != actual_patch_num:
            raise ValueError(
                f'官方 head 公式 patch_num={self.patch_num} 与自适应 patch 实际数 '
                f'{actual_patch_num} 不一致（seq_len={configs.seq_len}, '
                f'patch_len_list={self.patch_len_list}），请核对配方')
        self.head_nf = configs.d_model * self.patch_num

        self.heads = nn.ModuleList([
            FlattenHead(configs.enc_in, self.head_nf, self.seg_len, head_dropout=configs.dropout)
            for _ in range(self.num_segs)
        ])

        # 官方固定 revin=False（走去均值/除标准差路径），但仍实例化 RevIN；照搬以保持参数量一致
        self.revin = False
        self.revin_layer = TMSRevIN(configs.enc_in, affine=True, subtract_last=False)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        else:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        seg_outputs = []
        B, C = x_enc.shape[0], x_enc.shape[2]

        # 通道信息注入方式（官方 --channel）
        if self.channel == "CI":
            # simple CI
            extra_token = self.enc_embedding(x_enc, None)
            extra_token = extra_token.view(-1, 1, self.d_model)
        elif self.channel == "CD":
            # simple channel-wise
            x_pool = x_enc.mean(dim=2)  # [B, T]
            extra_token = self.enc_embedding(x_pool.unsqueeze(-1), None)  # [B, 1, d]
            extra_token = extra_token.mean(dim=1, keepdim=True)  # [B, 1, d]
            extra_token = extra_token.repeat_interleave(C, dim=0)  # [B*C, 1, d]
        elif self.channel == "CDP":
            x_pool = x_enc.mean(dim=2)  # [B, T]
            channel_token = self.enc_embedding(x_pool.unsqueeze(-1), None)  # [B, 1, d]
            channel_token = channel_token.mean(dim=1, keepdim=True)  # [B, 1, d]
            channel_token = channel_token.repeat_interleave(C, dim=0)  # [B*C, 1, d]

            global_tokens = self.enc_embedding(x_enc, x_mark_enc)  # [B, C+K, D]
            cal_tokens = global_tokens[:, C:, :]
            cal_tokens = cal_tokens.repeat_interleave(C, dim=0)

            extra_token = torch.cat([channel_token, cal_tokens], dim=1)
        elif self.channel == "CDA":
            extra_token = self.enc_embedding(x_enc, x_mark_enc)  # [B, C+K, D]
            extra_token = extra_token.repeat_interleave(C, dim=0)  # [B*C, C+K, D]
        elif self.channel == "CI+":
            global_tokens = self.enc_embedding(x_enc, x_mark_enc)  # [B, C+K, D]
            var_tokens = global_tokens[:, :C, :]
            cal_tokens = global_tokens[:, C:, :]

            var_tokens = var_tokens.reshape(-1, 1, self.d_model)  # [B*C, 1, D]
            cal_tokens = cal_tokens.repeat_interleave(C, dim=0)   # [B*C, K, D]

            extra_token = torch.cat([var_tokens, cal_tokens], dim=1)
        else:
            raise ValueError(f'unknown tms_channel: {self.channel} (CI/CD/CDP/CDA/CI+)')

        x_enc = x_enc.permute(0, 2, 1)
        # enc_out: [B*C, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        enc_out = torch.cat([enc_out, extra_token], dim=1)

        for i in range(self.num_segs):
            # Prompt embedding for segment i
            prompt = self.prompt_embeddings.weight[i * self.num_latent_token: (i + 1) * self.num_latent_token]
            prompt = prompt.unsqueeze(0).expand(B * C, -1, -1)  # [B*C, num_prompt, d_model]

            # Concatenate prompt, patch embeddings, global channel
            segment_input = torch.cat([prompt, enc_out], dim=1)  # [B*C, prompt+patch+extra, d_model]

            # Encoder forward
            segment_out, _ = self.encoder(segment_input)
            segment_out = segment_out[:, self.num_latent_token:self.num_latent_token + self.patch_num, :]

            # Reshape: [B*C, patch_num, d_model] -> [B, C, d_model, patch_num]
            segment_out = torch.reshape(segment_out, (B, C, self.d_model, self.patch_num))

            # Segment head: [B, C, seg_len]
            seg_out = self.heads[i](segment_out)
            seg_outputs.append(seg_out)

        # Concatenate all segment outputs: [B, C, pred_len]
        dec_out = torch.cat(seg_outputs, dim=2)
        dec_out = dec_out.permute(0, 2, 1)  # -> [B, pred_len, C]

        # De-normalize
        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        else:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc)
        dec_out = dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]
        if self.training:
            # 官方 Exp_TimeMosaic（非 AMP 分支）：粒度选择比例向均匀分布的 L1 正则
            cls_soft = self.patch_embedding.latest_cls_soft  # [B*C*R, num_classes]
            current_ratio = cls_soft.mean(dim=0)
            target_ratio = torch.full_like(current_ratio, 1.0 / current_ratio.numel()).detach()
            aux_loss = self.lambda_cls * F.l1_loss(current_ratio, target_ratio)
            return dec_out, aux_loss
        return dec_out
