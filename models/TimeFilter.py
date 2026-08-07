"""TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series
Forecasting (ICML 2025).

官方仓库：https://github.com/TROUBADOUR000/TimeFilter
论文：https://arxiv.org/pdf/2501.13041

移植改动（相对官方 models/TimeFilter.py + exp/exp_long_term_forecasting.py）：
  1. 签名适配：官方 forward(self, x, masks, is_training=False, target=None)
     改为本框架四参 forward(x_enc, x_mark_enc, x_dec, x_mark_dec)；
     x_mark_enc/x_dec/x_mark_dec 官方模型完全不使用，直接忽略。
     target 参数官方 exp 从未传入（train/vali/test 均为
     model(batch_x, self.masks, is_training=...)），安全丢弃。
  2. masks 预构建：官方在 Exp_Long_Term_Forecast.__init__ 里调 _get_mask() 按
     (seq_len, c_out, patch_len) 构建 [L, 3, L] 的 spatial(S)/temporal(T)/
     spatial-temporal(ST) 三区掩码并常驻 device，每次 forward 原样传入。
     此处把 _get_mask() 逐行搬进 Model.__init__（含 T 区排除自身 k、ST[k]=0
     两处细节，与 layers 里 masks=None 的兜底版本不同——训练实际用的是 exp 版），
     注册为 buffer：跟随 model.to(device)，不进 state_dict，不在 forward 重建。
  3. 辅助损失通路：官方 exp 训练时 loss = MSE + 0.05 * moe_loss，0.05 为
     exp_long_term_forecasting.py train() 内硬编码（变量名恰好也叫 alpha，
     与 run.py 的 --alpha=KNN 保留比例无关）。此处训练态返回
     (pred, 0.05 * moe_loss)，由本框架 exp 解包相加；eval 态只返回 pred。
     top_p=0.0 时官方 moe_loss 为 python float 0.0（无梯度贡献），此时不返回元组。
  4. is_training 用 self.training 传递（官方由 exp 显式传参，语义一致：控制
     mask_moe 的 noisy gating 是否加噪）。
  5. 设备无关：无任何 .to('cuda') 硬编码；masks 为 buffer，PositionalEmbedding
     内部自带 buffer。
  6. 专属超参改读 tfil_ 前缀键：tfil_alpha（KNN 图保留比例，官方 --alpha 默认
     0.1）、tfil_top_p（MoE 动态路由累计概率阈值，官方 --top_p 默认 0.5）、
     tfil_pos（patch 嵌入位置编码开关，官方 --pos 默认 1）。patch_len 复用
     model 节通用键；stride 恒等于 patch_len（官方行为），不读 configs.stride。
"""
import torch
from torch import nn

from layers.Embed import PositionalEmbedding
from layers.TimeFilter_layers import Normalize, TimeFilter_Backbone


class PatchEmbed(nn.Module):
    """官方 models/TimeFilter.py 的 PatchEmbed，逐行照搬。

    输入 [B, C*T] 的展平序列，unfold 成 [B, C*T/P, P] 后线性投影到 d_model，
    可选加正弦位置编码（PositionalEmbedding 与本框架 layers/Embed.py 逐行等价，
    直接复用；官方以 max_len=10000 实例化）。
    """

    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            pos_emb_theta = 10000
            self.pe = PositionalEmbedding(dim, pos_emb_theta)

    def forward(self, x):
        # x: [B, N, T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [B, N*L, P]
        x = self.patch_proj(x)  # [B, N*L, D]
        if self.pos:
            x += self.pe(x)
        return x


class Model(nn.Module):
    # 官方 exp_long_term_forecasting.py train() 内硬编码：
    #   alpha = 0.05; loss = criterion(outputs, batch_y) + alpha * moe_loss
    MOE_LOSS_WEIGHT = 0.05

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.c_out
        self.dim = configs.d_model
        self.d_ff = configs.d_ff
        self.patch_len = configs.patch_len
        self.stride = self.patch_len
        assert self.seq_len % self.patch_len == 0, \
            f'TimeFilter requires seq_len % patch_len == 0, got {self.seq_len} % {self.patch_len}'
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)  # L per var

        # Filter：tfil_alpha = KNN 图构建保留比例（官方 run.py --alpha，默认 0.1）
        self.alpha = configs.tfil_alpha
        self.top_p = configs.tfil_top_p

        # embed
        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, configs.tfil_pos)

        # TimeFilter Backbone
        self.backbone = TimeFilter_Backbone(
            self.dim, self.n_vars, self.d_ff,
            configs.n_heads, configs.e_layers, self.top_p, configs.dropout,
            self.seq_len * self.n_vars // self.patch_len)

        # head
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        # Without RevIN（官方固定 use_RevIN=False，即无 affine 参数的纯实例归一化）
        self.use_RevIN = False
        self.norm = Normalize(configs.enc_in, affine=self.use_RevIN)

        # 官方 exp _get_mask() 预构建的 spatial-temporal 掩码，注册为 buffer
        self.register_buffer('masks', self._build_masks(), persistent=False)

    def _build_masks(self):
        """逐行移植官方 exp_long_term_forecasting.py 的 _get_mask()。

        L = seq_len * c_out / patch_len（全部变量拼接后的 patch 总数），
        N = seq_len / patch_len（单变量 patch 数）。对每个 patch k：
          S  同一时间位置、不同变量（排除 k 自身）
          T  同一变量的其余 patch（排除 k 自身）
          ST 其余全部 patch（ST[k] 置 0，自连接由 mask_moe 内 mask_base=I 提供）
        返回 [L, 3, L]，float32。
        """
        dtype = torch.float32
        L = self.seq_len * self.n_vars // self.patch_len
        N = self.seq_len // self.patch_len
        idx = torch.arange(L)
        masks = []
        for k in range(L):
            S = ((idx % N == k % N) & (idx != k)).to(dtype)
            T = ((idx >= k // N * N) & (idx < k // N * N + N) & (idx != k)).to(dtype)
            ST = torch.ones(L, dtype=dtype) - S - T
            ST[k] = 0.0
            masks.append(torch.stack([S, T, ST], dim=0))
        return torch.stack(masks, dim=0)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, T, C]；x_mark_enc/x_dec/x_mark_dec 官方模型不使用
        B, T, C = x_enc.shape
        # Normalization
        x = self.norm(x_enc, 'norm')
        # x: [B, C, T] -> [B, C*T]
        x = x.permute(0, 2, 1).reshape(-1, C * T)
        x = self.patch_embed(x)  # [B, N, D]  N = C*T / P

        x, moe_loss = self.backbone(x, self.masks, self.alpha, self.training)

        # [B, C, T/P, D] -> [B, C, pred_len]
        x = self.head(x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2))
        x = x.permute(0, 2, 1)
        # De-Normalization
        x = self.norm(x, 'denorm')

        # top_p=0.0 时官方 moe_loss 为 float 0.0（无 MoE 正则），无需返回元组
        if self.training and torch.is_tensor(moe_loss):
            return x, self.MOE_LOSS_WEIGHT * moe_loss
        return x
