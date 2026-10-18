"""PathFormer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting.

论文: Pathformer (ICLR 2024), https://openreview.net/forum?id=lJkOCMP2aW
官方仓库: https://github.com/decisionintelligence/pathformer （models/PathFormer.py）

移植改动:
  - forward 适配本框架四参签名 (x_enc, x_mark_enc, x_dec, x_mark_dec)，仅使用 x_enc
    （官方 forward 只吃 x）；
  - 辅助损失通路：训练态返回 (outputs, balance_loss) 元组，eval 态只返回 outputs。
    balance_loss 已含官方权重（AMS.forward 内 loss_coef=1e-2），官方 exp_main.py 为
    loss = criterion(...) + balance_loss 直接相加、无额外系数，与本框架 exp 的
    loss += aux_loss 行为一致；
  - 去掉硬编码 self.device = torch.device('cuda:{gpu}')（官方仅透传给
    Transformer_Layer 且从未使用），全模型设备无关；
  - 超参改读 pf_ 前缀键（configs/default.yaml pathformer 节）：pf_layer_nums、
    pf_num_nodes、pf_k、pf_num_experts_list、pf_patch_size_list、
    pf_residual_connection、pf_revin、pf_batch_norm；d_model/d_ff 复用 model 节通用键；
  - pf_patch_size_list 在 yaml 里为扁平列表，此处按官方 run.py L85 的方式
    reshape 成 (layer_nums, -1) 的每层专家 patch 列表；
  - 依赖层（AMS/Transformer_Layer/RevIN 等）全部合并在 layers/PathFormer_layers.py，
    不复用 layers/PRISM_layers.py 的 RevIN（语义不同）。
"""
from torch import nn

from layers.PathFormer_layers import AMS, RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.pf_layer_nums  # pathway 层数
        self.num_nodes = configs.pf_num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.pf_k
        self.num_experts_list = list(configs.pf_num_experts_list)
        self.patch_size_list = self._reshape_patch_size_list(configs.pf_patch_size_list)
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.pf_residual_connection
        self.revin = configs.pf_revin
        if self.revin:
            self.revin_layer = RevIN(num_features=self.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.batch_norm = configs.pf_batch_norm

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def _reshape_patch_size_list(self, patch_size_list):
        """扁平列表 -> 每层 patch 列表（官方 run.py: np.reshape(layer_nums, -1)）。"""
        patch_size_list = list(patch_size_list)
        if patch_size_list and isinstance(patch_size_list[0], (list, tuple)):
            return [list(p) for p in patch_size_list]
        assert len(patch_size_list) % self.layer_nums == 0, \
            'pf_patch_size_list 长度必须能被 pf_layer_nums 整除'
        per_layer = len(patch_size_list) // self.layer_nums
        return [patch_size_list[i * per_layer:(i + 1) * per_layer] for i in range(self.layer_nums)]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = x_enc  # [B, L, C]，官方 forward 只吃 x
        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))  # [B, L, C, d_model]

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)  # [B, pred_len, C]

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        if self.training:
            return out, balance_loss
        return out
