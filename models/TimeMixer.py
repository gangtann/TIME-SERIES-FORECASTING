import torch
from torch import nn

from layers.TimeMixer_layers import (
    PastDecomposableMixing,
    TMX_DataEmbedding_wo_pos,
    TMX_Normalize,
    TMX_series_decomp,
)


class Model(nn.Module):
    """TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (ICLR 2024).

    官方仓库: https://github.com/kwuking/TimeMixer（models/TimeMixer.py @ main）

    数据流（[B, L, C] 输入）：
      多尺度下采样(avg/max/conv, 共 down_sampling_layers+1 个尺度) → 逐尺度 Normalize
      → (channel_independence=1 时展平为 [B*C, T, 1]) → DataEmbedding_wo_pos
      → e_layers × PastDecomposableMixing（逐尺度 moving_avg 分解，season 自顶向下、
        trend 自底向上跨尺度混合）→ 逐尺度线性预测头映射到 pred_len → 多预测器求和
      → 尺度 0 的 Normalize 反归一化 → [B, pred_len, c_out]

    移植改动（相对官方源码，逐行 diff 见移植报告）：
      1. 只保留 long_term_forecast 路径，删除 imputation/anomaly/classification 分支
         （本框架 exp 只跑长程预测，与 models/PatchTST.py 同规格）；
      2. 专属配置键改读 tmx_ 前缀（tmx_down_sampling_layers/window/method、
         tmx_channel_independence、tmx_decomp_method、tmx_moving_avg、tmx_top_k、
         tmx_use_future_temporal_feature）；use_norm 读共享键 configs.use_norm；
      3. 下采样池化子在 __init__ 构建并注册（官方在 forward 内临时 new；method=conv 时
         官方写法参数不在 GPU 上也不参与训练，此处修正为注册子模块，设备无关）；
      4. tmx_use_future_temporal_feature=1 时把 x_mark_dec 裁到最后 pred_len 步
         （官方协议 label_len=0，本框架 x_mark_dec 含 label_len+pred_len 步）；
      5. 无内部辅助损失，forward 恒返回单值 outputs。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.down_sampling_window = configs.tmx_down_sampling_window
        self.down_sampling_layers = configs.tmx_down_sampling_layers
        self.channel_independence = int(configs.tmx_channel_independence)
        self.use_future_temporal_feature = int(configs.tmx_use_future_temporal_feature)

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        self.preprocess = TMX_series_decomp(configs.tmx_moving_avg)

        if self.channel_independence == 1:
            self.enc_embedding = TMX_DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                          configs.dropout)
        else:
            self.enc_embedding = TMX_DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed,
                                                          configs.freq, configs.dropout)

        self.layer = configs.e_layers

        # 官方在 forward 内按需临时构建 down_pool；此处移到 __init__ 注册（见类 docstring 改动 3）
        if configs.tmx_down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif configs.tmx_down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif configs.tmx_down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            self.down_pool = nn.Conv1d(in_channels=configs.enc_in, out_channels=configs.enc_in,
                                       kernel_size=3, padding=padding,
                                       stride=self.down_sampling_window,
                                       padding_mode='circular',
                                       bias=False)
        else:
            self.down_pool = None  # 官方行为：未知 method 时不做多尺度下采样

        self.normalize_layers = torch.nn.ModuleList(
            [
                TMX_Normalize(configs.enc_in, affine=True, non_norm=not bool(configs.use_norm))
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (self.down_sampling_window ** i),
                    configs.pred_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (self.down_sampling_window ** i),
                    configs.seq_len // (self.down_sampling_window ** i),
                )
                for i in range(self.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (self.down_sampling_window ** i),
                        configs.pred_len,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_pool is None:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = self.down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_future_temporal_feature:
            # 官方协议 label_len=0，x_mark_dec 即未来 pred_len 步；本框架 batch_y_mark
            # 含 label_len+pred_len 步，这里裁到最后 pred_len 步对齐官方
            x_mark_dec = x_mark_dec[:, -self.pred_len:, :]
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                    dec_out = self.projection_layer(dec_out)
                else:
                    dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]
