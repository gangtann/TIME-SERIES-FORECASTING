import csv
import math
import os
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class ArctanMAELoss(nn.Module):
    """xPatch 官方 arctan 逐 horizon 加权 MAE（stitsyuk/xPatch exp/exp_main.py）。

    ratio_i = -atan(i+1) + π/4 + 1（i 从 0 起，首步恰为 1，远端衰减到 1-π/4），
    对 pred 与 true 同乘权重后取 L1。官方实现硬编码 .to('cuda')，此处跟随输入设备。
    """

    def __init__(self, pred_len):
        super().__init__()
        ratio = torch.tensor([-math.atan(i + 1) + math.pi / 4 + 1 for i in range(pred_len)],
                             dtype=torch.float32).unsqueeze(-1)
        self.register_buffer('ratio', ratio)
        self.l1 = nn.L1Loss()

    def forward(self, pred, true):
        r = self.ratio.to(device=pred.device, dtype=pred.dtype)
        return self.l1(pred * r, true * r)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 训练损失可切换（training.loss）；验证/早停恒用 MSE，保证模型选择口径不变
        name = str(self.args.loss).lower()
        if name in ('mse', 'l2'):
            return nn.MSELoss()
        if name in ('mae', 'l1'):
            return nn.L1Loss()
        if name == 'huber':
            return nn.HuberLoss(delta=1.0)
        if name == 'smooth_l1':
            return nn.SmoothL1Loss()
        if name == 'arctan_mae':
            return ArctanMAELoss(self.args.pred_len)
        raise ValueError(f'unknown training loss: {self.args.loss}')

    def _forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input: [label_len 真实值; pred_len 零]（PatchTST 不用，留给未来 decoder 类模型）
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        mb = getattr(self.args, 'micro_batch', 0) or 0
        if mb and not torch.is_grad_enabled() and batch_x.shape[0] > mb:
            # 验证/测试（no_grad）按 micro_batch 分块前向再拼回：逐样本独立的模型下与整批前向等价，
            # 只为装进显存（如 PRISM token 混合的跨通道注意力随 batch × C² 增长）
            outs = [self._model_call(batch_x[s:s + mb], batch_x_mark[s:s + mb], dec_inp[s:s + mb],
                                     batch_y_mark[s:s + mb]) for s in range(0, batch_x.shape[0], mb)]
            outputs = torch.cat([o[0] if isinstance(o, tuple) else o for o in outs], dim=0)
        else:
            outputs = self._model_call(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        # 模型可返回 (outputs, aux_loss) 元组携带内部正则项（如路由熵）；仅训练时使用
        aux_loss = None
        if isinstance(outputs, tuple):
            outputs, aux_loss = outputs
        return outputs, batch_y, aux_loss

    def _model_call(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _train_loss(self, batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, epoch):
        """一个（微）批的训练损失：E 步（oracle 挡）→ 前向 → 预测损失 [+ 频域辅助] [+ 模型内正则]。"""
        if getattr(self.model, 'needs_oracle', False):
            # oracle-EM 分块的 E 步：用真实预测损失为本 batch 每条样本选划分（见 models/PRISM.py）
            self.model.oracle_step(batch_x.float().to(self.device),
                                   batch_y.float().to(self.device), epoch)
        outputs, batch_y, aux_loss = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
        if self.args.auxi_lambda > 0:
            # FreDF 式频域辅助损失（wang2025fredf）；早停/验证仍用纯 MSE
            freq_diff = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
            loss = (self.args.rec_lambda * criterion(outputs, batch_y)
                    + self.args.auxi_lambda * freq_diff.abs().mean())
        else:
            loss = criterion(outputs, batch_y)
        if aux_loss is not None:
            loss = loss + aux_loss
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                outputs, batch_y, _ = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs.detach(), batch_y.detach())
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       save_to_disk=self.args.save_checkpoint)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        vali_criterion = nn.MSELoss()   # 与 loss 开关解耦：早停/模型选择恒按 MSE

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # PatchTST 原版 'TST' 配方：OneCycleLR 逐 step 调整
        scheduler = None
        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                mb = getattr(self.args, 'micro_batch', 0) or 0
                B = batch_x.shape[0]
                if not mb or B <= mb:
                    loss = self._train_loss(batch_x, batch_y, batch_x_mark, batch_y_mark, criterion, epoch)
                    train_loss.append(loss.item())
                else:
                    # 梯度累积：batch 组成、步数、学习率计划与整批一致，只把前向/反传拆成 ≤ micro_batch 的块；
                    # 各块损失按样本占比加权，梯度之和 = 整批均值损失的梯度（逐样本独立的模型下严格成立）
                    loss_sum = 0.0
                    for s in range(0, B, mb):
                        part = self._train_loss(batch_x[s:s + mb], batch_y[s:s + mb], batch_x_mark[s:s + mb],
                                                batch_y_mark[s:s + mb], criterion, epoch) * (min(mb, B - s) / B)
                        if self.args.use_amp:
                            scaler.scale(part).backward()
                        else:
                            part.backward()
                        loss_sum += part.item()
                    loss = torch.tensor(loss_sum)
                    train_loss.append(loss_sum)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                accumulated = bool(mb) and B > mb   # 累积分支已逐块反传
                if self.args.use_amp:
                    if not accumulated:
                        scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    if not accumulated:
                        loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, vali_criterion)
            test_loss = self.vali(test_data, test_loader, vali_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 从内存回载验证集最优权重（save_checkpoint: true 时磁盘上也有一份）
        self.model.load_state_dict(early_stopping.best_state)
        # 最优验证损失随结果入 csv：一切配置选择只看这一列，测试集只在最终报成绩时用一次
        self.best_vali = float(early_stopping.val_loss_min)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            # 仅测试模式（is_training: 0）依赖磁盘 checkpoint，训练时需开 save_checkpoint: true
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = os.path.join(self.args.results, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                outputs, batch_y, _ = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :]
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                preds.append(outputs)
                trues.append(batch_y)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        if getattr(self.model, 'needs_oracle', False):
            # oracle 诊断一律在验证集上算（用真值选划分会看到标签，测试集只在上面报成绩时用一次）。
            # 先 eval() 一次清零模型在测试循环里累计的直方图，让下面 diagnostics() 只反映验证集
            self.model.eval()
            _, vali_loader = self._get_data(flag='val')
            self._oracle_diag(vali_loader, f_dim)
        if hasattr(self.model, 'diagnostics'):
            # 模型自报的机制读数（门值、选段直方图等；oracle 挡下按验证集统计），只打印不入 csv
            for k, v in self.model.diagnostics().items():
                print(f'diag {k}: {v}')

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        if self.args.save_arrays:
            np.save(os.path.join(folder_path, 'pred.npy'), preds)
            np.save(os.path.join(folder_path, 'true.npy'), trues)

        self._append_result_csv(setting, mse, mae, rmse, mape, mspe)
        return

    def _oracle_diag(self, loader, f_dim):
        """oracle 挡的三项读数（验证集）：路由器 mse / oracle 上界（用真值选划分）/ 交叉验证 oracle。

        交叉验证 oracle：用预测前半段的误差挑划分、只在后半段打分。P 选 1 的全段 oracle 会把该条样本的
        具体噪声也挑进来；若收益是"这条样本本身更适合某种划分"的真信号，它应在没参与挑选的后半段依然
        存在；若只是挑噪声，后半段收益应接近 0 或为正（变差）。
        """
        H = self.args.pred_len
        h = H // 2
        r_preds, o_preds, cv_preds, trues = [], [], [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                bx = batch_x.float().to(self.device)
                by = batch_y.float().to(self.device)
                out, by_, _ = self._forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                r_preds.append(out[:, -H:, f_dim:].cpu().numpy())
                trues.append(by_[:, -H:, f_dim:].cpu().numpy())
                o_preds.append(self.model.oracle_forward(bx, by)[:, -H:, f_dim:].cpu().numpy())
                cv_preds.append(self.model.oracle_forward(bx, by, select=slice(0, h), record=False)
                                [:, -H:, f_dim:].cpu().numpy())
        r, o, cv, t = (np.concatenate(a, axis=0) for a in (r_preds, o_preds, cv_preds, trues))
        r_mse, o_mse = float(((r - t) ** 2).mean()), float(((o - t) ** 2).mean())
        cv_mse, r2_mse = float(((cv[:, h:] - t[:, h:]) ** 2).mean()), float(((r[:, h:] - t[:, h:]) ** 2).mean())
        print(f'diag [vali] router_mse: {r_mse:.6f}')
        print(f'diag [vali] oracle_mse: {o_mse:.6f} (gap {100 * (o_mse - r_mse) / r_mse:+.2f}%)')
        print(f'diag [vali] oracle_cv (pick on steps [0,{h}), score on [{h},{H})): '
              f'{cv_mse:.6f} vs router {r2_mse:.6f}, gap {100 * (cv_mse - r2_mse) / r2_mse:+.2f}%')

    CSV_HEADER = ['timestamp', 'setting', 'model', 'data', 'seq_len', 'pred_len',
                  'mse', 'mae', 'rmse', 'mape', 'mspe', 'best_vali']

    def _append_result_csv(self, setting, mse, mae, rmse, mape, mspe):
        csv_path = os.path.join(self.args.results, 'results.csv')
        os.makedirs(self.args.results, exist_ok=True)
        if os.path.exists(csv_path):
            # 老文件缺 best_vali 列：一次性重写表头，旧行补空值（本文件只由本进程追加，重写是安全的）
            with open(csv_path, newline='') as f:
                rows = list(csv.reader(f))
            if rows and rows[0] != self.CSV_HEADER:
                extra = len(self.CSV_HEADER) - len(rows[0])
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.CSV_HEADER)
                    for r in rows[1:]:
                        writer.writerow(r + [''] * max(0, extra))
        else:
            with open(csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(self.CSV_HEADER)
        best_vali = getattr(self, 'best_vali', None)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), setting,
                self.args.model, self.args.data, self.args.seq_len, self.args.pred_len,
                f'{mse:.6f}', f'{mae:.6f}', f'{rmse:.6f}', f'{mape:.6f}', f'{mspe:.6f}',
                '' if best_vali is None else f'{best_vali:.6f}'])
