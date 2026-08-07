import math
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend('agg')


class _Tee:
    def __init__(self, stream, file):
        self.stream = stream
        self.file = file

    def write(self, msg):
        self.stream.write(msg)
        self.file.write(msg)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


@contextmanager
def tee_output(log_path):
    """把 stdout/stderr 同时写入终端和日志文件（追加模式）。"""
    with open(log_path, 'a') as f:
        f.write('\n===== run at {} =====\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_stdout, f)
        sys.stderr = _Tee(old_stderr, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # 严格复现模式：禁用 cuDNN 的非确定性算法选择，训练会稍慢
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'cosine':
        lr_adjust = {epoch: args.learning_rate / 10 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'cosine2':
        # TimeFilter 官方调度（TROUBADOUR000/TimeFilter utils/tools.py）：lr/2*(1+cos)，
        # 与上面 cosine 挡（lr/10）公式不同，勿混用
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'sigmoid':
        # xPatch 官方调度（stitsyuk/xPatch utils/tools.py）：先 logistic 预热再平滑衰减
        k, s, w = 0.5, 10, 10
        lr_adjust = {epoch: args.learning_rate / (1 + math.exp(-k * (epoch - w)))
                     - args.learning_rate / (1 + math.exp(-k / s * (epoch - w * s)))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    else:
        raise ValueError(f'unknown lradj: {args.lradj}')

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """最优权重始终保留在内存（best_state）；save_to_disk=True 时才写 checkpoint.pth。"""

    def __init__(self, patience=7, verbose=False, delta=0, save_to_disk=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_to_disk = save_to_disk
        self.best_state = None

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if self.save_to_disk:
            torch.save(self.best_state, os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()
