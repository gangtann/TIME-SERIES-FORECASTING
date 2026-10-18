"""数据集周期结构探针（docs/dataset_periodicity.md 的复现脚本）。

训练段（前 70%）逐通道标准化后，报告：
(1) 平均自相关的峰（滞后 2..400，阈值 > 0.1，取最高 5 个）；
(2) 平均功率谱的最强 6 个频点换算的周期（按 4×seq_len 切窗）；
(3) seq_len 窗内的信息密度不均匀度：按块算方差，块间方差的变异系数（越大越不均匀）。
用法: python scripts/period_probe.py [--root ./dataset/]
"""
import argparse

import numpy as np
import pandas as pd

SETS = [('ETTh1', 'ETTh1.csv', 96, 8), ('ETTh2', 'ETTh2.csv', 96, 8),
        ('ETTm1', 'ETTm1.csv', 96, 8), ('ETTm2', 'ETTm2.csv', 96, 8),
        ('weather', 'weather.csv', 96, 8), ('electricity', 'electricity.csv', 96, 8),
        ('traffic', 'traffic.csv', 96, 8), ('exchange', 'exchange_rate.csv', 96, 8),
        ('illness', 'national_illness.csv', 36, 4)]


def probe(path, L, blk):
    df = pd.read_csv(path)
    x = df.iloc[:, 1:].to_numpy(np.float64)
    n = int(len(x) * 0.7)
    x = x[:n]
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    if x.shape[1] > 64:
        x = x[:, ::max(1, x.shape[1] // 64)]  # 大数据集等距抽 64 通道

    lmax = min(400, n // 4)
    xf = np.fft.rfft(x - x.mean(0), n=2 * n, axis=0)
    ac = np.fft.irfft(np.abs(xf) ** 2, axis=0)[:lmax]
    ac = (ac / ac[0]).mean(1)
    peaks = [k for k in range(2, lmax - 1)
             if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.1]
    peaks = sorted(sorted(peaks, key=lambda k: -ac[k])[:5])

    W = 4 * L
    segs = np.stack([x[i:i + W] for i in range(0, n - W, W)])
    ps = (np.abs(np.fft.rfft(segs - segs.mean(1, keepdims=True), axis=1)) ** 2).mean((0, 2))
    top = np.argsort(-ps[1:])[:6] + 1
    periods = [W / k for k in top]

    wins = np.stack([x[i:i + L] for i in range(0, n - L, L)])
    bv = wins.reshape(len(wins), L // blk, blk, -1).var(2)
    cv = float((bv.std(1) / (bv.mean(1) + 1e-8)).mean())
    return df.shape[1] - 1, n, [(k, ac[k]) for k in peaks], periods, cv


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./dataset/')
    args = ap.parse_args()
    for name, f, L, blk in SETS:
        try:
            C, n, peaks, periods, cv = probe(args.root + f, L, blk)
        except FileNotFoundError:
            print(f'{name:12s} (missing {f})')
            continue
        print(f'{name:12s} C={C:4d} n={n:6d}  ACF峰 '
              + ' '.join(f'{k}:{v:.2f}' for k, v in peaks)
              + '  | 谱峰周期 ' + ' '.join(f'{p:.1f}' for p in periods)
              + f'  | {L}窗{blk}点块方差CV {cv:.2f}')
