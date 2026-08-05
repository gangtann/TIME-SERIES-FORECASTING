#!/usr/bin/env bash
# PatchTST / ETTh1 全部四个预测长度（96/192/336/720），顺序执行。
# 用法：bash scripts/PatchTST/ETTh1.sh
# 每个实验的日志自动落盘到 checkpoints/<setting>/train.log，
# 指标汇总追加到 results/results.csv。
set -e
cd "$(dirname "$0")/../.."

for pl in 96 192 336 720; do
    echo "========== ETTh1 96 -> ${pl} =========="
    python run.py -c "configs/PatchTST/ETTh1_${pl}.yaml"
done
