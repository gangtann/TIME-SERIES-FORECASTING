#!/usr/bin/env bash
# 固定段长对照（ETTh1，{96,336} × 3 种子）：等分段数 N ∈ {4, 3, 8}，即段长 24 / 32 / 12。
# 目的：ETTh1 主周期实测为 24（ACF 0.80），现行等分 16 = 2/3 天、每个 token 相位不同；
# 4×24 让每个切点落在日边界、每个 token 是相位一致的完整一天——这是"网格与周期不对齐"
# 这一问题在固定网格下的解法，用来测量自适应分块在 ETTh1 上的余量：
# 若 4×24 已明显优于 6×16，自适应须超过 24 这条线；若三者都在噪声内，边界位置不是 ETTh1 的杠杆。
# 其余键沿用主配方（aux_stream=trend，与 5 种子 aux 基线同配方：96 0.3773 / 336 0.4665）。
# 用法: PYTHON=... bash scripts/PRISM/run_etth1_ntok.sh
cd "$(dirname "$0")/../.."
export DATA=ETTh1 PREDS="${PREDS:-96 336}" SEEDS="${SEEDS:-2021 2022 2023}"
R="prism.aux_stream=trend"   # 训练配方（无轮数上限/type3/耐心 5）已写入 configs/PRISM/*.yaml
fail=0
run () { TAG=$1 REC="$R $2" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1)); }
run ntok4 "prism.n_tokens=4"
run ntok3 "prism.n_tokens=3"
run ntok8 "prism.n_tokens=8"
echo "===== etth1 ntok battery finished, failing batches: $fail  ($(date '+%F %T')) ====="
echo NTOK_BATTERY_DONE
