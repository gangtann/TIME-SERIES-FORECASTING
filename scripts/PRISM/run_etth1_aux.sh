#!/usr/bin/env bash
# ETTh1 判决批：近端流（aux_stream 三挡）× 分关系门（token_mix_gate）析因，{96,336} × 3 种子。
# 判据：96 格相对 main(5 种子 0.3834) ≥ +2%（≈4σ，σ 实测 0.13~0.45%）且 336 格(0.4698) 退步 ≤ 1σ。
# 用法: PYTHON=... bash scripts/PRISM/run_etth1_aux.sh
cd "$(dirname "$0")/../.."
export DATA=ETTh1 PREDS="${PREDS:-96 336}" SEEDS="${SEEDS:-2021 2022 2023}"
R=""   # 训练配方（无轮数上限/type3/耐心 5）已写入 configs/PRISM/*.yaml
fail=0
run () { TAG=$1 REC="$R $2" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1)); }
run auxFull   "prism.aux_stream=full"
run auxTrend  "prism.aux_stream=trend"
run auxLast   "prism.aux_stream=last"
run gateSplit "prism.token_mix_gate=split"
run auxFullSplit  "prism.aux_stream=full  prism.token_mix_gate=split"
run auxTrendSplit "prism.aux_stream=trend prism.token_mix_gate=split"
run auxLastSplit  "prism.aux_stream=last  prism.token_mix_gate=split"
echo "===== etth1 aux battery finished, failing batches: $fail  ($(date '+%F %T')) ====="
echo AUX_BATTERY_DONE
