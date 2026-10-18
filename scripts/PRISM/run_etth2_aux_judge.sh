#!/usr/bin/env bash
# 近端流（aux_stream=trend）进不进最终架构：在 ETTh2 上只看验证集判定（用户决定 2026-09-10）。
# 背景：ETTh1 上该组件是按测试集确认的，ETTh1 验证集不支持（四视野最优验证损失之和 4.5430 vs 4.5257）。
# 判据：四视野 × 5 种子的最优验证损失（results.csv 的 best_vali 列）之和，frzAux < frz 则进架构，否则不进。
# 两组都用当前代码、当前配方（无轮数上限/type3/耐心 5）重跑，保证只差这一个开关。ETTh2 此后不再作 held-out。
# 用法: PYTHON=... bash scripts/PRISM/run_etth2_aux_judge.sh
cd "$(dirname "$0")/../.."
export DATA=ETTh2 PREDS="${PREDS:-96 192 336 720}" SEEDS="${SEEDS:-2021 2022 2023 2024 2025}"
fail=0
TAG=frz    REC="prism.aux_stream=none"  bash scripts/PRISM/run_seeds.sh || fail=$((fail+1))
TAG=frzAux REC="prism.aux_stream=trend" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1))
echo "===== etth2 aux judge finished, failing batches: $fail  ($(date '+%F %T')) ====="
echo AUXJUDGE_DONE
