#!/usr/bin/env bash
# oracle-EM 分块判决批：默认只跑 ETTh1（{96,336} × 3 种子）；DATASETS 可追加其他数据集（会先跑其同池固定 4×24 对照）。
# 配方：周期池 {12,24,36,48}、N=4、T0=24（等分 4×24 是划分集合之一，共 31 种划分），其余同主配方（aux_stream=trend）。
# 判据（只比数值）：
#   不输性：router-test ≤ 同池固定 4×24（ETTh1 0.3801 / 0.4641；weather 由本批 ntok4 给出）——设计保证，违反即 bug；
#   余量：测试日志 diag oracle_mse − 固定 = 该数据集逐样本自适应上界；
#   有效：router-test < 固定 且 < 6×16 现行主配方（ETTh1 0.3773 / 0.4665）。
# 标签：oracle = 首轮（oracle_train_on=oracle，已证实训练/测试划分分布不一致）；oracleR = M 步按路由器划分训练（默认）。
# 用法: PYTHON=... [DATASETS="ETTh1"] ORACLE_TAG=oracleR ORACLE_EXTRA="" bash scripts/PRISM/run_oracle.sh
cd "$(dirname "$0")/../.."
export PREDS="${PREDS:-96 336}" SEEDS="${SEEDS:-2021 2022 2023}"
R="prism.aux_stream=trend"   # 训练配方（无轮数上限/type3/耐心 5）已写入 configs/PRISM/*.yaml
POOL="prism.n_tokens=4 prism.min_patch_len=12 prism.patch_pool=[0,12,24,36] prism.adp_resample_len=24"
fail=0
for ds in ${DATASETS:-ETTh1}; do
  export DATA=$ds
  if [ "$ds" != "ETTh1" ]; then   # ETTh1 的固定 4×24 已有（ntok4）
    TAG=ntok4 REC="$R prism.n_tokens=4" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1))
  fi
  TAG=${ORACLE_TAG:-oracleR} REC="$R $POOL prism.patch_mode=oracle prism.patch_norm=resample $ORACLE_EXTRA" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1))
done
echo "===== oracle battery finished, failing batches: $fail  ($(date '+%F %T')) ====="
echo ORACLE_BATTERY_DONE
