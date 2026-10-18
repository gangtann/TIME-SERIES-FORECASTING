#!/usr/bin/env bash
# 多种子驱动。本模型 run 间相对标准差实测 0.2~5.1%（常大于待测缺口），
# 单种子结论不可用，凡要下判断的实验一律走本脚本。
#
# 用法: TAG=<后缀> REC="<--set 覆盖串>" bash scripts/PRISM/run_seeds.sh
# 环境变量: DATA(默认 ETTh1)、SEEDS(默认 2021 2022 2023)、PREDS(默认四格)、RESULTS、PYTHON
# 结果 model_id 形如 PRISM_<DATA>_96_<pl>_<TAG>s<seed>
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
DATA=${DATA:-ETTh1}
TAG=${TAG:?need TAG=<suffix>}
REC=${REC:-}
RESULTS=${RESULTS:-/root/autodl-tmp/results_frame/}
PREDS=${PREDS:-"96 192 336 720"}
SEEDS=${SEEDS:-"2021 2022 2023"}

fail=0
for pl in $PREDS; do
    for sd in $SEEDS; do
        mid="PRISM_${DATA}_96_${pl}_${TAG}s${sd}"
        if grep -q ",${mid}_exp_0," "${RESULTS}/results.csv" 2>/dev/null; then
            echo "[skip] $mid"; continue
        fi
        echo "[run ] $mid  ($(date '+%F %T'))"
        # shellcheck disable=SC2086
        if ! "$PYTHON" run.py -c "configs/PRISM/${DATA}_${pl}.yaml" \
             --set model_id="$mid" system.results="$RESULTS" basic.seed="$sd" $REC; then
            echo "[FAIL] $mid"; fail=$((fail + 1))
        fi
    done
done
echo "===== ${DATA} seeds [$TAG] finished, failures: $fail  ($(date '+%F %T')) ====="
