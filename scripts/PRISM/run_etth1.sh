#!/usr/bin/env bash
# ETTh1 四格实验驱动（本阶段唯一战场）。所有探针/消融复用本脚本。
#
# 用法: TAG=<结果后缀> REC="<--set 覆盖串>" bash scripts/PRISM/run_etth1.sh
# 例:   TAG=f1 REC="training.lradj=cosine training.train_epochs=30" bash scripts/PRISM/run_etth1.sh
# 环境变量: PREDS 限定视野（默认四格）、RESULTS 结果目录、PYTHON 解释器。
#
# 服务器: ssh -f autodl 'nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python TAG=f1 \
#           REC="..." bash /root/TIME-SERIES-FORECASTING/scripts/PRISM/run_etth1.sh \
#           > /root/autodl-tmp/logs/PRISM_f1.log 2>&1 < /dev/null; exit'
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
TAG=${TAG:?need TAG=<suffix>}
REC=${REC:-}
RESULTS=${RESULTS:-/root/autodl-tmp/results_frame/}
PREDS=${PREDS:-"96 192 336 720"}

fail=0
for pl in $PREDS; do
    cfg="configs/PRISM/ETTh1_${pl}.yaml"
    mid="PRISM_ETTh1_96_${pl}_${TAG}"
    if grep -q ",${mid}_exp_0," "${RESULTS}/results.csv" 2>/dev/null; then
        echo "[skip] $mid (already in results)"
        continue
    fi
    echo "[run ] $mid  ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    if ! "$PYTHON" run.py -c "$cfg" --set model_id="$mid" system.results="$RESULTS" $REC; then
        echo "[FAIL] $mid"
        fail=$((fail + 1))
    fi
done
echo "===== ETTh1 [$TAG] finished, failures: $fail  ($(date '+%F %T')) ====="
