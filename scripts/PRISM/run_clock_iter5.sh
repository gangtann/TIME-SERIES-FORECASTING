#!/usr/bin/env bash
# PRISM clock 第五轮（收尾微调）：ETTm1_96 单格，在 MAE+np16+lr1e-3（0.3167）
# 两侧探 lr 甜点并复核 dropout。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_iter5.sh > /root/autodl-tmp/clock_iter5.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

REC="training.auxi_lambda=0.3 model.dropout=0.1 training.loss=MAE prism.clock_n_patch=16 training.learning_rate=0.001"

EXPS=(
    "clk5_lr15e4|$REC training.learning_rate=0.0015"
    "clk5_lr2e3|$REC training.learning_rate=0.002"
    "clk5_drop02|$REC model.dropout=0.2"
)

fail=0
for e in "${EXPS[@]}"; do
    suffix=${e%%|*}
    overrides=${e#*|}
    mid="PRISM_ETTm1_96_96_${suffix}"
    if [ -f "$RESULTS_CSV" ] && grep -q ",${mid}_exp_0," "$RESULTS_CSV"; then
        echo "[skip] $suffix (already in results.csv)"
        continue
    fi
    echo "[run ] PRISM ETTm1_96 $suffix  ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    if ! "$PYTHON" run.py -c configs/PRISM/ETTm1_96.yaml --set model_id="$mid" $overrides; then
        echo "[FAIL] $suffix"
        fail=$((fail + 1))
    fi
done
echo "===== clock_iter5 finished, failures: $fail  ($(date '+%F %T') ) ====="
