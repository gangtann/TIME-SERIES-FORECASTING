#!/usr/bin/env bash
# PRISM clock 第四轮探针批：ETTm1_96 单格，在 MAE+np16（0.3196）之上重标定
# 与损失函数耦合的超参：lr / patch 预算上探 / 编码器深度 / FreDF 权重。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_iter4.sh > /root/autodl-tmp/clock_iter4.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

REC="training.auxi_lambda=0.3 training.learning_rate=0.0005 model.dropout=0.1 training.loss=MAE prism.clock_n_patch=16"

EXPS=(
    "clk4_lr1e3|$REC training.learning_rate=0.001"
    "clk4_np24|$REC prism.clock_n_patch=24"
    "clk4_el3|$REC model.e_layers=3"
    "clk4_aux05|$REC training.auxi_lambda=0.5"
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
echo "===== clock_iter4 finished, failures: $fail  ($(date '+%F %T')) ====="
