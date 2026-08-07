#!/usr/bin/env bash
# PRISM clock 配方探针批：ETTm1_96 单格，把 v1 迭代验证过的配方叠加到 clock 主干。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_iter.sh > /root/autodl-tmp/clock_iter.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

EXPS=(
    "clk_aux03|training.auxi_lambda=0.3"
    "clk_aux05|training.auxi_lambda=0.5"
    "clk_aux03_lr5e4|training.auxi_lambda=0.3 training.learning_rate=0.0005"
    "clk_aux03_drop01|training.auxi_lambda=0.3 model.dropout=0.1"
    "clk_aux03_mavg|training.auxi_lambda=0.3 prism.decomp_type=moving_avg"
    "clk_aux03_alpha015|training.auxi_lambda=0.3 prism.decomp_alpha=0.15"
    "clk_full|training.auxi_lambda=0.3 training.learning_rate=0.0005 model.dropout=0.1 prism.decomp_type=moving_avg"
    "clk_full_ema|training.auxi_lambda=0.3 training.learning_rate=0.0005 model.dropout=0.1"
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
echo "===== clock_iter finished, failures: $fail  ($(date '+%F %T')) ====="
