#!/usr/bin/env bash
# PRISM clock 第三轮探针批：ETTm1_96 单格，叠加第二轮三个独立大杠杆
# （MAE 损失 -2.5% / d_model128 -1.7% / n_patch16 -1.3%）的组合。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_iter3.sh > /root/autodl-tmp/clock_iter3.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

REC="training.auxi_lambda=0.3 training.learning_rate=0.0005 model.dropout=0.1 training.loss=MAE"

EXPS=(
    "clk3_mae_dm128|$REC model.d_model=128 model.d_ff=256"
    "clk3_mae_np16|$REC prism.clock_n_patch=16"
    "clk3_mae_dm128_np16|$REC model.d_model=128 model.d_ff=256 prism.clock_n_patch=16"
    "clk3_mae_dm128_np16_win24|$REC model.d_model=128 model.d_ff=256 prism.clock_n_patch=16 prism.clock_win=24"
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
echo "===== clock_iter3 finished, failures: $fail  ($(date '+%F %T')) ====="
