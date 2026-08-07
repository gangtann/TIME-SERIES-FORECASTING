#!/usr/bin/env bash
# PRISM clock 第二轮探针批：ETTm1_96 单格，在定稿配方（aux03+lr5e4+drop01=0.3299）
# 之上逐一试未测杠杆：损失函数 / 短时谱窗宽 / patch 预算 / 密度下界 / 容量 / epoch。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_iter2.sh > /root/autodl-tmp/clock_iter2.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

REC="training.auxi_lambda=0.3 training.learning_rate=0.0005 model.dropout=0.1"

EXPS=(
    "clk2_mae|$REC training.loss=MAE"
    "clk2_huber|$REC training.loss=huber"
    "clk2_smooth|$REC training.loss=smooth_l1"
    "clk2_win24|$REC prism.clock_win=24"
    "clk2_win32|$REC prism.clock_win=32"
    "clk2_np8|$REC prism.clock_n_patch=8"
    "clk2_np16|$REC prism.clock_n_patch=16"
    "clk2_floor01|$REC prism.clock_floor=0.1"
    "clk2_dm128|$REC model.d_model=128 model.d_ff=256"
    "clk2_ep60|$REC training.train_epochs=60"
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
echo "===== clock_iter2 finished, failures: $fail  ($(date '+%F %T')) ====="
