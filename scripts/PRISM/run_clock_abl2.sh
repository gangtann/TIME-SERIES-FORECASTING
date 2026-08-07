#!/usr/bin/env bash
# PRISM clock 三挡消融跨格补测：在 ETTh1_96 / weather_96 / ETTm1_720 上
# 跑 uniform 与 reweight_only 两挡（完整 adaptive 即 _clkf 行，已有）。
# 用于验证"边界自适应"在非 ETTm1_96 格上的贡献。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_abl2.sh > /root/autodl-tmp/clock_abl2.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

REC="training.auxi_lambda=0.3 training.loss=MAE prism.clock_n_patch=16 training.learning_rate=0.001"

fail=0
for name in ETTh1_96 weather_96 ETTm1_720; do
    cfg="configs/PRISM/${name}.yaml"
    base=$(grep -E '^\s*model_id:' "$cfg" | awk '{print $2}')
    for mode in uniform reweight_only; do
        mid="${base}_abl_${mode}"
        if [ -f "$RESULTS_CSV" ] && grep -q ",${mid}_exp_0," "$RESULTS_CSV"; then
            echo "[skip] $name $mode (already in results.csv)"
            continue
        fi
        echo "[run ] PRISM $name $mode  ($(date '+%F %T'))"
        # shellcheck disable=SC2086
        if ! "$PYTHON" run.py -c "$cfg" --set model_id="$mid" $REC prism.clock_mode="$mode"; then
            echo "[FAIL] $name $mode"
            fail=$((fail + 1))
        fi
    done
done
echo "===== clock_abl2 finished, failures: $fail  ($(date '+%F %T')) ====="
