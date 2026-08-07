#!/usr/bin/env bash
# PRISM clock 快速验证批：ETT×4 + weather × pred{96,336}，统一叠加探针批选出的最优配方。
# model_id 加 _clk 后缀，与 v1 结果行并存；results.csv 已有的自动跳过。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_quick.sh > /root/autodl-tmp/clock_quick.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

# 探针批（ETTm1_96×8 变体）最优配方：clk_full 0.3298 ≈ clk_full_ema 0.3299，
# mavg 在组合中无实质贡献，取更简的 EMA 版
RECIPE="training.auxi_lambda=0.3 training.learning_rate=0.0005 model.dropout=0.1"

CONFIGS=(
    ETTh1_96 ETTh1_336
    ETTh2_96 ETTh2_336
    ETTm1_96 ETTm1_336
    ETTm2_96 ETTm2_336
    weather_96 weather_336
)

fail=0
for name in "${CONFIGS[@]}"; do
    cfg="configs/PRISM/${name}.yaml"
    mid="$(grep -E '^\s*model_id:' "$cfg" | awk '{print $2}')_clk"
    if [ -f "$RESULTS_CSV" ] && grep -q ",${mid}_exp_0," "$RESULTS_CSV"; then
        echo "[skip] $name (already in results.csv)"
        continue
    fi
    echo "[run ] PRISM $name  ($(date '+%F %T'))"
    # shellcheck disable=SC2086
    if ! "$PYTHON" run.py -c "$cfg" --set model_id="$mid" $RECIPE; then
        echo "[FAIL] $name"
        fail=$((fail + 1))
    fi
done
echo "===== clock_quick finished, failures: $fail  ($(date '+%F %T')) ====="
