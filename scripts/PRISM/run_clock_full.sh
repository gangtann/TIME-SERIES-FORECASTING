#!/usr/bin/env bash
# PRISM clock 最终全网格批：5 数据集 × pred{96,192,336,720}，统一最终配方。
# model_id 后缀 _clkf（final）；快速扫描的 _clk 行（旧配方）保留待清理。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_full.sh > /root/autodl-tmp/clock_full.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

# 最终配方（ETTm1_96 五轮探针收敛结果：0.3167/0.3477；dropout 用各格 yaml 默认）
RECIPE="training.auxi_lambda=0.3 training.loss=MAE prism.clock_n_patch=16 training.learning_rate=0.001"

CONFIGS=(
    ETTh1_96 ETTh1_192 ETTh1_336 ETTh1_720
    ETTh2_96 ETTh2_192 ETTh2_336 ETTh2_720
    ETTm1_96 ETTm1_192 ETTm1_336 ETTm1_720
    ETTm2_96 ETTm2_192 ETTm2_336 ETTm2_720
    weather_96 weather_192 weather_336 weather_720
)

fail=0
for name in "${CONFIGS[@]}"; do
    cfg="configs/PRISM/${name}.yaml"
    mid="$(grep -E '^\s*model_id:' "$cfg" | awk '{print $2}')_clkf"
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
echo "===== clock_full finished, failures: $fail  ($(date '+%F %T')) ====="
