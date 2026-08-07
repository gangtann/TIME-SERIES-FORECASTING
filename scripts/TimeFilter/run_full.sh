#!/usr/bin/env bash
# TimeFilter (ICML 2025) 全网格批：5 数据集 × pred{96,192,336,720}，共 20 格。
# 官方超参全部在 configs/TimeFilter/*.yaml 内（出处见各 yaml 顶部注释），不叠 --set。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/TimeFilter/run_full.sh > /root/autodl-tmp/timefilter_full.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

CONFIGS=(
    ETTh1_96 ETTh1_192 ETTh1_336 ETTh1_720
    ETTh2_96 ETTh2_192 ETTh2_336 ETTh2_720
    ETTm1_96 ETTm1_192 ETTm1_336 ETTm1_720
    ETTm2_96 ETTm2_192 ETTm2_336 ETTm2_720
    weather_96 weather_192 weather_336 weather_720
)

fail=0
for name in "${CONFIGS[@]}"; do
    cfg="configs/TimeFilter/${name}.yaml"
    mid="$(grep -E '^\s*model_id:' "$cfg" | awk '{print $2}')"
    if [ -f "$RESULTS_CSV" ] && grep -q ",${mid}_exp_0," "$RESULTS_CSV"; then
        echo "[skip] $name (already in results.csv)"
        continue
    fi
    echo "[run ] TimeFilter $name  ($(date '+%F %T'))"
    if ! "$PYTHON" run.py -c "$cfg"; then
        echo "[FAIL] $name"
        fail=$((fail + 1))
    fi
done
echo "===== timefilter_full finished, failures: $fail  ($(date '+%F %T')) ====="
