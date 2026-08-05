#!/usr/bin/env bash
# 跑齐所有可用数据集的 PatchTST 实验；results.csv 里已有的自动跳过（中断后重跑可续接）。
#
# 本地：  bash scripts/PatchTST/run_all.sh
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PatchTST/run_all.sh > /root/autodl-tmp/run_all.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

# 按预计耗时升序排列，快的先出结果；traffic 最慢放最后
CONFIGS=(
    ETTh1_96 ETTh1_192 ETTh1_336 ETTh1_720
    ETTh2_96 ETTh2_192 ETTh2_336 ETTh2_720
    illness_24 illness_36 illness_48 illness_60
    exchange_96 exchange_192 exchange_336 exchange_720
    weather_96 weather_192 weather_336 weather_720
    ETTm1_96 ETTm1_192 ETTm1_336 ETTm1_720
    ETTm2_96 ETTm2_192 ETTm2_336 ETTm2_720
    electricity_96 electricity_192 electricity_336 electricity_720
    traffic_96 traffic_192 traffic_336 traffic_720
)

fail=0
for name in "${CONFIGS[@]}"; do
    cfg="configs/PatchTST/${name}.yaml"
    mid=$(grep -E '^\s*model_id:' "$cfg" | awk '{print $2}')
    if [ -f "$RESULTS_CSV" ] && grep -q ",${mid}_exp_0," "$RESULTS_CSV"; then
        echo "[skip] $name (already in results.csv)"
        continue
    fi
    echo "[run ] $name  ($(date '+%F %T'))"
    if ! "$PYTHON" run.py -c "$cfg"; then
        echo "[FAIL] $name"
        fail=$((fail + 1))
    fi
done
echo "===== run_all finished, failures: $fail  ($(date '+%F %T')) ====="
