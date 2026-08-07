#!/usr/bin/env bash
# PRISM clock 消融批：ETTm1_96 单格，最终配方下逐一关断核心模块。
# ①clock_mode 三挡拆"边界移动 vs 箱内重加权"；②clock_density 拆频域驱动贡献；
# ③clock_basis=1 拆基调制池化；④⑤⑥ use_graph/use_srm/use_fusion 结构消融。
#
# 服务器：nohup env PYTHON=/root/miniconda3/envs/tsf/bin/python \
#           RESULTS_CSV=/root/autodl-tmp/results/results.csv \
#           bash scripts/PRISM/run_clock_ablation.sh > /root/autodl-tmp/clock_abl.log 2>&1 &
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}

# 最终配方（与主表 _clkf 一致）
REC="training.auxi_lambda=0.3 training.loss=MAE prism.clock_n_patch=16 training.learning_rate=0.001"

EXPS=(
    "abl_uniform|$REC prism.clock_mode=uniform"
    "abl_reweight|$REC prism.clock_mode=reweight_only"
    "abl_mlp|$REC prism.clock_density=mlp"
    "abl_basis1|$REC prism.clock_basis=1"
    "abl_nograph|$REC prism.use_graph=false"
    "abl_nosrm|$REC prism.use_srm=false"
    "abl_nofusion|$REC prism.use_fusion=false"
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
echo "===== clock_ablation finished, failures: $fail  ($(date '+%F %T')) ====="
