#!/usr/bin/env bash
# Baseline 全网格 5 种子填充器（论文用）。
#
# 定位：低优先级后台填充。主线（PRISM 改进实验）绝对优先——检测到任何 PRISM 实验
# 运行时完全让行，仅在主线空闲且显存有余量时才启动下一个 run。
#
# 排序：按实测/估计耗时由快到慢——先快数据集（行数少）、再快模型，
# 使论文表格尽早填上大部分格子。
#
# 命名：一律 <Model>_<data>_96_<pred>_s<seed>，五个种子命名统一。
#   不复用历史上无种子后缀的单次结果——那些跑于配置演进之前，无法保证同条件。
#
# 不写 checkpoint / 预测数组（save_checkpoint、save_arrays 默认 false，
# 最优权重仅驻内存，见 utils/tools.py）。
#
# 用法: PYTHON=... RESULTS_CSV=... bash scripts/run_baseline_grid.sh
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
RESULTS_CSV=${RESULTS_CSV:-./results/results.csv}
# 结果目录 = RESULTS_CSV 所在目录。多分片并行时各分片一个目录，
# 因为 exp_long_term_forecasting.py 以无锁 append 写 csv，共用一个文件有交错风险。
RESULTS_DIR=$(dirname "$RESULTS_CSV")/
SEEDS=${SEEDS:-"2021 2022 2023 2024 2025"}
PREDS=${PREDS:-"96 192 336 720"}
# illness 走领域惯例的另一套协议：seq_len 36、pred 24/36/48/60（配置里已写死 seq_len/label_len）
PREDS_ILLNESS=${PREDS_ILLNESS:-"24 36 48 60"}
# 由轻到重：ETT(7 变量,17k~70k 行) → exchange(8) → illness(7,仅 966 行) → weather(21)
# → electricity(321) → traffic(862)。逐通道模型的开销随变量数近线性涨，故大数据集垫后。
DATASETS=${DATASETS:-"ETTh1 ETTh2 exchange illness weather ETTm1 ETTm2 electricity traffic"}
# weather 实测(分钟/run): TimeMixer 6.8 · iTransformer 7.0 · PatchMLP 13.5 · DUET 19.0 · PathFormer 33.9
# 未实测者按结构复杂度插入；顺序只影响先后，不影响结果
MODELS=${MODELS:-"DLinear TimeMixer iTransformer xPatch PatchTST TimeFilter SimpleTM TimeMosaic PatchMLP DUET PathFormer"}
MIN_FREE_MB=${MIN_FREE_MB:-6000}   # 显存余量低于此值时等待，把 GPU 让给主线实验

# 让行条件有两个，缺一不可：
#   1. 主线（PRISM）没有实验在跑 —— 这些 baseline 模型都很小，显存从不是瓶颈，
#      争的是 GPU 时间片。实测三任务并发时主线单 run 从 3 分钟涨到 25 分钟。
#   2. 显存仍有余量 —— 防止与偶发的大模型撞 OOM。
wait_for_slot() {
    while :; do
        local n used total free
        n=$(ps -eo args --no-headers | grep -c "run\.py -c configs/PRISM/")
        if [ "$n" -gt 0 ]; then
            echo "[wait] 主线有 $n 个 PRISM 实验在跑，让行 ($(date '+%H:%M:%S'))"
            sleep 60; continue
        fi
        read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total \
                               --format=csv,noheader,nounits | tr ',' ' ')
        free=$((total - used))
        if [ "$free" -lt "$MIN_FREE_MB" ]; then
            echo "[wait] 显存余量 ${free}MB < ${MIN_FREE_MB}MB ($(date '+%H:%M:%S'))"
            sleep 60; continue
        fi
        return 0
    done
}

total=0; ran=0; skipped=0; fail=0
for ds in $DATASETS; do
  preds="$PREDS"
  [ "$ds" = illness ] && preds="$PREDS_ILLNESS"
  for m in $MODELS; do
    for pl in $preds; do
      cfg="configs/$m/${ds}_${pl}.yaml"
      [ -f "$cfg" ] || { echo "[miss] $cfg 不存在，跳过"; continue; }
      for sd in $SEEDS; do
        total=$((total+1))
        sl=96; [ "$ds" = illness ] && sl=36
        mid="${m}_${ds}_${sl}_${pl}_s${sd}"
        if grep -q ",${mid}_exp_0," "$RESULTS_CSV" 2>/dev/null; then
          skipped=$((skipped+1)); continue
        fi
        wait_for_slot
        echo "[run ] $mid  ($(date '+%F %T'))"
        if "$PYTHON" run.py -c "$cfg" --set model_id="$mid" basic.seed="$sd" \
                system.results="$RESULTS_DIR"; then
          ran=$((ran+1))
        else
          echo "[FAIL] $mid"; fail=$((fail+1))
        fi
      done
    done
  done
  echo "===== 数据集 $ds 完成: 已跑 $ran / 跳过 $skipped / 失败 $fail ====="
done
echo "===== 全部完成: 总格子 $total, 新跑 $ran, 跳过 $skipped, 失败 $fail  ($(date '+%F %T')) ====="
echo BASELINE_GRID_DONE
