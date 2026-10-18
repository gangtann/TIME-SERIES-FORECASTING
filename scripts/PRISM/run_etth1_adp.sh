#!/usr/bin/env bash
# 自适应 v2 判决批（ETTh1，{96,336} × 3 种子）。每阶段一个可证伪判据：
#   阶段 0 边界无害化：nonuV2（fixed_random + 重采样 + 重栅格化）相对 uniform 的代价须从 +2.7%/+0.5%
#                     降到 ≤ 1σ（σ 实测 0.13~0.7%）——证明重采样 + 重栅格化消掉了"段长不等"与"槽位失配"。
#   阶段 1 自适应有益：adpV2 相对 uniform 至少一格显著变好（> 2σ）且另一格不退步；
#                     adpV2noReg 拆熵正则；adpV2noGrid 拆重栅格化；测试末尾打印 diag adp_len_hist（按槽位）。
# 基线 uniform 走主配方，已有 aux(5 种子) 结果：96 0.3773 / 336 0.4665（aux_stream=trend）。
# 本批全部叠 aux_stream=trend，与该基线同配方。
# 用法: PYTHON=... bash scripts/PRISM/run_etth1_adp.sh
cd "$(dirname "$0")/../.."
export DATA=ETTh1 PREDS="${PREDS:-96 336}" SEEDS="${SEEDS:-2021 2022 2023}"
R="prism.aux_stream=trend"   # 训练配方（无轮数上限/type3/耐心 5）已写入 configs/PRISM/*.yaml
fail=0
run () { TAG=$1 REC="$R $2" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1)); }
run nonuV2      "prism.patch_mode=fixed_random prism.patch_norm=resample"
run adpV2       "prism.patch_mode=greedy_v2 prism.patch_norm=resample"
run adpV2noReg  "prism.patch_mode=greedy_v2 prism.patch_norm=resample prism.adp_lambda=0"
run adpV2noGrid "prism.patch_mode=greedy_v2 prism.patch_norm=resample prism.adp_regrid=false"
echo "===== etth1 adp battery finished, failing batches: $fail  ($(date '+%F %T')) ====="
echo ADP_BATTERY_DONE
