#!/usr/bin/env bash
# 消融表（3 种子协议；数据集由 DATA 指定，默认 ETTh1）。本模型 run 间相对标准差实测 0.2~5.1%，单种子消融
# 结论不可用，故本脚本一律走 run_etth1_seeds.sh。
#
# 判据：某项消融"有效"当且仅当其掉分幅度显著超过该格的种子标准差。
#
# 用法: bash scripts/PRISM/run_ablation.sh
# 环境变量: BASE(完整模型配方覆盖串)、DATA、PREDS、SEEDS、RESULTS、PYTHON、
#           ONLY(只跑列出的消融项后缀，空=全跑)
# 结果 model_id 形如 PRISM_<DATA>_96_<pl>_abl<项>s<seed>；full 组为 ..._ablNones<seed>
#
# 种子策略：先用单种子扫全表定位量级，再只给"效应落在噪声里"的行补种子。
# 判据要用该格实测 σ：weather 相对 σ 实测 0.11%~0.93%（ETTh1 曾达 5.1%），
# 单种子只能判定效应 ≳3% 的行，1~2% 的行必须补种子才能下结论。
# 注意 ablNone 与主配方 (token_mix=hgnn, token_mix_layers=1) 逐位相同——weather 上
# 已验证 w_on 与 ablNone 同种子 mse/mae 完全一致——已有主配方多种子结果时可用 ONLY 跳过它。
cd "$(dirname "$0")/../.."

# 完整模型配方 = configs/default.yaml 的现行默认值（已全部改为修复版：tokenize=direct、
# gate_scope=token、patch_select=st、patch_mode=uniform、use_hgnn=false、token_mix=hgnn 等），
# 故此处只需给训练配方，不必再逐键覆盖。
BASE=${BASE:-""}   # 训练配方（无轮数上限/type3/耐心 5）已写入 configs/PRISM/*.yaml

# 消融项：后缀|覆盖。①~⑥ 对应 FRAME 的六条主张，⑦ 是本框架的修复项。
ABL=(
  "None|"                                   # full：与各消融项同配方的基准组
  "Freq|prism.sra_sparse=false"             # ① 频域稀疏匹配（top-k + 高斯平滑）
  "Rel|prism.token_mix=none"                # ② 关系建模：token 图混合（同通道跨时间 + 同时刻跨通道）
                                            #    官方把异构图用在专家滤波器上，那里它只是自由参数的
                                            #    重参数化、消融恒为零；移到 token 序列后才进入关键路径
  "MoE|prism.use_moe_gate=false"            # ③ 混合专家门控（退化为均匀 1/K）
  "Adp|prism.patch_mode=greedy_fixed prism.adp_regrid=false"   # ④ 自适应边界（原版定义：零填充、无重栅格化）：
                                            #    full 挡用等分，本行把官方的自适应"加回来"
                                            #    （已实测为负贡献，故 full 挡不含它；此行报告其代价）
  "NonU|prism.patch_mode=fixed_random prism.adp_regrid=false"  # ④b 仅"段长不等"、边界仍恒定，用于拆分④的代价来源
  "Adp2|prism.patch_mode=greedy_v2"         # ④v2 自适应 v2（重采样 + 可训练路由 + 重栅格化，见 layers 文件头）
  "Seas|prism.seasonal_alpha=0.0"           # ⑤ 谱分支（只留 patch 轴趋势）
  "Proto|prism.pattern_strategy=2"          # ⑥ 原型参数化（共享随机→逐通道随机）
  "Route|prism.gate_scope=channel"          # ⑦ 逐样本路由（退化为官方的静态逐通道门控）
)

ONLY=${ONLY:-}

fail=0
for e in "${ABL[@]}"; do
  suf="${e%%|*}"; ov="${e#*|}"
  if [ -n "$ONLY" ] && ! grep -qw "$suf" <<< "$ONLY"; then
    echo "[skip row] abl${suf}"; continue
  fi
  TAG="abl${suf}" REC="$BASE $ov" bash scripts/PRISM/run_seeds.sh || fail=$((fail+1))
done
echo "===== ablation finished, failing batches: $fail  ($(date '+%F %T')) ====="
echo ABLATION_DONE
