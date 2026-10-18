# TIME-SERIES-FORECASTING

[English](README.md) | 简体中文

最小可运行的时间序列长期预测框架。内置 **PRISM**（开发中的研究模型）、
**PatchTST** 与十个从官方仓库移植的 baseline（见"模型"一节），数据集覆盖
ETT×4、Electricity、Traffic、Weather、Exchange、Illness、Solar、PEMS×4
以及任意带 `date` 列的自定义 CSV。参考 Time-Series-Library / PatchTST 系项目实现。

## 快速开始

```bash
pip install -r requirements.txt

# ETTh1 96->96（仓库已内置 ETTh1.csv，开箱即跑）
python run.py -c configs/PatchTST/ETTh1_96.yaml

# 先做 1-epoch 快速 sanity check
python run.py -c configs/PatchTST/ETTh1_96.yaml --set train_epochs=1

# 命令行覆盖任意配置项（点号路径或裸键名均可）
python run.py -c configs/PatchTST/ETTh1_96.yaml --set training.learning_rate=1e-3 pred_len=192
```

随附实验配置的**模型与优化超参对齐论文官方仓库**
（PatchTST：[yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)），
评测协议（切分/标准化/指标/`drop_last=False`）与 `seq_len` 96（illness 36）用本框架设定。
轮数不作约束：每个模型在早停下自然收敛（`train_epochs` 默认 100；官方学习率计划按 epoch 数
定义的模型——OneCycle/cosine 类——保留官方轮数，改轮数即改计划）。ETTh1 96→96 参考结果
（标准化尺度）：MSE ≈ 0.37–0.40。
exchange 官方无脚本（论文未测），借用 ETTh 配方。

每次运行的产物（`<setting>` = `{model_id}_{des}_{itr序号}`，如 `ETTh1_96_96_exp_0`）：

- `checkpoints/<setting>/checkpoint.pth` — 验证集最优权重；
  **仅当 `save_checkpoint: true` 时写盘**（默认 `false`：最优权重保留在内存中用于
  训练后测试；若以后需要仅测试模式 `is_training: 0`，请开启此项）
- `checkpoints/<setting>/config.yaml` — 合并后完整配置的快照（含命令行覆盖）
- `checkpoints/<setting>/train.log` — 全部运行日志（终端输出的完整副本，追加模式）
- `results/<setting>/` — `metrics.npy` + 预测对比图；全量预测数组
  `pred.npy` / `true.npy` **仅当 `save_arrays: true` 时写盘**
  （默认 `false`——大数据集上单次运行可达数 GB）
- `results/results.csv` — 所有实验的指标汇总，每次测试追加一行

**复现某次实验**：配置快照本身就是合法的实验配置，直接指给 `-c` 即可原样重跑：

```bash
python run.py -c checkpoints/ETTh1_96_96_exp_0/config.yaml
```

注意：目录名不再编码超参数，用 `--set` 改超参做对比实验时，请同时改 `des`
（如 `--set d_model=256 des=dm256`）或 `model_id`，否则同名目录会被覆盖。

## 模型

**PatchTST** — 忠实复现；随附配置对齐官方仓库配方（见上）。

**移植 baseline** — 十个模型自官方仓库移植，每个为 `models/<X>.py`（+ 自包含的
`layers/<X>_layers.py`），`configs/<X>/` 下按官方最优超参逐格配置，
`scripts/<X>/run_full.sh` 一键批跑。评测协议（切分/标准化/指标/`drop_last=False`/
按验证集 MSE 早停）用本框架的；模型结构与训练配方（损失、LR 调度、耐心）对齐
各官方仓库，轮数不封顶（见上）。模型专属配置键放在 `configs/default.yaml` 带前缀的专属节（如
`xp_*`、`duet_*`）；内部正则项经 `(outputs, aux_loss)` 元组通路进训练损失。
出处（按 `MODEL_REGISTRY` 顺序）：

| 模型 | 会议 | 官方仓库 | 协议备注 |
|---|---|---|---|
| DLinear | AAAI 2023 | cure-lab/LTSF-Linear | 官方配方 seq_len 336，此处按 seq96 跑 |
| DUET | KDD 2025 | decisionintelligence/DUET | unified seq96 配方；MAE 训练损失 |
| PatchMLP | AAAI 2025 | TangPeiwang/PatchMLP | 官方脚本残缺，配方取自 DUET 仓库的 PatchMLP 脚本 |
| PathFormer | ICLR 2024 | decisionintelligence/pathformer | MAE 训练损失；balance 辅助损失 |
| SimpleTM | ICLR 2025 | vsingh-group/SimpleTM | 逐 pred_len 配方；需 `pywt` |
| TimeFilter | ICML 2025 | TROUBADOUR000/TimeFilter | `lradj: cosine2`（官方 lr/2 余弦）；moe 辅助损失 |
| TimeMixer | ICLR 2024 | kwuking/TimeMixer | "unify" seq96 配方 |
| TimeMosaic | AAAI 2026 | BenchCouncil/TimeMosaic | 官方配方 seq_len 320，此处按 seq96 跑 |
| iTransformer | ICLR 2024 | thuml/iTransformer | 完全兼容的移植 |
| xPatch | AAAI 2025 | stitsyuk/xPatch | `loss: arctan_mae` + `lradj: sigmoid`（官方协议）；本框架 `drop_last=False` 对应其论文 "fair" 表 |

额外 pip 依赖：`einops`（DUET、PathFormer）、`PyWavelets`（SimpleTM）。

**PRISM** — 研究模型，主干移植自 FRAME（频域感知的关系自适应混合专家）。
数据流：RevIN → 可学习谱模式原型 `[C, K, F]` → **并行动态分块**：逐通道游标
扫过序列，候选 patch 长度与原型做稀疏谱共振打分（top-k 频点稀疏化 + 高斯平滑
+ 余弦相似度），训练期经 Gumbel-Softmax 硬选择、推理期取 argmax → **谱混合
专家**：K 个时域专家滤波器先经异构图注意力（模式内变量↔变量、模式间模式↔模式）
编码，再做 rFFT 并按共振门控融合为逐通道有效滤波器，依卷积定理在频域相乘，
与滑动平均趋势分支按 `seasonal_alpha` 混合 → patch 散射回 `[B, C, L]` →
MLP 头 → RevIN 反归一化。消融开关集中在 `configs/default.yaml` 的 `prism` 节：
`pattern_strategy`（原型参数化五挡）、`seasonal_alpha`（`1.0` 纯谱分支、
`0.0` 纯趋势分支）、`patch_pool`（单元素即退化为定长 patch）、`freq_topk`、
`pattern_nums`；其他模型不读取该节。实验配置见 `configs/PRISM/*.yaml`；
批量脚本见 `scripts/PRISM/run_full.sh`；无数据集结构自检见
`scripts/PRISM/check_frame.py`；公开 baseline seq96 对照数字见
`docs/baselines_seq96.md`；各数据集周期结构实测（ACF 峰、谱峰、窗内密度不均匀度）
及由此推导的逐数据集候选池建议见 `docs/dataset_periodicity.md`
（复现脚本 `scripts/period_probe.py`）；轮数/早停协议、各模型上限与撞上限 run 的处理见
`docs/training_protocol.md`。

```bash
python run.py -c configs/PRISM/ETTh1_96.yaml
```

所有模型共享的训练损失开关（`training` 节）：

- `loss: MSE | MAE | huber | smooth_l1` — 仅切换训练损失；验证、早停与模型
  选择恒用 MSE，保证评测口径不变；
- `auxi_lambda` — FreDF 式频域辅助损失权重（`|rfft(pred) − rfft(true)|` 均值），
  `0` 为完全关闭；辅助损失开启时 `rec_lambda` 为时域项权重；
- 模型的 `forward` 可返回 `(outputs, aux_loss)` 元组，把内部正则项（如 SimpleTM
  的注意力 L1、TimeFilter 的 MoE 损失）注入训练损失；元组自动解包，eval 阶段忽略。

## 目录结构

```
run.py                       # 入口
configs/
  default.yaml               # 全量默认配置（分节、带注释）
  PatchTST/*.yaml            # 每实验一个配置，只写与 default 的差异
  PRISM/*.yaml
data_provider/
  data_factory.py            # 数据集注册表 + DataLoader 构建
  data_loader.py             # Dataset_ETT_hour/_minute/_Custom/_Solar/_PEMS
exp/
  exp_basic.py               # 设备选择、模型注册表
  exp_long_term_forecasting.py  # 训练/验证/测试循环
layers/                      # 每模型一个自包含的 <X>_layers.py，无跨模型公共层文件
                             # （DLinear 足够简单，无 layers 文件）
models/
  __init__.py                # MODEL_REGISTRY
  PatchTST.py
  PRISM.py
  <X>.py                     # 十个移植 baseline（见"模型"一节）
utils/                       # config / metrics / timefeatures / tools
dataset/                     # 数据文件（见 dataset/README.md）
docs/                        # 笔记文档：公开 baseline seq96 对照数字、数据集周期结构实测、训练协议
scripts/                     # 批量运行脚本，如 scripts/PatchTST/ETTh1.sh 一键跑四个预测长度
```

## 配置系统

- `configs/default.yaml` 定义全部默认值，实验 YAML 按节深合并覆盖；
- `configs/local.yaml`（可选，**不随代码同步**）存放本机特有配置，典型用途是把数据/
  权重目录指到大容量数据盘；合并优先级 `default < 实验 yaml < local < 命令行`；
  模板见 `configs/local.yaml.example`，每台机器 `cp` 一份自行修改，勿跨机器传输；
- 命令行 `--set key=value`（或直接 `key=value`）覆盖单项，值按 YAML 语法解析；
- 合并后扁平化为单层命名空间，代码中以 `configs.d_model` 访问，节名不参与寻址；
- `training.learning_rate=1e-3` 按路径定位，裸 `learning_rate=1e-3` 全节搜索（歧义时报错）。

## 扩展指南

**加模型**：新建 `models/X.py`，暴露 `class Model(nn.Module)`，签名
`forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, c_out]`；
在 `models/__init__.py` 的 `MODEL_REGISTRY` 注册；建 `configs/X/` 放实验配置。
辅助层放自包含的 `layers/X_layers.py`——不进公共文件，宁可重复类
（模块命名空间天然隔离）。

**加数据集**：带 `date` 列的 CSV 直接 `data: custom` 零代码接入；
其他格式在 `data_provider/data_loader.py` 新增 `Dataset_*` 类
（返回 `(seq_x, seq_y, seq_x_mark, seq_y_mark)` 4 元组），
并注册进 `data_factory.py` 的 `data_dict`。

**加任务**：`exp/exp_basic.py` 与任务无关；新增 `Exp_*` 子类，
在 `run.py` 中按 `task_name` 分发。

## 复现说明

依赖采用**宽松声明 + 锁定快照**双层策略：

- `requirements.txt` 只声明需要哪些包（不 pin 版本），保证任何 CUDA 环境都能装上；
- 环境装好并验证通过后，用 `pip freeze > requirements.lock.txt` 记录精确版本快照；
  他人需逐位复现时用 `pip install -r requirements.lock.txt` 还原同一环境。

对结果影响最大的三个环境要素，报告实验时务必注明：**torch 版本、CUDA 版本、GPU 型号**
（`python -c "import torch; print(torch.__version__, torch.version.cuda)"` + `nvidia-smi`）。
numpy/pandas/scikit-learn 的版本差异对指标的影响可忽略。

随机性控制：

- 全局种子默认 `seed: 2021`（random / numpy / torch / CUDA 全部覆盖）；
- GPU 上 cuDNN 默认会做非确定性的算法选择，同种子两次运行指标仍可能在小数点后
  第三、四位波动，这属于正常现象（与文献同口径）；
- 需要逐位一致时设 `--set deterministic=true`（启用 `cudnn.deterministic`，训练稍慢）。

## 实现约定（与 TSLib 对齐）

- StandardScaler 只在 train 切片上 fit，整条序列 transform；
- ETT 用固定 12/4/4 月边界切分，Custom 0.7/0.1/0.2，PEMS 0.6/0.2/0.2；
- 指标默认在标准化数据上计算（与文献口径一致），`inverse: true` 可反标准化；
- 所有 DataLoader `drop_last=False`，test 不 shuffle；
- 训练结束后加载验证集最优 checkpoint 再测试；
- `lradj: TST` 复现 PatchTST 原版 OneCycleLR 逐 step 调度。
