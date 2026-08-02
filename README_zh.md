# TIME-SERIES-FORECASTING

[English](README.md) | 简体中文

最小可运行的时间序列长期预测框架。当前支持 **PatchTST** 模型，数据集覆盖
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

ETTh1 96→96 参考结果（标准化尺度）：MSE ≈ 0.37–0.42，MAE ≈ 0.39–0.42。

每次运行的产物（`<setting>` = `{model_id}_{des}_{itr序号}`，如 `ETTh1_96_96_exp_0`）：

- `checkpoints/<setting>/checkpoint.pth` — 验证集最优权重
- `checkpoints/<setting>/config.yaml` — 合并后完整配置的快照（含命令行覆盖）
- `checkpoints/<setting>/train.log` — 全部运行日志（终端输出的完整副本，追加模式）
- `results/<setting>/` — `pred.npy` / `true.npy` / `metrics.npy` + 预测对比图
- `results/results.csv` — 所有实验的指标汇总，每次测试追加一行

**复现某次实验**：配置快照本身就是合法的实验配置，直接指给 `-c` 即可原样重跑：

```bash
python run.py -c checkpoints/ETTh1_96_96_exp_0/config.yaml
```

注意：目录名不再编码超参数，用 `--set` 改超参做对比实验时，请同时改 `des`
（如 `--set d_model=256 des=dm256`）或 `model_id`，否则同名目录会被覆盖。

## 目录结构

```
run.py                       # 入口
configs/
  default.yaml               # 全量默认配置（分节、带注释）
  PatchTST/*.yaml            # 每实验一个配置，只写与 default 的差异
data_provider/
  data_factory.py            # 数据集注册表 + DataLoader 构建
  data_loader.py             # Dataset_ETT_hour/_minute/_Custom/_Solar/_PEMS
exp/
  exp_basic.py               # 设备选择、模型注册表
  exp_long_term_forecasting.py  # 训练/验证/测试循环
layers/                      # PatchEmbedding、Attention、Encoder
models/
  __init__.py                # MODEL_REGISTRY
  PatchTST.py
utils/                       # config / metrics / timefeatures / tools
dataset/                     # 数据文件（见 dataset/README.md）
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
