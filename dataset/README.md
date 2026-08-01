# 数据集说明

把数据文件直接放到本目录（`./dataset/`）下，文件名需与下表一致。

| 数据集 | 文件名 | 变量数 (enc_in) | 采样频率 | 加载类 (`data`) |
|---|---|---|---|---|
| ETTh1 / ETTh2 | `ETTh1.csv` / `ETTh2.csv` | 7 | 1 小时 | `ETTh1` / `ETTh2` |
| ETTm1 / ETTm2 | `ETTm1.csv` / `ETTm2.csv` | 7 | 15 分钟 | `ETTm1` / `ETTm2` |
| Electricity | `electricity.csv` | 321 | 1 小时 | `custom` |
| Traffic | `traffic.csv` | 862 | 1 小时 | `custom` |
| Weather | `weather.csv` | 21 | 10 分钟 | `custom` |
| Exchange | `exchange_rate.csv` | 8 | 1 天 | `custom` |
| Illness (ILI) | `national_illness.csv` | 7 | 1 周 | `custom` |
| Solar-Energy | `solar_AL.txt` | 137 | 10 分钟 | `Solar` |
| PEMS03/04/07/08 | `PEMS03.npz` 等 | 358/307/883/170 | 5 分钟 | `PEMS` |

任何带 `date` 列、目标列默认名为 `OT` 的 CSV 都可以直接用 `data: custom` 接入，无需写代码
（目标列名不同时用 `target` 配置指定）。

## 下载来源

- **主数据包（ETT/electricity/traffic/weather/exchange/illness 等）**：
  Time-Series-Library README 提供的整包链接
  [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) /
  [百度云](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)
- **Solar-Energy**：[laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)（`solar-energy/solar_AL.txt.gz` 解压）
- **PEMS03/04/07/08**：随 SCINet/iTransformer 数据包分发（上面 TSLib 整包内也含 PEMS）
- **HuggingFace 自动兜底**：安装 `pip install datasets` 后，CSV 类数据集本地缺失时会自动从
  [`thuml/Time-Series-Library`](https://huggingface.co/datasets/thuml/Time-Series-Library) 下载（Solar/PEMS 不支持）。

本仓库已内置 `ETTh1.csv`（2.5MB）供开箱冒烟测试。
