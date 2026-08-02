# TIME-SERIES-FORECASTING

English | [简体中文](README_zh.md)

A minimal, runnable long-term time series forecasting framework. Currently ships
**PatchTST**, with dataset coverage of ETT×4, Electricity, Traffic, Weather,
Exchange, Illness, Solar, PEMS×4, plus any custom CSV with a `date` column.
Implementation references Time-Series-Library / PatchTST-family projects.

## Quick Start

```bash
pip install -r requirements.txt

# ETTh1 96->96 (ETTh1.csv is bundled, works out of the box)
python run.py -c configs/PatchTST/ETTh1_96.yaml

# 1-epoch quick sanity check
python run.py -c configs/PatchTST/ETTh1_96.yaml --set train_epochs=1

# Override any config key from the CLI (dotted path or bare key)
python run.py -c configs/PatchTST/ETTh1_96.yaml --set training.learning_rate=1e-3 pred_len=192
```

Reference result for ETTh1 96→96 (standardized scale): MSE ≈ 0.37–0.42, MAE ≈ 0.39–0.42.

Artifacts per run (`<setting>` = `{model_id}_{des}_{itr index}`, e.g. `ETTh1_96_96_exp_0`):

- `checkpoints/<setting>/checkpoint.pth` — best weights on the validation set
- `checkpoints/<setting>/config.yaml` — snapshot of the full merged config (incl. CLI overrides)
- `checkpoints/<setting>/train.log` — complete copy of terminal output (append mode)
- `results/<setting>/` — `pred.npy` / `true.npy` / `metrics.npy` + forecast plots
- `results/results.csv` — one metrics row appended per test across all experiments

**Reproducing a run**: the config snapshot is itself a valid experiment config —
point `-c` at it to rerun the experiment exactly:

```bash
python run.py -c checkpoints/ETTh1_96_96_exp_0/config.yaml
```

Note: run directory names no longer encode hyperparameters. When sweeping
hyperparameters via `--set`, also change `des` (e.g. `--set d_model=256 des=dm256`)
or `model_id`, otherwise the same-named directory gets overwritten.

## Project Layout

```
run.py                       # entry point
configs/
  default.yaml               # all defaults (sectioned, commented)
  PatchTST/*.yaml            # one config per experiment, only diffs from default
data_provider/
  data_factory.py            # dataset registry + DataLoader construction
  data_loader.py             # Dataset_ETT_hour/_minute/_Custom/_Solar/_PEMS
exp/
  exp_basic.py               # device selection, model registry
  exp_long_term_forecasting.py  # train/val/test loops
layers/                      # PatchEmbedding, Attention, Encoder
models/
  __init__.py                # MODEL_REGISTRY
  PatchTST.py
utils/                       # config / metrics / timefeatures / tools
dataset/                     # data files (see dataset/README.md)
```

## Config System

- `configs/default.yaml` defines every default; experiment YAMLs override it via
  sectioned deep-merge;
- `configs/local.yaml` (optional, **never synced across machines**) holds
  machine-specific settings — typically pointing data/checkpoint dirs to a large
  data disk; merge precedence: `default < experiment yaml < local < CLI`;
  see the template `configs/local.yaml.example`: `cp` a copy per machine and edit;
- CLI `--set key=value` (or bare `key=value`) overrides single keys, values parsed
  as YAML;
- after merging, everything is flattened into a single namespace consumed as
  `configs.d_model`; section names take no part in addressing;
- `training.learning_rate=1e-3` addresses by path; bare `learning_rate=1e-3`
  searches all sections (errors on ambiguity).

## Extension Guide

**Add a model**: create `models/X.py` exposing `class Model(nn.Module)` with
signature `forward(x_enc, x_mark_enc, x_dec, x_mark_dec) -> [B, pred_len, c_out]`;
register it in `MODEL_REGISTRY` in `models/__init__.py`; add configs under `configs/X/`.

**Add a dataset**: a CSV with a `date` column works with `data: custom` and zero
code; for other formats add a `Dataset_*` class in `data_provider/data_loader.py`
(returning the `(seq_x, seq_y, seq_x_mark, seq_y_mark)` 4-tuple) and register it
in `data_dict` in `data_factory.py`.

**Add a task**: `exp/exp_basic.py` is task-agnostic; add an `Exp_*` subclass and
dispatch on `task_name` in `run.py`.

## Reproducibility

Dependencies use a **loose declaration + locked snapshot** two-layer strategy:

- `requirements.txt` only declares which packages are needed (no pins), so it
  installs in any CUDA environment;
- after an environment is set up and verified, `pip freeze > requirements.lock.txt`
  records the exact version snapshot; reproduce bit-for-bit with
  `pip install -r requirements.lock.txt`.

The three environment factors that matter most for results — always report:
**torch version, CUDA version, GPU model**
(`python -c "import torch; print(torch.__version__, torch.version.cuda)"` + `nvidia-smi`).
Version differences in numpy/pandas/scikit-learn have negligible effect on metrics.

Randomness control:

- global seed defaults to `seed: 2021` (covers random / numpy / torch / CUDA);
- on GPU, cuDNN picks algorithms non-deterministically by default, so two runs
  with the same seed may still differ in the 3rd–4th decimal place — this is
  normal and matches the literature;
- for bit-exact runs set `--set deterministic=true` (enables
  `cudnn.deterministic`, slightly slower).

## Implementation Conventions (aligned with TSLib)

- StandardScaler is fit on the train slice only, then transforms the whole series;
- ETT splits use the fixed 12/4/4-month borders; Custom 0.7/0.1/0.2; PEMS 0.6/0.2/0.2;
- metrics are computed on standardized data by default (same protocol as the
  literature); `inverse: true` de-standardizes first;
- all DataLoaders use `drop_last=False`; test is never shuffled;
- after training, the best validation checkpoint is reloaded before testing;
- `lradj: TST` reproduces the original PatchTST OneCycleLR per-step schedule.
