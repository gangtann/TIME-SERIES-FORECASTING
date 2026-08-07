# TIME-SERIES-FORECASTING

English | [简体中文](README_zh.md)

A minimal, runnable long-term time series forecasting framework. Currently ships
**PatchTST** and **PRISM** (a research model under development), with dataset
coverage of ETT×4, Electricity, Traffic, Weather, Exchange, Illness, Solar,
PEMS×4, plus any custom CSV with a `date` column.
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

Shipped experiment configs follow the **official paper repo recipes**
(PatchTST: [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)) for all model
and optimization hyperparameters, while keeping this framework's evaluation protocol
(splits / scaling / metrics / `drop_last=False`) and `seq_len` 96 (illness 36), with
`train_epochs` capped at 30. Reference result for ETTh1 96→96 (standardized scale):
MSE ≈ 0.37–0.40. Exchange has no official script (not tested in the paper) and borrows
the ETTh recipe.

Artifacts per run (`<setting>` = `{model_id}_{des}_{itr index}`, e.g. `ETTh1_96_96_exp_0`):

- `checkpoints/<setting>/checkpoint.pth` — best weights on the validation set;
  **written only when `save_checkpoint: true`** (default `false`: the best state is
  kept in memory for the post-training test; enable it if you need the test-only
  mode `is_training: 0` later)
- `checkpoints/<setting>/config.yaml` — snapshot of the full merged config (incl. CLI overrides)
- `checkpoints/<setting>/train.log` — complete copy of terminal output (append mode)
- `results/<setting>/` — `metrics.npy` + forecast plots; the full prediction
  arrays `pred.npy` / `true.npy` are **written only when `save_arrays: true`**
  (default `false` — they can reach several GB per run on large datasets)
- `results/results.csv` — one metrics row appended per test across all experiments

**Reproducing a run**: the config snapshot is itself a valid experiment config —
point `-c` at it to rerun the experiment exactly:

```bash
python run.py -c checkpoints/ETTh1_96_96_exp_0/config.yaml
```

Note: run directory names no longer encode hyperparameters. When sweeping
hyperparameters via `--set`, also change `des` (e.g. `--set d_model=256 des=dm256`)
or `model_id`, otherwise the same-named directory gets overwritten.

## Models

**PatchTST** — faithful reimplementation; shipped configs follow the official
repo recipes (see above).

**PRISM** — research model (frequency-guided adaptive patching + spectral
relational modeling). Pipeline: RevIN → trend/season decomposition; the trend
goes through a linear head, while the season branch computes its amplitude
spectrum, matches it against a learnable spectral pattern dictionary (FAP
router), applies heterogeneous-graph spectral filtering across channels (SRM),
and feeds a **spectral information-clock adaptive patching** backbone: a local
short-time-spectrum affinity yields a per-position information density whose
integral fires patch boundaries — dense/resonant segments get more, finer
patches, with a fixed token budget and a strict uniform-patch degeneration
(`clock_mode: uniform`). All PRISM hyperparameters and ablation switches
(`clock_mode` / `clock_density` / `use_graph` / `use_srm` / `use_fusion`) live
in the `prism` section of `configs/default.yaml`; other models ignore that
section. Experiment configs: `configs/PRISM/*.yaml`; batch runners:
`scripts/PRISM/*.sh`; published seq96 baseline numbers for comparison:
`docs/baselines_seq96.md`.

```bash
python run.py -c configs/PRISM/ETTm1_96.yaml
```

Training-loss switches shared by all models (`training` section):

- `loss: MSE | MAE | huber | smooth_l1` — training loss only; validation,
  early stopping and model selection always use MSE so the protocol stays fixed;
- `auxi_lambda` — weight of a FreDF-style frequency-domain auxiliary loss
  (`|rfft(pred) − rfft(true)|` mean), `0` disables it; `rec_lambda` weights the
  time-domain term when the auxiliary loss is active;
- a model may return `(outputs, aux_loss)` from `forward` to inject internal
  regularizers (e.g. PRISM's routing entropy / clock KL band) into the training
  loss; the tuple is unpacked automatically and ignored at eval time.

## Project Layout

```
run.py                       # entry point
configs/
  default.yaml               # all defaults (sectioned, commented)
  PatchTST/*.yaml            # one config per experiment, only diffs from default
  PRISM/*.yaml
data_provider/
  data_factory.py            # dataset registry + DataLoader construction
  data_loader.py             # Dataset_ETT_hour/_minute/_Custom/_Solar/_PEMS
exp/
  exp_basic.py               # device selection, model registry
  exp_long_term_forecasting.py  # train/val/test loops
layers/                      # PatchEmbedding, Attention, Encoder, RevIN, Decomp, PRISM_blocks
models/
  __init__.py                # MODEL_REGISTRY
  PatchTST.py
  PRISM.py
utils/                       # config / metrics / timefeatures / tools
dataset/                     # data files (see dataset/README.md)
docs/                        # notes, e.g. published seq96 baseline numbers
scripts/                     # batch runners, e.g. scripts/PatchTST/ETTh1.sh runs all four horizons
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
