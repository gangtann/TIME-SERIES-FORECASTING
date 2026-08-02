"""YAML config loading: default.yaml + experiment yaml deep-merge + CLI overrides.

This module must stay torch-free so config logic can be tested on machines
without a deep-learning environment.
"""
import argparse
import copy
import os

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')
DEFAULT_CONFIG = os.path.join(_CONFIG_DIR, 'default.yaml')
# 机器级配置（可选，不随代码同步）：存放本机特有的路径等，如数据盘位置。
# 合并优先级：default.yaml < 实验 yaml < local.yaml < 命令行 --set
LOCAL_CONFIG = os.path.join(_CONFIG_DIR, 'local.yaml')


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_value(s):
    """'1e-3' -> float, 'true' -> bool, '[24,48]' -> list, plain text -> str."""
    try:
        v = yaml.safe_load(s)
    except yaml.YAMLError:
        return s
    # PyYAML 把不带小数点的科学计数法（如 '1e-3'）当字符串，这里补救
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def set_override(cfg, key, value):
    """Apply one override. 'training.learning_rate' addresses a section directly;
    a bare 'learning_rate' is located by searching all sections for a unique leaf."""
    parts = key.split('.')
    if len(parts) > 1:
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"config path '{key}' not found")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"config path '{key}' not found")
        node[parts[-1]] = value
        return

    hits = [sec for sec, body in cfg.items()
            if isinstance(body, dict) and key in body]
    if not hits:
        raise KeyError(f"config key '{key}' not found in any section")
    if len(hits) > 1:
        raise KeyError(f"config key '{key}' is ambiguous (in sections {hits}); "
                       f"use a dotted path like '{hits[0]}.{key}'")
    cfg[hits[0]][key] = value


def flatten_sections(cfg):
    flat = {}
    for sec, body in cfg.items():
        if not isinstance(body, dict):
            raise ValueError(f"top-level key '{sec}' must be a section (mapping)")
        for k, v in body.items():
            if k in flat:
                raise ValueError(f"duplicate config key '{k}' across sections")
            flat[k] = v
    return flat


class Config:
    def __init__(self, d, sections=None):
        self.__dict__.update(d)
        # 分节形式的完整配置（default + 实验 yaml + local + 命令行覆盖后），
        # 用于保存可直接重跑的快照；下划线开头，不参与属性访问语义
        self._sections = sections or {}

    def save(self, path):
        """把合并后的完整配置存为快照，可用 run.py -c <path> 原样重跑。"""
        with open(path, 'w') as f:
            yaml.safe_dump(self._sections, f, sort_keys=False, allow_unicode=True)

    def __repr__(self):
        items = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        width = max(len(k) for k in items)
        lines = ['{:<{w}}  {}'.format(k, v, w=width)
                 for k, v in sorted(items.items())]
        bar = '-' * 60
        return '\n'.join([bar, 'Experiment Configuration', bar] + lines + [bar])


def load_config(argv=None):
    parser = argparse.ArgumentParser(
        description='Time series long-term forecasting',
        epilog="Override any config key with --set key=value, e.g. "
               "--set training.learning_rate=1e-3 pred_len=192")
    parser.add_argument('-c', '--config', required=True, help='experiment yaml path')
    parser.add_argument('--set', nargs='*', default=[], dest='overrides',
                        metavar='KEY=VALUE', help='config overrides')
    args, extra = parser.parse_known_args(argv)

    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    with open(args.config) as f:
        exp_cfg = yaml.safe_load(f) or {}
    cfg = deep_merge(cfg, exp_cfg)

    if os.path.exists(LOCAL_CONFIG):
        with open(LOCAL_CONFIG) as f:
            local_cfg = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, local_cfg)
        print(f'Applied machine-local config: {LOCAL_CONFIG}')

    overrides = args.overrides + [e for e in extra if '=' in e]
    bad = [e for e in extra if '=' not in e]
    if bad:
        raise SystemExit(f"unrecognized arguments: {bad}")
    for item in overrides:
        key, _, val = item.partition('=')
        set_override(cfg, key, parse_value(val))

    return Config(flatten_sections(cfg), sections=cfg)
