"""最终对照表：按数据集 × 视野对所有模型排名（只比数值）。

公平口径（2026-09-10）：
- 只统计带种子后缀的行（setting 形如 <Model>_<data>_96_<pl>_s<seed>_exp_0，PRISM 为 ..._<tag>s<seed>_exp_0），
  且种子限定为 --seeds（默认 2021–2025）；无种子后缀的老 run 一律不计；
- baseline csv 里的 PRISM 行是旧模型，排除；PRISM 只从 --prism csv 读，并只取 --prism-tag 指定的配方；
- 同一 setting 出现多次时取时间戳最新的一条（重跑替换旧行时先删旧行，这里只是兜底）；
- 每格给出各模型 n 种子均值 MSE 与均值最优验证损失（best_vali 列，老行为空）。

用法:
  python scripts/rank_results.py --baseline bl1.csv bl2.csv --prism results_frame.csv --prism-tag aux [--data ETTh1] [--seeds 2021 2022 2023 2024 2025]
"""
import argparse
import collections
import csv
import re

BL_RE = re.compile(r'^(?P<model>[A-Za-z]+)_(?P<data>[A-Za-z0-9]+)_(?:96|36)_(?P<pl>\d+)_s(?P<seed>\d{4})_exp_0$')  # illness 窗口 36
PR_RE = re.compile(r'^PRISM_(?P<data>[A-Za-z0-9]+)_(?:96|36)_(?P<pl>\d+)_(?P<tag>.+?)s(?P<seed>\d{4})_exp_0$')


def load(paths):
    rows = {}
    for p in paths:
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                if not r.get('mse'):
                    continue
                key = r['setting']
                if key not in rows or r['timestamp'] > rows[key]['timestamp']:
                    rows[key] = r
    return rows.values()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', nargs='+', required=True)
    ap.add_argument('--prism', nargs='*', default=[])
    ap.add_argument('--prism-tag', default='aux')
    ap.add_argument('--data', default=None)
    ap.add_argument('--seeds', nargs='+', default=['2021', '2022', '2023', '2024', '2025'])
    ap.add_argument('--top', type=int, default=12)
    args = ap.parse_args()
    seeds = set(args.seeds)

    cells = collections.defaultdict(lambda: collections.defaultdict(dict))  # (data,pl) -> model -> seed -> (mse, vali)
    for r in load(args.baseline):
        m = BL_RE.match(r['setting'])
        if not m or m['model'] == 'PRISM' or m['seed'] not in seeds:
            continue
        cells[(m['data'], int(m['pl']))][m['model']][m['seed']] = (float(r['mse']), r.get('best_vali') or None)
    for r in load(args.prism):
        m = PR_RE.match(r['setting'])
        if not m or m['tag'] != args.prism_tag or m['seed'] not in seeds:
            continue
        cells[(m['data'], int(m['pl']))][f'PRISM[{args.prism_tag}]'][m['seed']] = (float(r['mse']), r.get('best_vali') or None)

    for (data, pl) in sorted(cells, key=lambda k: (k[0], k[1])):
        if args.data and data != args.data:
            continue
        print(f'--- {data} pred_len={pl} (seeds {sorted(seeds)}) ---')
        table = []
        for model, per_seed in cells[(data, pl)].items():
            mses = [v[0] for v in per_seed.values()]
            valis = [float(v[1]) for v in per_seed.values() if v[1]]
            table.append((sum(mses) / len(mses), len(mses),
                          sum(valis) / len(valis) if valis else None, model))
        for i, (mse, n, vali, model) in enumerate(sorted(table)[:args.top], 1):
            vs = f'{vali:.4f}' if vali is not None else '   -  '
            print(f'{i:2d}. {model:16s} mse {mse:.4f}  (n={n}, vali {vs})')


if __name__ == '__main__':
    main()
