"""合并多台机器 / 多分片的 results.csv。

背景：exp/exp_long_term_forecasting.py 以无锁 append 写 csv，所以并行实验一律
各写各的文件，事后用本脚本合并。按 setting 列去重（同一格子只保留一条），
并追加 source 列记录该行来自哪个文件，供论文的数据公开使用。同一 setting 多次出现时保留时间戳最新的一条。

用法:
    python scripts/merge_results.py -o merged.csv a/results.csv b/results.csv ...
    python scripts/merge_results.py -o merged.csv --label autodl2:sh_h1 h1/results.csv
"""
import argparse
import csv
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--label', action='append', default=[],
                    help='与 inputs 一一对应的来源标签；缺省用文件路径')
    ap.add_argument('--only-seeded', action='store_true',
                    help='只保留带种子后缀的行（..._s<seed>_exp_0 / ...<tag>s<seed>_exp_0），去掉无种子的老 run')
    ap.add_argument('--exclude-model', action='append', default=[],
                    help='按 model 列排除（如 baseline csv 里的旧 PRISM 行）')
    args = ap.parse_args()
    seeded = re.compile(r's\d{4}_exp_0$')

    labels = args.label or []
    rows, seen, header = {}, [], None
    dup = 0
    for i, path in enumerate(args.inputs):
        src = labels[i] if i < len(labels) else path
        if not os.path.exists(path):
            print(f'[miss] {path}')
            continue
        with open(path, newline='') as f:
            rd = csv.DictReader(f)
            header = header or rd.fieldnames
            for r in rd:
                key = r['setting']
                if r.get('model') in args.exclude_model or (args.only_seeded and not seeded.search(key)):
                    continue
                if key in rows:
                    dup += 1
                    # 同一格重跑替换（delete-after-success 尚未清到仍在写入的分片 csv 时会撞上）：取时间戳最新的
                    if r.get('timestamp', '') <= rows[key].get('timestamp', ''):
                        continue
                    seen.remove(key)
                r['source'] = src
                rows[key] = r
                seen.append(key)

    with open(args.out, 'w', newline='') as f:
        # 各文件表头可能不同（best_vali 列是后加的）：取并集，缺的字段留空
        fields = list(dict.fromkeys(k for r in rows.values() for k in r.keys() if k is not None))
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        for k in sorted(seen):
            wr.writerow(rows[k])
    print(f'合并 {len(args.inputs)} 个文件 -> {args.out}: {len(seen)} 行，去重丢弃 {dup} 行')


if __name__ == '__main__':
    main()
