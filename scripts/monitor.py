"""两台 GPU 服务器的实验进度看板（本机运行，只用标准库）。

设计：后台线程每 POLL 秒各发一次 ssh，把进程 / 显卡 / 结果 csv / 日志尾巴一次性
取回（每台只一次往返），解析后缓存；浏览器轮询 /api/status 读缓存，页面自身不触发 ssh。

用法:
    python3 scripts/monitor.py            # 默认 http://127.0.0.1:8765
    PORT=9000 POLL=20 python3 scripts/monitor.py

停止: kill 掉本进程即可；服务器上的实验不受影响。
"""
import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get('PORT', 8765))
POLL = int(os.environ.get('POLL', 30))

# ---- 两台机器的分工（彻底分开：autodl 只跑主线，autodl2 只跑 baseline）----
SHARDS = ['h1', 'h2', 'wf', 'ws', 'm1', 'm2', 'm1a', 'm1b', 'm1c', 'm1d', 'h2r', 'h1x', 'wsx',
          'rrA', 'rrB', 'ndA', 'ndB', 'ndC', 'nds1', 'nds2', 'ndl']  # 15:27 起 m1 拆四片并行；rrA/rrB = 2026-09-11 失败格补跑；ndA–ndC = 四个新数据集首轮三路；nds1/nds2 = 小数据集收尾，ndl = 大数据集单路
HOSTS = {
    'autodl': {
        'role': '主线 PRISM',
        'csv': {'main': '/root/autodl-tmp/results_frame/results.csv',
                'baseline': '/root/autodl-tmp/results/results.csv'},
        # 主线换批次会换日志文件名，取最新的那个，免得看板一直读着上一批的旧日志
        'logs': {'mainline': '$(ls -t /tmp/mainline*.log 2>/dev/null | head -1)'},
    },
    'autodl2': {
        'role': 'baseline 全网格',
        'csv': {f'sh_{s}': f'/root/autodl-tmp/results_bl/{s}/results.csv' for s in SHARDS},
        'logs': {f'sh_{s}': f'/root/autodl-tmp/logs/bl_{s}.log' for s in SHARDS},
    },
}

# ---- baseline 网格定义（与 scripts/run_baseline_grid.sh 保持一致）----
BL_MODELS = ['DLinear', 'TimeMixer', 'iTransformer', 'xPatch', 'PatchTST', 'TimeFilter',
             'SimpleTM', 'TimeMosaic', 'PatchMLP', 'DUET', 'PathFormer']
BL_DATASETS = ['ETTh1', 'ETTh2', 'weather', 'ETTm1', 'ETTm2', 'illness', 'exchange', 'electricity', 'traffic']
# 已定案不补跑的格不算剩余（PathFormer weather 官方实现数值溢出，按 15 种子计，见 docs/training_protocol.md）
BL_WAIVED = {('PathFormer', 'weather')}
# 暂缓的格（等全部 baseline 跑完再定梯度累积方案）单列，不进 ETA
BL_DEFERRED = {('TimeMosaic', 'electricity'), ('TimeMosaic', 'traffic')}
# 2026-09-14 用户决定：electricity/traffic 此后只要求种子 2021（DLinear 仍 5 种子），其余种子后续再补；
# 已多跑的种子保留在 csv 里，但不计入进度。PRISM 正式成绩同样按此计。
BL_ONE_SEED = {'electricity', 'traffic'}


def target_seeds(model, ds):
    return [2021] if ds in BL_ONE_SEED and model != 'DLinear' else SEEDS
# 分片 → 看板上的中文说明、数据集、模型（与 autodl2 上 /root/newds3.sh 保持一致；未列出的分片只显示名字）
LANES = {
    'nds1': ('小数据集 A', ['illness', 'exchange'],
             ['DLinear', 'TimeMixer', 'iTransformer', 'xPatch', 'PatchTST', 'TimeFilter']),
    'nds2': ('小数据集 B', ['illness', 'exchange'],
             ['SimpleTM', 'TimeMosaic', 'PatchMLP', 'DUET', 'PathFormer']),
    'ndl': ('大数据集（单种子，独占显卡）', ['electricity', 'traffic'],
            ['DLinear', 'TimeMixer', 'iTransformer', 'xPatch', 'PatchTST', 'TimeFilter',
             'SimpleTM', 'PatchMLP', 'DUET', 'PathFormer']),
}
# illness 走领域惯例协议：窗口 36、视野 24/36/48/60，结果名为 <Model>_illness_36_<pl>_s<seed>
BL_PREDS = {'illness': [24, 36, 48, 60]}
BL_SEQ = {'illness': 36}
PREDS = [96, 192, 336, 720]
SEEDS = [2021, 2022, 2023, 2024, 2025]

# ---- 主线 weather 战役（run_ablation.sh 的 9 项 + 主配方两挡）----
# 消融表按 5 种子协议：每格取已完成种子的均值比较，并标出相对 full 基准的代价。
# 单种子曾试过又改回——weather 上种子 2021 在 96/336/720 三格都偏差，且 1~2% 量级的
# 效应会被种子抖动盖住（关系建模 336/720 格单种子给出的符号与 5 种子均值相反）。
# ablNone/ablRel 不单独跑：它们与主配方 w_on/w_off 的覆盖串逐位相同（weather 上已验证
# 同种子 mse/mae 完全一致），直接复用。
MAIN_DATA = 'weather'          # 消融表所在数据集
NEED_SEEDS = 5
# PRISM 主表：全部数据集 × 4 视野 × 5 种子。weather 复用 w_on 组（覆盖串与 main 逐位相同）。
MAIN_TABLE = [('ETTh1', [96,192,336,720]), ('ETTh2', [96,192,336,720]),
              ('exchange', [96,192,336,720]), ('illness', [24,36,48,60]),
              ('weather', [96,192,336,720]), ('ETTm1', [96,192,336,720]),
              ('ETTm2', [96,192,336,720]), ('electricity', [96,192,336,720]),
              ('traffic', [96,192,336,720])]
MAIN_TAGS = [('w_on', '主配方 = 消融基准 full'),
             ('w_off', '② 关系建模（= 主配方关掉）'),
             ('ablFreq', '① 频域稀疏匹配'),
             ('ablMoE', '③ 专家门控'),
             ('ablAdp', '④ 自适应边界'),
             ('ablNonU', '④b 仅段长不等'),
             ('ablSeas', '⑤ 谱分支'),
             ('ablProto', '⑥ 原型参数化'),
             ('ablRoute', '⑦ 逐样本路由')]

# ETTh1 判决批（3 种子 × {96,336}）：自适应 v2 各配方相对 aux 基线（5 种子）的相对代价
JUDGE_DATA, JUDGE_BASE, JUDGE_PREDS, JUDGE_SEEDS = 'ETTh1', 'aux', [96, 336], 3
JUDGE_TAGS = [('ntok4', '固定对照 4×24（日周期对齐）'),
              ('ntok3', '固定对照 3×32'),
              ('ntok8', '固定对照 8×12'),
              ('nonuV2', '阶段0 段长不等+重采样+重栅格化（须 ≤1σ）'),
              ('adpV2', '阶段1 自适应 v2 全量（旧池 N=6）'),
              ('adpV2p', '自适应 v2 + 周期池 {12,24,36,48} N=4 T0=24'),
              ('oracle', 'oracle-EM 首轮（M 步按 oracle 标签训练；训练/测试划分不一致）'),
              ('oracleR', 'oracle-EM v2（M 步按路由器划分训练；路由器初始化未隔离）'),
              ('oracleR2', 'oracle-EM v2（路由器随机数隔离；交叉熵仍经 RevIN 泄漏）'),
              ('oracleR3', 'oracle-EM v2（+路由器输入 detach、全等分走原生路径：须与 ntok4 逐种子相同）'),
              ('adpV2noReg', '阶段1 拆熵正则'),
              ('adpV2noGrid', '阶段1 拆重栅格化')]

PROBE = r'''
set -u
echo "###NOW"; date '+%F %T'
echo "###PROC"; ps -eo pid,ppid,args --no-headers | grep "[r]un[.]py -c" || true
echo "###GPU"; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null || true
echo "###LOAD"; cat /proc/loadavg
__CSV__
__LOG__
'''

SETTING_RE = re.compile(r'^PRISM_(?P<ds>[A-Za-z0-9]+)_96_(?P<pl>\d+)_(?P<tag>.+?)s(?P<seed>\d{4})_exp_0$')

_state = {'ts': 0, 'hosts': {}, 'error': {}}
_lock = threading.Lock()


def probe(host, spec):
    csv_part = '\n'.join(
        f'echo "###CSV {k}"; [ -f {p} ] && tail -n +2 {p} | cut -d, -f1,2,7,8 || true'
        for k, p in spec['csv'].items())
    log_part = '\n'.join(
        f'echo "###CUR {k}"; grep "^\\[run \\]" {p} 2>/dev/null | tail -1;'
        f' echo "###LOG {k}"; tail -n 3 {p} 2>/dev/null || true;'
        f' echo "###CNT {k}"; grep -c "^\\[run \\]" {p} 2>/dev/null || echo 0;'
        f' grep -c "^\\[FAIL\\]" {p} 2>/dev/null || echo 0'
        for k, p in spec['logs'].items())
    out = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host,
         PROBE.replace('__CSV__', csv_part).replace('__LOG__', log_part)],
        capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:300] or f'ssh exit {out.returncode}')
    return parse(out.stdout)


def parse(text):
    sec, buf = None, {}
    for line in text.splitlines():
        if line.startswith('###'):
            sec = line[3:].strip()
            buf.setdefault(sec, [])
            continue
        if sec:
            buf[sec].append(line)

    d = {'now': (buf.get('NOW') or [''])[0], 'gpu': None, 'load': None,
         'procs': [], 'rows': {}, 'logs': {}}

    g = (buf.get('GPU') or [''])[0].split(',')
    if len(g) == 3:
        d['gpu'] = {'used': int(g[0]), 'total': int(g[1]), 'util': int(g[2])}
    la = (buf.get('LOAD') or [''])[0].split()
    if la:
        d['load'] = float(la[0])

    # 只留顶层 run.py（DataLoader worker 是 fork 出来的，argv 与父进程相同）
    procs = []
    for line in buf.get('PROC', []):
        parts = line.split(None, 2)
        if len(parts) == 3:
            procs.append((parts[0], parts[1], parts[2]))
    pids = {p[0] for p in procs}
    for pid, ppid, args in procs:
        if ppid in pids:
            continue
        cfg = re.search(r'-c (\S+)', args)
        mid = re.search(r'model_id=(\S+)', args)
        d['procs'].append({'cfg': cfg.group(1) if cfg else '?',
                           'model_id': mid.group(1) if mid else '?'})

    for k, v in buf.items():
        if k.startswith('CSV '):
            rows = []
            for line in v:
                f = line.split(',')
                if len(f) >= 4:
                    rows.append({'ts': f[0], 'setting': f[1], 'mse': f[2], 'mae': f[3]})
            d['rows'][k[4:]] = rows
        elif k.startswith('LOG '):
            d['logs'].setdefault(k[4:], {})['tail'] = [x for x in v if x.strip()][-2:]
        elif k.startswith('CUR '):
            m = re.match(r'\[run \] (\S+)\s+\((.+)\)', v[-1]) if v else None
            if m:
                d['logs'].setdefault(k[4:], {}).update(cur_id=m.group(1), cur_ts=m.group(2))
        elif k.startswith('CNT '):
            nums = [x for x in v if x.strip().isdigit()]
            d['logs'].setdefault(k[4:], {})['run'] = int(nums[0]) if nums else 0
            d['logs'].setdefault(k[4:], {})['fail'] = int(nums[1]) if len(nums) > 1 else 0
    return d


def poller():
    while True:
        hosts, errs = {}, {}
        for h, spec in HOSTS.items():
            try:
                hosts[h] = probe(h, spec)
            except Exception as e:  # 网络抖动不该让看板整体挂掉
                errs[h] = str(e)[:300]
        with _lock:
            _state['hosts'] = hosts
            _state['error'] = errs
            _state['ts'] = time.time()
        time.sleep(POLL)


def has_cfg(model, ds, pl):
    return os.path.exists(os.path.join(ROOT, 'configs', model, f'{ds}_{pl}.yaml'))


def build_view():
    with _lock:
        st = json.loads(json.dumps(_state))  # 快照，避免持锁做重活

    # ---- baseline：合并两台机器所有 csv，按 setting 去重 ----
    # 各分片 csv 都以同一份主表播种，故按 setting 去重后再统计时间戳，否则吞吐会虚高
    done, stamp_of = set(), {}
    for h, d in st['hosts'].items():
        for _, rows in d['rows'].items():
            for r in rows:
                done.add(r['setting'])
                stamp_of.setdefault(r['setting'], r['ts'])
    stamps = list(stamp_of.values())

    grid, bl_total, bl_done = [], 0, 0
    for m in BL_MODELS:
        row = {'model': m, 'cells': []}
        for ds in BL_DATASETS:
            tot = n = 0
            sl = BL_SEQ.get(ds, 96)
            # 5 种子已全部跑完的格照实显示 20/20；没跑完的才按单种子目标计
            seeds = target_seeds(m, ds)
            if all(f'{m}_{ds}_{sl}_{pl}_s{sd}_exp_0' in done
                   for pl in BL_PREDS.get(ds, PREDS) if has_cfg(m, ds, pl) for sd in SEEDS):
                seeds = SEEDS
            for pl in BL_PREDS.get(ds, PREDS):
                if not has_cfg(m, ds, pl):
                    continue
                for sd in seeds:
                    tot += 1
                    if f'{m}_{ds}_{sl}_{pl}_s{sd}_exp_0' in done:
                        n += 1
            row['cells'].append({'ds': ds, 'n': n, 'tot': tot})
            bl_total += tot
            bl_done += n
        grid.append(row)

    # ---- 主线：按 tag × pred 统计种子数，并给出该格已完成种子的 mse 均值 ----
    main_rows = st['hosts'].get('autodl', {}).get('rows', {}).get('main', [])
    agg = {}
    for r in main_rows:
        m = SETTING_RE.match(r['setting'])
        if not m:
            continue
        key = (m.group('ds'), m.group('tag'), int(m.group('pl')))
        try:
            agg.setdefault(key, {})[int(m.group('seed'))] = float(r['mse'])
        except ValueError:
            pass
    def mean(d):
        return sum(d.values()) / len(d) if d else None

    def rsd(d):   # 相对标准差(%)，判定消融效应是否显著要用它
        if len(d) < 2:
            return None
        v = list(d.values())
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        return round(100 * sd / m, 2)

    main_tbl, main_done, main_total = [], 0, 0
    for tag, label in MAIN_TAGS:
        cells = []
        for pl in PREDS:
            v = agg.get((MAIN_DATA, tag, pl), {})
            b = agg.get((MAIN_DATA, 'w_on', pl), {})
            mv, mb = mean(v), mean(b)
            cost = round((mv - mb) / mb * 100, 2) if mv and mb and tag != 'w_on' else None
            cells.append({'pl': pl, 'n': len(v), 'need': NEED_SEEDS,
                          'mse': round(mv, 4) if mv else None,
                          'rsd': rsd(v), 'cost': cost})
            main_done += min(len(v), NEED_SEEDS)
            main_total += NEED_SEEDS
        main_tbl.append({'tag': tag, 'label': label, 'cells': cells})

    # ---- ETTh1 判决批：v2 各配方 vs aux 基线 ----
    judge_tbl = []
    for tag, label in JUDGE_TAGS:
        cells = []
        for pl in JUDGE_PREDS:
            v = agg.get((JUDGE_DATA, tag, pl), {})
            b = agg.get((JUDGE_DATA, JUDGE_BASE, pl), {})
            mv, mb = mean(v), mean(b)
            cost = round((mv - mb) / mb * 100, 2) if mv and mb else None
            cells.append({'pl': pl, 'n': len(v), 'need': JUDGE_SEEDS,
                          'mse': round(mv, 4) if mv else None, 'rsd': rsd(v), 'cost': cost,
                          'base': round(mb, 4) if mb else None, 'base_rsd': rsd(b)})
        judge_tbl.append({'tag': tag, 'label': label, 'cells': cells})

    # ---- PRISM 主表：weather 用 w_on 组，其余用 main 组 ----
    prism_tbl, pm_done, pm_total = [], 0, 0
    for ds, preds in MAIN_TABLE:
        # 2026-09-10 起 ETT×4 + weather 重跑含近端流的 aux 组；有 aux 行就展示 aux，否则回落到旧组
        if any(k[0] == ds and k[1] == 'aux' for k in agg):
            tag = 'aux'
        else:
            tag = 'w_on' if ds == MAIN_DATA else 'main'
        cells = []
        for pl in preds:
            v = agg.get((ds, tag, pl), {})
            mv = mean(v)
            cells.append({'pl': pl, 'n': len(v), 'need': NEED_SEEDS,
                          'mse': round(mv, 4) if mv else None, 'rsd': rsd(v)})
            pm_done += min(len(v), NEED_SEEDS)
            pm_total += NEED_SEEDS
        prism_tbl.append({'ds': ds + (' (aux)' if tag == 'aux' else ''), 'cells': cells})

    # ---- 吞吐与 ETA：两台服务器分开，各用自己最近一小时实际落库的行数外推 ----
    now = time.time()

    def recent_by_ds(rows_by_csv):
        seen = {}   # 各分片 csv 以同一主表播种，按 setting 去重，否则吞吐虚高
        for rows in rows_by_csv.values():
            for r in rows:
                seen.setdefault(r['setting'], r['ts'])
        c = {}
        for setting, ts in seen.items():
            try:
                t = time.mktime(time.strptime(ts.strip(), '%Y-%m-%d %H:%M:%S'))
            except ValueError:
                continue
            if now - t < 3600:
                ds = setting.split('_')[1]
                c[ds] = c.get(ds, 0) + 1
        return c

    # 各数据集跑速差上百倍（illness 几秒一格、traffic 数小时一格），整机总吞吐外推会严重偏乐观，
    # 所以 ETA 按数据集分开算：剩余格 / 该数据集近一小时落库格数
    # autodl 的下一项工作 = PRISM 正式成绩（tag final，MAIN_TABLE 全数据集 × 视野 × 5 种子）
    main_cells = {(ds, pl) for ds, preds in MAIN_TABLE for pl in preds}
    final_done = set()
    for r in main_rows:
        m = SETTING_RE.match(r['setting'])
        if (m and m.group('tag') == 'final' and int(m.group('seed')) in SEEDS
                and (m.group('ds'), int(m.group('pl'))) in main_cells):
            final_done.add((m.group('ds'), m.group('pl'), m.group('seed')))
    prism_remain = {ds: len(preds) * len(target_seeds('PRISM', ds))
                        - sum(1 for k in final_done
                              if k[0] == ds and int(k[2]) in target_seeds('PRISM', ds))
                    for ds, preds in MAIN_TABLE}
    bl_remain, bl_deferred = {}, 0
    for row in grid:
        for c in row['cells']:
            r = c['tot'] - c['n']
            if r <= 0 or (row['model'], c['ds']) in BL_WAIVED:
                continue
            if (row['model'], c['ds']) in BL_DEFERRED:
                bl_deferred += r
            else:
                bl_remain[c['ds']] = bl_remain.get(c['ds'], 0) + r
    eta_by_host = {}
    for h, remain, deferred, label in (('autodl', prism_remain, 0, 'PRISM 正式成绩（final）'),
                                       ('autodl2', bl_remain, bl_deferred, 'baseline 全网格')):
        rate = recent_by_ds(st['hosts'].get(h, {}).get('rows', {}))
        items = [{'ds': ds, 'remain': r, 'rate': rate.get(ds, 0),
                  'eta': round(r / rate[ds], 1) if rate.get(ds) else None}
                 for ds, r in remain.items() if r > 0]
        eta_by_host[h] = {'label': label, 'remain': sum(remain.values()), 'deferred': deferred,
                          'items': items}

    # ---- 队列：当前实验还在进程表里才算在跑，展开显示；已结束的历史分片只计数，不再逐条列出 ----
    cell = {(row['model'], c['ds']): c for row in grid for c in row['cells']}
    queue_by_host = {}
    for h, d in st['hosts'].items():
        live = {p['model_id'] for p in d['procs']}
        lanes, ended = [], 0
        for k, l in d['logs'].items():
            if l.get('cur_id') not in live:
                ended += 1
                continue
            name = k[3:] if k.startswith('sh_') else k
            title, dss, models = LANES.get(name, (name, [], []))
            cs = [cell[(m, ds)] for m in models for ds in dss if (m, ds) in cell]
            lanes.append({'title': title, 'datasets': dss, 'models': models,
                          'done': sum(c['n'] for c in cs), 'total': sum(c['tot'] for c in cs),
                          'fail': l.get('fail', 0), 'cur': l['cur_id'], 'since': l['cur_ts']})
        curs = {l['cur'] for l in lanes}
        queue_by_host[h] = {'lanes': lanes, 'ended': ended,
                            'other': [p['model_id'] for p in d['procs'] if p['model_id'] not in curs]}

    return {'ts': st['ts'], 'errors': st['error'],
            'hosts': {h: {'role': HOSTS[h]['role'], 'gpu': d['gpu'], 'load': d['load'],
                          'now': d['now'], 'procs': d['procs'], 'queue': queue_by_host[h]}
                      for h, d in st['hosts'].items()},
            'grid': grid, 'bl_done': bl_done, 'bl_total': bl_total,
            'main': main_tbl, 'main_done': main_done, 'main_total': main_total,
            'prism': prism_tbl, 'pm_done': pm_done, 'pm_total': pm_total,
            'judge': judge_tbl, 'judge_preds': JUDGE_PREDS,
            'eta_by_host': eta_by_host,
            'datasets': BL_DATASETS, 'preds': PREDS}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith('/api/status'):
            body = json.dumps(build_view()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


PAGE = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>实验看板</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e9ef;--dim:#8b93a7;
      --ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--acc:#60a5fa}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"PingFang SC",sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;
       gap:16px;align-items:baseline;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:16px;margin:0;font-weight:600}
.dim{color:var(--dim)}
main{padding:20px;display:grid;gap:16px;max-width:1400px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card h2{font-size:14px;margin:0 0 10px;font-weight:600}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 8px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500}
td.c{text-align:center}
.bar{height:6px;background:#232833;border-radius:3px;overflow:hidden;margin-top:4px}
.bar>i{display:block;height:100%;background:var(--acc)}
.full{color:var(--ok)} .part{color:var(--warn)} .none{color:var(--dim)} .bad{color:var(--bad)}
.badge{padding:1px 7px;border-radius:99px;font-size:12px;background:#232833}
pre{margin:4px 0 0;font-size:12px;color:var(--dim);white-space:pre-wrap;word-break:break-all}
.scroll{overflow-x:auto}
</style></head><body>
<header><h1>实验看板</h1>
<span class="dim" id="stamp">加载中…</span>
<span class="dim" id="rate"></span></header>
<main>
  <div class="two" id="hosts"></div>
  <div class="card"><h2>Baseline 全网格（5 种子 × 4 视野）<span id="blpct" class="dim"></span></h2>
    <div class="bar"><i id="blbar" style="width:0"></i></div>
    <div class="scroll"><table id="bltbl"></table></div></div>
</main>
<script>
const cls = (n,t)=> n===0?'none':(n>=t?'full':'part');
function esc(s){return String(s).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))}
function etaText(e){
  let s=e.label+' 剩 '+e.remain+' 格'+(e.deferred?'（另暂缓 '+e.deferred+'）':'');
  if(!e.remain) return s+' · 已全部完成';
  const run=e.items.filter(i=>i.eta!=null), idle=e.items.filter(i=>i.eta==null);
  if(!run.length) return s+' · 近 1 小时无产出，暂无法估算';
  s+=' · 在跑的数据集最慢约 '+Math.max(...run.map(i=>i.eta))+' 小时';
  if(idle.length) s+=' · '+idle.map(i=>i.ds).join('/')+' 尚无产出，无法估算';
  return s;
}
function mins(now,since){
  if(!now||!since) return '—';
  const m=Math.round((new Date(now.replace(' ','T'))-new Date(since.replace(' ','T')))/60000);
  return m<60? m+' 分钟' : (m/60).toFixed(1)+' 小时';
}
function etaRows(e){
  return e.items.map(i=>esc(i.ds)+' 剩 '+i.remain+' · 近 1 小时 '+i.rate+' 格'+
    (i.eta!=null?' · 约 '+i.eta+' 小时':' · 无产出')).join('<br>');
}

function render(d){
  document.getElementById('stamp').textContent =
    '采样于 ' + new Date(d.ts*1000).toLocaleTimeString('zh-CN');
  document.getElementById('rate').textContent =
    Object.entries(d.eta_by_host||{}).map(([h,e])=>h+'：'+etaText(e)).join('  ｜  ');

  let h='';
  for(const [name,v] of Object.entries(d.hosts)){
    const g=v.gpu;
    const eh=(d.eta_by_host||{})[name];
    h+=`<div class="card"><h2>${esc(name)} <span class="badge">${esc(v.role)}</span></h2>
      ${eh?'<div class="dim">剩余：'+esc(etaText(eh))+'<br>'+etaRows(eh)+'</div>':''}
      <table><tr><th>显存</th><td>${g?g.used+' / '+g.total+' MiB':'—'}</td>
      <th>GPU</th><td>${g?g.util+'%':'—'}</td>
      <th>负载</th><td>${v.load ?? '—'}</td></tr></table>
      `;
    const q=v.queue||{lanes:[],ended:0,other:[]};
    h+='<table><tr><th>正在跑的队列</th><th>进度</th><th>失败</th><th>当前实验</th><th>本格已跑</th></tr>'+
      (q.lanes.map(l=>`<tr><td>${esc(l.title)}<div class="dim">${esc(l.datasets.join(' → '))} · ${esc(l.models.join(' / '))}</div></td>
        <td class="c">${l.done}/${l.total}</td><td class="c ${l.fail?'bad':''}">${l.fail}</td>
        <td>${esc(l.cur)}</td><td class="c">${mins(v.now,l.since)}</td></tr>`).join('')
       ||'<tr><td colspan="5" class="dim">空闲，没有队列在跑</td></tr>')+'</table>';
    if(q.other.length) h+='<div class="dim">队列之外的进程：'+q.other.map(esc).join('，')+'</div>';
    if(q.ended) h+=`<div class="dim">另有 ${q.ended} 个历史队列已结束（结果已计入下方全网格表）</div>`;
    h+='</div>';
  }
  for(const [k,e] of Object.entries(d.errors||{}))
    h+=`<div class="card"><h2>${esc(k)}</h2><pre>连接失败: ${esc(e)}</pre></div>`;
  document.getElementById('hosts').innerHTML=h;

  const bp = d.bl_total? Math.round(100*d.bl_done/d.bl_total):0;
  document.getElementById('blpct').textContent=` ${d.bl_done}/${d.bl_total} · ${bp}%`;
  document.getElementById('blbar').style.width=bp+'%';
  document.getElementById('bltbl').innerHTML =
    '<tr><th>模型</th>'+d.datasets.map(s=>`<th class="c">${s}</th>`).join('')+'</tr>'+
    d.grid.map(r=>`<tr><td>${esc(r.model)}</td>`+
      r.cells.map(c=>`<td class="c ${cls(c.n,c.tot)}">${c.tot?c.n+'/'+c.tot:'—'}</td>`).join('')+
      '</tr>').join('');
}
async function tick(){
  try{ render(await (await fetch('/api/status')).json()); }
  catch(e){ document.getElementById('stamp').textContent='看板服务已停止'; }
}
tick(); setInterval(tick, 10000);
</script></body></html>'''


if __name__ == '__main__':
    threading.Thread(target=poller, daemon=True).start()
    print(f'看板: http://127.0.0.1:{PORT}   (每 {POLL}s 采样一次)')
    HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
