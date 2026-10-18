"""PRISM（FRAME 主干）结构冒烟检查（CPU 可跑，无需数据集）。

用法: python scripts/PRISM/check_frame.py
断言项：输出形状 / 推理确定性 / 消融开关可前向 / 无死参数 / 形状断言生效 /
小样本可过拟合。全部通过打印 ALL CHECKS PASSED。
"""

import copy
import functools
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from models.PRISM import Model  # noqa: E402
from utils.config import load_config  # noqa: E402

ARGS = ['-c', 'configs/PRISM/smoke.yaml']


def make_cfg(**overrides):
    cfg = copy.deepcopy(BASE_CFG)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def open_gates(m):
    """把所有零初始化门拨离零点后再查梯度。

    零初始化门（token_mix 的 mix_gate、residual 的 res_gate、hgnn_ctx 的 ctx_proj）
    在初值处让其下游子树梯度恰为零——这是该设计的固有性质（门自身有梯度，会先离开
    零点，子树随后才开始学），不是死参数。故梯度检查须先开门，验证的是通路本身。
    """
    with torch.no_grad():
        if hasattr(m, 'mix_gate'):
            m.mix_gate.fill_(0.5)
        if hasattr(m, 'res_gate'):
            m.res_gate.fill_(0.5)
        if hasattr(m, 'aux_gate'):
            m.aux_gate.fill_(0.5)
        if hasattr(m, 'spec_moe') and hasattr(m.spec_moe, 'ctx_proj'):
            m.spec_moe.ctx_proj.weight.normal_(std=0.1)
    return m


def dummy(cfg, batch=4):
    torch.manual_seed(0)
    return torch.randn(batch, cfg.seq_len, cfg.enc_in)


BASE_CFG = load_config(ARGS)
BASE_CFG.enc_in = 7

print('== 1. forward shape ==')
torch.manual_seed(0)
model = Model(make_cfg())
x = dummy(BASE_CFG)
y = model(x, None, None, None)
assert y.shape == (4, BASE_CFG.pred_len, BASE_CFG.enc_in), y.shape
n_params = sum(p.numel() for p in model.parameters())
print(f'   ok: out {tuple(y.shape)}, params {n_params:,}')

print('== 2. eval determinism (gumbel -> argmax) ==')
model.eval()
with torch.no_grad():
    y1 = model(x, None, None, None)
    y2 = model(x, None, None, None)
assert torch.equal(y1, y2), 'eval forward is non-deterministic'
model.train()
print('   ok: two eval forwards bit-identical')

print('== 3. ablation switches forward ==')
for strategy in (0, 1, 2, 3, 4):
    m = Model(make_cfg(pattern_strategy=strategy))
    out = m(x, None, None, None)
    assert out.shape == y.shape
    print(f'   ok: pattern_strategy={strategy}')
for alpha in (0.0, 1.0):
    m = Model(make_cfg(seasonal_alpha=alpha))
    assert m(x, None, None, None).shape == y.shape
    print(f'   ok: seasonal_alpha={alpha}')
m = Model(make_cfg(patch_pool=[0]))
assert m(x, None, None, None).shape == y.shape
print('   ok: patch_pool=[0] (fixed-length patching degeneration)')
m = Model(make_cfg(pattern_nums=1))
assert m(x, None, None, None).shape == y.shape
print('   ok: pattern_nums=1 (single expert)')
# 关掉某分支后按定义应失活的参数（不算 bug），其余一律不得无梯度
# 门控被均匀 1/K 覆盖后，逆温度失效；且在新默认值（use_hgnn=false + patch_mode=uniform）下
# 谱原型的唯一消费者就是这个门控——边界不再由共振打分决定、图也不再读原型，
# 故 pat_emb 一并失活。这是结构事实（①频域稀疏匹配与⑦原型参数化在门控关闭时同样无消费者），
# 不是 bug；消融表里这三项须理解为共享同一条通路。
EXPECTED_INERT = {'use_moe_gate': {'pdp.log_gate_temp',
                                   'pat_emb.pattern_shared', 'pat_emb.pattern_specific'}}
for key in ('sra_sparse', 'use_hgnn', 'use_moe_gate'):
    m = open_gates(Model(make_cfg(**{key: False})))
    assert m(x, None, None, None).shape == y.shape
    m.zero_grad()
    m(x, None, None, None).sum().backward()
    dead = {n for n, p in m.named_parameters()
            if p.grad is None or torch.count_nonzero(p.grad) == 0}
    unexpected = dead - EXPECTED_INERT.get(key, set())
    assert not unexpected, f'{key}=False leaves dead params: {sorted(unexpected)}'
    names = {n for n, _ in m.named_parameters()}
    missing = (EXPECTED_INERT.get(key, set()) & names) - dead
    assert not missing, f'{key}=False 未使 {sorted(missing)} 失活，消融未真正生效'
    print(f'   ok: {key}=False (no unexpected dead params)')

# 零初始化门：gate_input 起点必须与官方（无残差）逐位等价
# residual_mode 仅对 recon 数据流有意义：direct 挡下残差会把原始序列重新喂进预测头，
# 构造期守卫已禁止（见 == 11 非法组合）。故此处显式切回 recon 测试。
RECON = dict(tokenize='recon', patch_mode='greedy', token_mix='none')
torch.manual_seed(0)
m_res = Model(make_cfg(residual_mode='gate_input', **RECON)).eval()
torch.manual_seed(0)
m_off = Model(make_cfg(residual_mode='none', **RECON)).eval()
with torch.no_grad():
    assert torch.allclose(m_res(x, None, None, None), m_off(x, None, None, None), atol=1e-6), \
        'zero-init residual gate is not identity at init'
print('   ok: residual_mode=gate_input (identity at init)')
for mode in ('gate_backbone',):
    m = Model(make_cfg(residual_mode=mode, **RECON))
    assert m(x, None, None, None).shape == y.shape
    print(f'   ok: residual_mode={mode} (recon 数据流)')
for mi in ('allpass',):
    m = Model(make_cfg(moe_init=mi))
    assert m(x, None, None, None).shape == y.shape
    print(f'   ok: moe_init={mi}')

print('== 4. no dead parameters ==')
open_gates(model)
model.zero_grad()
model(x, None, None, None).sum().backward()
dead = [n for n, p in model.named_parameters()
        if p.grad is None or torch.count_nonzero(p.grad) == 0]
assert not dead, f'parameters without gradient: {dead}'
print(f'   ok: all {len(list(model.parameters()))} parameter tensors receive gradient')

print('== 5. shape assertions fire ==')
for bad, msg in [(dict(p_model=65), 'p_model % n_heads'),
                 (dict(freq_topk=999), 'freq_topk'),
                 (dict(min_patch_len=1000), 'min_patch_len')]:
    try:
        Model(make_cfg(**bad))
    except AssertionError:
        print(f'   ok: {msg} guarded')
    else:
        raise SystemExit(f'FAIL: no assertion for {bad}')

print('== 6. overfit 2 samples ==')
torch.manual_seed(0)
m = Model(make_cfg())
xs = dummy(BASE_CFG, batch=2)
ys = torch.randn(2, BASE_CFG.pred_len, BASE_CFG.enc_in)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)
loss = None
for _ in range(300):
    opt.zero_grad()
    loss = torch.nn.functional.mse_loss(m(xs, None, None, None), ys)
    loss.backward()
    opt.step()
assert loss.item() < 0.05, f'overfit loss too high: {loss.item():.4f}'
print(f'   ok: final loss {loss.item():.5f}')

print('== 7. tokenize=direct / 固定段数分块 ==')
DIRECT = dict(tokenize='direct', n_tokens=6, head_type='linear', residual_mode='none')


def _pdp_out(m, xin):
    """抓 ParallelDynamicPatching 的输出 (patches, lengths, var_ids, sel_moe)。"""
    box = {}
    h = m.pdp.register_forward_hook(lambda mod, i, o: box.update(o=o))
    with torch.no_grad():
        m(xin, None, None, None)
    h.remove()
    return box['o']


for pm in ('uniform', 'fixed_random', 'greedy_fixed'):
    m = open_gates(Model(make_cfg(patch_mode=pm, **DIRECT)))
    assert m(x, None, None, None).shape == y.shape
    m.zero_grad()
    m(x, None, None, None).sum().backward()
    dead = [n for n, prm in m.named_parameters()
            if prm.grad is None or torch.count_nonzero(prm.grad) == 0]
    assert not dead, f'patch_mode={pm} leaves dead params: {dead}'
    m.eval()
    assert torch.equal(_pdp_out(m, x)[0], _pdp_out(m, x)[0]), f'{pm} eval non-deterministic'
    _, lens, var_ids, _ = _pdp_out(m, x)
    C, N = BASE_CFG.enc_in, 6
    # direct 挡按通道展平，var_ids 必须是通道主序；否则 token 会串通道
    assert torch.equal(var_ids, torch.arange(C).repeat_interleave(N)), \
        f'{pm}: var_ids 非通道主序 {var_ids.tolist()}'
    # 逐样本边界下段长是 [B, C*N]，整批共享时是 [C*N]；两种形状都要求每通道恰好铺满
    per_ch = lens.reshape(-1, C, N)
    assert (per_ch.sum(-1) == BASE_CFG.seq_len).all(), \
        f'{pm}: 段长和 != seq_len ({per_ch.sum(-1).tolist()})'
    print(f'   ok: patch_mode={pm} (每通道恰 {N} 段、和为 {BASE_CFG.seq_len}、通道主序、无死参数)')

# 逐样本边界：段长张量为 [B, C*N]，且同一条样本的分段不得随同批伙伴改变
m_ps = Model(make_cfg(patch_mode='greedy_fixed', boundary_scope='sample', **DIRECT)).eval()
x_one = x[:1]
seg_alone = _pdp_out(m_ps, x_one)[1]
assert seg_alone.shape == (1, BASE_CFG.enc_in * 6), f'逐样本段长形状 {tuple(seg_alone.shape)}'
assert (seg_alone.reshape(BASE_CFG.enc_in, 6).sum(1) == BASE_CFG.seq_len).all()
for extra in (torch.randn(7, BASE_CFG.seq_len, BASE_CFG.enc_in),
              torch.randn(7, BASE_CFG.seq_len, BASE_CFG.enc_in)):
    seg_batch = _pdp_out(m_ps, torch.cat([x_one, extra]))[1][:1]
    assert torch.equal(seg_alone, seg_batch), \
        f'boundary_scope=sample 仍受同批样本影响: {seg_alone.tolist()} vs {seg_batch.tolist()}'
print('   ok: boundary_scope=sample 分段与同批伙伴无关（batch 挡实测会变）')

# greedy_fixed 必须真的自适应：不同通道/不同步的段长应出现差异
_, lens, _, _ = _pdp_out(Model(make_cfg(patch_mode='greedy_fixed', **DIRECT)).eval(), x)
assert lens.unique().numel() > 1, 'greedy_fixed 段长全同，退化为等分'
assert lens.dim() == 2, f'默认 boundary_scope=sample 下段长应为 [B, C*N]，实得 {tuple(lens.shape)}'
print(f'   ok: greedy_fixed 段长自适应（出现 {lens.unique().numel()} 种长度）')

# 可达性表须与暴力枚举逐格一致（区间松弛在离散池上会给出假可行）
pdp = Model(make_cfg(patch_mode='greedy_fixed', **DIRECT)).pdp
steps, L_, N_ = pdp.steps_list, pdp.seq_len, pdp.n_tokens


@functools.lru_cache(None)
def _brute(m_, r_):
    if m_ == 0:
        return r_ == 0
    return any(_brute(m_ - 1, r_ - st) for st in steps if st <= r_)


bad = [(mm, rr) for mm in range(N_ + 1) for rr in range(L_ + 1)
       if _brute(mm, rr) != bool(pdp.reach[mm, rr])]
assert not bad, f'reach 表与暴力枚举不一致: {bad[:5]}'
print(f'   ok: reach 表 {N_ + 1}x{L_ + 1} 与暴力枚举逐格一致')

# fixed_random 必须"非等分且恒定"：段长有多种取值，且两次前向逐位一致
m_fr = Model(make_cfg(patch_mode='fixed_random', **DIRECT)).eval()
l1, l2 = _pdp_out(m_fr, x)[1], _pdp_out(m_fr, x[:2])[1]
assert l1.unique().numel() > 1, 'fixed_random 段长全同，退化为等分'
assert torch.equal(l1, l2), 'fixed_random 段长随批次变化，不是固定分段'
print(f'   ok: fixed_random 非等分且恒定（{l1.unique().numel()} 种长度）')

# token_pos_enc 让预测头多读到 (起点, 段长)：头部输入维应增加 2*N
for pm in ('uniform', 'greedy_fixed'):
    m_pe = Model(make_cfg(patch_mode=pm, token_pos_enc=True, **DIRECT))
    m_no = Model(make_cfg(patch_mode=pm, token_pos_enc=False, **DIRECT))
    assert m_pe(x, None, None, None).shape == y.shape
    d = m_pe.model[-1].in_features - m_no.model[-1].in_features
    assert d == 2 * 6, f'{pm}: token_pos_enc 头部输入维只增加了 {d}，应为 12'
print('   ok: token_pos_enc 头部输入维 +2N（uniform / greedy_fixed 均然）')

# 贪心挡 + 逐 token 门控：段数随 batch 涌现，门控须按 patch 顺序拼接（此前形状不匹配）
for pm in ('greedy', 'greedy_fixed', 'fixed_random', 'uniform'):
    kw = (dict(tokenize='recon', boundary_scope='batch', token_mix='none')
          if pm == 'greedy' else DIRECT)
    mm = Model(make_cfg(patch_mode=pm, gate_scope='token', **kw))
    assert mm(x, None, None, None).shape == y.shape, f'{pm} + gate_scope=token 前向失败'
print('   ok: 四种 patch_mode 均可配 gate_scope=token')

print('== 8. gate_scope=token 逐样本路由 ==')
for pm in ('uniform', 'greedy_fixed'):
    m = Model(make_cfg(patch_mode=pm, gate_scope='token', **DIRECT))
    sel = _pdp_out(m, x)[3]
    assert sel.dim() == 3 and sel.shape[0] == x.shape[0], f'{pm}: sel_moe {tuple(sel.shape)}'
    # channel 挡把逐样本/逐段分数 mean 掉，门控退化为静态滤波器的重参数化（消融效应恒为 0）
    assert sel.std(dim=0).mean() > 1e-4, f'{pm}: 门控不随样本变化'
    assert sel.std(dim=1).mean() > 1e-4, f'{pm}: 门控不随 patch 变化'
    print(f'   ok: patch_mode={pm} 门控 {tuple(sel.shape)} 随样本与 patch 变化')
    sel_ch = _pdp_out(Model(make_cfg(patch_mode=pm, gate_scope='channel', **DIRECT)), x)[3]
    assert sel_ch.dim() == 2, f'{pm}: channel 挡 sel_moe 应为 [C, K]'

print('== 9. hgnn_ctx=spectral 逐样本专家滤波器 ==')
m_sp = Model(make_cfg(patch_mode='uniform', hgnn_ctx='spectral', use_hgnn=True, **DIRECT)).eval()
# 同一实例切到 static 比较：另建实例会因多构造两个模块而错开 RNG 流，其余权重本就不同
with torch.no_grad():
    out_sp = m_sp(x, None, None, None)
    m_sp.spec_moe.hgnn_ctx = 'static'
    out_st = m_sp(x, None, None, None)
    m_sp.spec_moe.hgnn_ctx = 'spectral'
# 批量 matmul 与非批量走不同 kernel，归约顺序不同，故比数值等价而非逐位相等
_d = (out_sp - out_st).abs().max().item()
assert _d < 1e-5, \
    f'hgnn_ctx=spectral 在初值下与 static 不等价（max diff {_d:.2e}，ctx_proj 未零初始化？）'
print(f'   ok: 初值下与 static 数值等价（max diff {_d:.1e}；谱上下文为零初始化修正量）')

with torch.no_grad():  # 扰动投影后，专家滤波器必须随样本变化
    m_sp.spec_moe.ctx_proj.weight.normal_(std=0.1)
box = {}
m_sp.spec_moe.hgnn.register_forward_hook(lambda mo, i, o: box.update(w=o))
with torch.no_grad():
    m_sp(x, None, None, None)
w = box['w']
assert w.dim() == 4, f'专家滤波器仍无 batch 维: {tuple(w.shape)}'
assert w.std(dim=0).mean() > 1e-5, '专家滤波器不随样本变化'
print(f'   ok: 专家滤波器 {tuple(w.shape)} 随样本变化（跨样本 σ={float(w.std(0).mean()):.4f}）')

# ctx_proj 零初始化时 ctx_norm 在初值处梯度必为零（下游权重为 0），属预期而非死参数；
# 故先扰动投影再查梯度，验证的是通路本身而不是初值
m_g = open_gates(Model(make_cfg(patch_mode='uniform', hgnn_ctx='spectral',
                                use_hgnn=True, **DIRECT)))
m_g(x, None, None, None).sum().backward()
dead = [n for n, prm in m_g.named_parameters()
        if prm.grad is None or torch.count_nonzero(prm.grad) == 0]
assert not dead, f'hgnn_ctx=spectral 留下死参数: {dead}'
print('   ok: 无死参数')

print('== 10. token_mix=hgnn（token 序列上的图混合）==')
m_mix = Model(make_cfg(patch_mode='uniform', token_mix='hgnn', **DIRECT)).eval()
m_mix.spec_moe_out = None
with torch.no_grad():
    out_mix = m_mix(x, None, None, None)
    m_mix.token_mix = 'none'
    out_non = m_mix(x, None, None, None)
    m_mix.token_mix = 'hgnn'
_d = (out_mix - out_non).abs().max().item()
assert _d < 1e-6, f'零初始化门未生效：初值下与 none 差 {_d:.2e}'
print(f'   ok: 初值下与 none 逐位等价（max diff {_d:.1e}）')

# 门打开后，token 必须真的跨通道、跨时间混合：扰动一个通道的输入应影响其他通道的 token
with torch.no_grad():
    m_mix.mix_gate.fill_(1.0)
box = {}
m_mix.token_hgnn[-1].register_forward_hook(lambda mo, i, o: box.update(o=o))
with torch.no_grad():
    m_mix(x, None, None, None)
    base = box['o'].clone()
    x_pert = x.clone()
    # 必须改变波形：RevIN 会把加常数/乘系数这类扰动整个归一化掉
    torch.manual_seed(1)
    x_pert[:, :, 0] = torch.randn_like(x_pert[:, :, 0])
    m_mix(x_pert, None, None, None)
    diff = (box['o'] - base).abs().mean(dim=(0, 2, 3))   # 每通道的变化量
assert diff[1:].max() > 1e-6, f'扰动通道 0 未影响其他通道，跨通道通路未生效: {diff.tolist()}'
print(f'   ok: 扰动通道 0 影响到其他通道（其余通道平均变化 {diff[1:].mean():.4f}）')

for n_lay in (1, 2):
    m_grad = Model(make_cfg(patch_mode='uniform', token_mix='hgnn',
                            token_mix_layers=n_lay, **DIRECT))
    assert len(m_grad.token_hgnn) == n_lay and m_grad.mix_gate.numel() == n_lay
    with torch.no_grad():
        m_grad.mix_gate.fill_(0.5)   # 门为 0 时上层无梯度属预期，故先打开再查
    m_grad(x, None, None, None).sum().backward()
    dead = [n for n, prm in m_grad.named_parameters()
            if prm.grad is None or torch.count_nonzero(prm.grad) == 0]
    assert not dead, f'token_mix_layers={n_lay} 留下死参数: {dead}'
    print(f'   ok: token_mix_layers={n_lay} 无死参数')

print('== 11. 非法组合报错 ==')
for bad_cfg, msg in [
        (dict(tokenize='direct', patch_mode='greedy'), 'direct + greedy'),
        (dict(tokenize='direct', patch_mode='uniform', residual_mode='gate_backbone'),
         'direct + residual'),
        (dict(tokenize='direct', patch_mode='uniform', n_tokens=7), 'uniform 段数不整除'),
        (dict(patch_mode='greedy_fixed', n_tokens=12, patch_pool=[0], min_patch_len=8),
         'greedy_fixed 无自适应空间(12x8=96 唯一解)'),
        (dict(patch_mode='greedy_fixed', n_tokens=5, patch_pool=[0, 2],
              min_patch_len=7), 'greedy_fixed 无解'),
        (dict(gate_scope='bogus'), 'gate_scope 非法值'),
        (dict(boundary_scope='sample', patch_mode='greedy_fixed', tokenize='recon'),
         'sample 边界 + greedy_fixed + recon 挡'),
        (dict(hgnn_ctx='bogus', use_hgnn=True), 'hgnn_ctx 非法值'),
        (dict(token_mix='hgnn', tokenize='recon'), 'token_mix + recon 挡'),
        (dict(token_mix='bogus'), 'token_mix 非法值'),
        (dict(aux_stream='bogus'), 'aux_stream 非法值'),
        (dict(token_mix='hgnn', token_mix_gate='bogus'), 'token_mix_gate 非法值')]:
    try:
        Model(make_cfg(**bad_cfg))
    except ValueError as e:
        print(f'   ok: {msg} -> ValueError({str(e)[:48]}...)')
    else:
        raise SystemExit(f'FAIL: no ValueError for {bad_cfg}')

print('== 12. aux_stream 近端流 ==')
UNI = dict(patch_mode='uniform', **DIRECT)
torch.manual_seed(0)
m_non = Model(make_cfg(aux_stream='none', **UNI)).eval()
for mode in ('full', 'last'):
    m_aux = Model(make_cfg(aux_stream=mode, **UNI)).eval()
    # aux_head 的随机初始化会移动后续参数的 RNG 状态，故拷贝共有权重后再比（缺 aux_* 属预期）
    missing, unexpected = m_aux.load_state_dict(m_non.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('aux_') for k in missing), (missing, unexpected)
    with torch.no_grad():
        _d = (m_aux(x, None, None, None) - m_non(x, None, None, None)).abs().max().item()
    assert _d < 1e-6, f'aux_stream={mode} 零初始化门未生效：初值下与 none 差 {_d:.2e}'
    print(f'   ok: aux_stream={mode} 初值下与 none 逐位等价（max diff {_d:.1e}）')
m_tr = Model(make_cfg(aux_stream='trend', **UNI)).eval()
assert m_tr(x, None, None, None).shape == y.shape
print('   ok: aux_stream=trend 前向形状正确（主干输入为季节项，初值不要求与 none 等价）')
for mode in ('full', 'trend', 'last'):
    m_aux = open_gates(Model(make_cfg(aux_stream=mode, **UNI)))
    m_aux(x, None, None, None).sum().backward()
    dead = [n for n, prm in m_aux.named_parameters()
            if prm.grad is None or torch.count_nonzero(prm.grad) == 0]
    assert not dead, f'aux_stream={mode} 留下死参数: {dead}'
    assert m_aux.aux_head.in_features == (BASE_CFG.seq_len // BASE_CFG.n_tokens
                                          if mode == 'last' else BASE_CFG.seq_len)
    # 门打开后近端流必须改变输出
    with torch.no_grad():
        o1 = m_aux(x, None, None, None)
        m_aux.aux_gate.zero_()
        o0 = m_aux(x, None, None, None)
    assert (o1 - o0).abs().max() > 1e-6, f'aux_stream={mode} 门打开后输出不变'
    print(f'   ok: aux_stream={mode} 无死参数，头输入维 {m_aux.aux_head.in_features}，门生效')

print('== 13. token_mix_gate=split 分关系门 ==')
torch.manual_seed(0)
m_sh = Model(make_cfg(token_mix='hgnn', token_mix_gate='shared', **UNI)).eval()
m_sp = Model(make_cfg(token_mix='hgnn', token_mix_gate='split', **UNI)).eval()
missing, unexpected = m_sp.load_state_dict(m_sh.state_dict(), strict=False)
assert not unexpected and all('gate_int' in k for k in missing), (missing, unexpected)
with torch.no_grad():
    m_sh.mix_gate.fill_(1.0)
    m_sp.mix_gate.fill_(1.0)   # 外门打开，比较的是内门初值 1 是否等价于不分门
    _d = (m_sp(x, None, None, None) - m_sh(x, None, None, None)).abs().max().item()
assert _d < 1e-6, f'split 初值下与 shared 差 {_d:.2e}'
print(f'   ok: split 初值下与 shared 逐位等价（max diff {_d:.1e}）')
m_sp.train()
m_sp(x, None, None, None).sum().backward()
for lay in m_sp.token_hgnn:
    assert lay.gate_intra.grad is not None and lay.gate_intra.grad.abs().item() > 0
    assert lay.gate_inter.grad is not None and lay.gate_inter.grad.abs().item() > 0
print('   ok: gate_intra / gate_inter 均有梯度')
# 主干里另有一条与 token 混合无关的跨通道泄漏：SpectralMoE 的趋势分支沿展平后的
# patch 轴 [B, C*N] 做滑动平均（原版行为，见 layers 文件头备案），核 25 会越过通道边界
# 把通道 0 的 token 平均进相邻 1~2 个通道。故先在 token 混合关闭时量出泄漏范围，
# 再断言：只关 inter 门时泄漏范围之外的通道纹丝不动；inter 门打开时它们必须动。
m_sp.eval()
box = {}
m_sp.token_hgnn[-1].register_forward_pre_hook(lambda mo, i: box.update(i=i[0]))
m_sp.token_hgnn[-1].register_forward_hook(lambda mo, i, o: box.update(o=o))
x_pert = x.clone()
torch.manual_seed(1)
x_pert[:, :, 0] = torch.randn_like(x_pert[:, :, 0])
def _chan_diff(m_, key):
    with torch.no_grad():
        m_(x, None, None, None)
        base = box[key].clone()
        m_(x_pert, None, None, None)
        return (box[key] - base).abs().mean(dim=(0, 2, 3))
leak = _chan_diff(m_sp, 'i') > 1e-6        # 进入图混合之前的 token：主干自身的泄漏范围
outside = ~leak
assert outside.any() and leak[0], f'泄漏范围异常: {leak.tolist()}'
with torch.no_grad():
    for lay in m_sp.token_hgnn:
        lay.gate_inter.zero_()             # 只关跨通道
d_off = _chan_diff(m_sp, 'o')
assert d_off[outside].max() < 1e-6, f'关掉 inter 门后仍有跨通道影响: {d_off.tolist()}'
with torch.no_grad():
    for lay in m_sp.token_hgnn:
        lay.gate_inter.fill_(1.0)
d_on = _chan_diff(m_sp, 'o')
assert d_on[outside].max() > 1e-6, f'inter 门打开后跨通道通路未生效: {d_on.tolist()}'
print(f'   ok: 关掉 gate_inter 后泄漏范围外的 {int(outside.sum())} 个通道不变，打开后受影响'
      f'（趋势分支沿展平 patch 轴平滑的固有泄漏波及通道 {leak.nonzero().flatten().tolist()}）')

print('== 14. 自适应 v2：重采样 / 路由 / 重栅格化 ==')
V2 = dict(patch_mode='greedy_v2', patch_norm='resample', **DIRECT)
# (1) uniform 不受新键影响：adp_regrid / patch_norm 对等分挡逐位无关
torch.manual_seed(0)
m_u0 = Model(make_cfg(adp_regrid=False, **UNI)).eval()
torch.manual_seed(0)
m_u1 = Model(make_cfg(adp_regrid=True, patch_norm='resample', **UNI)).eval()
with torch.no_grad():
    _d = (m_u0(x, None, None, None) - m_u1(x, None, None, None)).abs().max().item()
assert _d == 0, f'uniform 挡受 adp_regrid/patch_norm 影响: {_d:.2e}'
print('   ok: uniform 挡对 adp_regrid / patch_norm 逐位不变')
# (2) 重采样表：ℓ=T0 恒等，每行和为 1，ℓ>T0 为平均池化
pdp2 = Model(make_cfg(**V2)).pdp
W = pdp2.resample_w
p16 = pdp2.steps_list.index(pdp2.resample_len)
assert torch.equal(W[p16, :, :pdp2.resample_len], torch.eye(pdp2.resample_len)), 'ℓ=T0 重采样非恒等'
assert torch.allclose(W.sum(-1), torch.ones_like(W.sum(-1))), '重采样权重行和 != 1'
p_long = pdp2.steps_list.index(max(pdp2.steps_list))
assert (W[p_long] > 0).sum(-1).min() >= 3, '长段重采样未做平均池化'
print(f'   ok: 重采样表 [P={W.size(0)}, T0={W.size(1)}, M={W.size(2)}]：ℓ=T0 恒等、行和为 1、长段平均池化')
# (3) 前向形状 / 每通道恰 N 段 / 推理确定性 / 无死参数 / 熵正则进 aux_loss
torch.manual_seed(0)
m2 = open_gates(Model(make_cfg(**V2)))
out = m2(x, None, None, None)
assert isinstance(out, tuple) and out[0].shape == y.shape and out[1].dim() == 0, '训练期应返回 (out, aux)'
assert out[1].item() >= 0, f'熵正则应 ≥ 0，得到 {out[1].item()}'
m2.zero_grad()
(out[0].sum() + out[1]).backward()
dead = [n for n, prm in m2.named_parameters() if prm.grad is None or torch.count_nonzero(prm.grad) == 0]
assert not dead, f'greedy_v2 留下死参数: {dead}'
assert all(m2.pdp.router[i].weight.grad.abs().sum() > 0 for i in (0, 2)), '路由器无梯度'
m2.eval()
o_eval = m2(x, None, None, None)
assert not isinstance(o_eval, tuple), 'eval 期不应返回 aux'
assert torch.equal(_pdp_out(m2, x)[0], _pdp_out(m2, x)[0]), 'greedy_v2 eval 非确定'
patches2, lens2, var2, moe2 = _pdp_out(m2, x)
C_, N_ = BASE_CFG.enc_in, 6
assert patches2.shape == (x.size(0), C_ * N_, pdp2.resample_len), f'patches 形状 {tuple(patches2.shape)}'
assert lens2.shape == (x.size(0), C_ * N_) and (lens2.reshape(-1, C_, N_).sum(-1) == BASE_CFG.seq_len).all()
assert torch.equal(var2, torch.arange(C_).repeat_interleave(N_))
print('   ok: greedy_v2 形状/每通道恰 N 段且和为 L/通道主序/推理确定/无死参数/路由有梯度/aux_loss 生效')
# (4) 选段随输入变化（逐样本），且同一样本不受同批伙伴影响
lens_a = _pdp_out(m2, x)[1]
lens_b = _pdp_out(m2, torch.randn_like(x))[1]
assert not torch.equal(lens_a, lens_b), 'greedy_v2 分段不随输入变化'
seg_alone = _pdp_out(m2, x[:1])[1]
seg_batch = _pdp_out(m2, torch.cat([x[:1], torch.randn(5, BASE_CFG.seq_len, BASE_CFG.enc_in)]))[1][:1]
assert torch.equal(seg_alone, seg_batch), 'greedy_v2 分段受同批伙伴影响'
print(f'   ok: 分段逐样本（出现 {lens_a.unique().numel()} 种长度），与同批伙伴无关')
# (5) 重栅格化：权重每行和为 1；等分段长下恒等；把 token 打乱到别的边界后槽位内容随之变
tok = torch.randn(x.size(0), C_ * N_, 5)
lens_uni = torch.full((C_ * N_,), BASE_CFG.seq_len // N_)
assert torch.allclose(m2._regrid(tok, lens_uni, x.size(0), C_, BASE_CFG.seq_len), tok), '等分下 regrid 非恒等'
lens_one = lens2[:1]
n_b = lens_one.reshape(1, C_, N_).to(tok.dtype)
e_i = n_b.cumsum(-1); s_i = e_i - n_b
seg = BASE_CFG.seq_len / N_
s_j = (torch.arange(N_).to(tok.dtype) * seg).view(1, 1, N_, 1)
w = (torch.minimum(e_i.unsqueeze(2), s_j + seg) - torch.maximum(s_i.unsqueeze(2), s_j)).clamp_min(0) / seg
assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1))), 'regrid 权重行和 != 1'
r1 = m2._regrid(tok[:1], lens_one, 1, C_, BASE_CFG.seq_len)
assert not torch.allclose(r1, tok[:1]), '非等分下 regrid 未改变槽位内容'
print('   ok: 重栅格化 等分恒等 / 权重行和为 1 / 非等分下按重叠混合')
# (6) fixed_random + resample + regrid（阶段 0 对照挡）可前向，patches 宽为 T0
m_fr = Model(make_cfg(patch_mode='fixed_random', patch_norm='resample', **DIRECT)).eval()
assert _pdp_out(m_fr, x)[0].shape[-1] == m_fr.pdp.resample_len
assert m_fr(x, None, None, None).shape == y.shape
print('   ok: fixed_random + resample + regrid 可前向（阶段 0 对照挡）')
# (7) 非法组合
for bad in (dict(patch_mode='greedy_v2', patch_norm='pad', **DIRECT),
            dict(patch_mode='greedy_fixed', patch_norm='resample', **DIRECT),
            dict(patch_mode='greedy_v2', patch_norm='resample', boundary_scope='batch', **DIRECT),
            dict(patch_mode='greedy_v2', patch_norm='resample', tokenize='recon', residual_mode='none')):
    try:
        Model(make_cfg(**bad)); raise AssertionError(f'未拦截非法组合 {bad}')
    except ValueError:
        pass
print('   ok: greedy_v2+pad / greedy_fixed+resample / v2+batch / v2+recon 均被拦截')

print('== 15. oracle-EM 分块 ==')
D4 = {**DIRECT, 'n_tokens': 4}     # DIRECT 自带 n_tokens=6，这里覆盖为 4
ORC = dict(patch_mode='oracle', patch_norm='resample', min_patch_len=12,
           patch_pool=[0, 12, 24, 36], adp_resample_len=24, **D4)
# (1) 划分枚举与暴力递归一致、含等分
torch.manual_seed(0)
m_o = Model(make_cfg(**ORC))
pdp_o = m_o.pdp


def _brute_parts(rem, n):
    if n == 0:
        return [[]] if rem == 0 else []
    return [[st] + rest for st in pdp_o.steps_list if st <= rem for rest in _brute_parts(rem - st, n - 1)]


bp = _brute_parts(BASE_CFG.seq_len, 4)
assert pdp_o.n_parts == len(bp) == 31, f'划分数 {pdp_o.n_parts} vs 暴力 {len(bp)}'
assert sorted(map(tuple, pdp_o.part_lens.tolist())) == sorted(map(tuple, bp))
assert pdp_o.part_lens[pdp_o.uniform_part].tolist() == [24, 24, 24, 24]
print(f'   ok: 枚举 {pdp_o.n_parts} 种划分与暴力一致，等分下标 {pdp_o.uniform_part}')
# (2) 等分划分下与 patch_mode=uniform n_tokens=4 逐位一致（把 uniform 模型的权重装进 oracle 模型）
torch.manual_seed(0)
m_u4 = Model(make_cfg(**{**UNI, 'n_tokens': 4})).eval()
missing, unexpected = m_o.load_state_dict(m_u4.state_dict(), strict=False)
assert not unexpected and all(k.startswith('router.') for k in missing), (missing, unexpected)
m_o.eval()
with torch.no_grad():
    part_u = torch.full((x.size(0),), pdp_o.uniform_part, dtype=torch.long)
    _d = (m_o._core(x, part_u)[0] - m_u4(x, None, None, None)).abs().max().item()
assert _d < 1e-5, f'等分划分下与 uniform 4×24 差 {_d:.2e}'
print(f'   ok: 等分划分前向与 patch_mode=uniform(4×24) 一致（max diff {_d:.1e}）')
# (2b) 同种子直接构造（不装权重）即逐位一致：路由器用分叉随机数初始化，不扰动其余层的初始化
torch.manual_seed(0)
m_u4b = Model(make_cfg(**{**UNI, 'n_tokens': 4})).eval()
torch.manual_seed(0)
m_ob = Model(make_cfg(**ORC)).eval()
sd_u, sd_o = m_u4b.state_dict(), m_ob.state_dict()
assert set(sd_o) - set(sd_u) == {k for k in sd_o if k.startswith('router.')} and set(sd_u) <= set(sd_o)
assert all(torch.equal(sd_u[k], sd_o[k]) for k in sd_u), '同种子构造下非路由器参数初始化不一致'
with torch.no_grad():
    _d = (m_ob._core(x, part_u)[0] - m_u4b(x, None, None, None)).abs().max().item()
assert _d == 0, f'同种子构造下等分划分前向与 uniform 4×24 差 {_d:.2e}'
print('   ok: 同种子构造即与 uniform 4×24 参数逐位相同、前向逐位相同（路由器随机数已隔离）')
# (2c) 训练一步后仍逐位相同：路由器恒选等分时，其交叉熵不得改动任何非路由器参数（曾经经 RevIN 仿射参数泄漏）
m_u4b.train(); m_ob.train()
opt_u = torch.optim.Adam(m_u4b.parameters(), lr=1e-2)
opt_o = torch.optim.Adam(m_ob.parameters(), lr=1e-2)
yt = y.detach()                                           # y 是第 1 节的模型输出，带着计算图
for _step in range(3):
    m_ob.oracle_step(x, yt, epoch=0)                      # 暖启动期：标签恒为等分
    torch.manual_seed(100 + _step)
    lu = ((m_u4b(x, None, None, None) - yt) ** 2).mean()
    torch.manual_seed(100 + _step)
    oo = m_ob(x, None, None, None)
    lo = ((oo[0] - yt) ** 2).mean() + oo[1]
    opt_u.zero_grad(); lu.backward(); opt_u.step()
    opt_o.zero_grad(); lo.backward(); opt_o.step()
sd_u, sd_o = m_u4b.state_dict(), m_ob.state_dict()
bad = [k for k in sd_u if not torch.equal(sd_u[k], sd_o[k])]
assert not bad, f'路由器恒选等分时训练 3 步后非路由器参数与 uniform 4×24 不同: {bad[:5]}'
m_u4b.eval(); m_ob.eval()
print('   ok: 暖启动（恒等分）训练 3 步后，非路由器参数与 uniform 4×24 逐位相同（交叉熵不泄漏到主干）')
# (3) oracle_step：暖启动恒等分；之后返回 [B, P] 且标签 = argmin
m_o.train()
assert m_o.oracle_step(x, y, epoch=0) is None and (m_o._oracle_part == pdp_o.uniform_part).all()
L_ = m_o.oracle_step(x, y, epoch=99)
assert L_.shape == (x.size(0), pdp_o.n_parts) and torch.equal(m_o._oracle_part, L_.argmin(1))
assert m_o.training, 'oracle_step 后应恢复训练态'
print(f'   ok: oracle_step 暖启动恒等分；E 步损失矩阵 {tuple(L_.shape)}，标签取 argmin')
# (4) 训练前向返回 (out, aux)，路由器有梯度，无死参数
out = m_o(x, None, None, None)
assert isinstance(out, tuple) and out[0].shape == y.shape
m_o.zero_grad(); (out[0].sum() + out[1]).backward()
assert m_o.router.out.weight.grad.abs().sum() > 0, '路由器无梯度'
dead = [n for n, prm in open_gates(m_o).named_parameters() if prm.grad is None]
assert not dead, f'oracle 挡留下无梯度参数: {dead}'
print('   ok: 训练期返回 (out, aux)，路由器交叉熵有梯度，无死参数')
# (4b) M 步划分来源：router 挡下模型用路由器 argmax（与 oracle 标签无关）；margin 把不显著的非等分标签压回等分
m_o.oracle_step(x, y, epoch=99)
box = {}
h = m_o.pdp.register_forward_pre_hook(lambda mod, args, kw: box.update(p=kw.get('part_idx')), with_kwargs=True)
m_o(x, None, None, None)
with torch.no_grad():
    exp_part = m_o.router(m_o._normalize(x)[0]).argmax(-1)
assert torch.equal(box['p'], exp_part), 'oracle_train_on=router 下 M 步未用路由器 argmax'
m_o.oracle_margin = 1.0
m_o.oracle_step(x, y, epoch=99)
assert (m_o._oracle_part == pdp_o.uniform_part).all(), 'margin=1 时标签应全为等分'
m_o.oracle_margin = 0.0
m_o.oracle_train_on = 'oracle'
m_o.oracle_step(x, y, epoch=99); tgt = m_o._oracle_part.clone()
m_o(x, None, None, None)
assert torch.equal(box['p'], tgt), 'oracle_train_on=oracle 下 M 步未用 E 步标签'
h.remove(); m_o.oracle_train_on = 'router'
print('   ok: oracle_train_on=router/oracle 的 M 步划分来源正确；oracle_margin 生效')
# (5) eval 走路由器：确定、不返回 aux；oracle_forward 累计一致率与直方图，且不清零测试期计数
m_o.eval()
o1 = m_o(x, None, None, None); o2 = m_o(x, None, None, None)
assert not isinstance(o1, tuple) and torch.equal(o1, o2)
assert m_o.router_hist.sum() == 2 * x.size(0)
_ = m_o.oracle_forward(x, y)
assert m_o.router_hist.sum() == 2 * x.size(0) and m_o.oracle_hist.sum() == x.size(0) and m_o._agree[1] == x.size(0)
d = m_o.diagnostics()
assert 'router_oracle_agree(%)' in d and 'oracle_uniform_share(%)' in d
print(f'   ok: eval 走路由器且确定；oracle_forward 计数正确（{d["router_oracle_agree(%)"]}% 一致）')
# (5b) 交叉验证 oracle：只用前半段选划分、record=False 不改计数；训练期标签统计已产出
n_before = (m_o.oracle_hist.sum().item(), m_o._agree[1])
o_cv = m_o.oracle_forward(x, y, select=slice(0, BASE_CFG.pred_len // 2), record=False)
assert o_cv.shape == y.shape and (m_o.oracle_hist.sum().item(), m_o._agree[1]) == n_before
assert 'train_label_nonuniform(%)' in d and 'train_router_agree(%)' in d, d.keys()
print(f'   ok: 交叉验证 oracle 可前向且不污染计数；训练期标签统计 {d["train_label_nonuniform(%)"]}% 非等分')
# (6) 非法组合
for bad in (dict(patch_mode='oracle', patch_norm='pad', min_patch_len=12, patch_pool=[0, 12, 24, 36], **D4),
            dict(patch_mode='oracle', patch_norm='resample', min_patch_len=12, patch_pool=[0, 24, 36], adp_resample_len=24, **D4),
            dict(patch_mode='oracle', patch_norm='resample', tokenize='recon', residual_mode='none')):
    try:
        Model(make_cfg(**bad)); raise AssertionError(f'未拦截非法组合 {bad}')
    except ValueError:
        pass
print('   ok: oracle+pad / 等分不在池内 / oracle+recon 均被拦截')

print('\nALL CHECKS PASSED')
