"""PRISM v2 信息时钟冒烟检查（CPU 可跑，无需数据集）。

用法：python scripts/PRISM/check_clock.py
断言清单见计划：质量守恒 / 均匀退化 / 平移不变 / 梯度 / 确定性 / batch 无关 /
token 数恒定 / 参数预算 / 小样本过拟合。
"""
import copy
import sys

import torch

sys.path.insert(0, '.')
from utils.config import load_config                       # noqa: E402
from models.PRISM import Model                             # noqa: E402
from layers.PRISM_layers import InfoClockPatchEmbedding    # noqa: E402

torch.manual_seed(2021)
ARGS = ['-c', 'configs/PRISM/smoke.yaml']


def make(overrides=()):
    cfg = load_config(ARGS + list(overrides))
    return cfg, Model(cfg).eval()


def main():
    cfg, model = make()
    B, C, L, K = 4, cfg.enc_in, cfg.seq_len, cfg.n_patterns
    N = cfg.clock_n_patch
    x = torch.randn(B, L, C)
    mark = torch.zeros(B, L, 4)

    # --- 内部不变量：直接检查 InfoClockPatchEmbedding ---
    embed = model.backbone.embed
    season = torch.randn(B, C, L)
    dic = model.router.dictionary.detach()

    a_raw = embed._density(season.reshape(B * C, 1, L),
                           embed.norm(embed.frontend(season.reshape(B * C, 1, L)).transpose(1, 2)),
                           dic)
    p = (1 - embed.floor) * torch.softmax(a_raw, dim=-1) + embed.floor / L
    m, u = embed._soft_bins(embed.N * p)
    assert torch.allclose(m.sum(-1), torch.ones_like(m.sum(-1)), atol=1e-4), '每箱质量应为 1'
    assert torch.allclose(m.sum(1), embed.N * p, atol=1e-4), '每步质量应等于 α'
    assert (m >= 0).all() and (u >= 0).all() and (u <= 1).all()
    print(f'[ok] 质量守恒  m∈[{m.min():.4f},{m.max():.4f}]')

    # --- 均匀退化：uniform 模式 & β 零初始化的第 0 步 adaptive 都应严格等分 ---
    w = L // N
    m_uni, _ = embed._soft_bins(torch.full((1, L), N / L))
    target = torch.zeros(1, N, L)
    for i in range(N):
        target[0, i, i * w:(i + 1) * w] = 1.0
    assert torch.allclose(m_uni.sum(-1), torch.ones(1, N), atol=1e-5)
    assert (m_uni * (1 - target)).abs().max() < 1e-4, 'α 常数时应严格等分'
    assert a_raw.abs().max() < 1e-6, 'β 零初始化时 a_raw 应为 0（初始即均匀）'
    print('[ok] 均匀退化（uniform 目标 + 零初始化起点）')

    # --- 平移不变：a_raw 加常数不改变 p ---
    p2 = (1 - embed.floor) * torch.softmax(a_raw + 3.7, dim=-1) + embed.floor / L
    assert torch.allclose(p, p2, atol=1e-6), 'softmax 重参数化应对常数平移不变'
    print('[ok] 重参数化平移不变')

    # --- 前向形状 / token 数 / 参数预算 ---
    y = model(x, mark, None, None)
    assert y.shape == (B, cfg.pred_len, C), y.shape
    tok, _ = embed(season, dic)
    assert tok.shape == (B * C, N, cfg.d_model)
    n_params = sum(pa.numel() for pa in model.parameters())
    assert n_params < 1_000_000, n_params
    print(f'[ok] 形状 {tuple(y.shape)}  tokens {N}  参数量 {n_params:,}')

    # --- 确定性 ---
    assert torch.equal(model(x, mark, None, None), y), 'eval 两次前向应逐位一致'
    print('[ok] 确定性')

    # --- batch 无关（eval；train 下 encoder BatchNorm 天然耦合，不检） ---
    y1 = model(x[:1], mark[:1], None, None)
    assert torch.allclose(y1, y[:1], atol=1e-5), 'eval 下应 batch 无关'
    print('[ok] batch 无关')

    # --- 梯度阶段 1（初始状态）：β 与内容前端从第一步就有梯度 ---
    # 注：β/res_gate 零初始化时，字典与 log_temp 的梯度恰为 0（乘子为零），
    # 这是刻意的"从均匀基线出发"设计——两个门自身有梯度，动起来后路径即打通。
    mt = copy.deepcopy(model).train()
    out = mt(x, mark, None, None)
    out = out[0] if isinstance(out, tuple) else out
    out.pow(2).mean().backward()
    for name in ['backbone.embed.beta', 'backbone.embed.frontend.0.weight']:
        g = dict(mt.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f'{name} 梯度异常'
    assert all(torch.isfinite(pa.grad).all() for pa in mt.parameters() if pa.grad is not None)
    print('[ok] 梯度阶段 1（初始：β/前端非零有限，无 NaN）')

    # --- 梯度阶段 2（预热状态）：β/res_gate 打开后，字典与温度获得梯度 ---
    mw = copy.deepcopy(model).train()
    with torch.no_grad():
        mw.backbone.embed.beta.uniform_(-0.5, 0.5)
        mw.srm.res_gate.fill_(0.1)
    out = mw(x, mark, None, None)
    out = out[0] if isinstance(out, tuple) else out
    out.pow(2).mean().backward()
    for name in ['router.dictionary', 'backbone.embed.log_temp']:
        g = dict(mw.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f'{name} 梯度异常'
    print('[ok] 梯度阶段 2（预热后：字典/温度非零有限）')

    # --- 三挡模式 + 密度源均可前向 ---
    for ov in (['clock_mode=uniform'], ['clock_mode=reweight_only'],
               ['clock_density=mlp'], ['clock_detach_dict=true'], ['clock_reverse=false']):
        _, mv = make(ov)
        assert mv(x, mark, None, None).shape == (B, cfg.pred_len, C), ov
    print('[ok] 消融开关全部可前向')

    # --- 小样本过拟合（机制端到端可学习性） ---
    _, mo = make()
    mo.train()
    xs, ys_t = torch.randn(2, L, C), torch.randn(2, cfg.pred_len, C)
    opt = torch.optim.Adam(mo.parameters(), lr=1e-3)
    for step in range(300):
        out = mo(xs, torch.zeros(2, L, 4), None, None)
        out = out[0] if isinstance(out, tuple) else out
        loss = (out - ys_t).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.05, f'过拟合失败 loss={loss.item():.4f}'
    print(f'[ok] 2 样本过拟合 → {loss.item():.5f}')

    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
