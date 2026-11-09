# -*- coding: utf-8 -*-
"""
区域NDVI趋势的归因分析（不含预测）
=====================================================
沿用论文 §3.5 的精确分解恒等式:

    NDVI_reg(y) = Σ_k w_k(y)·NDVI_k(y),   Σ_k w_k(y) = 1
    β_reg = β_within + β_struct + β_int
    β_within = Σ_k w̄_k·β_k          —— 类型内变绿（份额加权的各地类绿度趋势）
    β_struct = Σ_k NDVĪ_k·γ_k       —— 土地利用结构转换（绿度加权的份额趋势）
    β_int    = Σ_k Σ_y (y-ȳ)(w_k-w̄_k)(NDVI_k-NDVĪ_k) / Σ_y (y-ȳ)²

三个输出层次:
  1) 全期（2001-2025）分解 —— 复现正文 Table 1
  2) 分期分解（5个锚点区间）—— 看 β_within / β_struct 随政策阶段的变化
  3) 分期 × 分类型结构贡献 —— 中心化到区域均值，正负可解释
另附:
  - Bennett（中点权重）加性分解，用于检验 β_struct 对估计量的稳健性
  - 末段留一年法（leave-one-year-out），检验 2020-2025 结论的稳定性

数据:
  各类型/区域月均NDVI: data/class_ndvi_series_2001_2025.csv
  面积（锚点年）      : data/anchor_area_shares_percent.csv
运行:
  python code/02_regional_decomposition.py
"""

import os
import numpy as np
import pandas as pd

import config as cfg

CACHE = cfg.CLASS_SERIES
AREA_CSV = cfg.ANCHOR_SHARES
OUT_DIR = cfg.OUT

KEYS = ['forest', 'grass', 'crop', 'impervious']
LABELS = {'forest': '森林', 'grass': '草地', 'crop': '农田',
          'impervious': '不透水面', 'overall': '区域总体'}
YEARS = list(range(2001, 2026))


def load_annual():
    """月度缓存 -> 年均矩阵（只保留12个月齐全的年份）"""
    m = pd.read_csv(CACHE, encoding='utf-8-sig')
    m['year'] = m['date'].str[:4].astype(int)
    m['month'] = m['date'].str[5:7].astype(int)
    cnt = m.groupby(['type', 'year'])['month'].count()
    full = cnt[cnt == 12].reset_index()[['type', 'year']]
    m = m.merge(full, on=['type', 'year'])
    ann = m.groupby(['type', 'year'])['mean'].mean().unstack(0).sort_index()
    return ann


def load_shares():
    """锚点年面积 -> 归一化面积份额（四类合计为1）"""
    A = pd.read_csv(AREA_CSV).set_index('year')[KEYS].astype(float).sort_index()
    W = A.div(A.sum(axis=1), axis=0)
    return A, W


def anchor_of(year):
    """分段锚定: 2001-2004→2001, 2005-2009→2005, ..., 2025→2025"""
    return 2001 if year <= 2004 else 2005 + ((year - 2005) // 5) * 5


def ols(x, z):
    """OLS 斜率"""
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    xm, zm = x.mean(), z.mean()
    return float(((x - xm) * (z - zm)).sum() / ((x - xm) ** 2).sum())


def decompose(years, W, N, keys=KEYS):
    """
    精确趋势分解（与论文 §3.5 同一估计量）
    years: 年份序列; W/N: (n_years, n_types) 份额矩阵与绿度矩阵
    返回: 总量、三项、分类型明细（含中心化结构贡献）
    """
    y = np.asarray(years, float)
    yc = y - y.mean()
    D = (yc ** 2).sum()

    reg = (W * N).sum(axis=1)          # Σ w_k·NVNI_k
    beta_reg = ols(y, reg)
    ndvi_bar_reg = float(reg.mean())

    rows = []
    b_within = b_struct = b_int = 0.0
    for i, k in enumerate(keys):
        w, n = W[:, i], N[:, i]
        wbar, nbar = w.mean(), n.mean()
        b_k = ols(y, n)                # 类型内绿度趋势 β_k
        g_k = ols(y, w)                # 面积份额趋势 γ_k
        i_k = float((yc * (w - wbar) * (n - nbar)).sum() / D)
        b_within += wbar * b_k
        b_struct += nbar * g_k
        b_int += i_k
        rows.append({
            'type': k, 'w_bar': wbar, 'ndvi_bar': nbar,
            'beta_k': b_k, 'gamma_k': g_k,
            'within_contrib': wbar * b_k,
            'struct_contrib': nbar * g_k,
            # 中心化: 扣除区域均值后各类型结构贡献可单独解释（Σγ_k=0，总和不变）
            'struct_centered': (nbar - ndvi_bar_reg) * g_k,
            'int_contrib': i_k,
        })

    detail = pd.DataFrame(rows)
    return {
        'beta_reg': beta_reg,
        'ndvi_bar_reg': ndvi_bar_reg,
        'beta_within': b_within,
        'beta_struct': b_struct,
        'beta_int': b_int,
        'within_share': 100 * b_within / beta_reg if beta_reg else np.nan,
        'struct_share': 100 * b_struct / beta_reg if beta_reg else np.nan,
        'detail': detail,
        'reg_series': reg,
    }


def bennett(a, b, ann, W, keys=KEYS):
    """
    Bennett（中点权重）加性分解，检验对估计量的稳健性:
      ΔNDVI_reg = Σ_k w̄_k·ΔNDVI_k + Σ_k NDVĪ_k·Δw_k   （无交互项，精确）
    返回年均速率（Δ/年数）
    """
    wa, wb = W.loc[a].values, W.loc[b].values
    na, nb = ann.loc[a, keys].values, ann.loc[b, keys].values
    wm, nm = (wa + wb) / 2, (na + nb) / 2
    yrs = b - a
    within = float((wm * (nb - na)).sum()) / yrs
    struct = float((nm * (wb - wa)).sum()) / yrs
    return within, struct


def stage_table(ann, W, keys=KEYS):
    """逐锚点区间做论文口径分解"""
    anchors = list(W.index)
    rows = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        yrs = [y for y in range(a, b + 1) if y in ann.index]
        N = ann.loc[yrs, keys].values
        Ws = np.array([W.loc[anchor_of(y)].values for y in yrs])
        res = decompose(yrs, Ws, N, keys)
        bw_bennett, bs_bennett = bennett(a, b, ann, W, keys)
        rows.append({
            'period': f'{a}–{b}',
            'years': b - a,
            'beta_reg': res['beta_reg'],
            'beta_within': res['beta_within'],
            'beta_struct': res['beta_struct'],
            'beta_int': res['beta_int'],
            'within_share_pct': res['within_share'],
            'struct_share_pct': res['struct_share'],
            'within_bennett': bw_bennett,
            'struct_bennett': bs_bennett,
            'ndvi_bar_reg': res['ndvi_bar_reg'],
        })
    return pd.DataFrame(rows)


def stage_by_type(ann, W, keys=KEYS):
    """分期 × 分类型的中心化结构贡献"""
    anchors = list(W.index)
    out = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        yrs = [y for y in range(a, b + 1) if y in ann.index]
        N = ann.loc[yrs, keys].values
        Ws = np.array([W.loc[anchor_of(y)].values for y in yrs])
        res = decompose(yrs, Ws, N, keys)
        d = res['detail'].copy()
        d.insert(0, 'period', f'{a}–{b}')
        out.append(d)
    return pd.concat(out, ignore_index=True)


def loo_last_stage(ann, W, keys=KEYS):
    """末段（2020-2025）留一年法，检验 within 项符号是否稳定"""
    a, b = list(W.index)[-2], list(W.index)[-1]
    yrs = [y for y in range(a, b + 1) if y in ann.index]
    rows = []
    for drop in [None] + yrs:
        yy = [y for y in yrs if y != drop]
        N = ann.loc[yy, keys].values
        Ws = np.array([W.loc[anchor_of(y)].values for y in yy])
        res = decompose(yy, Ws, N, keys)
        rows.append({'dropped_year': drop if drop else 'none',
                     'beta_reg': res['beta_reg'],
                     'beta_within': res['beta_within'],
                     'beta_struct': res['beta_struct']})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ann = load_annual()
    A, W = load_shares()
    anchors = list(W.index)
    print('锚点年份额 (%):')
    print((W * 100).round(2).to_string())

    yrs = [y for y in YEARS if y in ann.index]
    N = ann.loc[yrs, KEYS].values
    Ws = np.array([W.loc[anchor_of(y)].values for y in yrs])
    full = decompose(yrs, Ws, N)

    print('\n' + '=' * 72)
    print(f'全期分解 2001–2025  (n={len(yrs)})')
    print('=' * 72)
    print(f"  β_reg    = {full['beta_reg']:.5f} a⁻¹")
    print(f"  β_within = {full['beta_within']:.5f} a⁻¹  ({full['within_share']:.1f}%)")
    print(f"  β_struct = {full['beta_struct']:.5f} a⁻¹  ({full['struct_share']:.1f}%)")
    print(f"  β_int    = {full['beta_int']:+.2e} a⁻¹")
    print(f"  恒等式校验 = {full['beta_within'] + full['beta_struct'] + full['beta_int'] - full['beta_reg']:+.2e}")
    d = full['detail'].copy()
    d['type'] = d['type'].map(LABELS)
    print('\n  分类型明细:')
    print(d[['type', 'w_bar', 'ndvi_bar', 'beta_k', 'gamma_k',
             'within_contrib', 'struct_centered']]
          .to_string(index=False, float_format=lambda v: f'{v:.5f}'))

    st = stage_table(ann, W)
    print('\n' + '=' * 72)
    print('分期分解')
    print('=' * 72)
    print(st.to_string(index=False, float_format=lambda v: f'{v:.5f}'))

    sbt = stage_by_type(ann, W)
    piv = sbt.pivot(index='type', columns='period', values='struct_centered')
    piv = piv.reindex(KEYS)
    piv.index = [LABELS[k] for k in KEYS]
    print('\n分期 × 分类型 中心化结构贡献 (×10⁻³ a⁻¹):')
    print((piv * 1e3).round(3).to_string())

    loo = loo_last_stage(ann, W)
    print('\n末段 2020–2025 留一年法:')
    print(loo.to_string(index=False, float_format=lambda v: f'{v:.5f}'))

    xlsx = cfg.DECOMPOSITION_XLSX
    with pd.ExcelWriter(xlsx, engine='openpyxl') as w:
        ann.to_excel(w, sheet_name='年均NDVI')
        (W * 100).round(3).to_excel(w, sheet_name='锚点年份额%')
        full['detail'].to_excel(w, sheet_name='全期_分类型', index=False)
        pd.DataFrame([{k: v for k, v in full.items()
                       if k not in ('detail', 'reg_series')}]).to_excel(
            w, sheet_name='全期_汇总', index=False)
        st.to_excel(w, sheet_name='分期_汇总', index=False)
        sbt.to_excel(w, sheet_name='分期_分类型', index=False)
        loo.to_excel(w, sheet_name='稳健性_末段LOO', index=False)

    st.to_csv(cfg.PERIOD_SUMMARY, index=False, encoding='utf-8-sig')
    piv.to_csv(cfg.PERIOD_BY_CLASS, encoding='utf-8-sig')
    print(f'\n[已保存] {xlsx}')
    print(f'[已保存] {cfg.PERIOD_SUMMARY}, {cfg.PERIOD_BY_CLASS}')


if __name__ == '__main__':
    main()
