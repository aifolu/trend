# -*- coding: utf-8 -*-
"""
稳定像元检验：类型内项对"像元归属更替"的敏感性
=================================================
问题
----
主分解中，每一年的类别均值用的是当年土地覆盖归属。转换后的像元会以
低于类别均值的绿度进入新地类，把该类的均值拉低，从而低估"原地类内部
的变绿"。本脚本把这一影响量化出来。

做法
----
1. 基准：各年按当年份额加权，得到各类别年均 NDVI 序列（与主分解同口径）。
2. 检验：只保留 2001 与 2025 年属于同一地类的像元（主导类相同），
   并把归属固定在 2001 年，逐年计算类别均值序列。
3. 两者相减，得到"归属更替造成的稀释量"。

输出
----
  控制台报告
  输出_归因/稳定像元检验.csv

运行
----
  python code/06_stable_pixel_check.py
"""

import os
import numpy as np
import pandas as pd

import config as cfg

RES = cfg.GRID_ATTRIBUTION
OUT = cfg.STABLE_PIXEL_CHECK

NAMES = ['forest', 'grass', 'crop', 'impervious']
LABELS = ['林地', '草地', '耕地', '不透水面']

# 与主分解一致的研究区掩膜
MIN_VALID_YEARS = 18
MIN_FRAC = 0.05


def ols_slope(a, t):
    """一维 OLS 斜率"""
    a = np.asarray(a, float)
    t = np.asarray(t, float)
    t = t - t.mean()
    return float((t * (a - a.mean())).sum() / (t * t).sum())


def class_series(nd, fy, valid, mask):
    """
    给定掩膜，返回 (年份, 4) 的份额与类别均值。
    份额按有效 NDVI 像元归一化，保证求和为 1。
    """
    f = np.where(valid[:, None, :, :] & mask[None, None, :, :], fy, np.nan)
    nd0 = np.nan_to_num(nd)
    num = np.nansum(f * nd0[:, None, :, :], axis=(2, 3))   # Σ 份额×NDVI
    den = np.nansum(f, axis=(2, 3))                        # Σ 份额
    w = den / den.sum(1, keepdims=True)
    m = num / den
    return w, m


def within_term(w, m, years):
    """β_within = Σ_k w̄_k · β_k"""
    bk = np.array([ols_slope(m[:, k], years) for k in range(4)])
    return float((w.mean(0) * bk).sum()), bk


def main():
    z = np.load(RES)
    nd = z['ndvi_year'].astype(np.float64)      # (25, H, W)
    fy = z['frac_year'].astype(np.float64)      # (25, 4, H, W)
    T = nd.shape[0]
    years = np.arange(2001, 2001 + T, dtype=float)
    valid = np.isfinite(nd)

    study = (valid.sum(0) >= MIN_VALID_YEARS) & \
            (fy[0].sum(0) > MIN_FRAC) & (fy[-1].sum(0) > MIN_FRAC)

    dom0 = np.argmax(fy[0], axis=0)
    dom1 = np.argmax(fy[-1], axis=0)
    stable = study & (dom0 == dom1)

    print('=' * 68)
    print('稳定像元检验')
    print('=' * 68)
    print(f'  研究区像元      {study.sum():,}')
    print(f'  稳定像元        {stable.sum():,}  ({100 * stable.sum() / study.sum():.1f}%)')

    w_base, m_base = class_series(nd, fy, valid, study)
    within_base, bk_base = within_term(w_base, m_base, years)

    # 【主结果】份额按分数覆盖加权，与主分解完全同口径：
    #   把份额图冻结在 2001 年，只在稳定像元上求类别序列。
    fy_fixed = np.repeat(fy[[0]], T, axis=0)
    w_stab, m_stab = class_series(nd, fy_fixed, valid, stable)
    within_stab, bk_stab = within_term(w_stab, m_stab, years)

    # 【变体】二值主导类归属。仅用于说明结论不依赖权重口径，
    #   不作为正文数字——它与主分解的加权方式不同。
    masks = [stable & (dom0 == k) for k in range(4)]
    w_bin = np.array([masks[k].sum() / stable.sum() for k in range(4)])
    m_bin = np.zeros((T, 4))
    for k in range(4):
        sel = masks[k]
        m_bin[:, k] = np.nansum(np.where(sel[None], nd, 0.0), axis=(1, 2)) / sel.sum()
    bk_bin = np.array([ols_slope(m_bin[:, k], years) for k in range(4)])
    within_bin = float((w_bin * bk_bin).sum())

    print()
    print(f'  基准类型内项           {within_base:.5f} a⁻¹   [表 1: 0.00280]')
    print(f'  稳定像元类型内项       {within_stab:.5f} a⁻¹   <- 正文采用')
    print(f'  相对变化               {100 * (within_stab - within_base) / within_base:+.2f}%')
    print(f'  变体（二值归属）       {within_bin:.5f} a⁻¹   '
          f'({100 * (within_bin - within_base) / within_base:+.2f}%)')
    print()
    print(f'  {"地类":<8}{"基准":>10}{"稳定像元":>12}{"差":>10}{"权重":>10}')
    print('  ' + '-' * 52)
    rows = []
    for k in range(4):
        d = bk_stab[k] - bk_base[k]
        print(f'  {LABELS[k]:<8}{bk_base[k]*1e3:>10.2f}{bk_stab[k]*1e3:>12.2f}'
              f'{d*1e3:>+10.2f}{w_stab.mean(0)[k]:>10.3f}')
        rows.append(dict(land_cover=NAMES[k], beta_base=bk_base[k],
                         beta_stable=bk_stab[k], diff=bk_stab[k] - bk_base[k],
                         weight_stable=w_stab.mean(0)[k],
                         beta_binary=bk_bin[k]))
    print('  ' + '-' * 52)
    print(f'  {"合计":<8}{within_base*1e3:>10.2f}{within_stab*1e3:>12.2f}'
          f'{(within_stab-within_base)*1e3:>+10.2f}{1.0:>10.3f}')
    print('  （地类列为 10⁻³ a⁻¹，合计行为加权后的类型内项）')

    df = pd.DataFrame(rows)
    df.loc[len(df)] = dict(land_cover='TOTAL', beta_base=within_base,
                           beta_stable=within_stab,
                           diff=within_stab - within_base, weight_stable=1.0,
                           beta_binary=within_bin)
    df['stable_pixel_share'] = stable.sum() / study.sum()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False, encoding='utf-8-sig')
    print(f'\n[已保存] {OUT}')


if __name__ == '__main__':
    main()
