# -*- coding: utf-8 -*-
"""
网格尺度归因制图（聚合守恒版）
=========================================================
为什么要这样定义
----------------
指数分解的局地版本（窗口内用局地类别均值）不满足"局地平均 = 区域值"，
因为 β_within/β_struct 是 w̄·β 的乘积，窗口内类别均值随窗口构成变化。
逐事件跳变法又有分辨率损失：1 km 多数类翻转只捕捉到部分转换。

本脚本采用第三种定义，两个性质同时成立：

    structural_p = Σ_k Δw_k(p) · NDVĪ_k^G     （构成变化 × 全区类别水平）
    within_p     = Σ_k w̄_k(p) · ΔNDVI_k^G     （局地构成权重 × 全区类别趋势）

其中 ^G 表示全区（study mask 面积加权）的类别量：
    NDVĪ_k^G = 全区第 k 类年均 NDVI 的时期均值
    ΔNDVI_k^G = 该序列 OLS 斜率 × 24
    Δw_k(p) = 像元 p 的第 k 类面积比例 OLS 斜率 × 24

性质:
  1) 全区平均严格等于 Table 1 的分解（因为求平均与 Σ 可交换，斜率对均值线性）；
  2) 逐像元可解释：转换像元的结构贡献 = 面积变化 × 区域类别绿度差；
  3) 残差 residual_p = ΔNDVI_p − within_p − structural_p 逐像元闭合，
     全区平均 ≈ 0（交互项）。

单位: 结果除以 24 转为 a⁻¹，绘图时乘 1000。
运行: python code/03_grid_decomposition.py
"""

import os
import numpy as np
from math import erf, sqrt

import config as cfg

RES = cfg.GRID_ATTRIBUTION
OUT = cfg.GRID_CONTRIBUTION
YEARS = list(range(2001, 2026))
NY = len(YEARS)
KEYS = ['forest', 'grass', 'crop', 'impervious']


def slope_p(a):
    """(T, ...) 沿第0轴的 OLS 斜率，NaN 安全"""
    v = np.isfinite(a)
    x = np.where(v, a, 0.0)
    n = v.sum(0)
    t = np.asarray(YEARS, float).reshape((NY,) + (1,) * (a.ndim - 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        mt = (t * v).sum(0) / np.maximum(n, 1)
        mx = x.sum(0) / np.maximum(n, 1)
        dy = (t - mt) * v
        sxx = (dy * dy).sum(0)
        b = (dy * (x - mx)).sum(0) / sxx
        pred = mx + b * (t - mt)
        sse = (v * (x - pred) ** 2).sum(0)
        se = np.sqrt(np.maximum(sse / np.maximum(n - 2, 1) / sxx, 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            tval = np.abs(b / se)
    return b, tval


def pval(tval):
    z = np.nan_to_num(tval, nan=0.0)
    return np.where(np.isfinite(tval), 1 - np.vectorize(lambda q: erf(q / sqrt(2)))(z), np.nan)


def main():
    z = np.load(RES)
    nd = z['ndvi_year'].astype(np.float64)      # (25,H,W)
    fy = z['frac_year'].astype(np.float64)      # (25,4,H,W)
    T, H, W = nd.shape
    study = (np.isfinite(nd).sum(0) >= 20) & (fy[0].sum(0) > 0.05) & (fy[-1].sum(0) > 0.05)

    valid = np.isfinite(nd)
    f = np.where(valid[:, None, :, :] & study[None, None], fy, np.nan)   # 只保留研究区且NDVI有效
    nd0 = np.nan_to_num(nd)
    num = np.nansum(f * nd0[:, None, :, :], axis=(2, 3))                 # (25,4)
    den = np.nansum(f, axis=(2, 3))
    m_glob = num / den                                                    # (25,4) 类别年均NDVI
    beta_glob, _ = slope_p(m_glob)                                        # (4,)
    ndvi_bar = m_glob.mean(0)                                             # (4,)

    print('全区类别序列（对照正文）:')
    for k, nm in enumerate(KEYS):
        print(f'  {nm:10s} NDVĪ={ndvi_bar[k]:.4f}  β={beta_glob[k]:.5f} a⁻¹')

    fk = np.where(study[None, None], fy, np.nan)
    gamma_p, _ = slope_p(fk)              # (4,H,W) 各像元各类份额趋势
    wbar_p = np.nanmean(fk, axis=0)       # (4,H,W)
    beta_px, tval = slope_p(nd)           # (H,W)
    p_px = pval(tval)

    structural = (gamma_p * ndvi_bar[:, None, None]).sum(0)   # NDVI/年
    within = (wbar_p * beta_glob[:, None, None]).sum(0)
    total = beta_px
    resid = total - within - structural

    m = study & np.isfinite(total) & np.isfinite(structural)
    print('\n' + '=' * 64)
    print('聚合守恒检查（a⁻¹）')
    print('=' * 64)
    print(f'  有效像元 {int(m.sum())} ({100*m.mean():.1f}%)')
    print(f'  区域总趋势      {np.nanmean(total[m]):+.5f}   [正文 0.00316]')
    print(f'  类型内 within   {np.nanmean(within[m]):+.5f}   [正文 0.00280]')
    print(f'  结构 struct     {np.nanmean(structural[m]):+.5f}   [正文 0.00037]')
    print(f'  残差 residual   {np.nanmean(resid[m]):+.5f}（应≈0）')
    print(f'  结构占比        {100*np.nanmean(structural[m])/np.nanmean(total[m]):.1f}%   [正文 11.6%]')

    share = np.abs(structural) / np.maximum(np.abs(within) + np.abs(structural), 1e-9)
    sig = p_px < 0.05
    cat = np.zeros((H, W), np.int8)
    cat[study] = 1
    cat[study & sig & (total > 0)] = 2
    cat[study & sig & (total < 0)] = 3
    cat[study & sig & (total > 0) & (share > 0.3)] = 4
    labels = ['非研究区', '无显著变化', '显著变绿·类型内主导', '显著变褐', '显著变绿·较高中转换占比']
    for c in range(5):
        print(f'  {labels[c]}: {int((cat == c).sum())} px ({100*np.mean(cat == c):.1f}%)')

    np.savez_compressed(OUT, total=total, within=within, structural=structural,
                        resid=resid, share=share, cat=cat, study=study,
                        p=p_px, ndvi_bar=ndvi_bar, beta_glob=beta_glob)
    print('\n[已保存]', OUT)


if __name__ == '__main__':
    main()
