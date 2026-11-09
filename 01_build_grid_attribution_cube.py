# -*- coding: utf-8 -*-
"""
网格尺度归因制图：局地窗口指数分解
=========================================================
把论文 §3.5 的恒等式逐像元化：

    NDVI_reg(p,y) = Σ_k w_k(p,y)·NDVI_k(p,y)
    β_within(p) = Σ_k w̄_k(p)·β_k(p)
    β_struct(p) = Σ_k NDVĪ_k(p)·γ_k(p)

w_k(p,y) 是以 p 为中心、半径 R 的窗口内第 k 类的面积份额；
NDVI_k(p,y) 是该窗口内第 k 类的均值绿度；β_k、γ_k 为二者对年份的 OLS 斜率。
窗口足够大时，全区面积加权平均应回到区域整体分解（0.00280 / 0.00037）。

与区域分解的区别（写论文时必须说明）：
  区域分解 = 全区分成 4 个总量序列；局地分解 = 每个像元看自己邻域的构成。
  前者是账目，后者是制图；两者符号与空间格局一致，数值不完全相同。

数据:
  NDVI 与土地利用栅格路径见 config.py
运行:
  python code/01_build_grid_attribution_cube.py
输出:
  输出_归因/网格归因.npz  +  诊断信息
"""

import os
import re
import glob
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter

import config as cfg

NDVI_OVERALL = cfg.NDVI_MONTHLY
NDVI_CLASS_DIR = cfg.NDVI_CLASS
CLASS_FOLDERS = cfg.CLASS_FOLDERS
LC_DIR = cfg.LC_DIR
LC_CODES = {'crop': (1,), 'forest': (2, 3), 'grass': (4,), 'impervious': (8,)}
KEYS = ['forest', 'grass', 'crop', 'impervious']
ANCHOR_RASTER = {2001: 2000, 2002: 2000, 2003: 2000, 2004: 2000,
                 2005: 2005, 2006: 2005, 2007: 2005, 2008: 2005, 2009: 2005,
                 2010: 2010, 2011: 2010, 2012: 2010, 2013: 2010, 2014: 2010,
                 2015: 2015, 2016: 2015, 2017: 2015, 2018: 2015, 2019: 2015,
                 2020: 2020, 2021: 2020, 2022: 2020, 2023: 2020, 2024: 2020,
                 2025: 2020}
YEARS = list(range(2001, 2026))
BLOCK = 28          # 30 m -> ~840 m 先做整数块聚合（28×0.000316°≈0.00885°）
WINDOW_KM = 10      # 局地窗口半径（km），1 像元≈1 km
OUT_DIR = cfg.OUT
CACHE = cfg.GRID_CUBE_CACHE


def _idx(path):
    m = re.search(r'(\d{4})[-_](\d{1,2})(?!\d)', os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _read(p):
    with rasterio.open(p) as s:
        return s.read(1).astype(np.float32)


def load_monthly_cube():
    """(300, H, W) 月 NDVI，2025 由四类掩膜合并补齐"""
    files = {}
    for f in glob.glob(os.path.join(NDVI_OVERALL, '*.tif')):
        k = _idx(f)
        if k:
            files[k] = f
    cls_files = {k: {} for k in CLASS_FOLDERS}
    for k, d in CLASS_FOLDERS.items():
        for f in glob.glob(os.path.join(NDVI_CLASS_DIR, d, '*.tif')):
            kk = _idx(f)
            if kk:
                cls_files[k].setdefault(kk, f)

    ref = _read(files[(2001, 1)])
    H, W = ref.shape
    cube = np.full((len(YEARS), 12, H, W), np.nan, np.float32)
    for yi, y in enumerate(YEARS):
        for m in range(1, 13):
            if (y, m) in files:
                a = _read(files[(y, m)])
            else:
                a = np.full((H, W), np.nan, np.float32)
                for k in CLASS_FOLDERS:
                    f = cls_files[k].get((y, m))
                    if not f:
                        continue
                    b = _read(f)
                    ok = np.isnan(a) & np.isfinite(b) & (b > -3e38)
                    a[ok] = b[ok]
            a[a <= -3e38] = np.nan
            a[(a < -1) | (a > 1)] = np.nan
            cube[yi, m - 1] = a
        print(f'  NDVI {y} 完成', flush=True)
    return np.nanmean(cube, axis=1), cube.shape[2:]


def ndvi_profile():
    with rasterio.open(glob.glob(os.path.join(NDVI_OVERALL, '*.tif'))[0]) as s:
        return s.bounds, s.transform, s.crs, s.width, s.height


def lc_fractions(year, dst_shape, dst_transform, dst_crs, bounds):
    """锚点年 30m 土地覆盖 -> 1km 四类面积比例 (4, H, W)"""
    with rasterio.open(os.path.join(LC_DIR, f'extracted_{year}.tif')) as s:
        win = from_bounds(*bounds, transform=s.transform).round_offsets().round_lengths()
        a = s.read(1, window=win)
        wtr = s.window_transform(win)
        src_crs = s.crs
    h, w = a.shape
    hb, wb = h // BLOCK, w // BLOCK
    a = a[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK)
    out = np.zeros((4, dst_shape[0], dst_shape[1]), np.float32)
    src_tr = wtr * Affine.scale(BLOCK)
    for i, k in enumerate(KEYS):
        frac = np.isin(a, LC_CODES[k]).sum(axis=(1, 3)).astype(np.float32) / (BLOCK * BLOCK)
        dst = np.zeros(dst_shape, np.float32)
        reproject(frac, dst, src_transform=src_tr, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=Resampling.bilinear)
        out[i] = np.clip(dst, 0, 1)
    return out


def slope(arr):
    """沿时间轴的 OLS 斜率，NaN 安全；arr (T,H,W)"""
    v = np.isfinite(arr)
    x = np.where(v, arr, 0.0)
    n = v.sum(0)
    t = np.asarray(YEARS, float)[:, None, None]
    my = (t * v).sum(0) / np.maximum(n, 1)
    mx = x.sum(0) / np.maximum(n, 1)
    dy = (t - my) * v
    num = (dy * (x - mx)).sum(0)
    den = (dy * dy).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def local_decompose(ndvi_year, frac_year, radius):
    """逐像元局地指数分解，返回 beta_within / beta_struct 等"""
    T, H, W = ndvi_year.shape
    s = 2 * radius + 1
    S = np.zeros((T, 4, H, W), np.float32)
    M = np.zeros((T, 4, H, W), np.float32)
    nd = np.nan_to_num(ndvi_year, nan=0.0)
    val = np.isfinite(ndvi_year).astype(np.float32)
    for yi in range(T):
        for k in range(4):
            f = frac_year[yi, k]
            S[yi, k] = uniform_filter(f, size=s, mode='constant', cval=0.0)
            num = uniform_filter(f * nd[yi], size=s, mode='constant', cval=0.0)
            den = uniform_filter(f * val[yi], size=s, mode='constant', cval=0.0)
            with np.errstate(invalid='ignore', divide='ignore'):
                M[yi, k] = np.where(den > 1e-6, num / den, np.nan)
    tot = S.sum(1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        Wk = np.where(tot > 1e-6, S / tot, np.nan)
    wbar = np.nanmean(Wk, axis=0)
    nbar = np.nanmean(M, axis=0)
    beta_within = np.zeros((H, W), np.float32)
    beta_struct = np.zeros((H, W), np.float32)
    for k in range(4):
        bk = slope(M[:, k])
        gk = slope(Wk[:, k])
        beta_within += np.where(np.isfinite(bk) & np.isfinite(wbar[k]), wbar[k] * bk, 0.0)
        beta_struct += np.where(np.isfinite(gk) & np.isfinite(nbar[k]), nbar[k] * gk, 0.0)
    valid = tot[0, 0] > 0.02
    return beta_within, beta_struct, valid


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    bounds, transform, crs, W, H = ndvi_profile()
    radius = max(1, int(round(WINDOW_KM / 1.0)))

    if os.path.exists(CACHE):
        z = np.load(CACHE)
        ndvi_year = z['ndvi_year']
        frac_anchor = z['frac_anchor']
        print('[缓存] 读取', CACHE)
    else:
        print('[1/3] 读取月 NDVI ...')
        ndvi_year, (H, W) = load_monthly_cube()
        np.save(os.path.join(OUT_DIR, '_tmp_ndvi.npy'), ndvi_year)
        print('[2/3] 聚合 30m 土地覆盖 ...')
        frac_anchor = np.zeros((6, 4, H, W), np.float32)
        for i, y in enumerate([2000, 2005, 2010, 2015, 2020, 2025]):
            frac_anchor[i] = lc_fractions(y, (H, W), transform, crs, bounds)
            print(f'  LC {y} 完成', flush=True)
        np.savez_compressed(CACHE, ndvi_year=ndvi_year, frac_anchor=frac_anchor)
        os.remove(os.path.join(OUT_DIR, '_tmp_ndvi.npy'))

    # 逐年的地类比例（分段锚定）
    anchors = [2000, 2005, 2010, 2015, 2020, 2025]
    a_index = {y: anchors.index(ANCHOR_RASTER.get(y, 2000)) for y in YEARS}
    frac_year = np.stack([frac_anchor[a_index[y]] for y in YEARS])

    print(f'[3/3] 局地分解（窗口半径 {radius} km）...')
    bw, bs, valid = local_decompose(ndvi_year, frac_year, radius)

    m = valid & np.isfinite(bw) & np.isfinite(bs)
    print('\n' + '=' * 60)
    print('局地分解诊断')
    print('=' * 60)
    print(f'  有效像元: {int(m.sum())} / {m.size} ({100*m.mean():.1f}%)')
    print(f'  面积加权 β_within = {np.nanmean(bw[m]):.5f} a⁻¹   [区域值 0.00280]')
    print(f'  面积加权 β_struct = {np.nanmean(bs[m]):.5f} a⁻¹   [区域值 0.00037]')
    print(f'  β_struct>0 的像元占比: {100*np.mean(bs[m] > 0):.1f}%')
    print(f'  β_within 分位: {np.nanpercentile(bw[m], [5, 25, 50, 75, 95]).round(5)}')
    print(f'  β_struct 分位: {np.nanpercentile(bs[m], [5, 25, 50, 75, 95]).round(5)}')

    np.savez_compressed(cfg.GRID_ATTRIBUTION,
                        beta_within=bw, beta_struct=bs, valid=m,
                        ndvi_year=ndvi_year, frac_year=frac_year)
    print('\n[已保存]', cfg.GRID_ATTRIBUTION)


if __name__ == '__main__':
    main()
