# -*- coding: utf-8 -*-
"""
第二层归因：把类型内变绿拆成气候可解释部分与非气候残差
=========================================================
逐像元：NDVI_p(y) ~ a + bT·T_p(y) + bP·P_p(y)

    β_clim,p = bT·slope(T_p) + bP·slope(P_p)
    β_res,p  = slope(NDVI_p) − β_clim,p

由于 OLS 斜率对样本均值是线性的，按地类聚合后：
    β_k = mean_p∈k(β_clim,p) + mean_p∈k(β_res,p)
    β_within = Σ_k w̄_k·β_k = β_climate + β_nonclimate
拆分是精确的、可加的。

气候：生长期（4–10 月）均温与累计降水。
数据：气温与降水栅格路径见 config.py（2000-01~2024-12）
NDVI/份额：输出_归因/网格归因结果.npz
输出：输出_归因/气候归因.npz + 控制台报告
运行：python code/04_climate_attribution.py
"""

import os, glob, re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

import config as cfg

T_DIR = cfg.TEMP_DIR
P_DIR = cfg.PRECIP_DIR
RES = cfg.GRID_ATTRIBUTION
OUT = cfg.CLIMATE_ATTRIBUTION
NDVI_REF = cfg.NDVI_REFERENCE
YEARS = list(range(2001, 2025))      # 气候与NDVI的共同期
GS = list(range(4, 11))              # 生长季 4–10 月


def parse(f):
    m = re.search(r'(\d{4})(\d{2})', os.path.basename(f))
    return int(m.group(1)), int(m.group(2))


def build_seasonal(folder, how):
    """逐月读入并重采样到NDVI网格，按生长季聚合 -> (years, H, W)"""
    files = {}
    for f in glob.glob(os.path.join(folder, '*.tif')):
        y, m = parse(f)
        files[(y, m)] = f
    with rasterio.open(NDVI_REF) as r:
        H, W = r.height, r.width
        dst_tr, dst_crs = r.transform, r.crs
    cube = np.full((len(YEARS), H, W), np.nan, np.float32)
    for i, y in enumerate(YEARS):
        acc = []
        for m in GS:
            f = files.get((y, m))
            if not f:
                continue
            with rasterio.open(f) as s:
                a = s.read(1).astype(np.float32)
                nod = s.nodata
                if nod is not None:
                    a[a == nod] = np.nan
                dst = np.full((H, W), np.nan, np.float32)
                reproject(a, dst, src_transform=s.transform, src_crs=s.crs,
                          dst_transform=dst_tr, dst_crs=dst_crs,
                          resampling=Resampling.bilinear,
                          src_nodata=np.nan, dst_nodata=np.nan)
            acc.append(dst)
        arr = np.stack(acc)
        cube[i] = np.nanmean(arr, 0) if how == 'mean' else np.nansum(arr, 0)
        print(f'  {os.path.basename(folder)} {y} 完成', flush=True)
    return cube


def ols3(Y, T, P):
    """逐像元 Y ~ a + bT·T + bP·P，返回 bT,bP,R2,p,n"""
    Y = Y.astype(np.float64); T = T.astype(np.float64); P = P.astype(np.float64)
    v = np.isfinite(Y) & np.isfinite(T) & np.isfinite(P)
    n = v.sum(0)
    yy = np.where(v, Y, 0.0); tt = np.where(v, T, 0.0); pp = np.where(v, P, 0.0)
    s1 = n
    sT = tt.sum(0); sP = pp.sum(0); sY = yy.sum(0)
    sTT = (tt*tt).sum(0); sPP = (pp*pp).sum(0); sTP = (tt*pp).sum(0)
    sTY = (tt*yy).sum(0); sPY = (pp*yy).sum(0)
    M = np.empty(n.shape + (3, 3))
    M[..., 0, 0] = s1; M[..., 0, 1] = sT; M[..., 0, 2] = sP
    M[..., 1, 0] = sT; M[..., 1, 1] = sTT; M[..., 1, 2] = sTP
    M[..., 2, 0] = sP; M[..., 2, 1] = sTP; M[..., 2, 2] = sPP
    rhs = np.stack([sY, sTY, sPY], -1)
    ok = n >= 12
    beta = np.full(n.shape + (3,), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        b = np.linalg.solve(M[ok], rhs[ok][..., None])[..., 0]
    beta[ok] = b
    pred = beta[..., 0][None] + beta[..., 1][None]*T + beta[..., 2][None]*P
    with np.errstate(invalid='ignore'):
        sse = (v*(Y - pred)**2).sum(0)
        ybar = np.where(n > 0, sY/np.maximum(n, 1), np.nan)
        sst = (v*(Y - ybar)**2).sum(0)
        r2 = 1 - sse/np.maximum(sst, 1e-12)
    return beta[..., 0], beta[..., 1], beta[..., 2], r2, n


def slope(a, years):
    """沿时间轴OLS斜率"""
    t = np.asarray(years, float)[:, None, None]
    v = np.isfinite(a)
    x = np.where(v, a, 0.0); n = v.sum(0)
    mt = (t*v).sum(0)/np.maximum(n, 1); mx = x.sum(0)/np.maximum(n, 1)
    dy = (t-mt)*v
    with np.errstate(invalid='ignore', divide='ignore'):
        return (dy*(x-mx)).sum(0)/ (dy*dy).sum(0)


def main():
    print('[1/3] 读取并聚合气候 ...')
    cache = cfg.CLIMATE_CACHE
    if os.path.exists(cache):
        cc = np.load(cache); T, P = cc['T'], cc['P']
        print('  [缓存] 读取', cache)
    else:
        T = build_seasonal(T_DIR, 'mean')
        P = build_seasonal(P_DIR, 'sum')
        np.savez_compressed(cache, T=T, P=P)

    z = np.load(RES)
    nd_all = z['ndvi_year']            # 2001-2025
    fy_all = z['frac_year']
    nd = nd_all[:len(YEARS)]
    fy = fy_all[:len(YEARS)]
    study = np.isfinite(nd).sum(0) >= 18

    print('[2/3] 逐像元回归 ...')
    A, bT, bP, r2, nval = ols3(nd, T, P)
    sT = slope(T, YEARS); sP = slope(P, YEARS); sY = slope(nd, YEARS)
    beta_clim = bT*sT + bP*sP
    beta_res = sY - beta_clim

    print('[3/3] 按地类聚合 ...')
    pred = A[None] + bT[None]*T + bP[None]*P
    resid = nd - pred
    fbar = np.nanmean(fy, 0)
    fbar_s = fbar * study
    wbar = fbar_s.sum((1, 2)) / fbar_s.sum()

    def wmean(fk, arr):
        return np.nansum(fk*np.nan_to_num(arr), axis=(1, 2)) / np.nansum(fk, axis=(1, 2))

    within_clim = within_res = within_total = 0.0
    for k, name in enumerate(['forest', 'grass', 'crop', 'impervious']):
        fk = np.where(study[None], fy[:, k], np.nan)
        b_tot = slope(wmean(fk, nd)[:, None, None], YEARS)[0, 0]
        b_clim = slope(wmean(fk, pred)[:, None, None], YEARS)[0, 0]
        b_res = slope(wmean(fk, resid)[:, None, None], YEARS)[0, 0]
        within_clim += wbar[k]*b_clim
        within_res += wbar[k]*b_res
        within_total += wbar[k]*b_tot
        print(f'  {name:10s} 合计 {b_tot:+.5f} = 气候 {b_clim:+.5f} + 非气候 {b_res:+.5f} a⁻¹')

    print()
    print('='*66)
    print(f'共同期 {YEARS[0]}–{YEARS[-1]}（气候数据止于2024）')
    print('='*66)
    print(f'  类型内合计 β_within = {within_total:+.5f} a⁻¹   [表1 全期 0.00280]')
    print(f'    气候可解释   = {within_clim:+.5f} a⁻¹  ({100*within_clim/within_total:.1f}%)')
    print(f'    非气候残差   = {within_res:+.5f} a⁻¹  ({100*within_res/within_total:.1f}%)')
    print(f'  按全期 0.00280 等比折算：气候 {0.00280*within_clim/within_total:+.5f}，'
          f'非气候 {0.00280*within_res/within_total:+.5f} a⁻¹')
    print(f'  气候回归 R² 中位数 = {np.nanmedian(r2[study]):.3f}')
    print(f'  R²>0.3 的像元占比  = {100*np.mean(r2[study] > 0.3):.1f}%')

    # 口径校验：由类别均值序列重算 β_within
    m = np.zeros((len(YEARS), 4))
    w = np.zeros((len(YEARS), 4))
    for k in range(4):
        fk = np.where(study[None], fy[:, k], np.nan)
        num = np.nansum(fk*np.nan_to_num(nd), axis=(1, 2))
        den = np.nansum(fk, axis=(1, 2))
        m[:, k] = num/den
        w[:, k] = den
    w = w/w.sum(1, keepdims=True)
    betak = slope(m[:, :, None], YEARS)[:, 0]
    within_ref = float((np.nanmean(w, 0)*betak).sum())
    print(f'  校验：类别序列重算 β_within = {within_ref:+.5f} a⁻¹（与上表差 {within_total-within_ref:+.1e}）')

    np.savez_compressed(OUT, beta_clim=beta_clim, beta_res=beta_res,
                        r2=r2, nval=nval, study=study, T=T, P=P)
    print('\n[已保存]', OUT)


if __name__ == '__main__':
    main()
