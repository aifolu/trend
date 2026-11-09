# -*- coding: utf-8 -*-
"""
气候归因的三组敏感性 + 空间制图 + 分期拆分
输出目录: outputs/
运行: python code/05_climate_sensitivity.py
"""
import os, glob, re
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

import config as cfg

T_DIR = cfg.TEMP_DIR
P_DIR = cfg.PRECIP_DIR
NDVI_REF = cfg.NDVI_REFERENCE
RES = cfg.GRID_ATTRIBUTION
CACHE_GS = cfg.CLIMATE_CACHE
OUT = cfg.ensure_output_dir()
YEARS = list(range(2001, 2025))
GS = list(range(4, 11))
os.makedirs(OUT, exist_ok=True)


def parse(f):
    m = re.search(r'(\d{4})(\d{2})', os.path.basename(f))
    return int(m.group(1)), int(m.group(2))


def build_cube(folder, months, how, tag):
    cache = os.path.join(OUT, f'climate_{tag}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        print(f'  [缓存] {tag}')
        return z['A']
    files = {}
    for f in glob.glob(os.path.join(folder, '*.tif')):
        y, m = parse(f)
        files[(y, m)] = f
    with rasterio.open(NDVI_REF) as r:
        H, W, tr, crs = r.height, r.width, r.transform, r.crs
    cube = np.full((len(YEARS), H, W), np.nan, np.float32)
    for i, y in enumerate(YEARS):
        acc = []
        for m in months:
            f = files.get((y, m))
            if not f:
                continue
            with rasterio.open(f) as s:
                a = s.read(1).astype(np.float32)
                if s.nodata is not None:
                    a[a == s.nodata] = np.nan
                dst = np.full((H, W), np.nan, np.float32)
                reproject(a, dst, src_transform=s.transform, src_crs=s.crs,
                          dst_transform=tr, dst_crs=crs,
                          resampling=Resampling.bilinear,
                          src_nodata=np.nan, dst_nodata=np.nan)
            acc.append(dst)
        arr = np.stack(acc)
        cube[i] = np.nanmean(arr, 0) if how == 'mean' else np.nansum(arr, 0)
        print(f'  {tag} {y}', flush=True)
    np.savez_compressed(cache, A=cube)
    return cube


def slope(a):
    t = np.asarray(YEARS[:a.shape[0]], float)[:, None, None]
    v = np.isfinite(a); x = np.where(v, a, 0.0); n = v.sum(0)
    mt = (t * v).sum(0) / np.maximum(n, 1); mx = x.sum(0) / np.maximum(n, 1)
    dy = (t - mt) * v
    with np.errstate(invalid='ignore', divide='ignore'):
        return (dy * (x - mx)).sum(0) / (dy * dy).sum(0)


def slope_years(a, yrs):
    t = np.asarray(yrs, float)[:, None, None]
    v = np.isfinite(a); x = np.where(v, a, 0.0); n = v.sum(0)
    mt = (t * v).sum(0) / np.maximum(n, 1); mx = x.sum(0) / np.maximum(n, 1)
    dy = (t - mt) * v
    with np.errstate(invalid='ignore', divide='ignore'):
        return (dy * (x - mx)).sum(0) / (dy * dy).sum(0)


def ols2(Y, T, P):
    Y = Y.astype(np.float64); T = T.astype(np.float64); P = P.astype(np.float64)
    v = np.isfinite(Y) & np.isfinite(T) & np.isfinite(P)
    n = v.sum(0)
    yy = np.where(v, Y, 0.); tt = np.where(v, T, 0.); pp = np.where(v, P, 0.)
    s1 = n; sT = tt.sum(0); sP = pp.sum(0); sY = yy.sum(0)
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
        sse = (v*(Y-pred)**2).sum(0)
        ybar = sY/np.maximum(n, 1)
        sst = (v*(Y-ybar)**2).sum(0)
        r2 = np.clip(1 - sse/np.maximum(sst, 1e-12), 0, 1)
    from scipy.stats import f as fdist
    F = (r2/2)/((1-r2)/np.maximum(n-3, 1))
    p = fdist.sf(F, 2, np.maximum(n-3, 1))
    return beta[..., 0], beta[..., 1], beta[..., 2], r2, p, n


def wmean(fk, arr):
    return np.nansum(fk*np.nan_to_num(arr), axis=(1, 2)) / np.nansum(fk, axis=(1, 2))


def run_case(name, Tc, Pc, nd, fy, study, years, subset=None):
    m = len(years)
    ny = nd[:m]
    yidx = [YEARS.index(y) for y in years]
    Tc = Tc[yidx]; Pc = Pc[yidx]
    A, bT, bP, r2, p, n = ols2(ny, Tc, Pc)
    pred = A[None] + bT[None]*Tc + bP[None]*Pc
    resid = ny - pred
    fbar = np.nanmean(fy[:m], 0)
    extra = study if subset is None else (study & subset)
    fbar_s = fbar * extra
    wbar = fbar_s.sum((1, 2)) / fbar_s.sum()
    out = {'case': name, 'years': f'{years[0]}-{years[-1]}',
           'sig_frac': 100*np.mean((p < 0.05)[extra]) if subset is not None else np.nan}
    tot = cl = rs = 0.0
    for k, nm in enumerate(['forest', 'grass', 'crop', 'impervious']):
        fk = np.where(extra[None], fy[:m, k], np.nan)
        b_tot = slope_years(wmean(fk, ny)[:, None, None], years)[0, 0]
        b_cl = slope_years(wmean(fk, pred)[:, None, None], years)[0, 0]
        b_rs = slope_years(wmean(fk, resid)[:, None, None], years)[0, 0]
        out[f'{nm}_tot'] = b_tot; out[f'{nm}_clim'] = b_cl; out[f'{nm}_res'] = b_rs
        tot += wbar[k]*b_tot; cl += wbar[k]*b_cl; rs += wbar[k]*b_rs
    out.update(dict(within_tot=tot, climate=cl, nonclimate=rs,
                    climate_pct=100*cl/tot if tot else np.nan,
                    nonclimate_pct=100*rs/tot if tot else np.nan))
    return out, pred, resid, extra


def main():
    print('[1/4] 准备气候序列 ...')
    z = np.load(RES)
    nd_all = z['ndvi_year']; fy_all = z['frac_year']
    m = len(YEARS)
    nd = nd_all[:m]; fy = fy_all[:m]
    study = np.isfinite(nd).sum(0) >= 18

    if os.path.exists(CACHE_GS):
        cc = np.load(CACHE_GS); T_gs, P_gs = cc['T'], cc['P']
        print('  [缓存] 生长期')
    else:
        T_gs = build_cube(T_DIR, GS, 'mean', 'gsT')
        P_gs = build_cube(P_DIR, GS, 'sum', 'gsP')
    T_an = build_cube(T_DIR, range(1, 13), 'mean', 'anT')
    P_an = build_cube(P_DIR, range(1, 13), 'sum', 'anP')

    print('[2/4] 三组敏感性 ...')
    main_res, pred_m, resid_m, _ = run_case('主结果：生长期 T+P，全像元', T_gs, P_gs, nd, fy, study, YEARS)
    _, _, _, r2m, pm, _ = ols2(nd, T_gs, P_gs)
    resA, *_ = run_case('敏感性A：年均 T+P', T_an, P_an, nd, fy, study, YEARS)
    resB, *_ = run_case('敏感性B：仅 p<0.05 显著像元', T_gs, P_gs, nd, fy, study, YEARS,
                        subset=(pm < 0.05))
    yrsC = list(range(2001, 2025))
    T_lag = np.concatenate([T_gs[:1], T_gs[:-1]], 0)
    P_lag = np.concatenate([P_gs[:1], P_gs[:-1]], 0)
    resC, *_ = run_case('敏感性C：滞后一年气候', T_lag, P_lag, nd, fy, study, YEARS)

    tab = pd.DataFrame([main_res, resA, resB, resC])
    cols = ['case', 'years', 'within_tot', 'climate', 'nonclimate',
            'climate_pct', 'nonclimate_pct', 'sig_frac']
    tab = tab[cols].round(6)
    tab.to_csv(os.path.join(OUT, cfg.CLIMATE_SENSITIVITY), index=False, encoding='utf-8-sig')
    print(tab.to_string(index=False))

    print('\n[3/4] 分期气候拆分 ...')
    periods = [(2001, 2005), (2005, 2010), (2010, 2015), (2015, 2020), (2020, 2024)]
    fbar = np.nanmean(fy, 0); fbar_s = fbar * study
    wbar = fbar_s.sum((1, 2)) / fbar_s.sum()
    rows = []
    for a, b in periods:
        yrs = [y for y in YEARS if a <= y <= b]
        idx = [YEARS.index(y) for y in yrs]
        tot = cl = rs = 0.0
        for k in range(4):
            fk = np.where(study[None], fy[idx, k], np.nan)
            tot += wbar[k]*slope_years(wmean(fk, nd[idx])[:, None, None], yrs)[0, 0]
            cl += wbar[k]*slope_years(wmean(fk, pred_m[idx])[:, None, None], yrs)[0, 0]
            rs += wbar[k]*slope_years(wmean(fk, resid_m[idx])[:, None, None], yrs)[0, 0]
        rows.append(dict(period=f'{a}-{b}', within=tot, climate=cl, nonclimate=rs,
                         climate_pct=100*cl/tot if tot else np.nan))
    per = pd.DataFrame(rows).round(6)
    per.to_csv(os.path.join(OUT, cfg.PERIOD_CLIMATE_SPLIT), index=False, encoding='utf-8-sig')
    print(per.to_string(index=False))

    print('\n[4/4] 制图 ...')
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    mpl.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'axes.linewidth': 0.7,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    A, bT, bP, r2, p, n = ols2(nd, T_gs, P_gs)
    sT = slope(T_gs); sP = slope(P_gs)
    b_clim = (bT*sT + bP*sP)*1e3
    b_res = (slope(nd) - bT*sT - bP*sP)*1e3
    sh = np.abs(b_clim)/(np.abs(b_clim)+np.abs(b_res))*100
    msk = study & np.isfinite(b_clim) & np.isfinite(b_res)
    b_clim = np.where(msk, b_clim, np.nan); b_res = np.where(msk, b_res, np.nan)
    sh = np.where(msk, sh, np.nan)
    CB = LinearSegmentedColormap.from_list('b', ['#F4F1EC', '#7FA3B8', '#2F5F7A'])
    CO = LinearSegmentedColormap.from_list('o', ['#F4F1EC', '#E0B48F', '#A9552F'])
    fig, axs = plt.subplots(1, 3, figsize=(7.09, 2.25))
    lim1 = np.nanpercentile(np.abs(b_clim), 98)
    lim2 = np.nanpercentile(np.abs(b_res), 98)
    specs = [(axs[0], b_clim, CB, Normalize(-lim1, lim1), 'Climate-explained (×10$^{-3}$ a$^{-1}$)', 'a'),
             (axs[1], b_res, CO, Normalize(-lim2, lim2), 'Non-climatic residual (×10$^{-3}$ a$^{-1}$)', 'b'),
             (axs[2], sh, CO, Normalize(0, 100), 'Climate share of within-type (%)', 'c')]
    for ax, dat, cm, nm, ttl, lab in specs:
        im = ax.imshow(dat, cmap=cm, norm=nm, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(ttl, fontsize=6.8, pad=3)
        ax.text(0.005, 1.06, lab, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='right')
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=5.8, width=0.5, length=1.8); cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.02, right=0.96, top=0.85, bottom=0.02, wspace=0.28)
    fig.savefig(os.path.join(OUT, 'Fig_climate_attribution.pdf'))
    fig.savefig(os.path.join(OUT, 'Fig_climate_attribution.png'), dpi=600)
    np.savez_compressed(os.path.join(OUT, cfg.output('climate_attribution_per_pixel.npz')),
                        beta_clim=b_clim/1e3, beta_res=b_res/1e3, share=sh,
                        r2=r2, p=p, study=study)
    print(f'[已保存] {OUT}')


if __name__ == '__main__':
    main()
