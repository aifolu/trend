# -*- coding: utf-8 -*-
"""
Counterfactual land-use compositions (Table 3).

Class-level greenness is held at its observed 2025 value while the area
composition is replaced by each alternative. Because area shares enter the
identity explicitly, no model has to be retrained: every scenario is a direct
arithmetic substitution.

Scenarios, expressed as a fraction of the total study area:

    S0  observed 2025 composition
    S1  3% grassland  -> forest
    S2  5% grassland  -> forest
    S3  5% forest     -> grassland
    S4  5% cropland   -> forest
    S5  2% cropland   -> impervious
"""

import numpy as np
import pandas as pd

import config as cfg


def load_observed_2025():
    """Return class-mean NDVI and area shares for 2025."""
    series = pd.read_csv(cfg.CLASS_SERIES, encoding='utf-8-sig')
    series['year'] = series['date'].str[:4].astype(int)
    ndvi = series[series['year'] == 2025].groupby('type')['mean'].mean()

    shares = pd.read_csv(cfg.ANCHOR_SHARES).set_index('year')
    w = shares.loc[2025, cfg.CLASSES].astype(float)
    w = w / w.sum()

    return np.array([ndvi[k] for k in cfg.CLASSES]), w.values


def regional_mean(ndvi, w):
    """Area-weighted regional mean NDVI."""
    return float((ndvi * w).sum())


def shift_area(w, src, dst, fraction):
    """Move ``fraction`` of the total area from class ``src`` to class ``dst``."""
    w = w.copy()
    w[cfg.CLASSES.index(src)] -= fraction
    w[cfg.CLASSES.index(dst)] += fraction
    return w


SCENARIOS = [
    ('S0', None, None, 0.00, 'observed 2025 composition'),
    ('S1', 'grass', 'forest', 0.03, '3% grassland -> forest'),
    ('S2', 'grass', 'forest', 0.05, '5% grassland -> forest'),
    ('S3', 'forest', 'grass', 0.05, '5% forest -> grassland'),
    ('S4', 'crop', 'forest', 0.05, '5% cropland -> forest'),
    ('S5', 'crop', 'impervious', 0.02, '2% cropland -> impervious'),
]


def main():
    ndvi, w0 = load_observed_2025()
    base = regional_mean(ndvi, w0)

    print('Class-mean NDVI in 2025')
    for name, value in zip(cfg.CLASSES, ndvi):
        print(f'  {name:<11s} {value:.4f}')
    print(f'  regional mean  {base:.4f}\n')

    rows = []
    for name, src, dst, fraction, label in SCENARIOS:
        w = w0 if src is None else shift_area(w0, src, dst, fraction)
        delta = regional_mean(ndvi, w) - base
        rows.append({'scenario': name, 'composition_change': label,
                     'delta_regional_mean_ndvi': delta})
        print(f'  {name}  {label:<32s} {delta:+.4f}')

    out = cfg.output('counterfactual_scenarios.csv')
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print(f'\n[wrote] {out}')


if __name__ == '__main__':
    main()
